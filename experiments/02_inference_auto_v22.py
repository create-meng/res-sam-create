"""
Res-SAM v22 - Step 2: fully automatic inference

V22 说明：
- 基于 V21 仅方案1（当前最优：F1=0.321, Precision=0.346）
- 测试多尺度分数图优化

环境变量：
- USE_MULTI_SCALE: 是否使用多尺度分数图 (0/1, 默认0)
- MULTI_SCALE_STRIDES: 多尺度stride，逗号分隔 (默认"3,5,8")
- MULTI_SCALE_WEIGHTS: 多尺度权重，逗号分隔 (默认"0.3,0.4,0.3")
- OUTPUT_SUFFIX: 输出文件后缀 (默认"")
- MAX_IMAGES_PER_CATEGORY: 每类最大图像数 (默认None，处理全部)

固定配置（V21基线）：
- top_k_per_image = 1
- nms_iou_threshold = 0.3
- min_bbox_area = 5000
- hidden_size = 30
- background_removal_method = "both"
- 验证集驱动的 beta 校准

目标性能（基于V21 仅方案1）：
- F1 = 0.321, Precision = 0.346, Recall = 0.300
- TP = 18, FP = 34, FN = 42, AUC = 0.824
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_02_03
from experiments.paper_constants import DEFAULT_BETA_THRESHOLD, preflight_faiss_or_raise
from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess
from PatchRes.logger import setup_global_logger, log_config, log_section, log_finish


def _to_abs(base_dir: str, path: str) -> str:
    if not path: return path
    if os.path.isabs(path): return path
    return os.path.abspath(os.path.join(base_dir, path))


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v22", "feature_bank_v22.pth"),
    "metadata_path":     os.path.join(BASE_DIR, "outputs", "feature_banks_v22", "metadata.json"),
    "test_data_dirs": {
        "cavities":   os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "utilities":  os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
        "normal_auc": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "annotation_dirs": {
        "cavities":  os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities",
                                  "annotations", "VOC_XML_format"),
        "utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities",
                                  "annotations", "VOC_XML_format"),
    },
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions_v22"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v22"),
    "output_suffix": os.getenv("OUTPUT_SUFFIX", ""),
    
    # 核心参数
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,
    
    # 全图 patch 分数图
    "use_score_map": True,
    "score_map_smooth_sigma": 2.0,
    
    # V22 固定配置（V21基线）
    "min_bbox_area": 5000,
    "max_bbox_area": 80000,
    "bbox_expand_pixels": 15,
    "nms_iou_threshold": 0.3,
    "top_k_per_image": 1,
    
    # V22 测试：多尺度分数图
    "use_multi_scale": int(os.getenv("USE_MULTI_SCALE", "0")),
    "multi_scale_strides": [int(x) for x in os.getenv("MULTI_SCALE_STRIDES", "3,5,8").split(",")],
    "multi_scale_weights": [float(x) for x in os.getenv("MULTI_SCALE_WEIGHTS", "0.3,0.4,0.3").split(",")],
    
    # 自适应阈值
    "use_per_image_threshold": True,
    "adaptive_threshold_strategy": "dynamic",
    "per_image_threshold_ratio": 0.7,
    
    # 形状和分数联合过滤
    "aspect_ratio_threshold": 2.9,
    "score_threshold_p80": 0.3283,
    
    # GPR 背景去除
    "gpr_background_removal": True,
    "background_removal_method": "both",
    
    # 其他
    "device": "auto",
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "max_images_per_category": int(os.getenv("MAX_IMAGES_PER_CATEGORY")) if os.getenv("MAX_IMAGES_PER_CATEGORY") else None,
    "checkpoint_interval": 50,
    "random_seed": 11,
    "version": "v22",
    "feature_with_bias": True,
}


def remove_gpr_background(arr: np.ndarray, method: str = "both") -> np.ndarray:
    """GPR B-scan 背景去除（V22：移除局部对比度增强）"""
    result = arr.copy().astype(np.float32)
    
    # 行列均值去除
    if method in ("row_mean", "both"):
        result = result - result.mean(axis=1, keepdims=True)
    if method in ("col_mean", "both"):
        result = result - result.mean(axis=0, keepdims=True)
    if method == "row_median":
        result = arr - np.median(arr, axis=1, keepdims=True)
    
    # 全局标准化
    std = result.std()
    if std > 1e-8:
        result = result / std
    
    return result.astype(np.float32)


def parse_voc_xml(xml_path: str) -> dict | None:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        bboxes = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            bboxes.append([int(bbox.find("xmin").text), int(bbox.find("ymin").text),
                           int(bbox.find("xmax").text), int(bbox.find("ymax").text)])
        return {"width": width, "height": height, "bboxes": bboxes}
    except Exception as exc:
        print(f"解析 XML 失败：{xml_path} | {exc}")
        return None


def load_image_with_orig_size(path: str, size_hw: tuple[int, int] | None,
                               config: dict) -> tuple[int, int, np.ndarray]:
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        img = im.convert("L")
        if size_hw:
            img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        if config.get("gpr_background_removal", False):
            arr = remove_gpr_background(arr, config.get("background_removal_method", "both"))
    return orig_w, orig_h, arr


def compute_iou(box1, box2):
    """计算两个框的 IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def generate_score_map_single_scale(image_shape: tuple, patch_positions: list, 
                                    scores: np.ndarray, window_size: int) -> np.ndarray:
    """生成单尺度分数图"""
    img_h, img_w = image_shape
    score_map = np.zeros((img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.int32)
    
    half_win = window_size // 2
    
    for (cx, cy), score in zip(patch_positions, scores):
        x1 = max(0, cx - half_win)
        y1 = max(0, cy - half_win)
        x2 = min(img_w, cx + half_win)
        y2 = min(img_h, cy + half_win)
        
        score_map[y1:y2, x1:x2] += score
        count_map[y1:y2, x1:x2] += 1
    
    mask = count_map > 0
    score_map[mask] = score_map[mask] / count_map[mask]
    
    return score_map


def nms(bboxes, scores, iou_threshold=0.5):
    """计算两个框的 IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def nms(bboxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression
    
    Parameters:
    -----------
    bboxes : list of [x1, y1, x2, y2]
    scores : list of float
    iou_threshold : float
    
    Returns:
    --------
    list of int : 保留的索引
    """
    if len(bboxes) == 0:
        return []
    
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]
        
        indices = [
            i for i in indices
            if compute_iou(bboxes[current], bboxes[i]) <= iou_threshold
        ]
    
    return keep



def detect_with_score_map(model, image: np.ndarray, config: dict, 
                          logger: logging.Logger, img_name: str) -> dict:
    """
    V22 核心：使用全图 patch 分数图进行异常检测（支持多尺度）
    
    流程：
    1. 滑窗提取所有 patch（支持多尺度）
    2. ESN 特征提取
    3. Feature Bank 比较得到 patch 分数
    4. 生成全图分数图（支持多尺度融合）
    5. 高斯平滑
    6. 自适应阈值
    7. 阈值化
    8. 连通分量分析
    9. V22 环境变量控制：优化的 FP 抑制
    """
    img_h, img_w = image.shape
    window_size = config["window_size"]
    stride = config["stride"]
    beta = config["beta_threshold"]
    
    use_multi_scale = config.get("use_multi_scale", 0)
    
    logger.debug(f"  [V22] 开始生成全图 patch 分数图")
    logger.debug(f"  [V22] 多尺度: {use_multi_scale}")
    
    # 多尺度分数图生成
    if use_multi_scale:
        strides = config.get("multi_scale_strides", [3, 5, 8])
        weights = config.get("multi_scale_weights", [0.3, 0.4, 0.3])
        
        logger.debug(f"  [V22] 多尺度 strides: {strides}, weights: {weights}")
        
        score_maps = []
        all_scores_list = []
        original_stride = model.stride  # 保存原始 stride
        
        for stride_val in strides:
            # 临时修改 stride
            model.stride = stride_val
            
            # 提取 patch
            patches = model._extract_patches(image)
            if patches.shape[0] == 0:
                continue
            
            # 计算 patch 中心位置
            half_win = window_size // 2
            patch_positions = []
            for y in range(half_win, img_h - half_win, stride_val):
                for x in range(half_win, img_w - half_win, stride_val):
                    patch_positions.append((x, y))
            
            # ESN 特征提取
            features = model._fit_patches(patches)
            features_np = features.detach().cpu().numpy()
            
            # Feature Bank 比较
            scores = model._score_features_against_bank(features_np)
            all_scores_list.extend(scores.tolist())
            
            # 生成分数图
            score_map = generate_score_map_single_scale(
                (img_h, img_w), patch_positions, scores, window_size
            )
            score_maps.append(score_map)
            
            logger.debug(f"  [V22] stride={stride_val}: {len(scores)} patches, "
                        f"max_score={np.max(scores):.4f}")
        
        # 恢复原始 stride
        model.stride = original_stride
        
        if len(score_maps) == 0:
            logger.debug(f"  [V22] 未提取到任何 patch")
            return {
                "pred_bboxes": [],
                "anomaly_scores": [],
                "num_patches": 0,
                "num_anomaly_patches": 0,
                "num_connected_components": 0,
                "max_patch_score": 0.0,
                "mean_patch_score": 0.0,
            }
        
        # 融合多尺度分数图（加权平均）
        score_map = np.zeros_like(score_maps[0])
        for w, sm in zip(weights[:len(score_maps)], score_maps):
            score_map += w * sm
        
        all_scores = np.array(all_scores_list)
        num_patches = len(all_scores)
        
    else:
        # 单尺度（原始逻辑）
        # Step 1: 滑窗提取所有 patch
        patches = model._extract_patches(image)
        if patches.shape[0] == 0:
            logger.debug(f"  [V22] 未提取到任何 patch")
            return {
                "pred_bboxes": [],
                "anomaly_scores": [],
                "num_patches": 0,
                "num_anomaly_patches": 0,
                "num_connected_components": 0,
                "max_patch_score": 0.0,
                "mean_patch_score": 0.0,
            }
        
        # 计算 patch 中心位置
        half_win = window_size // 2
        patch_positions = []
        for y in range(half_win, img_h - half_win, stride):
            for x in range(half_win, img_w - half_win, stride):
                patch_positions.append((x, y))
        
        num_patches = len(patch_positions)
        
        if num_patches != patches.shape[0]:
            logger.error(f"  [V22] Patch 数量不匹配: positions={num_patches}, patches={patches.shape[0]}")
            num_patches = patches.shape[0]
            if num_patches < len(patch_positions):
                patch_positions = patch_positions[:num_patches]
            else:
                logger.error(f"  [V22] 无法修复 patch 位置，跳过该图像")
                return {
                    "pred_bboxes": [],
                    "anomaly_scores": [],
                    "num_patches": 0,
                    "num_anomaly_patches": 0,
                    "num_connected_components": 0,
                    "max_patch_score": 0.0,
                    "mean_patch_score": 0.0,
                }
        
        logger.debug(f"  [V22] 提取 {num_patches} 个 patch")
        
        # Step 2: ESN 特征提取
        features = model._fit_patches(patches)
        features_np = features.detach().cpu().numpy()
        
        # Step 3: Feature Bank 比较得到 patch 分数
        all_scores = model._score_features_against_bank(features_np)
        
        # Step 4: 生成全图分数图
        score_map = generate_score_map_single_scale(
            (img_h, img_w), patch_positions, all_scores, window_size
        )
    
    max_score = float(np.max(all_scores))
    mean_score = float(np.mean(all_scores))
    std_score = float(np.std(all_scores))
    logger.debug(f"  [V22] Patch 分数: max={max_score:.4f}, mean={mean_score:.4f}, std={std_score:.4f}")
    
    # Step 5: 高斯平滑
    sigma = config.get("score_map_smooth_sigma", 2.0)
    if sigma > 0:
        score_map = ndimage.gaussian_filter(score_map, sigma=sigma)
        logger.debug(f"  [V22] 高斯平滑 (sigma={sigma})")
    
    # Step 6: 自适应阈值
    use_per_image_threshold = config.get("use_per_image_threshold", True)
    if use_per_image_threshold:
        adaptive_strategy = config.get("adaptive_threshold_strategy", "dynamic")
        
        if max_score < beta:
            logger.debug(f"  [V22] 最高分 {max_score:.4f} < beta {beta:.4f}，判定为正常图")
            return {
                "pred_bboxes": [],
                "anomaly_scores": [],
                "num_patches": num_patches,
                "num_anomaly_patches": 0,
                "num_connected_components": 0,
                "max_patch_score": max_score,
                "mean_patch_score": mean_score,
            }
        
        if adaptive_strategy == "dynamic":
            score_range = max_score - mean_score
            
            if std_score < 0.1:
                adaptive_ratio = 0.8
                logger.debug(f"  [V22] 分数分布集中 (std={std_score:.4f})，使用严格阈值比例 {adaptive_ratio}")
            elif score_range > 0.5:
                adaptive_ratio = 0.6
                logger.debug(f"  [V22] 分数范围大 (range={score_range:.4f})，使用宽松阈值比例 {adaptive_ratio}")
            else:
                adaptive_ratio = config.get("per_image_threshold_ratio", 0.7)
                logger.debug(f"  [V22] 使用默认阈值比例 {adaptive_ratio}")
        else:
            adaptive_ratio = config.get("per_image_threshold_ratio", 0.7)
            logger.debug(f"  [V22] 使用固定阈值比例 {adaptive_ratio}")
        
        adaptive_threshold = max_score * adaptive_ratio
        logger.debug(f"  [V22] 自适应阈值: {adaptive_threshold:.4f} (max_score * {adaptive_ratio:.2f})")
    else:
        adaptive_threshold = beta
        logger.debug(f"  [V22] 使用全局阈值: {beta:.4f}")
    
    # Step 7: 阈值化
    binary_map = (score_map > adaptive_threshold).astype(np.uint8)
    num_anomaly_pixels = int(np.sum(binary_map))
    logger.debug(f"  [V22] 阈值化后异常像素数: {num_anomaly_pixels}")
    
    if num_anomaly_pixels == 0:
        logger.debug(f"  [V22] 无异常像素")
        return {
            "pred_bboxes": [],
            "anomaly_scores": [],
            "num_patches": num_patches,
            "num_anomaly_patches": 0,
            "num_connected_components": 0,
            "max_patch_score": max_score,
            "mean_patch_score": mean_score,
        }
    
    # Step 8: 连通分量分析
    labeled_map, num_components = ndimage.label(binary_map)
    logger.debug(f"  [V22] 连通分量数: {num_components}")
    
    # Step 9: 提取每个连通分量的 bbox
    pred_bboxes = []
    pred_scores = []
    
    min_area = config.get("min_bbox_area", 5000)
    max_area = config.get("max_bbox_area", 80000)
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled_map == comp_id)
        
        rows, cols = np.where(comp_mask)
        if len(rows) == 0:
            continue
        
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1
        
        # bbox 膨胀
        expand_px = config.get("bbox_expand_pixels", 15)
        if expand_px > 0:
            x1 = max(0, x1 - expand_px)
            y1 = max(0, y1 - expand_px)
            x2 = min(img_w, x2 + expand_px)
            y2 = min(img_h, y2 + expand_px)
        
        # V22 环境变量控制：尺寸过滤
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            logger.debug(f"  [V22] 过滤连通分量 {comp_id}: area={area} (min={min_area}, max={max_area})")
            continue
        
        region_score = float(score_map[comp_mask].mean())
        
        pred_bboxes.append([x1, y1, x2, y2])
        pred_scores.append(region_score)
        
        logger.debug(f"  [V22] 连通分量 {comp_id}: bbox=[{x1},{y1},{x2},{y2}], area={area}, score={region_score:.4f}")
    
    logger.debug(f"  [V22] 尺寸过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 10: V22 环境变量控制 - NMS
    if len(pred_bboxes) > 1:
        nms_threshold = config.get("nms_iou_threshold", 0.3)
        keep_indices = nms(pred_bboxes, pred_scores, nms_threshold)
        pred_bboxes = [pred_bboxes[i] for i in keep_indices]
        pred_scores = [pred_scores[i] for i in keep_indices]
        logger.debug(f"  [V22] NMS (threshold={nms_threshold}) 后剩余 {len(pred_bboxes)} 个框")
    
    # Step 11: V22 环境变量控制 - Top-K 过滤
    top_k = config.get("top_k_per_image", 1)
    if len(pred_bboxes) > top_k:
        sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        pred_bboxes = [pred_bboxes[i] for i in sorted_indices[:top_k]]
        pred_scores = [pred_scores[i] for i in sorted_indices[:top_k]]
        logger.debug(f"  [V22] Top-{top_k} 过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 12: 形状和分数联合过滤
    if len(pred_bboxes) > 0:
        aspect_ratio_threshold = config.get("aspect_ratio_threshold", 2.9)
        score_threshold_p80 = config.get("score_threshold_p80", 0.3283)
        
        filtered_bboxes = []
        filtered_scores = []
        
        for bbox, score in zip(pred_bboxes, pred_scores):
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
            
            if aspect_ratio < aspect_ratio_threshold and score >= score_threshold_p80:
                filtered_bboxes.append(bbox)
                filtered_scores.append(score)
            else:
                logger.debug(f"  [V22] 过滤框: bbox=[{x1},{y1},{x2},{y2}], aspect={aspect_ratio:.2f}, score={score:.4f}")
        
        num_filtered = len(pred_bboxes) - len(filtered_bboxes)
        if num_filtered > 0:
            logger.debug(f"  [V22] 形状+分数过滤: 移除 {num_filtered} 个框，剩余 {len(filtered_bboxes)} 个")
        
        pred_bboxes = filtered_bboxes
        pred_scores = filtered_scores
    
    num_anomaly_patches = int(np.sum(all_scores > beta))
    
    return {
        "pred_bboxes": pred_bboxes,
        "anomaly_scores": pred_scores,
        "num_patches": num_patches,
        "num_anomaly_patches": num_anomaly_patches,
        "num_connected_components": num_components,
        "max_patch_score": max_score,
        "mean_patch_score": mean_score,
    }



def run_inference(config: dict) -> dict:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    logger = setup_global_logger(base_dir, "02_inference_auto_v22")
    
    config = dict(config)
    for key in ["feature_bank_path", "metadata_path", "output_dir", "checkpoint_dir"]:
        config[key] = _to_abs(base_dir, config.get(key, ""))
    config["test_data_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("test_data_dirs", {}).items()}
    config["annotation_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("annotation_dirs", {}).items()}

    # 从 metadata 读取自适应 beta
    effective_beta = config["beta_threshold"]
    if config.get("use_adaptive_beta", True) and os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "adaptive_beta" in metadata:
            effective_beta = float(metadata["adaptive_beta"])
            logger.info(f"自适应 beta = {effective_beta:.4f}（从 metadata 读取）")
        else:
            logger.info(f"无 adaptive_beta，使用默认值 {effective_beta}")
    config["beta_threshold"] = effective_beta

    log_section("Res-SAM V22：固化最优配置（基于V21 仅方案1）", logger)
    log_config(config, logger)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(f"Feature bank not found: {config['feature_bank_path']}\n"
                                f"请先运行 01_build_feature_bank_v22.py")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    logger.info("初始化 ResSAM（V22）...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config["beta_threshold"]),
        device=config.get("device", "auto"),
        feature_with_bias=bool(config.get("feature_with_bias", True)),
    )

    logger.info(f"加载 Feature Bank：{config['feature_bank_path']}")
    model.load_feature_bank(config["feature_bank_path"])

    all_results: dict[str, list] = {}

    for category, data_dir in config["test_data_dirs"].items():
        log_section(f"处理类别：{category}", logger)
        
        if not os.path.isdir(data_dir):
            logger.warning(f"数据目录不存在：{data_dir}")
            continue

        image_files = sorted(f for f in os.listdir(data_dir)
                             if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if config["max_images_per_category"]:
            image_files = image_files[:int(config["max_images_per_category"])]
        logger.info(f"共找到 {len(image_files)} 张图像")

        checkpoint_file = os.path.join(config["checkpoint_dir"], f"checkpoint_auto_{category}{config.get('output_suffix', '')}.json")
        category_results: list[dict] = []
        start_idx = 0
        last_completed = 0
        category_success = 0
        category_failed = 0
        fail_reasons: dict[str, int] = {}

        # 读取断点
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            if checkpoint.get("completed", False):
                logger.info(f"Checkpoint completed, skip: {checkpoint_file}")
                all_results[category] = checkpoint.get("results", []) or []
                continue
            if isinstance(checkpoint.get("processed_count"), int):
                start_idx = int(checkpoint.get("processed_count", 0) or 0)
                category_results = checkpoint.get("results", []) or []
                last_completed = start_idx
                logger.info(f"从断点继续：start_idx={start_idx}")

        annotation_dir = config["annotation_dirs"].get(category, "")
        checkpoint_interval = int(config.get("checkpoint_interval", 50) or 0)
        effective_interval = 1 if image_files and len(image_files) <= checkpoint_interval else checkpoint_interval

        try:
            for i in tqdm(range(start_idx, len(image_files)), desc=f"处理 {category}", file=sys.stdout):
                img_file = image_files[i]
                img_path = os.path.join(data_dir, img_file)
                xml_name = os.path.splitext(img_file)[0] + ".xml"
                xml_path = os.path.join(annotation_dir, xml_name) if annotation_dir else ""
                gt = parse_voc_xml(xml_path) if xml_path and os.path.exists(xml_path) else None
                target_hw = target_hw_for_preprocess(config, gt)

                try:
                    orig_w, orig_h, img = load_image_with_orig_size(img_path, target_hw, config)
                    proc_h, proc_w = int(img.shape[0]), int(img.shape[1])

                    logger.debug(f"[{category}] 处理图像 {i+1}/{len(image_files)}: {img_file}")
                    logger.debug(f"  原始尺寸: {orig_w}×{orig_h}, 处理尺寸: {proc_w}×{proc_h}")

                    # V22 核心：全图 patch 分数图 + 固化的最优参数
                    result = detect_with_score_map(
                        model, img, config, logger, img_file
                    )

                    pred_bboxes_resized = result["pred_bboxes"]
                    anomaly_scores = result["anomaly_scores"]

                    logger.debug(f"  检测到 {len(pred_bboxes_resized)} 个异常区域")
                    if len(pred_bboxes_resized) > 0:
                        logger.debug(f"  分数范围: {min(anomaly_scores):.4f} - {max(anomaly_scores):.4f}")

                    # 缩放到原始尺寸
                    target_w = int(gt["width"]) if gt and gt.get("width") else int(orig_w)
                    target_h = int(gt["height"]) if gt and gt.get("height") else int(orig_h)
                    scale_x = target_w / float(proc_w)
                    scale_y = target_h / float(proc_h)
                    pred_bboxes = [[int(b[0] * scale_x), int(b[1] * scale_y),
                                    int(b[2] * scale_x), int(b[3] * scale_y)]
                                   for b in pred_bboxes_resized]

                    record = {
                        "image_name": img_file, "image_path": img_path,
                        "pred_bboxes_resized": pred_bboxes_resized,
                        "pred_bboxes": pred_bboxes,
                        "anomaly_scores": anomaly_scores,
                        "num_patches": result.get("num_patches", 0),
                        "num_anomaly_patches": result.get("num_anomaly_patches", 0),
                        "num_connected_components": result.get("num_connected_components", 0),
                        "max_patch_score": result.get("max_patch_score", 0.0),
                        "mean_patch_score": result.get("mean_patch_score", 0.0),
                    }

                    if gt:
                        inv_scale_x = proc_w / float(gt["width"])
                        inv_scale_y = proc_h / float(gt["height"])
                        gt_boxes_resized = [[int(b[0] * inv_scale_x), int(b[1] * inv_scale_y),
                                             int(b[2] * inv_scale_x), int(b[3] * inv_scale_y)]
                                            for b in gt["bboxes"]]
                        record.update({
                            "gt_bboxes": gt["bboxes"], "gt_bboxes_resized": gt_boxes_resized,
                            "num_gt": len(gt["bboxes"]), "gt_width": gt["width"],
                            "gt_height": gt["height"],
                            "exclude_from_det_metrics": False, "exclude_from_auc": False,
                        })
                    elif category == "normal_auc":
                        record.update({"gt_bboxes": [], "gt_bboxes_resized": [], "num_gt": 0,
                                       "exclude_from_det_metrics": True, "exclude_from_auc": False})
                    else:
                        record.update({"gt_bboxes": [], "gt_bboxes_resized": [], "num_gt": 0,
                                       "exclude_from_det_metrics": True, "exclude_from_auc": True})

                    category_results.append(record)
                    category_success += 1
                    last_completed = i + 1

                    # 保存断点
                    if effective_interval > 0 and last_completed % effective_interval == 0:
                        with open(checkpoint_file, "w", encoding="utf-8") as f:
                            json.dump({"category": category,
                                       "processed_count": int(last_completed),
                                       "results": category_results,
                                       "timestamp": datetime.now().isoformat(),
                                       "completed": False}, f, ensure_ascii=False)

                except Exception as exc:
                    category_failed += 1
                    err_low = str(exc).lower()
                    reason = ("image_open_failed" if "cannot identify" in err_low else "unknown")
                    fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + 1
                    logger.error(f"处理图像失败：{img_file} | {exc}", exc_info=True)

        except KeyboardInterrupt:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"category": category, "processed_count": int(last_completed),
                           "results": category_results, "timestamp": datetime.now().isoformat(),
                           "completed": False}, f, ensure_ascii=False)
            raise

        all_results[category] = category_results
        logger.info(f"类别 {category} 完成，共 {len(category_results)} 张")
        
        # 保存完成标记
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"category": category, "processed_count": int(last_completed),
                       "results": category_results, "timestamp": datetime.now().isoformat(),
                       "completed": True,
                       "summary": {"total": len(image_files), "success": category_success,
                                   "failed": category_failed, "fail_reasons": fail_reasons}},
                      f, ensure_ascii=False)

    # 保存最终结果
    suffix = config.get("output_suffix", "")
    output_filename = f"auto_predictions_v22{suffix}.json" if suffix else "auto_predictions_v22.json"
    output_file = os.path.join(config["output_dir"], output_filename)
    output_data = {
        "meta": {
            "version": "v22",
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
            "hidden_size": int(config["hidden_size"]),
            "beta_threshold": float(config["beta_threshold"]),
            "use_score_map": bool(config.get("use_score_map", True)),
            "score_map_smooth_sigma": float(config.get("score_map_smooth_sigma", 2.0)),
            # V22 环境变量控制的配置
            "min_bbox_area": int(config.get("min_bbox_area", 5000)),
            "max_bbox_area": int(config.get("max_bbox_area", 80000)),
            "bbox_expand_pixels": int(config.get("bbox_expand_pixels", 15)),
            "nms_iou_threshold": float(config.get("nms_iou_threshold", 0.3)),
            "top_k_per_image": int(config.get("top_k_per_image", 1)),
            # 新增优化方案
            "use_multi_scale": int(config.get("use_multi_scale", 0)),
            "multi_scale_strides": config.get("multi_scale_strides", [3, 5, 8]),
            "multi_scale_weights": config.get("multi_scale_weights", [0.3, 0.4, 0.3]),
            # 自适应阈值
            "use_per_image_threshold": bool(config.get("use_per_image_threshold", True)),
            "per_image_threshold_ratio": float(config.get("per_image_threshold_ratio", 0.7)),
        },
        "results": all_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_images = sum(len(r) for r in all_results.values())
    total_detections = sum(len(r["pred_bboxes"]) for recs in all_results.values() for r in recs)
    
    logger.info(f"结果保存至：{output_file}")
    logger.info(f"处理图像数：{total_images}，检出框数：{total_detections}")
    
    log_finish("02_inference_auto_v22", logger)
    
    return all_results


if __name__ == "__main__":
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v22")

    # 根据环境变量决定 Feature Bank 路径
    use_rotation = int(os.getenv("USE_ROTATION_AUG", "0"))
    if use_rotation:
        rotation_angles = os.getenv("ROTATION_ANGLES", "90,180,270")
        angles_str = "_".join(rotation_angles.split(","))
        suffix = f"_rotation_{angles_str}"
        CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", f"feature_banks_v22{suffix}", f"feature_bank_v22{suffix}.pth")
        CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", f"feature_banks_v22{suffix}", "metadata.json")
    else:
        CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v22", "feature_bank_v22.pth")
        CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", "feature_banks_v22", "metadata.json")

    # 输出目录
    CONFIG["output_dir"]     = os.path.join(BASE_DIR, "outputs", "predictions_v22")
    CONFIG["checkpoint_dir"] = os.path.join(BASE_DIR, "outputs", "checkpoints_v22")

    with torch.no_grad():
        run_inference(CONFIG)

