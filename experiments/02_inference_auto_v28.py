"""
Res-SAM v28 - Step 2: fully automatic inference

V28 说明：
- 固定为当前最优检测主线
- 不再展开 patchcore / threshold learner 对比分支
- 新增 `utilities` 分类别后处理
- 新增可选的细小目标增强融合分支
- 新增 `utilities` 专用框精筛与缩框分支

固定配置：
- top_k_per_image = 1
- nms_iou_threshold = 0.3
- min_bbox_area = 5000
- hidden_size = 30
- background_removal_method = "both"
- use_multi_scale = 1
- multi_scale_strides = [3, 5, 8]
- multi_scale_weights = [0.33, 0.33, 0.34]
- 验证集驱动的 beta 校准
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


def _env_flag(name: str, default: str = "0") -> int:
    value = (os.getenv(name, default) or default).strip().lower()
    return 1 if value in {"1", "true", "yes", "on"} else 0


def _normalize_suffix(env_name: str, fallback: str = "") -> str:
    value = (os.getenv(env_name, fallback) or "").strip()
    if not value:
        return ""
    invalid_chars = set('/\\:')
    if any(ch in value for ch in invalid_chars):
        raise ValueError(f"{env_name} 包含非法路径字符: {value!r}")
    return value


def _to_abs(base_dir: str, path: str) -> str:
    if not path: return path
    if os.path.isabs(path): return path
    return os.path.abspath(os.path.join(base_dir, path))


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v28", "feature_bank_v28.pth"),
    "metadata_path":     os.path.join(BASE_DIR, "outputs", "feature_banks_v28", "metadata.json"),
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
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions_v28"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v28"),
    "output_suffix": _normalize_suffix("OUTPUT_SUFFIX", ""),
    "bank_suffix": _normalize_suffix("BANK_SUFFIX", ""),
    
    # 核心参数
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,
    
    # 全图 patch 分数图
    "use_score_map": True,
    "score_map_smooth_sigma": 2.0,
    
    # V23 固定配置（继承 V22 最优版本）
    "min_bbox_area": 5000,
    "max_bbox_area": 80000,
    "bbox_expand_pixels": 15,
    "nms_iou_threshold": 0.3,
    "top_k_per_image": 1,
    
    # V23 固定：采用 V22 multiscale_uniform
    "use_multi_scale": 1,
    "multi_scale_strides": [3, 5, 8],
    "multi_scale_weights": [0.33, 0.33, 0.34],
    
    # 自适应阈值
    "use_per_image_threshold": True,
    "adaptive_threshold_strategy": "dynamic",
    "per_image_threshold_ratio": 0.7,
    
    # 形状和分数联合过滤
    "aspect_ratio_threshold": 2.9,
    "score_threshold_p80": 0.3283,
    "use_utilities_category_filter": _env_flag("USE_UTILITIES_CATEGORY_FILTER", "0"),
    "utilities_aspect_ratio_threshold": float(os.getenv("UTILITIES_ASPECT_RATIO_THRESHOLD", "5.5")),
    "utilities_score_threshold_p80": float(os.getenv("UTILITIES_SCORE_THRESHOLD_P80", "0.20")),

    # V28 固化基线：在 V25 二次筛选上继续收紧 mean_patch 门槛
    "use_secondary_filter": 1,
    "secondary_filter_min_area_orig": 3500,
    "secondary_filter_min_mean_patch": float(os.getenv("SECONDARY_FILTER_MIN_MEAN_PATCH", "0.275")),
    "secondary_filter_box_score_min": float(os.getenv("SECONDARY_FILTER_BOX_SCORE_MIN", "0.0")),
    "utilities_secondary_filter_min_area_orig": int(os.getenv("UTILITIES_SECONDARY_FILTER_MIN_AREA_ORIG", "1200")),
    "utilities_secondary_filter_min_mean_patch": float(os.getenv("UTILITIES_SECONDARY_FILTER_MIN_MEAN_PATCH", "0.18")),
    "utilities_secondary_filter_box_score_min": float(os.getenv("UTILITIES_SECONDARY_FILTER_BOX_SCORE_MIN", "0.0")),

    # utilities 专用小目标增强
    "use_utilities_small_target_enhance": _env_flag("USE_UTILITIES_SMALL_TARGET_ENHANCE", "0"),
    "utilities_small_target_blend_alpha": float(os.getenv("UTILITIES_SMALL_TARGET_BLEND_ALPHA", "0.35")),
    "utilities_small_target_sigma": float(os.getenv("UTILITIES_SMALL_TARGET_SIGMA", "1.2")),
    "utilities_small_target_percentile": float(os.getenv("UTILITIES_SMALL_TARGET_PERCENTILE", "85.0")),
    "use_utilities_box_refiner": _env_flag("USE_UTILITIES_BOX_REFINER", "0"),
    "utilities_refiner_percentile": float(os.getenv("UTILITIES_REFINER_PERCENTILE", "90.0")),
    "utilities_refiner_expand_pixels": int(os.getenv("UTILITIES_REFINER_EXPAND_PIXELS", "8")),
    "utilities_refiner_min_peak_ratio": float(os.getenv("UTILITIES_REFINER_MIN_PEAK_RATIO", "1.10")),
    "utilities_refiner_max_core_area_ratio": float(os.getenv("UTILITIES_REFINER_MAX_CORE_AREA_RATIO", "0.55")),
    "utilities_refiner_min_core_pixels": int(os.getenv("UTILITIES_REFINER_MIN_CORE_PIXELS", "24")),
    "use_patchcore_topk_agg": 0,
    "patchcore_topk": int(os.getenv("PATCHCORE_TOPK", "3")),
    "patchcore_reweight_lambda": float(os.getenv("PATCHCORE_REWEIGHT_LAMBDA", "0.35")),
    "use_threshold_learner": 0,
    "threshold_normal_std_mult": float(os.getenv("THRESHOLD_NORMAL_STD_MULT", "2.5")),
    "threshold_image_std_mult": float(os.getenv("THRESHOLD_IMAGE_STD_MULT", "1.0")),
    "threshold_min_floor_delta": float(os.getenv("THRESHOLD_MIN_FLOOR_DELTA", "0.0")),

    # GPR 背景去除
    "gpr_background_removal": True,
    "background_removal_method": "both",
    
    # 其他
    "device": "auto",
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "max_images_per_category": (
        int(os.getenv("MAX_IMAGES_PER_CATEGORY", "").strip())
        if os.getenv("MAX_IMAGES_PER_CATEGORY", "").strip()
        else None
    ),
    "checkpoint_interval": 50,
    "random_seed": 11,
    "version": "v28",
    "feature_with_bias": True,
}


def remove_gpr_background(arr: np.ndarray, method: str = "both") -> np.ndarray:
    """GPR B-scan 背景去除（V23：固定最佳预处理）"""
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


def apply_utilities_small_target_enhance(score_map: np.ndarray, config: dict,
                                         logger: logging.Logger | None = None) -> np.ndarray:
    """对 utilities 的细小/尖峰响应做温和增强，不改变主流程结构。"""
    sigma = max(0.1, float(config.get("utilities_small_target_sigma", 1.2)))
    alpha = float(config.get("utilities_small_target_blend_alpha", 0.35))
    percentile = float(config.get("utilities_small_target_percentile", 85.0))

    # 用 LoG 提取局部小尺度峰值，再做非负归一化融合。
    log_response = -ndimage.gaussian_laplace(score_map, sigma=sigma)
    log_response = np.maximum(log_response, 0.0)
    hi = np.percentile(log_response, percentile) if np.any(log_response > 0) else 0.0
    if hi <= 1e-8:
        return score_map
    log_response = np.clip(log_response / hi, 0.0, 1.0)

    base_max = float(np.max(score_map))
    if base_max <= 1e-8:
        return score_map
    enhanced = score_map + alpha * log_response * base_max
    if logger is not None:
        logger.debug(
            f"  [V28] utilities 小目标增强已启用: sigma={sigma:.2f}, alpha={alpha:.2f}, p={percentile:.1f}"
        )
    return enhanced.astype(np.float32)


def get_category_filter_params(category: str, config: dict) -> dict[str, float]:
    if category == "utilities" and int(config.get("use_utilities_category_filter", 0)):
        return {
            "aspect_ratio_threshold": float(config.get("utilities_aspect_ratio_threshold", 5.5)),
            "score_threshold_p80": float(config.get("utilities_score_threshold_p80", 0.20)),
            "secondary_filter_min_area_orig": int(config.get("utilities_secondary_filter_min_area_orig", 1200)),
            "secondary_filter_min_mean_patch": float(config.get("utilities_secondary_filter_min_mean_patch", 0.18)),
            "secondary_filter_box_score_min": float(config.get("utilities_secondary_filter_box_score_min", 0.0)),
        }
    return {
        "aspect_ratio_threshold": float(config.get("aspect_ratio_threshold", 2.9)),
        "score_threshold_p80": float(config.get("score_threshold_p80", 0.3283)),
        "secondary_filter_min_area_orig": int(config.get("secondary_filter_min_area_orig", 3500)),
        "secondary_filter_min_mean_patch": float(config.get("secondary_filter_min_mean_patch", 0.275)),
        "secondary_filter_box_score_min": float(config.get("secondary_filter_box_score_min", 0.0)),
    }


def refine_utilities_bbox(bbox: list[int], score: float, score_map: np.ndarray, config: dict,
                          logger: logging.Logger | None = None) -> tuple[list[int] | None, float]:
    """对 utilities 框做缩框和 diffuse FP 抑制。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = score_map.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    patch = score_map[y1:y2, x1:x2]
    if patch.size == 0:
        return None, score

    patch_mean = float(np.mean(patch))
    patch_max = float(np.max(patch))
    peak_ratio = patch_max / (patch_mean + 1e-6)
    if peak_ratio < float(config.get("utilities_refiner_min_peak_ratio", 1.10)):
        if logger is not None:
            logger.debug(f"  [V28] utilities 精筛移除框: peak_ratio={peak_ratio:.3f} 过低")
        return None, score

    percentile = float(config.get("utilities_refiner_percentile", 90.0))
    threshold = np.percentile(patch, percentile)
    core_mask = patch >= threshold
    core_pixels = int(np.sum(core_mask))
    min_core_pixels = int(config.get("utilities_refiner_min_core_pixels", 24))
    if core_pixels < min_core_pixels:
        if logger is not None:
            logger.debug(f"  [V28] utilities 精筛移除框: core_pixels={core_pixels} < {min_core_pixels}")
        return None, score

    core_ratio = core_pixels / float(patch.size)
    max_core_ratio = float(config.get("utilities_refiner_max_core_area_ratio", 0.55))
    if core_ratio > max_core_ratio:
        if logger is not None:
            logger.debug(f"  [V28] utilities 精筛移除框: core_ratio={core_ratio:.3f} > {max_core_ratio:.3f}")
        return None, score

    rows, cols = np.where(core_mask)
    if len(rows) == 0:
        return None, score

    expand = int(config.get("utilities_refiner_expand_pixels", 8))
    rx1 = max(0, x1 + int(cols.min()) - expand)
    ry1 = max(0, y1 + int(rows.min()) - expand)
    rx2 = min(w, x1 + int(cols.max()) + 1 + expand)
    ry2 = min(h, y1 + int(rows.max()) + 1 + expand)

    refined = [rx1, ry1, rx2, ry2]
    refined_score = max(score, patch_max)
    return refined, float(refined_score)


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


def knn_scores_from_bank(model, features_np: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return model._score_features_against_bank(features_np)

    if getattr(model, "nn_backend", "sklearn") == "sklearn":
        distances, _ = model.nn_searcher.kneighbors(features_np, n_neighbors=k)
        return distances.astype(np.float32, copy=False)

    xq = np.ascontiguousarray(features_np.astype(np.float32, copy=False))
    dists, _ = model.nn_searcher.search(xq, k)
    return np.sqrt(np.maximum(dists, 0.0)).astype(np.float32, copy=False)


def score_features_with_variant(model, features_np: np.ndarray, config: dict) -> np.ndarray:
    if features_np is None or getattr(features_np, "size", 0) == 0:
        return np.zeros((0,), dtype=np.float32)

    if not int(config.get("use_patchcore_topk_agg", 0)):
        return model._score_features_against_bank(features_np)

    topk = max(2, int(config.get("patchcore_topk", 3)))
    blend_lambda = float(config.get("patchcore_reweight_lambda", 0.35))
    knn_dists = knn_scores_from_bank(model, features_np, topk)
    if knn_dists.ndim == 1:
        return knn_dists.astype(np.float32, copy=False)
    d1 = knn_dists[:, 0]
    dmean = knn_dists.mean(axis=1)
    scores = (1.0 - blend_lambda) * d1 + blend_lambda * dmean
    return scores.astype(np.float32, copy=False)


def apply_threshold_learner(adaptive_threshold: float, beta: float, max_score: float,
                            mean_score: float, std_score: float, config: dict) -> float:
    if not int(config.get("use_threshold_learner", 0)):
        return adaptive_threshold

    normal_mean = float(config.get("val_normal_patch_mean", 0.0))
    normal_std = float(config.get("val_normal_patch_std", 0.0))
    normal_floor = normal_mean + float(config.get("threshold_normal_std_mult", 2.5)) * normal_std
    image_floor = mean_score + float(config.get("threshold_image_std_mult", 1.0)) * std_score
    min_delta = float(config.get("threshold_min_floor_delta", 0.0))
    learned_floor = max(beta + min_delta, normal_floor, image_floor)
    return float(min(max_score, max(adaptive_threshold, learned_floor)))


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
                          logger: logging.Logger, img_name: str,
                          category: str = "") -> dict:
    """
    V23 核心：使用固定多尺度全图 patch 分数图进行异常检测
    
    流程：
    1. 滑窗提取所有 patch（支持多尺度）
    2. ESN 特征提取
    3. Feature Bank 比较得到 patch 分数
    4. 生成全图分数图（支持多尺度融合）
    5. 高斯平滑
    6. 自适应阈值
    7. 阈值化
    8. 连通分量分析
    9. 固定最优配置下的 FP 抑制
    """
    img_h, img_w = image.shape
    window_size = config["window_size"]
    stride = config["stride"]
    beta = config["beta_threshold"]
    
    use_multi_scale = config.get("use_multi_scale", 0)
    
    logger.debug(f"  [V23] 开始生成全图 patch 分数图")
    logger.debug(f"  [V23] 多尺度: {use_multi_scale}")
    
    # 多尺度分数图生成
    if use_multi_scale:
        strides = config.get("multi_scale_strides", [3, 5, 8])
        weights = config.get("multi_scale_weights", [0.3, 0.4, 0.3])
        
        logger.debug(f"  [V23] 多尺度 strides: {strides}, weights: {weights}")
        
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
            scores = score_features_with_variant(model, features_np, config)
            all_scores_list.extend(scores.tolist())
            
            # 生成分数图
            score_map = generate_score_map_single_scale(
                (img_h, img_w), patch_positions, scores, window_size
            )
            score_maps.append(score_map)
            
            logger.debug(f"  [V23] stride={stride_val}: {len(scores)} patches, "
                        f"max_score={np.max(scores):.4f}")
        
        # 恢复原始 stride
        model.stride = original_stride
        
        if len(score_maps) == 0:
            logger.debug(f"  [V23] 未提取到任何 patch")
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
            logger.debug(f"  [V23] 未提取到任何 patch")
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
            logger.error(f"  [V23] Patch 数量不匹配: positions={num_patches}, patches={patches.shape[0]}")
            num_patches = patches.shape[0]
            if num_patches < len(patch_positions):
                patch_positions = patch_positions[:num_patches]
            else:
                logger.error(f"  [V23] 无法修复 patch 位置，跳过该图像")
                return {
                    "pred_bboxes": [],
                    "anomaly_scores": [],
                    "num_patches": 0,
                    "num_anomaly_patches": 0,
                    "num_connected_components": 0,
                    "max_patch_score": 0.0,
                    "mean_patch_score": 0.0,
                }
        
        logger.debug(f"  [V23] 提取 {num_patches} 个 patch")
        
        # Step 2: ESN 特征提取
        features = model._fit_patches(patches)
        features_np = features.detach().cpu().numpy()
        
        # Step 3: Feature Bank 比较得到 patch 分数
        all_scores = score_features_with_variant(model, features_np, config)
        
        # Step 4: 生成全图分数图
        score_map = generate_score_map_single_scale(
            (img_h, img_w), patch_positions, all_scores, window_size
        )
    
    max_score = float(np.max(all_scores))
    mean_score = float(np.mean(all_scores))
    std_score = float(np.std(all_scores))
    logger.debug(f"  [V23] Patch 分数: max={max_score:.4f}, mean={mean_score:.4f}, std={std_score:.4f}")
    
    # Step 5: 高斯平滑
    sigma = config.get("score_map_smooth_sigma", 2.0)
    if sigma > 0:
        score_map = ndimage.gaussian_filter(score_map, sigma=sigma)
        logger.debug(f"  [V23] 高斯平滑 (sigma={sigma})")

    if category == "utilities" and int(config.get("use_utilities_small_target_enhance", 0)):
        score_map = apply_utilities_small_target_enhance(score_map, config, logger)

    # Step 6: 自适应阈值
    use_per_image_threshold = config.get("use_per_image_threshold", True)
    if use_per_image_threshold:
        adaptive_strategy = config.get("adaptive_threshold_strategy", "dynamic")
        
        if max_score < beta:
            logger.debug(f"  [V23] 最高分 {max_score:.4f} < beta {beta:.4f}，判定为正常图")
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
                logger.debug(f"  [V23] 分数分布集中 (std={std_score:.4f})，使用严格阈值比例 {adaptive_ratio}")
            elif score_range > 0.5:
                adaptive_ratio = 0.6
                logger.debug(f"  [V23] 分数范围大 (range={score_range:.4f})，使用宽松阈值比例 {adaptive_ratio}")
            else:
                adaptive_ratio = config.get("per_image_threshold_ratio", 0.7)
                logger.debug(f"  [V23] 使用默认阈值比例 {adaptive_ratio}")
        else:
            adaptive_ratio = config.get("per_image_threshold_ratio", 0.7)
            logger.debug(f"  [V23] 使用固定阈值比例 {adaptive_ratio}")
        
        adaptive_threshold = max_score * adaptive_ratio
        adaptive_threshold = apply_threshold_learner(
            adaptive_threshold, beta, max_score, mean_score, std_score, config
        )
        logger.debug(f"  [V23] 自适应阈值: {adaptive_threshold:.4f} (max_score * {adaptive_ratio:.2f})")
    else:
        adaptive_threshold = beta
        adaptive_threshold = apply_threshold_learner(
            adaptive_threshold, beta, max_score, mean_score, std_score, config
        )
        logger.debug(f"  [V23] 使用全局阈值: {beta:.4f}")
    
    # Step 7: 阈值化
    binary_map = (score_map > adaptive_threshold).astype(np.uint8)
    num_anomaly_pixels = int(np.sum(binary_map))
    logger.debug(f"  [V23] 阈值化后异常像素数: {num_anomaly_pixels}")

    if num_anomaly_pixels == 0:
        logger.debug(f"  [V23] 无异常像素")
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
    logger.debug(f"  [V23] 连通分量数: {num_components}")
    
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
        
        # V23 固定尺寸过滤
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            logger.debug(f"  [V23] 过滤连通分量 {comp_id}: area={area} (min={min_area}, max={max_area})")
            continue
        
        region_score = float(score_map[comp_mask].mean())
        
        pred_bboxes.append([x1, y1, x2, y2])
        pred_scores.append(region_score)
        
        logger.debug(f"  [V23] 连通分量 {comp_id}: bbox=[{x1},{y1},{x2},{y2}], area={area}, score={region_score:.4f}")
    
    logger.debug(f"  [V23] 尺寸过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 10: 固定 NMS
    if len(pred_bboxes) > 1:
        nms_threshold = config.get("nms_iou_threshold", 0.3)
        keep_indices = nms(pred_bboxes, pred_scores, nms_threshold)
        pred_bboxes = [pred_bboxes[i] for i in keep_indices]
        pred_scores = [pred_scores[i] for i in keep_indices]
        logger.debug(f"  [V23] NMS (threshold={nms_threshold}) 后剩余 {len(pred_bboxes)} 个框")

    if category == "utilities" and int(config.get("use_utilities_box_refiner", 0)) and len(pred_bboxes) > 0:
        refined_bboxes = []
        refined_scores = []
        for bbox, score in zip(pred_bboxes, pred_scores):
            refined_bbox, refined_score = refine_utilities_bbox(bbox, float(score), score_map, config, logger)
            if refined_bbox is None:
                continue
            refined_bboxes.append(refined_bbox)
            refined_scores.append(refined_score)
        pred_bboxes = refined_bboxes
        pred_scores = refined_scores
        logger.debug(f"  [V28] utilities 精筛后剩余 {len(pred_bboxes)} 个框")
    
    # Step 11: 固定 Top-K 过滤
    top_k = config.get("top_k_per_image", 1)
    if len(pred_bboxes) > top_k:
        sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        pred_bboxes = [pred_bboxes[i] for i in sorted_indices[:top_k]]
        pred_scores = [pred_scores[i] for i in sorted_indices[:top_k]]
        logger.debug(f"  [V23] Top-{top_k} 过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 12: 形状和分数联合过滤
    if len(pred_bboxes) > 0:
        filter_params = get_category_filter_params(category, config)
        aspect_ratio_threshold = float(filter_params["aspect_ratio_threshold"])
        score_threshold_p80 = float(filter_params["score_threshold_p80"])
        
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
                logger.debug(
                    f"  [V23] 过滤框[{category or 'default'}]: bbox=[{x1},{y1},{x2},{y2}], "
                    f"aspect={aspect_ratio:.2f}, score={score:.4f}"
                )
        
        num_filtered = len(pred_bboxes) - len(filtered_bboxes)
        if num_filtered > 0:
            logger.debug(f"  [V23] 形状+分数过滤: 移除 {num_filtered} 个框，剩余 {len(filtered_bboxes)} 个")
        
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

    logger = setup_global_logger(base_dir, "02_inference_auto_v28")
    
    config = dict(config)
    for key in ["feature_bank_path", "metadata_path", "output_dir", "checkpoint_dir"]:
        config[key] = _to_abs(base_dir, config.get(key, ""))
    config["test_data_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("test_data_dirs", {}).items()}
    config["annotation_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("annotation_dirs", {}).items()}

    # 从 metadata 读取自适应 beta
    effective_beta = config["beta_threshold"]
    metadata = None
    if config.get("use_adaptive_beta", True) and os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "adaptive_beta" in metadata:
            effective_beta = float(metadata["adaptive_beta"])
            logger.info(f"自适应 beta = {effective_beta:.4f}（从 metadata 读取）")
        else:
            logger.info(f"无 adaptive_beta，使用默认值 {effective_beta}")
    elif os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
    config["beta_threshold"] = effective_beta
    if metadata:
        normal_stats = (((metadata.get("validation_set") or {}).get("stats") or {}).get("normal_scores") or {})
        config["val_normal_patch_mean"] = float(normal_stats.get("mean", 0.0) or 0.0)
        config["val_normal_patch_std"] = float(normal_stats.get("std", 0.0) or 0.0)

    log_section("Res-SAM V28：继承 V25 归档基线，准备继续优化检测指标", logger)
    log_config(config, logger)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(f"Feature bank not found: {config['feature_bank_path']}\n"
                                f"请先运行 01_build_feature_bank_v28.py")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    logger.info("初始化 ResSAM（V28）...")
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

                    # V23 核心：全图 patch 分数图 + 固定最优参数
                    result = detect_with_score_map(
                        model, img, config, logger, img_file, category=category
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

                    # V24: 原图坐标系下的二次筛选，只使用已有输出统计量
                    if config.get("use_secondary_filter", 1) and len(pred_bboxes) > 0:
                        filter_params = get_category_filter_params(category, config)
                        min_area_orig = int(filter_params["secondary_filter_min_area_orig"])
                        min_mean_patch = float(filter_params["secondary_filter_min_mean_patch"])
                        min_box_score = float(filter_params["secondary_filter_box_score_min"])
                        mean_patch_score = float(result.get("mean_patch_score", 0.0))

                        kept_pred = []
                        kept_pred_resized = []
                        kept_scores = []
                        for bbox_orig, bbox_resized, score in zip(pred_bboxes, pred_bboxes_resized, anomaly_scores):
                            area_orig = max(0, (bbox_orig[2] - bbox_orig[0])) * max(0, (bbox_orig[3] - bbox_orig[1]))
                            if area_orig < min_area_orig:
                                logger.debug(
                                    f"  [V28] 二次筛选移除框[{category}]: 原图面积 {area_orig} < {min_area_orig}"
                                )
                                continue
                            if mean_patch_score < min_mean_patch:
                                logger.debug(
                                    f"  [V28] 二次筛选移除框[{category}]: "
                                    f"mean_patch_score {mean_patch_score:.4f} < {min_mean_patch:.4f}"
                                )
                                continue
                            if float(score) < min_box_score:
                                logger.debug(
                                    f"  [V28] 二次筛选移除框[{category}]: "
                                    f"box_score {float(score):.4f} < {min_box_score:.4f}"
                                )
                                continue
                            kept_pred.append(bbox_orig)
                            kept_pred_resized.append(bbox_resized)
                            kept_scores.append(score)

                        pred_bboxes = kept_pred
                        pred_bboxes_resized = kept_pred_resized
                        anomaly_scores = kept_scores

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
    output_filename = f"auto_predictions_v28{suffix}.json" if suffix else "auto_predictions_v28.json"
    output_file = os.path.join(config["output_dir"], output_filename)
    output_data = {
        "meta": {
            "version": "v28",
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "bank_suffix": config.get("bank_suffix", ""),
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
            "hidden_size": int(config["hidden_size"]),
            "beta_threshold": float(config["beta_threshold"]),
            "use_score_map": bool(config.get("use_score_map", True)),
            "score_map_smooth_sigma": float(config.get("score_map_smooth_sigma", 2.0)),
            # V23 固定最优配置
            "min_bbox_area": int(config.get("min_bbox_area", 5000)),
            "max_bbox_area": int(config.get("max_bbox_area", 80000)),
            "bbox_expand_pixels": int(config.get("bbox_expand_pixels", 15)),
            "nms_iou_threshold": float(config.get("nms_iou_threshold", 0.3)),
            "top_k_per_image": int(config.get("top_k_per_image", 1)),
            # 新增优化方案
            "use_multi_scale": int(config.get("use_multi_scale", 0)),
            "multi_scale_strides": config.get("multi_scale_strides", [3, 5, 8]),
            "multi_scale_weights": config.get("multi_scale_weights", [0.3, 0.4, 0.3]),
            "use_secondary_filter": int(config.get("use_secondary_filter", 0)),
            "secondary_filter_min_area_orig": int(config.get("secondary_filter_min_area_orig", 0)),
            "secondary_filter_min_mean_patch": float(config.get("secondary_filter_min_mean_patch", 0.0)),
            "secondary_filter_box_score_min": float(config.get("secondary_filter_box_score_min", 0.0)),
            "use_utilities_category_filter": int(config.get("use_utilities_category_filter", 0)),
            "utilities_aspect_ratio_threshold": float(config.get("utilities_aspect_ratio_threshold", 5.5)),
            "utilities_score_threshold_p80": float(config.get("utilities_score_threshold_p80", 0.20)),
            "utilities_secondary_filter_min_area_orig": int(config.get("utilities_secondary_filter_min_area_orig", 1200)),
            "utilities_secondary_filter_min_mean_patch": float(config.get("utilities_secondary_filter_min_mean_patch", 0.18)),
            "utilities_secondary_filter_box_score_min": float(config.get("utilities_secondary_filter_box_score_min", 0.0)),
            "use_utilities_small_target_enhance": int(config.get("use_utilities_small_target_enhance", 0)),
            "utilities_small_target_blend_alpha": float(config.get("utilities_small_target_blend_alpha", 0.35)),
            "utilities_small_target_sigma": float(config.get("utilities_small_target_sigma", 1.2)),
            "utilities_small_target_percentile": float(config.get("utilities_small_target_percentile", 85.0)),
            "use_utilities_box_refiner": int(config.get("use_utilities_box_refiner", 0)),
            "utilities_refiner_percentile": float(config.get("utilities_refiner_percentile", 90.0)),
            "utilities_refiner_expand_pixels": int(config.get("utilities_refiner_expand_pixels", 8)),
            "utilities_refiner_min_peak_ratio": float(config.get("utilities_refiner_min_peak_ratio", 1.10)),
            "utilities_refiner_max_core_area_ratio": float(config.get("utilities_refiner_max_core_area_ratio", 0.55)),
            "utilities_refiner_min_core_pixels": int(config.get("utilities_refiner_min_core_pixels", 24)),
            "use_patchcore_topk_agg": int(config.get("use_patchcore_topk_agg", 0)),
            "patchcore_topk": int(config.get("patchcore_topk", 3)),
            "patchcore_reweight_lambda": float(config.get("patchcore_reweight_lambda", 0.35)),
            "use_threshold_learner": int(config.get("use_threshold_learner", 0)),
            "threshold_normal_std_mult": float(config.get("threshold_normal_std_mult", 2.5)),
            "threshold_image_std_mult": float(config.get("threshold_image_std_mult", 1.0)),
            "threshold_min_floor_delta": float(config.get("threshold_min_floor_delta", 0.0)),
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
    
    log_finish("02_inference_auto_v28", logger)
    
    return all_results


if __name__ == "__main__":
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v28")
    bank_suffix = CONFIG.get("bank_suffix", "")
    if bank_suffix:
        CONFIG["feature_bank_path"] = os.path.join(
            BASE_DIR, "outputs", f"feature_banks_v28{bank_suffix}", f"feature_bank_v28{bank_suffix}.pth"
        )
        CONFIG["metadata_path"] = os.path.join(
            BASE_DIR, "outputs", f"feature_banks_v28{bank_suffix}", "metadata.json"
        )
    else:
        CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v28", "feature_bank_v28.pth")
        CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", "feature_banks_v28", "metadata.json")
    CONFIG["output_dir"]        = os.path.join(BASE_DIR, "outputs", "predictions_v28")
    CONFIG["checkpoint_dir"]    = os.path.join(BASE_DIR, "outputs", "checkpoints_v28")

    with torch.no_grad():
        run_inference(CONFIG)



