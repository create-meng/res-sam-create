"""
Res-SAM v18 - Step 2: fully automatic inference with patch-level score map.

V18 相对 V16 的核心改进：
1. 不使用 SAM 生成候选区域（解决速度和稳定性问题）
2. 全图 patch-level 异常分数图 + 连通分量分析（解决框太小问题）
3. 多级 FP 抑制：尺寸过滤 + per-image 自适应阈值 + NMS

理论依据：
- AnomalyDINO (WACV 2025): patch-level deep nearest neighbor without SAM
- PatchCore (CVPR 2022): memory bank + patch matching

继承 V16 最优配置：
- hidden_size = 30
- background_removal_method = "both"（行列双向背景去除）
- 验证集驱动的 beta 校准

预期效果：
- 解决 pred 框太小问题（74×75 → 接近 GT 的 224×171）
- TP (IoU>0.5) 从 10 提升到 20-30
- FP 从 170 减少到 40-80
- F1 从 0.083 提升到 0.25-0.40
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from logging.handlers import RotatingFileHandler

import cv2
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
from PatchRes.logger import setup_global_logger, log_config, log_section, log_step, log_finish


def _to_abs(base_dir: str, path: str) -> str:
    if not path: return path
    if os.path.isabs(path): return path
    return os.path.abspath(os.path.join(base_dir, path))


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id
        return True


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v18", "feature_bank_v18.pth"),
    "metadata_path":     os.path.join(BASE_DIR, "outputs", "feature_banks_v18", "metadata.json"),
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
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions_v18"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v18"),
    # v18：继承 V16 最优配置
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    # beta 从 metadata 读取
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,
    # v18 核心：全图 patch 分数图
    "use_score_map": True,
    "score_map_smooth_sigma": 2.0,  # 高斯平滑 sigma
    # v18：连通分量过滤
    "min_bbox_area": 3000,   # 最小 bbox 面积（GT 的 10%）
    "max_bbox_area": 80000,  # 最大 bbox 面积（GT 的 250%）
    # v18：per-image 自适应阈值
    "use_per_image_threshold": True,
    "per_image_threshold_ratio": 0.7,  # max_score * 0.7
    # v18：NMS
    "nms_iou_threshold": 0.5,
    "top_k_per_image": 3,
    # GPR 背景去除
    "gpr_background_removal": True,
    "background_removal_method": "both",
    # 其他
    "device": "auto",
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "max_images_per_category": None,
    "checkpoint_interval": 50,
    "random_seed": 11,
    "version": "v18",
    "feature_with_bias": True,
    "alignment_notes": (
        "v18: 全图patch分数图 + 连通分量 + 多级FP抑制; "
        "不使用SAM，解决框太小问题"
    ),
}

_max_images_env = (os.environ.get("MAX_IMAGES_PER_CATEGORY") or "").strip()
if _max_images_env:
    try:
        v = int(_max_images_env)
        if v > 0:
            CONFIG["max_images_per_category"] = v
    except Exception:
        pass


def remove_gpr_background(arr: np.ndarray, method: str = "both") -> np.ndarray:
    """GPR B-scan 背景去除"""
    result = arr.copy().astype(np.float32)
    if method in ("row_mean", "both"):
        result = result - result.mean(axis=1, keepdims=True)
    if method in ("col_mean", "both"):
        result = result - result.mean(axis=0, keepdims=True)
    if method == "row_median":
        result = arr - np.median(arr, axis=1, keepdims=True)
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
    
    # 按分数降序排序
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        current = indices[0]
        keep.append(current)
        indices = indices[1:]
        
        # 移除与当前框 IoU > threshold 的框
        indices = [
            i for i in indices
            if compute_iou(bboxes[current], bboxes[i]) <= iou_threshold
        ]
    
    return keep


def run_inference(config: dict) -> dict:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 设置日志
    logger = setup_global_logger(base_dir, "02_inference_auto_v18")
    
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

    log_section("Res-SAM v18：全自动推理（全图 patch 分数图）", logger)
    log_config(config, logger)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(f"Feature bank not found: {config['feature_bank_path']}\n"
                                f"请先运行 01_build_feature_bank_v18.py")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    logger.info("初始化 ResSAM（v18）...")
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

        checkpoint_file = os.path.join(config["checkpoint_dir"], f"checkpoint_auto_{category}.json")
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

                    # V18 核心：生成全图 patch 分数图
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
    output_file = os.path.join(config["output_dir"], "auto_predictions_v18.json")
    output_data = {
        "meta": {
            "version": "v18",
            "alignment_notes": config["alignment_notes"],
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
            "hidden_size": int(config["hidden_size"]),
            "beta_threshold": float(config["beta_threshold"]),
            "use_score_map": bool(config.get("use_score_map", True)),
            "score_map_smooth_sigma": float(config.get("score_map_smooth_sigma", 2.0)),
            "min_bbox_area": int(config.get("min_bbox_area", 3000)),
            "max_bbox_area": int(config.get("max_bbox_area", 80000)),
            "use_per_image_threshold": bool(config.get("use_per_image_threshold", True)),
            "per_image_threshold_ratio": float(config.get("per_image_threshold_ratio", 0.7)),
            "nms_iou_threshold": float(config.get("nms_iou_threshold", 0.5)),
            "top_k_per_image": int(config.get("top_k_per_image", 3)),
        },
        "results": all_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_images = sum(len(r) for r in all_results.values())
    total_detections = sum(len(r["pred_bboxes"]) for recs in all_results.values() for r in recs)
    
    logger.info(f"结果保存至：{output_file}")
    logger.info(f"处理图像数：{total_images}，检出框数：{total_detections}")
    
    log_finish("02_inference_auto_v18", logger)
    
    return all_results


def detect_with_score_map(model, image: np.ndarray, config: dict, 
                          logger: logging.Logger, img_name: str) -> dict:
    """
    V18 核心：使用全图 patch 分数图进行异常检测
    
    流程：
    1. 滑窗提取所有 patch
    2. ESN 特征提取
    3. Feature Bank 比较得到 patch 分数
    4. 生成全图分数图
    5. 高斯平滑
    6. 阈值化 + 连通分量分析
    7. 多级 FP 抑制
    """
    img_h, img_w = image.shape
    window_size = config["window_size"]
    stride = config["stride"]
    beta = config["beta_threshold"]
    
    logger.debug(f"  [V18] 开始生成全图 patch 分数图")
    
    # Step 1: 滑窗提取所有 patch 及其位置
    patches = model._extract_patches(image)  # [N, window_size, window_size]
    if patches.shape[0] == 0:
        logger.debug(f"  [V18] 未提取到任何 patch")
        return {
            "pred_bboxes": [],
            "anomaly_scores": [],
            "num_patches": 0,
            "num_anomaly_patches": 0,
            "num_connected_components": 0,
            "max_patch_score": 0.0,
            "mean_patch_score": 0.0,
        }
    
    # 计算 patch 中心位置（必须与 _extract_patches 的逻辑一致）
    half_win = window_size // 2
    patch_positions = []
    for y in range(half_win, img_h - half_win, stride):
        for x in range(half_win, img_w - half_win, stride):
            patch_positions.append((x, y))
    
    num_patches = len(patch_positions)
    
    # 验证 patch 数量一致性
    if num_patches != patches.shape[0]:
        logger.error(f"  [V18] Patch 数量不匹配: positions={num_patches}, patches={patches.shape[0]}")
        logger.error(f"  [V18] 图像尺寸: {img_h}×{img_w}, window_size={window_size}, stride={stride}")
        # 使用实际提取的 patch 数量
        num_patches = patches.shape[0]
        # 重新计算位置（如果数量不匹配，可能是边界处理问题）
        if num_patches < len(patch_positions):
            patch_positions = patch_positions[:num_patches]
        else:
            logger.error(f"  [V18] 无法修复 patch 位置，跳过该图像")
            return {
                "pred_bboxes": [],
                "anomaly_scores": [],
                "num_patches": 0,
                "num_anomaly_patches": 0,
                "num_connected_components": 0,
                "max_patch_score": 0.0,
                "mean_patch_score": 0.0,
            }
    
    logger.debug(f"  [V18] 提取 {num_patches} 个 patch")
    
    # Step 2: ESN 特征提取
    features = model._fit_patches(patches)  # [N, hidden_size+1]
    features_np = features.detach().cpu().numpy()
    
    # Step 3: Feature Bank 比较得到 patch 分数
    scores = model._score_features_against_bank(features_np)  # [N]
    
    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    logger.debug(f"  [V18] Patch 分数: max={max_score:.4f}, mean={mean_score:.4f}")
    
    # Step 4: 生成全图分数图
    # 创建分数图（初始化为 0）
    score_map = np.zeros((img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.int32)
    
    # 将每个 patch 的分数填入对应位置
    for (cx, cy), score in zip(patch_positions, scores):
        x1 = max(0, cx - half_win)
        y1 = max(0, cy - half_win)
        x2 = min(img_w, cx + half_win)
        y2 = min(img_h, cy + half_win)
        
        score_map[y1:y2, x1:x2] += score
        count_map[y1:y2, x1:x2] += 1
    
    # 平均化（避免重叠区域分数过高）
    mask = count_map > 0
    score_map[mask] = score_map[mask] / count_map[mask]
    
    # Step 5: 高斯平滑
    sigma = config.get("score_map_smooth_sigma", 2.0)
    if sigma > 0:
        score_map = ndimage.gaussian_filter(score_map, sigma=sigma)
        logger.debug(f"  [V18] 高斯平滑 (sigma={sigma})")
    
    # Step 6: per-image 自适应阈值
    use_per_image_threshold = config.get("use_per_image_threshold", True)
    if use_per_image_threshold:
        per_image_ratio = config.get("per_image_threshold_ratio", 0.7)
        if max_score < beta:
            # 最高分都低于 beta，认为是正常图，不输出任何框
            logger.debug(f"  [V18] 最高分 {max_score:.4f} < beta {beta:.4f}，判定为正常图")
            return {
                "pred_bboxes": [],
                "anomaly_scores": [],
                "num_patches": num_patches,
                "num_anomaly_patches": 0,
                "num_connected_components": 0,
                "max_patch_score": max_score,
                "mean_patch_score": mean_score,
            }
        else:
            # 使用 max_score * ratio 作为阈值
            adaptive_threshold = max_score * per_image_ratio
            logger.debug(f"  [V18] 自适应阈值: {adaptive_threshold:.4f} (max_score * {per_image_ratio})")
    else:
        adaptive_threshold = beta
        logger.debug(f"  [V18] 使用全局阈值: {beta:.4f}")
    
    # Step 7: 阈值化 + 连通分量分析
    binary_map = (score_map > adaptive_threshold).astype(np.uint8)
    num_anomaly_pixels = int(np.sum(binary_map))
    logger.debug(f"  [V18] 异常像素数: {num_anomaly_pixels}")
    
    if num_anomaly_pixels == 0:
        logger.debug(f"  [V18] 无异常像素")
        return {
            "pred_bboxes": [],
            "anomaly_scores": [],
            "num_patches": num_patches,
            "num_anomaly_patches": 0,
            "num_connected_components": 0,
            "max_patch_score": max_score,
            "mean_patch_score": mean_score,
        }
    
    # 连通分量分析
    labeled_map, num_components = ndimage.label(binary_map)
    logger.debug(f"  [V18] 连通分量数: {num_components}")
    
    # Step 8: 提取每个连通分量的 bbox
    pred_bboxes = []
    pred_scores = []
    
    min_area = config.get("min_bbox_area", 3000)
    max_area = config.get("max_bbox_area", 80000)
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled_map == comp_id)
        
        # 计算外接矩形
        rows, cols = np.where(comp_mask)
        if len(rows) == 0:
            continue
        
        y1, y2 = int(rows.min()), int(rows.max()) + 1
        x1, x2 = int(cols.min()), int(cols.max()) + 1
        
        # 尺寸过滤
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            logger.debug(f"  [V18] 过滤连通分量 {comp_id}: area={area} (min={min_area}, max={max_area})")
            continue
        
        # 计算该区域的平均分数
        region_score = float(score_map[comp_mask].mean())
        
        pred_bboxes.append([x1, y1, x2, y2])
        pred_scores.append(region_score)
        
        logger.debug(f"  [V18] 连通分量 {comp_id}: bbox=[{x1},{y1},{x2},{y2}], area={area}, score={region_score:.4f}")
    
    logger.debug(f"  [V18] 尺寸过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 9: NMS
    if len(pred_bboxes) > 1:
        nms_threshold = config.get("nms_iou_threshold", 0.5)
        keep_indices = nms(pred_bboxes, pred_scores, nms_threshold)
        pred_bboxes = [pred_bboxes[i] for i in keep_indices]
        pred_scores = [pred_scores[i] for i in keep_indices]
        logger.debug(f"  [V18] NMS 后剩余 {len(pred_bboxes)} 个框")
    
    # Step 10: Top-K 过滤
    top_k = config.get("top_k_per_image", 3)
    if len(pred_bboxes) > top_k:
        # 按分数降序排序，保留 top-k
        sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        pred_bboxes = [pred_bboxes[i] for i in sorted_indices[:top_k]]
        pred_scores = [pred_scores[i] for i in sorted_indices[:top_k]]
        logger.debug(f"  [V18] Top-{top_k} 过滤后剩余 {len(pred_bboxes)} 个框")
    
    # 统计异常 patch 数量
    num_anomaly_patches = int(np.sum(scores > beta))
    
    return {
        "pred_bboxes": pred_bboxes,
        "anomaly_scores": pred_scores,
        "num_patches": num_patches,
        "num_anomaly_patches": num_anomaly_patches,
        "num_connected_components": num_components,
        "max_patch_score": max_score,
        "mean_patch_score": mean_score,
    }


if __name__ == "__main__":
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v18")
    CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v18", "feature_bank_v18.pth")
    CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", "feature_banks_v18", "metadata.json")
    CONFIG["output_dir"]        = os.path.join(BASE_DIR, "outputs", "predictions_v18")
    CONFIG["checkpoint_dir"]    = os.path.join(BASE_DIR, "outputs", "checkpoints_v18")

    with torch.no_grad():
        run_inference(CONFIG)
