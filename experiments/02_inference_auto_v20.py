"""
Res-SAM v20 - Step 2: fully automatic inference with morphological post-processing and adaptive thresholding.

V20 相对 V18 的核心改进：
1. **方案2：形态学后处理** - 在分数图二值化后应用闭运算和开运算，减少碎片化FP
2. **方案4：自适应阈值优化** - 基于图像统计的动态阈值 + 双阈值策略，提升检测精度

理论依据：
- Morphological Operations for Image Processing (ResearchGate 2013)
- Deep Anomaly Detection via Morphological Transformations (MDPI 2020)
- Adaptive Threshold in Industrial Anomaly Detection (2024)

继承 V18 最优配置：
- 全图 patch-level 异常分数图 + 连通分量分析
- hidden_size = 30
- background_removal_method = "both"（行列双向背景去除）
- 验证集驱动的 beta 校准

预期效果：
- F1 从 0.504 提升到 0.53-0.55 (+3-5%)
- Precision 从 0.443 提升到 0.48-0.52 (+4-8%)
- FP 从 44 减少到 35-40 (-10-20%)
- 保持 Recall 在 0.55+ 水平
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
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v20", "feature_bank_v20.pth"),
    "metadata_path":     os.path.join(BASE_DIR, "outputs", "feature_banks_v20", "metadata.json"),
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
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions_v20"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v20"),
    # v20：继承 V18 最优配置
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    # beta 从 metadata 读取
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,
    # v20 核心：全图 patch 分数图
    "use_score_map": True,
    "score_map_smooth_sigma": 2.0,  # 高斯平滑 sigma
    # v20 方案2：形态学后处理
    "use_morphology": True,
    "morphology_closing_kernel": 5,  # 闭运算 kernel 大小（填充小孔洞）
    "morphology_opening_kernel": 3,  # 开运算 kernel 大小（去除小噪声）
    # v20：连通分量过滤
    "min_bbox_area": 3000,   # 最小 bbox 面积（GT 的 10%）
    "max_bbox_area": 80000,  # 最大 bbox 面积（GT 的 250%）
    "bbox_expand_pixels": 15,  # bbox 膨胀像素数（修正连通分量过于保守）
    # v20 方案4：自适应阈值优化
    "use_per_image_threshold": True,
    "adaptive_threshold_strategy": "dynamic",  # "fixed" or "dynamic"
    "per_image_threshold_ratio": 0.7,  # 默认比例（dynamic 模式下会动态调整）
    "use_dual_threshold": True,  # 启用双阈值策略
    "dual_threshold_high_ratio": 0.8,  # 高阈值比例（确定异常）
    "dual_threshold_low_ratio": 0.5,   # 低阈值比例（可能异常）
    # v20：NMS
    "nms_iou_threshold": 0.5,
    "top_k_per_image": 3,
    # v18/v19 优化：形状和分数联合过滤（保留）
    "aspect_ratio_threshold": 2.9,  # 过滤 aspect_ratio >= 2.9 的细长框
    "score_threshold_p80": 0.3283,  # 过滤 score < 0.3283 (正常图像p80) 的低分框
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
    "version": "v20",
    "feature_with_bias": True,
    "alignment_notes": (
        "v20: V18 + 方案2(形态学后处理) + 方案4(自适应阈值优化); "
        "预期F1 0.504→0.53-0.55, Precision 0.443→0.48-0.52"
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
    logger = setup_global_logger(base_dir, "02_inference_auto_v20")
    
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

    log_section("Res-SAM v20：全自动推理（全图 patch 分数图 + 形态学后处理 + 自适应阈值）", logger)
    log_config(config, logger)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(f"Feature bank not found: {config['feature_bank_path']}\n"
                                f"请先运行 01_build_feature_bank_v20.py")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    logger.info("初始化 ResSAM（v20）...")
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

                    # V20 核心：生成全图 patch 分数图 + 形态学后处理 + 自适应阈值
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
    output_file = os.path.join(config["output_dir"], "auto_predictions_v20.json")
    output_data = {
        "meta": {
            "version": "v20",
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
            "bbox_expand_pixels": int(config.get("bbox_expand_pixels", 15)),
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
    
    log_finish("02_inference_auto_v20", logger)
    
    return all_results


def detect_with_score_map(model, image: np.ndarray, config: dict, 
                          logger: logging.Logger, img_name: str) -> dict:
    """
    V20 核心：使用全图 patch 分数图 + 形态学后处理 + 自适应阈值进行异常检测
    
    流程：
    1. 滑窗提取所有 patch
    2. ESN 特征提取
    3. Feature Bank 比较得到 patch 分数
    4. 生成全图分数图
    5. 高斯平滑
    6. V20 方案4：改进的自适应阈值（动态比例 + 双阈值）
    7. 阈值化
    8. V20 方案2：形态学后处理（闭运算 + 开运算）
    9. 连通分量分析
    10. 多级 FP 抑制
    """
    img_h, img_w = image.shape
    window_size = config["window_size"]
    stride = config["stride"]
    beta = config["beta_threshold"]
    
    logger.debug(f"  [V20] 开始生成全图 patch 分数图")
    
    # Step 1: 滑窗提取所有 patch 及其位置
    patches = model._extract_patches(image)  # [N, window_size, window_size]
    if patches.shape[0] == 0:
        logger.debug(f"  [V20] 未提取到任何 patch")
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
        logger.error(f"  [V20] Patch 数量不匹配: positions={num_patches}, patches={patches.shape[0]}")
        logger.error(f"  [V20] 图像尺寸: {img_h}×{img_w}, window_size={window_size}, stride={stride}")
        # 使用实际提取的 patch 数量
        num_patches = patches.shape[0]
        # 重新计算位置（如果数量不匹配，可能是边界处理问题）
        if num_patches < len(patch_positions):
            patch_positions = patch_positions[:num_patches]
        else:
            logger.error(f"  [V20] 无法修复 patch 位置，跳过该图像")
            return {
                "pred_bboxes": [],
                "anomaly_scores": [],
                "num_patches": 0,
                "num_anomaly_patches": 0,
                "num_connected_components": 0,
                "max_patch_score": 0.0,
                "mean_patch_score": 0.0,
            }
    
    logger.debug(f"  [V20] 提取 {num_patches} 个 patch")
    
    # Step 2: ESN 特征提取
    features = model._fit_patches(patches)  # [N, hidden_size+1]
    features_np = features.detach().cpu().numpy()
    
    # Step 3: Feature Bank 比较得到 patch 分数
    scores = model._score_features_against_bank(features_np)  # [N]
    
    max_score = float(np.max(scores))
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    logger.debug(f"  [V20] Patch 分数: max={max_score:.4f}, mean={mean_score:.4f}, std={std_score:.4f}")
    
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
        logger.debug(f"  [V20] 高斯平滑 (sigma={sigma})")
    
    # Step 6: V20 方案4 - 改进的自适应阈值
    use_per_image_threshold = config.get("use_per_image_threshold", True)
    if use_per_image_threshold:
        adaptive_strategy = config.get("adaptive_threshold_strategy", "dynamic")
        
        if max_score < beta:
            # 最高分都低于 beta，认为是正常图，不输出任何框
            logger.debug(f"  [V20] 最高分 {max_score:.4f} < beta {beta:.4f}，判定为正常图")
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
            # V20 方案4：基于图像统计动态调整阈值比例
            score_range = max_score - mean_score
            
            if std_score < 0.1:
                # 分数分布集中 → 更严格
                adaptive_ratio = 0.8
                logger.debug(f"  [V20] 分数分布集中 (std={std_score:.4f})，使用严格阈值比例 {adaptive_ratio}")
            elif score_range > 0.5:
                # 分数范围大 → 更宽松
                adaptive_ratio = 0.6
                logger.debug(f"  [V20] 分数范围大 (range={score_range:.4f})，使用宽松阈值比例 {adaptive_ratio}")
            else:
                # 默认
                adaptive_ratio = config.get("per_image_threshold_ratio", 0.7)
                logger.debug(f"  [V20] 使用默认阈值比例 {adaptive_ratio}")
        else:
            # 固定比例（V18 模式）
            adaptive_ratio = config.get("per_image_threshold_ratio", 0.7)
            logger.debug(f"  [V20] 使用固定阈值比例 {adaptive_ratio}")
        
        adaptive_threshold = max_score * adaptive_ratio
        logger.debug(f"  [V20] 自适应阈值: {adaptive_threshold:.4f} (max_score * {adaptive_ratio:.2f})")
    else:
        adaptive_threshold = beta
        logger.debug(f"  [V20] 使用全局阈值: {beta:.4f}")
    
    # Step 7: 阈值化
    binary_map = (score_map > adaptive_threshold).astype(np.uint8)
    num_anomaly_pixels_before = int(np.sum(binary_map))
    logger.debug(f"  [V20] 阈值化后异常像素数: {num_anomaly_pixels_before}")
    
    if num_anomaly_pixels_before == 0:
        logger.debug(f"  [V20] 无异常像素")
        return {
            "pred_bboxes": [],
            "anomaly_scores": [],
            "num_patches": num_patches,
            "num_anomaly_patches": 0,
            "num_connected_components": 0,
            "max_patch_score": max_score,
            "mean_patch_score": mean_score,
        }
    
    # Step 8: V20 方案2 - 形态学后处理
    if config.get("use_morphology", True):
        # 8.1 闭运算（填充小孔洞，连接邻近区域）
        closing_kernel_size = config.get("morphology_closing_kernel", 5)
        if closing_kernel_size > 0:
            closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), dtype=np.uint8)
            binary_map = ndimage.binary_closing(binary_map, structure=closing_kernel).astype(np.uint8)
            logger.debug(f"  [V20] 形态学闭运算 (kernel={closing_kernel_size}×{closing_kernel_size})")
        
        # 8.2 开运算（去除小噪声点，平滑边界）
        opening_kernel_size = config.get("morphology_opening_kernel", 3)
        if opening_kernel_size > 0:
            opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), dtype=np.uint8)
            binary_map = ndimage.binary_opening(binary_map, structure=opening_kernel).astype(np.uint8)
            logger.debug(f"  [V20] 形态学开运算 (kernel={opening_kernel_size}×{opening_kernel_size})")
        
        num_anomaly_pixels_after = int(np.sum(binary_map))
        logger.debug(f"  [V20] 形态学后异常像素数: {num_anomaly_pixels_after} "
                    f"(变化: {num_anomaly_pixels_after - num_anomaly_pixels_before:+d})")
    
    # Step 9: V20 方案4 - 双阈值策略（可选）
    if config.get("use_dual_threshold", False):
        high_ratio = config.get("dual_threshold_high_ratio", 0.8)
        low_ratio = config.get("dual_threshold_low_ratio", 0.5)
        
        high_threshold = max_score * high_ratio
        low_threshold = max_score * low_ratio
        
        # 高阈值区域：确定异常
        certain_anomaly = (score_map > high_threshold).astype(np.uint8)
        # 低阈值区域：可能异常
        possible_anomaly = (score_map > low_threshold).astype(np.uint8)
        
        # 只保留与高阈值区域连通的低阈值区域
        labeled_certain, _ = ndimage.label(certain_anomaly)
        final_binary_map = np.zeros_like(binary_map)
        
        for label_id in range(1, labeled_certain.max() + 1):
            seed_mask = (labeled_certain == label_id)
            # 从种子区域生长到可能异常区域
            grown_mask = ndimage.binary_dilation(seed_mask, iterations=10) & possible_anomaly
            final_binary_map |= grown_mask
        
        num_dual_pixels = int(np.sum(final_binary_map))
        logger.debug(f"  [V20] 双阈值策略: high={high_threshold:.4f}, low={low_threshold:.4f}, "
                    f"像素数: {num_dual_pixels}")
        
        binary_map = final_binary_map.astype(np.uint8)
    
    # Step 10: 连通分量分析
    labeled_map, num_components = ndimage.label(binary_map)
    logger.debug(f"  [V20] 连通分量数: {num_components}")
    
    # Step 11: 提取每个连通分量的 bbox
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
        
        # V20 改进：bbox 膨胀（修正连通分量过于保守的问题）
        expand_px = config.get("bbox_expand_pixels", 15)
        if expand_px > 0:
            x1 = max(0, x1 - expand_px)
            y1 = max(0, y1 - expand_px)
            x2 = min(img_w, x2 + expand_px)
            y2 = min(img_h, y2 + expand_px)
        
        # 尺寸过滤
        area = (x2 - x1) * (y2 - y1)
        if area < min_area or area > max_area:
            logger.debug(f"  [V20] 过滤连通分量 {comp_id}: area={area} (min={min_area}, max={max_area})")
            continue
        
        # 计算该区域的平均分数
        region_score = float(score_map[comp_mask].mean())
        
        pred_bboxes.append([x1, y1, x2, y2])
        pred_scores.append(region_score)
        
        logger.debug(f"  [V20] 连通分量 {comp_id}: bbox=[{x1},{y1},{x2},{y2}], area={area}, score={region_score:.4f}")
    
    logger.debug(f"  [V20] 尺寸过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 12: NMS
    if len(pred_bboxes) > 1:
        nms_threshold = config.get("nms_iou_threshold", 0.5)
        keep_indices = nms(pred_bboxes, pred_scores, nms_threshold)
        pred_bboxes = [pred_bboxes[i] for i in keep_indices]
        pred_scores = [pred_scores[i] for i in keep_indices]
        logger.debug(f"  [V20] NMS 后剩余 {len(pred_bboxes)} 个框")
    
    # Step 13: Top-K 过滤
    top_k = config.get("top_k_per_image", 3)
    if len(pred_bboxes) > top_k:
        # 按分数降序排序，保留 top-k
        sorted_indices = sorted(range(len(pred_scores)), key=lambda i: pred_scores[i], reverse=True)
        pred_bboxes = [pred_bboxes[i] for i in sorted_indices[:top_k]]
        pred_scores = [pred_scores[i] for i in sorted_indices[:top_k]]
        logger.debug(f"  [V20] Top-{top_k} 过滤后剩余 {len(pred_bboxes)} 个框")
    
    # Step 14: V18/V19 优化 - 形状和分数联合过滤（保留）
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
            
            # 应用联合过滤条件
            if aspect_ratio < aspect_ratio_threshold and score >= score_threshold_p80:
                filtered_bboxes.append(bbox)
                filtered_scores.append(score)
            else:
                logger.debug(f"  [V20] 过滤框: bbox=[{x1},{y1},{x2},{y2}], aspect={aspect_ratio:.2f}, score={score:.4f}")
        
        num_filtered = len(pred_bboxes) - len(filtered_bboxes)
        if num_filtered > 0:
            logger.debug(f"  [V20] 形状+分数过滤: 移除 {num_filtered} 个框，剩余 {len(filtered_bboxes)} 个")
        
        pred_bboxes = filtered_bboxes
        pred_scores = filtered_scores
    
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
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v20")
    CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v20", "feature_bank_v20.pth")
    CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", "feature_banks_v20", "metadata.json")
    CONFIG["output_dir"]        = os.path.join(BASE_DIR, "outputs", "predictions_v20")
    CONFIG["checkpoint_dir"]    = os.path.join(BASE_DIR, "outputs", "checkpoints_v20")

    with torch.no_grad():
        run_inference(CONFIG)
