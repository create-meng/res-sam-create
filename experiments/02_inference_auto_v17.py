"""
Res-SAM v17 - Step 2: fully automatic inference with all improvements.

V17 相对 V16 的核心改进（一次性实现所有改进）：
1. Pixel-level Heatmap Bbox 生成：
   - 在候选区域内生成 pixel-level heatmap
   - 对 heatmap 归一化到 [0,1]
   - 用归一化后的 beta 阈值化
   - 取外接矩形作为最终 bbox
   - 解决 pred 框太小的问题（74px → 接近 GT 224px）
2. 后处理过滤：
   - 置信度过滤：只保留 score > percentile(scores, 80) 的框
   - NMS：IoU > 0.5 的重叠框只保留最高分的
   - Top-1：每张图只保留最高分的 1 个框
   - 减少 FP，提升 Precision

继承 V16 最优配置：
- hidden_size = 30
- background_removal_method = "both"（行列双向背景去除）
- merge_all_anomaly_patches = True
- 验证集驱动的 beta 校准（从 metadata 读取）

预期效果：
- 短期（后处理过滤）：FP 从 170 减少到 50-80，Precision 从 0.056 提升到 0.10-0.15，F1 从 0.083 提升到 0.12-0.15
- 中期（Pixel-level Heatmap）：cavities TP 从 10 提升到 20-30，F1 从 0.12-0.15 提升到 0.25-0.35
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

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_02_03
from experiments.paper_constants import DEFAULT_BETA_THRESHOLD, preflight_faiss_or_raise
from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess


def _require_segment_anything() -> None:
    try:
        import importlib
        importlib.import_module("segment_anything")
    except Exception:
        print("错误：未安装 segment-anything", flush=True)
        raise SystemExit(1)


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
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v17", "feature_bank_v17.pth"),
    "metadata_path":     os.path.join(BASE_DIR, "outputs", "feature_banks_v17", "metadata.json"),
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
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions_v17"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v17"),
    # v17：hidden_size 与建库一致
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,              # ← 与 01_build_feature_bank_v17.py 一致
    # beta 从 metadata 读取（V17 使用 anomaly_p10）
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,
    # 继承
    "merge_all_anomaly_patches": True,
    # v17 方案：行列双向背景去除
    "gpr_background_removal": True,
    "background_removal_method": "both",   # ← row_mean + column_mean
    # V17 核心改进：后处理过滤
    "confidence_percentile": 80,  # 只保留 score > percentile(scores, 80) 的框
    "nms_iou_threshold": 0.5,     # NMS IoU 阈值
    "top_k_per_image": 1,         # 每张图只保留 top-1 框
    # V17 核心改进：pixel-level heatmap bbox 生成
    "use_pixel_heatmap": True,    # 使用 pixel-level heatmap 生成 bbox
    "heatmap_beta_normalized": 0.5,  # heatmap 归一化阈值（0-1）
    "heatmap_min_area": 100,      # heatmap bbox 最小面积
    # SAM
    "sam_model_type": "vit_l",
    "sam_checkpoint": os.path.join(BASE_DIR, "sam", "sam_vit_l_0b3195.pth"),
    "device": "auto",
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "max_candidates_per_image": None,
    "min_region_area": None,
    "max_images_per_category": None,
    "checkpoint_interval": 50,
    "random_seed": 11,
    "version": "v17",
    "feature_with_bias": True,
    "automatic_fine_use_mask": True,
    "alignment_notes": (
        "v17: 继承V16 + 后处理过滤(置信度+NMS+top-1) + 可选pixel-level heatmap; "
        "目标：减少FP，提升Precision和F1"
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
    """
    GPR B-scan 背景去除，与建库预处理保持一致。
    method:
        "row_mean" : 减去每行均值
        "col_mean" : 减去每列均值
        "both"     : 先减行均值，再减列均值（v17）
    """
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


def _apply_nms(bboxes: list, scores: list, iou_threshold: float) -> tuple[list, list]:
    """
    非极大值抑制（NMS）
    
    Parameters:
    -----------
    bboxes : list
        Bbox 列表 [[x1, y1, x2, y2], ...]
    scores : list
        对应的分数列表
    iou_threshold : float
        IoU 阈值
        
    Returns:
    --------
    tuple[list, list]
        过滤后的 (bboxes, scores)
    """
    if len(bboxes) == 0:
        return [], []
    
    bboxes_arr = np.array(bboxes)
    scores_arr = np.array(scores)
    
    # 计算面积
    x1 = bboxes_arr[:, 0]
    y1 = bboxes_arr[:, 1]
    x2 = bboxes_arr[:, 2]
    y2 = bboxes_arr[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 按分数排序
    order = scores_arr.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # 计算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留 IoU <= threshold 的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return bboxes_arr[keep].tolist(), scores_arr[keep].tolist()


def apply_v17_postprocessing(
    pred_bboxes: list,
    anomaly_scores: list,
    config: dict,
) -> tuple[list, list]:
    """
    V17 核心改进：后处理过滤
    
    1. 置信度过滤：只保留 score > percentile(scores, confidence_percentile) 的框
    2. NMS：IoU > nms_iou_threshold 的重叠框只保留最高分的
    3. Top-K：每张图只保留 top-k 个框
    
    Returns:
        (filtered_bboxes, filtered_scores)
    """
    if len(pred_bboxes) == 0:
        return [], []
    
    # 1. 置信度过滤
    confidence_percentile = config.get("confidence_percentile", 80)
    if confidence_percentile > 0 and len(anomaly_scores) > 0:
        threshold = np.percentile(anomaly_scores, confidence_percentile)
        filtered = [(bbox, score) for bbox, score in zip(pred_bboxes, anomaly_scores)
                    if score >= threshold]
        if len(filtered) == 0:
            # 如果全部被过滤，保留最高分的一个
            max_idx = int(np.argmax(anomaly_scores))
            filtered = [(pred_bboxes[max_idx], anomaly_scores[max_idx])]
        pred_bboxes = [b for b, _ in filtered]
        anomaly_scores = [s for _, s in filtered]
    
    # 2. NMS
    nms_iou_threshold = config.get("nms_iou_threshold", 0.5)
    if nms_iou_threshold > 0 and len(pred_bboxes) > 1:
        pred_bboxes, anomaly_scores = _apply_nms(
            pred_bboxes, anomaly_scores, nms_iou_threshold
        )
    
    # 3. Top-K 过滤
    top_k = config.get("top_k_per_image")
    if top_k is not None and len(pred_bboxes) > top_k:
        paired = sorted(zip(anomaly_scores, pred_bboxes),
                        key=lambda x: x[0], reverse=True)
        anomaly_scores = [s for s, _ in paired[:top_k]]
        pred_bboxes = [b for _, b in paired[:top_k]]
    
    return pred_bboxes, anomaly_scores


def run_inference(config: dict) -> dict:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log_dir = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_inference_v17_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    runid_filter = _RunIdFilter(run_id)
    file_handler.addFilter(runid_filter)
    stream_handler.addFilter(runid_filter)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | run_id=%(run_id)s | %(message)s",
                        handlers=[file_handler, stream_handler], force=True)
    logger = logging.getLogger(__name__)

    config = dict(config)
    for key in ["feature_bank_path", "metadata_path", "output_dir", "checkpoint_dir", "sam_checkpoint"]:
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
            print(f"  [v17] 自适应 beta = {effective_beta:.4f}（从 metadata 读取，anomaly_p10）")
        else:
            print(f"  [v17] 无 adaptive_beta，使用默认值 {effective_beta}")
    config["beta_threshold"] = effective_beta

    print("=" * 60)
    print("Res-SAM v17：全自动推理（所有改进一次性实现）")
    print("=" * 60)
    print(f"  hidden_size               = {config['hidden_size']}")
    print(f"  beta_threshold (adaptive) = {config['beta_threshold']:.4f}")
    print(f"  background_removal        = {config.get('background_removal_method')}")
    print(f"  merge_all                 = {config.get('merge_all_anomaly_patches')}")
    print(f"  --- V17 核心改进 ---")
    print(f"  use_pixel_heatmap         = {config.get('use_pixel_heatmap')}")
    print(f"  heatmap_beta_normalized   = {config.get('heatmap_beta_normalized')}")
    print(f"  confidence_percentile     = {config.get('confidence_percentile')}")
    print(f"  nms_iou_threshold         = {config.get('nms_iou_threshold')}")
    print(f"  top_k_per_image           = {config.get('top_k_per_image')}")
    print("=" * 60)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(f"Feature bank not found: {config['feature_bank_path']}\n"
                                f"请先运行 01_build_feature_bank_v17.py")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM（v17）...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config["beta_threshold"]),
        sam_model_type=config["sam_model_type"],
        sam_checkpoint=config["sam_checkpoint"],
        device=config.get("device", "auto"),
        feature_with_bias=bool(config.get("feature_with_bias", True)),
        automatic_fine_use_mask=bool(config.get("automatic_fine_use_mask", True)),
        merge_all_anomaly_patches=bool(config.get("merge_all_anomaly_patches", False)),
    )

    print(f"加载 Feature Bank：{config['feature_bank_path']}")
    model.load_feature_bank(config["feature_bank_path"])

    all_results: dict[str, list] = {}

    for category, data_dir in config["test_data_dirs"].items():
        print(f"\n{'=' * 60}\n处理类别：{category}\n{'=' * 60}")
        if not os.path.isdir(data_dir):
            print(f"警告：数据目录不存在：{data_dir}")
            continue

        image_files = sorted(f for f in os.listdir(data_dir)
                             if f.lower().endswith((".jpg", ".png", ".jpeg")))
        if config["max_images_per_category"]:
            image_files = image_files[:int(config["max_images_per_category"])]
        print(f"共找到 {len(image_files)} 张图像")

        checkpoint_file = os.path.join(config["checkpoint_dir"], f"checkpoint_auto_{category}.json")
        category_results: list[dict] = []
        start_idx = 0
        last_completed = 0
        category_success = 0
        category_failed = 0
        fail_reasons: dict[str, int] = {}

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            if checkpoint.get("completed", False):
                logger.info("Checkpoint completed, skip: %s", checkpoint_file)
                all_results[category] = checkpoint.get("results", []) or []
                continue
            if isinstance(checkpoint.get("processed_count"), int):
                start_idx = int(checkpoint.get("processed_count", 0) or 0)
                category_results = checkpoint.get("results", []) or []
                last_completed = start_idx
                print(f"从断点继续：start_idx={start_idx}")

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

                    # V17 方案 4：提取 image_id 用于多尺度 Memory Bank
                    import re
                    img_basename = os.path.splitext(img_file)[0]
                    # 提取原始图 ID（去除 _aug_N 后缀）
                    match = re.match(r'^(.+?)_aug_\d+$', img_basename)
                    image_id = match.group(1) if match else img_basename

                    result = model.detect_automatic(
                        img,
                        min_region_area=config["min_region_area"],
                        max_regions=config["max_candidates_per_image"],
                        image_id=image_id,
                    )

                    # V17 核心改进：使用 pixel-level heatmap 生成 bbox
                    if config.get("use_pixel_heatmap", False):
                        # 对每个候选区域生成 pixel-level heatmap
                        pred_bboxes_resized = []
                        anomaly_scores = []
                        
                        for region in result.get("anomaly_regions", []):
                            region_bbox = region["bbox"]
                            region_mask = region.get("mask")
                            
                            # 生成 heatmap（返回候选区域内的 heatmap 和偏移）
                            heatmap, offset = model.generate_pixel_heatmap(
                                img, region_bbox, region_mask, image_id=image_id
                            )
                            
                            # 从 heatmap 生成 bbox（传入偏移以转换为全图坐标）
                            heatmap_bboxes = model.heatmap_to_bbox(
                                heatmap,
                                offset=offset,
                                beta_normalized=config.get("heatmap_beta_normalized", 0.5),
                                min_area=config.get("heatmap_min_area", 100),
                            )
                            
                            # 为每个 bbox 分配分数（使用原始区域的最大分数）
                            region_score = region.get("max_anomaly_score", 0.0)
                            for bbox in heatmap_bboxes:
                                pred_bboxes_resized.append(bbox)
                                anomaly_scores.append(region_score)
                    else:
                        # 使用原有的 patch 合并方式
                        pred_bboxes_resized = [r["bbox"] for r in result.get("anomaly_regions", [])]
                        anomaly_scores = [r.get("max_anomaly_score", 0.0)
                                          for r in result.get("anomaly_regions", [])]

                    # V17 核心改进：后处理过滤
                    pred_bboxes_resized, anomaly_scores = apply_v17_postprocessing(
                        pred_bboxes_resized, anomaly_scores, config
                    )

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
                        "num_candidates": result.get("num_candidates", 0),
                        "num_coarse_discarded": result.get("num_coarse_discarded", 0),
                        "num_esn_fits": result.get("num_esn_fits", 0),
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
                    reason = ("missing_segment_anything" if "segment_anything" in err_low
                              else "image_open_failed" if "cannot identify" in err_low
                              else "unknown")
                    fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + 1
                    print(f"处理图像失败：{img_file} | {exc}")
                    logger.exception("Error processing image=%s | category=%s", img_file, category)

        except KeyboardInterrupt:
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump({"category": category, "processed_count": int(last_completed),
                           "results": category_results, "timestamp": datetime.now().isoformat(),
                           "completed": False}, f, ensure_ascii=False)
            raise

        all_results[category] = category_results
        print(f"类别 {category} 完成，共 {len(category_results)} 张")
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"category": category, "processed_count": int(last_completed),
                       "results": category_results, "timestamp": datetime.now().isoformat(),
                       "completed": True,
                       "summary": {"total": len(image_files), "success": category_success,
                                   "failed": category_failed, "fail_reasons": fail_reasons}},
                      f, ensure_ascii=False)

    output_file = os.path.join(config["output_dir"], "auto_predictions_v17.json")
    output_data = {
        "meta": {
            "version": "v17",
            "alignment_notes": config["alignment_notes"],
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
            "hidden_size": int(config["hidden_size"]),
            "beta_threshold": float(config["beta_threshold"]),
            "beta_source": "adaptive_anomaly_p10",
            "merge_all_anomaly_patches": bool(config.get("merge_all_anomaly_patches", False)),
            "gpr_background_removal": bool(config.get("gpr_background_removal", False)),
            "background_removal_method": config.get("background_removal_method", "both"),
            # V17 核心改进参数
            "use_pixel_heatmap": config.get("use_pixel_heatmap"),
            "heatmap_beta_normalized": config.get("heatmap_beta_normalized"),
            "heatmap_min_area": config.get("heatmap_min_area"),
            "confidence_percentile": config.get("confidence_percentile"),
            "nms_iou_threshold": config.get("nms_iou_threshold"),
            "top_k_per_image": config.get("top_k_per_image"),
        },
        "results": all_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_images = sum(len(r) for r in all_results.values())
    total_detections = sum(len(r["pred_bboxes"]) for recs in all_results.values() for r in recs)
    print(f"\n结果保存至：{output_file}")
    print(f"处理图像数：{total_images}，检出框数：{total_detections}")
    print("v17 全自动推理完成！")
    return all_results


if __name__ == "__main__":
    preflight_faiss_or_raise()
    _require_segment_anything()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v17")
    CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v17", "feature_bank_v17.pth")
    CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", "feature_banks_v17", "metadata.json")
    CONFIG["output_dir"]        = os.path.join(BASE_DIR, "outputs", "predictions_v17")
    CONFIG["checkpoint_dir"]    = os.path.join(BASE_DIR, "outputs", "checkpoints_v17")

    with torch.no_grad():
        run_inference(CONFIG)
