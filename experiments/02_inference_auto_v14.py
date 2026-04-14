"""
Res-SAM v14 - Step 2: fully automatic inference.

V14 相对 V13 的改进：
1. 自适应 beta（方案1）：从 metadata 读取建库时计算的 adaptive_beta，替换固定 beta=0.1
   - 解决 V13 粗筛丢弃率=0% 的问题
2. top-1 pred 框（方案2）：每张图只保留 anomaly_score 最高的 1 个框
   - 每张图 GT 只有 1 个，多余的框全是 FP，直接过滤
   - 预期大幅提升 Precision

继承 V13：
- GPR row_mean 背景去除
- merge_all_anomaly_patches=True
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
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v14", "feature_bank_v14.pth"),
    "metadata_path":     os.path.join(BASE_DIR, "outputs", "feature_banks_v14", "metadata.json"),
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
    "output_dir":     os.path.join(BASE_DIR, "outputs", "predictions_v14"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v14"),
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    # V14：beta 从 metadata 读取（adaptive_beta），此处为 fallback
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
    "use_adaptive_beta": True,          # True = 从 metadata 读取 adaptive_beta
    # 继承 V13
    "merge_all_anomaly_patches": True,
    "gpr_background_removal": True,
    "background_removal_method": "row_mean",
    # V14 新增：top-1 框过滤
    "top_k_preds": 1,                   # 每张图只保留 score 最高的 k 个框，None = 不过滤
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
    "version": "v14",
    "feature_with_bias": True,
    "automatic_fine_use_mask": True,
    "alignment_notes": (
        "v14: 自适应beta(p95 1-NN) + top-1框 + GPR背景去除 + merge_all; "
        "解决V13粗筛丢弃率=0%和FP过多问题"
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


def remove_gpr_background(arr: np.ndarray, method: str = "row_mean") -> np.ndarray:
    if method == "row_mean":
        bg = arr.mean(axis=1, keepdims=True)
    elif method == "row_median":
        bg = np.median(arr, axis=1, keepdims=True)
    else:
        return arr
    result = arr - bg
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
            arr = remove_gpr_background(arr, config.get("background_removal_method", "row_mean"))
    return orig_w, orig_h, arr


def apply_top_k_filter(pred_bboxes: list, anomaly_scores: list, k: int | None):
    """
    V14 方案2：只保留 score 最高的 k 个 pred 框。
    每张图 GT 只有 1 个，多余的框全是 FP。
    """
    if k is None or len(pred_bboxes) <= k:
        return pred_bboxes, anomaly_scores
    # 按 score 降序排列，取前 k 个
    paired = sorted(zip(anomaly_scores, pred_bboxes), key=lambda x: x[0], reverse=True)
    top_scores = [s for s, _ in paired[:k]]
    top_bboxes = [b for _, b in paired[:k]]
    return top_bboxes, top_scores


def run_inference(config: dict) -> dict:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log_dir = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_inference_v14_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

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

    # V14：从 metadata 读取自适应 beta
    effective_beta = config["beta_threshold"]
    if config.get("use_adaptive_beta", True) and os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "adaptive_beta" in metadata:
            effective_beta = float(metadata["adaptive_beta"])
            print(f"  [V14] 自适应 beta = {effective_beta:.4f}（从 metadata 读取，"
                  f"V13 固定值 = {metadata.get('fixed_beta_v13', DEFAULT_BETA_THRESHOLD)}）")
        else:
            print(f"  [V14] metadata 中无 adaptive_beta，使用默认值 {effective_beta}")
    config["beta_threshold"] = effective_beta

    print("=" * 60)
    print("Res-SAM v14：全自动推理")
    print("=" * 60)
    print(f"  beta_threshold (adaptive) = {config['beta_threshold']:.4f}")
    print(f"  merge_all                 = {config.get('merge_all_anomaly_patches')}")
    print(f"  top_k_preds               = {config.get('top_k_preds')}")
    print(f"  gpr_background_removal    = {config.get('gpr_background_removal')}")
    print("=" * 60)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(f"Feature bank not found: {config['feature_bank_path']}\n"
                                f"请先运行 01_build_feature_bank_v14.py")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM（v14）...")
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
    top_k = config.get("top_k_preds")

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

                    result = model.detect_automatic(
                        img,
                        min_region_area=config["min_region_area"],
                        max_regions=config["max_candidates_per_image"],
                    )

                    pred_bboxes_resized = [r["bbox"] for r in result.get("anomaly_regions", [])]
                    anomaly_scores = [r.get("max_anomaly_score", 0.0)
                                      for r in result.get("anomaly_regions", [])]

                    # V14 方案2：top-k 过滤
                    pred_bboxes_resized, anomaly_scores = apply_top_k_filter(
                        pred_bboxes_resized, anomaly_scores, top_k
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
                        "num_preds_before_topk": len([r for r in result.get("anomaly_regions", [])]),
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

    output_file = os.path.join(config["output_dir"], "auto_predictions_v14.json")
    output_data = {
        "meta": {
            "version": "v14",
            "alignment_notes": config["alignment_notes"],
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
            "hidden_size": int(config["hidden_size"]),
            "beta_threshold": float(config["beta_threshold"]),
            "beta_source": "adaptive_p95" if config.get("use_adaptive_beta") else "fixed",
            "merge_all_anomaly_patches": bool(config.get("merge_all_anomaly_patches", False)),
            "gpr_background_removal": bool(config.get("gpr_background_removal", False)),
            "background_removal_method": config.get("background_removal_method", "row_mean"),
            "top_k_preds": config.get("top_k_preds"),
        },
        "results": all_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_images = sum(len(r) for r in all_results.values())
    total_detections = sum(len(r["pred_bboxes"]) for recs in all_results.values() for r in recs)
    print(f"\n结果保存至：{output_file}")
    print(f"处理图像数：{total_images}，检出框数：{total_detections}")
    print("v14 全自动推理完成！")
    return all_results


if __name__ == "__main__":
    preflight_faiss_or_raise()
    _require_segment_anything()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v14")
    CONFIG["feature_bank_path"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v14", "feature_bank_v14.pth")
    CONFIG["metadata_path"]     = os.path.join(BASE_DIR, "outputs", "feature_banks_v14", "metadata.json")
    CONFIG["output_dir"]        = os.path.join(BASE_DIR, "outputs", "predictions_v14")
    CONFIG["checkpoint_dir"]    = os.path.join(BASE_DIR, "outputs", "checkpoints_v14")
    with torch.no_grad():
        run_inference(CONFIG)
