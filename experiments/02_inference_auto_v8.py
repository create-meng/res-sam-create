"""
Res-SAM V8 - Step 2: fully automatic inference.

This script implements the V8 automatic mainline corresponding to the
paper's fully automatic mode. Feature definition follows f=[W_out, b].
Image preprocessing is unified to a fixed 369x369 resize, and the V8
runtime seed is unified to 11.
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
        print(
            "错误：未安装 segment-anything，无法运行 Res-SAM 推理。\n"
            "请先安装：pip install segment-anything\n"
            "或参考：https://github.com/facebookresearch/segment-anything",
            flush=True,
        )
        raise SystemExit(1)


def _to_abs(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
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
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v8", "feature_bank_v8.pth"),
    "metadata_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v8", "metadata.json"),
    "test_data_dirs": {
        "cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
        "normal_auc": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "annotation_dirs": {
        "cavities": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_cavities", "annotations", "VOC_XML_format"
        ),
        "utilities": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_utilities", "annotations", "VOC_XML_format"
        ),
    },
    "output_dir": os.path.join(BASE_DIR, "outputs", "predictions_v8"),
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v8"),
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": DEFAULT_BETA_THRESHOLD,
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
    "version": "V8",
    "feature_with_bias": True,
    "automatic_coarse_use_patch_aggregation": True,
    "alignment_notes": (
        "V8 experiment B: solve true f=[W_out,b]; keep rectangle-wide fine-stage sampling, but use patch-aggregation coarse filtering; "
        "fb_source=augmented_intact, eval=augmented_intact + current annotated anomaly sets"
    ),
}


_max_images_env = (os.environ.get("MAX_IMAGES_PER_CATEGORY") or "").strip()
if _max_images_env:
    try:
        _max_images_val = int(_max_images_env)
        if _max_images_val > 0:
            CONFIG["max_images_per_category"] = _max_images_val
    except Exception:
        pass


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
            bboxes.append(
                [
                    int(bbox.find("xmin").text),
                    int(bbox.find("ymin").text),
                    int(bbox.find("xmax").text),
                    int(bbox.find("ymax").text),
                ]
            )
        return {"width": width, "height": height, "bboxes": bboxes}
    except Exception as exc:
        print(f"解析 XML 失败：{xml_path} | {exc}")
        return None


def load_image_with_orig_size(path: str, size_hw: tuple[int, int] | None) -> tuple[int, int, np.ndarray]:
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        img = im.convert("L")
        if size_hw:
            img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return orig_w, orig_h, arr


def run_inference(config: dict) -> dict:
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log_dir = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_inference_v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    runid_filter = _RunIdFilter(run_id)
    file_handler.addFilter(runid_filter)
    stream_handler.addFilter(runid_filter)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | run_id=%(run_id)s | %(message)s",
        handlers=[file_handler, stream_handler],
        force=True,
    )
    logger = logging.getLogger(__name__)

    config = dict(config)
    config["feature_bank_path"] = _to_abs(base_dir, config.get("feature_bank_path", ""))
    config["metadata_path"] = _to_abs(base_dir, config.get("metadata_path", ""))
    config["output_dir"] = _to_abs(base_dir, config.get("output_dir", ""))
    config["checkpoint_dir"] = _to_abs(base_dir, config.get("checkpoint_dir", ""))
    config["sam_checkpoint"] = _to_abs(base_dir, config.get("sam_checkpoint", ""))
    config["test_data_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("test_data_dirs", {}).items()}
    config["annotation_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("annotation_dirs", {}).items()}

    print("=" * 60)
    print("Res-SAM V8：全自动推理")
    print("=" * 60)
    print(f"日志文件：{log_file}")
    print(f"run_id: {run_id}")
    print(f"  window_size = {config['window_size']}")
    print(f"  stride = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    print(f"  Expected feature dim = {2 * config['hidden_size'] + 1}")
    print(f"  beta_threshold = {config['beta_threshold']}")
    print("=" * 60)

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(
            f"Feature bank not found: {config['feature_bank_path']}\n"
            "Please run 01_build_feature_bank_v8.py first."
        )

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM（V8）...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        sam_model_type=config["sam_model_type"],
        sam_checkpoint=config["sam_checkpoint"],
        device=config.get("device", "auto"),
        feature_with_bias=bool(config.get("feature_with_bias", False)),
        automatic_coarse_use_patch_aggregation=bool(config.get("automatic_coarse_use_patch_aggregation", False)),
    )

    print(f"加载 Feature Bank：{config['feature_bank_path']}")
    model.load_feature_bank(config["feature_bank_path"])

    metadata = {}
    if os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Feature Bank 形状：{metadata.get('feature_bank_shape', 'Unknown')}")

    all_results: dict[str, list] = {}

    for category, data_dir in config["test_data_dirs"].items():
        print(f"\n{'=' * 60}")
        print(f"处理类别：{category}")
        print(f"数据目录：{data_dir}")
        print("=" * 60)

        if not os.path.isdir(data_dir):
            print(f"警告：数据目录不存在：{data_dir}")
            continue

        image_files = sorted(
            f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))
        )
        if config["max_images_per_category"]:
            image_files = image_files[: int(config["max_images_per_category"])]
        print(f"共找到 {len(image_files)} 张图像")

        checkpoint_file = os.path.join(config["checkpoint_dir"], f"checkpoint_auto_{category}.json")
        category_results: list[dict] = []
        start_idx = 0
        last_completed = 0
        legacy_processed_files: set[str] = set()
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
            else:
                legacy_processed_files = set(checkpoint.get("processed_files", []) or [])
                if legacy_processed_files:
                    print(f"从旧断点继续：processed_files={len(legacy_processed_files)}")

        annotation_dir = config["annotation_dirs"].get(category, "")
        checkpoint_interval = int(config.get("checkpoint_interval", 50) or 0)
        effective_interval = 1 if image_files and len(image_files) <= checkpoint_interval else checkpoint_interval

        try:
            for i in tqdm(range(start_idx, len(image_files)), desc=f"处理 {category}", file=sys.stdout):
                img_file = image_files[i]
                if legacy_processed_files and img_file in legacy_processed_files:
                    last_completed = i + 1
                    continue

                img_path = os.path.join(data_dir, img_file)
                xml_name = os.path.splitext(img_file)[0] + ".xml"
                xml_path = os.path.join(annotation_dir, xml_name) if annotation_dir else ""
                gt = parse_voc_xml(xml_path) if xml_path and os.path.exists(xml_path) else None
                target_hw = target_hw_for_preprocess(config, gt)

                try:
                    orig_w, orig_h, img = load_image_with_orig_size(img_path, target_hw)
                    proc_h, proc_w = int(img.shape[0]), int(img.shape[1])

                    result = model.detect_automatic(
                        img,
                        min_region_area=config["min_region_area"],
                        max_regions=config["max_candidates_per_image"],
                    )

                    pred_bboxes_resized = [r["bbox"] for r in result.get("anomaly_regions", [])]
                    anomaly_scores = [r.get("max_anomaly_score", 0.0) for r in result.get("anomaly_regions", [])]

                    target_w = int(gt["width"]) if gt and gt.get("width") else int(orig_w)
                    target_h = int(gt["height"]) if gt and gt.get("height") else int(orig_h)
                    scale_x = target_w / float(proc_w)
                    scale_y = target_h / float(proc_h)
                    pred_bboxes = [
                        [
                            int(b[0] * scale_x),
                            int(b[1] * scale_y),
                            int(b[2] * scale_x),
                            int(b[3] * scale_y),
                        ]
                        for b in pred_bboxes_resized
                    ]

                    record = {
                        "image_name": img_file,
                        "image_path": img_path,
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
                        gt_boxes_resized = [
                            [
                                int(bbox[0] * inv_scale_x),
                                int(bbox[1] * inv_scale_y),
                                int(bbox[2] * inv_scale_x),
                                int(bbox[3] * inv_scale_y),
                            ]
                            for bbox in gt["bboxes"]
                        ]
                        record.update(
                            {
                                "gt_bboxes": gt["bboxes"],
                                "gt_bboxes_resized": gt_boxes_resized,
                                "num_gt": int(len(gt["bboxes"])),
                                "gt_width": gt["width"],
                                "gt_height": gt["height"],
                                "exclude_from_det_metrics": False,
                                "exclude_from_auc": False,
                            }
                        )
                    elif category == "normal_auc":
                        record.update(
                            {
                                "gt_bboxes": [],
                                "gt_bboxes_resized": [],
                                "num_gt": 0,
                                "exclude_from_det_metrics": True,
                                "exclude_from_auc": False,
                            }
                        )
                    else:
                        record.update(
                            {
                                "gt_bboxes": [],
                                "gt_bboxes_resized": [],
                                "num_gt": 0,
                                "exclude_from_det_metrics": True,
                                "exclude_from_auc": True,
                            }
                        )

                    category_results.append(record)
                    category_success += 1
                    last_completed = i + 1

                    if effective_interval > 0 and last_completed % effective_interval == 0:
                        with open(checkpoint_file, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "category": category,
                                    "processed_count": int(last_completed),
                                    "results": category_results,
                                    "timestamp": datetime.now().isoformat(),
                                    "completed": False,
                                },
                                f,
                                ensure_ascii=False,
                            )

                except Exception as exc:
                    category_failed += 1
                    err_low = str(exc).lower()
                    if "segment-anything" in err_low or "segment_anything" in err_low:
                        reason = "missing_segment_anything"
                    elif "cannot identify image file" in err_low or "unidentifiedimageerror" in err_low:
                        reason = "image_open_failed"
                    elif "no such file" in err_low or "not found" in err_low:
                        reason = "file_not_found"
                    else:
                        reason = "unknown"
                    fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + 1
                    print(f"处理图像失败：{img_file} | {exc}")
                    logger.exception("Error processing image=%s | category=%s", img_file, category)

        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt：保存断点后中止 | category=%s | processed_count=%s | path=%s",
                category,
                int(last_completed),
                checkpoint_file,
            )
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "category": category,
                        "processed_count": int(last_completed),
                        "results": category_results,
                        "timestamp": datetime.now().isoformat(),
                        "completed": False,
                    },
                    f,
                    ensure_ascii=False,
                )
            raise

        all_results[category] = category_results
        print(f"类别 {category} 已完成，共处理 {len(category_results)} 张图像")

        completed_ok = not (category_success == 0 and category_failed > 0)
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "category": category,
                    "processed_count": int(last_completed),
                    "results": category_results,
                    "timestamp": datetime.now().isoformat(),
                    "completed": bool(completed_ok),
                    "summary": {
                        "total": int(len(image_files)),
                        "success": int(category_success),
                        "failed": int(category_failed),
                        "fail_reasons": dict(fail_reasons),
                    },
                },
                f,
                ensure_ascii=False,
            )

    version_tag = str(config.get("version", "V8")).strip().lower()
    output_file = os.path.join(config["output_dir"], f"auto_predictions_{version_tag}.json")
    output_data = {
        "meta": {
            "version": config["version"],
            "alignment_notes": config["alignment_notes"],
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "feature_bank_metadata_path": config.get("metadata_path", ""),
            "preprocess_signature": metadata.get("preprocess_signature", None),
            "resize_policy": config.get("resize_policy"),
            "fixed_image_size_hw": list(config.get("image_size", (369, 369))),
            "image_preprocess_note": ("V8 mainline uses fixed resize; all images are resized to fixed_image_size_hw before SAM, 2D-ESN, and evaluation coordinate mapping."),
            "window_size": int(config.get("window_size", 50)),
            "stride": int(config.get("stride", 5)),
            "hidden_size": int(config.get("hidden_size", 30)),
            "beta_threshold": float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
            "feature_with_bias": bool(config.get("feature_with_bias", False)),
            "automatic_coarse_use_patch_aggregation": bool(config.get("automatic_coarse_use_patch_aggregation", False)),
        },
        "results": all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    total_images = sum(len(results) for results in all_results.values())
    total_detections = sum(len(r["pred_bboxes"]) for results in all_results.values() for r in results)
    total_coarse_discarded = sum(int(r["num_coarse_discarded"]) for results in all_results.values() for r in results)

    print(f"\n结果已保存到：{output_file}")
    print("\n统计汇总：")
    print(f"  处理图像数：{total_images}")
    print(f"  检出异常框数：{total_detections}")
    print(f"  Region 级粗筛剔除数：{total_coarse_discarded}")
    print("V8 全自动推理完成！")
    print("=" * 60)

    logger.info("结果已保存：%s", output_file)
    return all_results


if __name__ == "__main__":
    preflight_faiss_or_raise()
    _require_segment_anything()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v8")
    with torch.no_grad():
        run_inference(CONFIG)

