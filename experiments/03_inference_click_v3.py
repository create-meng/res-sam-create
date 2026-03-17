"""Res-SAM 复现实验 V3 - Click-guided 模式推理

功能：
- 按 Table 1 的 click 配置（5/5、5/3、3/1）模拟用户点击
  - 正点击：从 GT bbox 内采样
  - 负点击：从 GT bbox 外采样
- 使用 ResSAM.detect_click_guided() 执行点击引导异常检测
- 支持断点继续（每个 click_config + category 独立断点）
- 输出到 outputs/predictions_v3/click_predictions_v3.json

论文对应：Table 1
"""

import sys
import os
import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime
import uuid
import logging
from logging.handlers import RotatingFileHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


CONFIG = {
    # Feature Bank 路径（V3）
    "feature_bank_path": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "feature_banks_v3",
        "feature_bank_v3.pth",
    ),
    "metadata_path": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "feature_banks_v3",
        "metadata.json",
    ),
    # 测试数据（open-source annotated 子集）
    "test_data_dirs": {
        "cavities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "GPR_data",
            "augmented_cavities",
        ),
        "utilities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "GPR_data",
            "augmented_utilities",
        ),
    },
    "annotation_dirs": {
        "cavities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "GPR_data",
            "augmented_cavities",
            "annotations",
            "VOC_XML_format",
        ),
        "utilities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "GPR_data",
            "augmented_utilities",
            "annotations",
            "VOC_XML_format",
        ),
    },
    # 输出
    "output_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "predictions_v3",
    ),
    "checkpoint_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "checkpoints_v3",
    ),
    # 论文参数（V3 严格对齐）
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": 0.5,
    "anomaly_threshold": 0.5,
    "region_coarse_threshold": 0.5,
    # SAM
    "sam_model_type": "vit_b",
    "sam_checkpoint": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "sam", "sam_vit_b_01ec64.pth"
    ),
    # 图像预处理
    "image_size": (369, 369),
    # click 配置（Table 1 的 5/5、5/3、3/1）
    "click_configs": [
        {"name": "5/5", "pos_clicks": 5, "neg_clicks": 5},
        {"name": "5/3", "pos_clicks": 5, "neg_clicks": 3},
        {"name": "3/1", "pos_clicks": 3, "neg_clicks": 1},
    ],
    # 断点
    "checkpoint_interval": 20,
    "max_images_per_category": None,
    # 随机种子
    "random_seed": 42,
}


_max_images_env = os.environ.get('MAX_IMAGES_PER_CATEGORY', '').strip()
if _max_images_env:
    try:
        _max_images_val = int(_max_images_env)
        if _max_images_val > 0:
            CONFIG['max_images_per_category'] = _max_images_val
    except Exception:
        pass


CONFIG["test_data_dirs"]["normal_auc"] = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "GPR_data",
    "intact",
)
CONFIG["annotation_dirs"]["normal_auc"] = ""


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id
        return True


def parse_voc_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            objects.append(
                {
                    "name": name,
                    "xmin": int(bbox.find("xmin").text),
                    "ymin": int(bbox.find("ymin").text),
                    "xmax": int(bbox.find("xmax").text),
                    "ymax": int(bbox.find("ymax").text),
                }
            )

        return {"width": img_width, "height": img_height, "objects": objects}
    except Exception:
        return None


def load_image(path: str, size: tuple = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
    return img_array


def _to_abs(base_dir: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p

    base_name = os.path.basename(base_dir)
    p_norm = p.replace("/", os.sep)
    if p_norm == base_name or p_norm.startswith(base_name + os.sep):
        p_norm = p_norm[len(base_name) + 1 :]

    return os.path.abspath(os.path.join(base_dir, p_norm))


def _iter_checkpoint_files(checkpoint_dir: str):
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return
    for root, _, files in os.walk(checkpoint_dir):
        for fn in files:
            if fn.lower().endswith(".json"):
                yield os.path.join(root, fn)


def _load_checkpoint_if_valid(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and ("results" in obj or "processed_count" in obj or "completed" in obj):
            return obj
    except Exception:
        return None
    return None


def _find_legacy_checkpoint(checkpoint_dir: str, cfg_name: str, category: str) -> str:
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return ""

    cfg_variants = [
        f"click_{cfg_name}",
        f"click_{cfg_name.replace('/', '_')}",
    ]
    if cfg_name and "/" in cfg_name:
        cfg_variants.append(f"click_{cfg_name.split('/')[0]}")

    needles = [category.lower(), "checkpoint", "v3"] + [v.lower() for v in cfg_variants]

    candidates = []
    for p in _iter_checkpoint_files(checkpoint_dir):
        path_l = p.lower()
        base_l = os.path.basename(p).lower()
        if not all(n in path_l or n in base_l for n in needles[:3]):
            continue
        if not any(v in path_l or v in base_l for v in needles[3:]):
            continue
        candidates.append(p)

    if not candidates:
        return ""

    candidates.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0.0, reverse=True)
    return candidates[0]


def _scale_box_to_resized(gt_box: list, gt_w: int, gt_h: int, resized_h: int, resized_w: int) -> list:
    scale_x = resized_w / float(gt_w)
    scale_y = resized_h / float(gt_h)
    xmin, ymin, xmax, ymax = gt_box
    return [
        int(xmin * scale_x),
        int(ymin * scale_y),
        int(xmax * scale_x),
        int(ymax * scale_y),
    ]


def _scale_box_to_original(pred_box: list, gt_w: int, gt_h: int, resized_h: int, resized_w: int) -> list:
    scale_x = gt_w / float(resized_w)
    scale_y = gt_h / float(resized_h)
    x1, y1, x2, y2 = pred_box
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    ]


def _sample_clicks_from_gt_bbox(gt_box_resized: list, pos_clicks: int, neg_clicks: int, img_w: int, img_h: int):
    xmin, ymin, xmax, ymax = gt_box_resized
    # IMPORTANT: This repo's VOC annotations behave like half-open intervals:
    # [xmin, ymin, xmax, ymax) where xmax/ymax may equal image width/height.
    xmin = max(0, min(int(xmin), img_w))
    xmax = max(0, min(int(xmax), img_w))
    ymin = max(0, min(int(ymin), img_h))
    ymax = max(0, min(int(ymax), img_h))

    pos_points = []
    neg_points = []

    for _ in range(max(0, pos_clicks)):
        # Sample strictly inside bbox. Since bbox is [xmin, xmax), the max valid x is xmax-1.
        if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
            break
        x = random.randint(xmin, max(xmin, xmax - 1))
        y = random.randint(ymin, max(ymin, ymax - 1))
        pos_points.append((x, y))

    attempts = 0
    while len(neg_points) < max(0, neg_clicks) and attempts < 200:
        x = random.randint(0, img_w - 1)
        y = random.randint(0, img_h - 1)
        # Outside check uses the same half-open containment.
        if not (xmin <= x < xmax and ymin <= y < ymax):
            neg_points.append((x, y))
        attempts += 1

    return pos_points, neg_points


def run_click_guided(config: dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    log_dir = os.path.join(base_dir, 'outputs', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'click_inference_v3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
    stream_handler = logging.StreamHandler()

    runid_filter = _RunIdFilter(run_id)
    file_handler.addFilter(runid_filter)
    stream_handler.addFilter(runid_filter)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | run_id=%(run_id)s | %(message)s',
        handlers=[
            file_handler,
            stream_handler,
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(f"run_id={run_id}")

    logger.info('=' * 60)
    logger.info('Res-SAM V3: Click-guided Inference (Strict Paper Alignment)')
    logger.info('=' * 60)
    logger.info(f"window_size={config.get('window_size')}, stride={config.get('stride')}, hidden_size={config.get('hidden_size')}")
    logger.info(f"beta_threshold={config.get('beta_threshold')}, anomaly_threshold={config.get('anomaly_threshold')}, region_coarse_threshold={config.get('region_coarse_threshold')}")
    logger.info(f"image_size_hw={config.get('image_size')}")
    logger.info(f"feature_bank_path={config.get('feature_bank_path')}")
    logger.info(f"sam_checkpoint={config.get('sam_checkpoint')}")
    logger.info(f"click_configs={config.get('click_configs', [])}")
    logger.info(f"max_images_per_category={config.get('max_images_per_category')}")
    logger.info(f"log_file={log_file}")
    logger.info('=' * 60)

    print("=" * 60)
    print("Res-SAM V3: Click-guided Inference")
    print("=" * 60)
    print(f"日志文件: {log_file}")

    config = dict(config)
    config["feature_bank_path"] = _to_abs(base_dir, config.get("feature_bank_path", ""))
    config["metadata_path"] = _to_abs(base_dir, config.get("metadata_path", ""))
    config["output_dir"] = _to_abs(base_dir, config.get("output_dir", ""))
    config["checkpoint_dir"] = _to_abs(base_dir, config.get("checkpoint_dir", ""))
    config["sam_checkpoint"] = _to_abs(base_dir, config.get("sam_checkpoint", ""))
    config["test_data_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("test_data_dirs", {}).items()}
    config["annotation_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("annotation_dirs", {}).items()}

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(
            f"Feature bank not found: {config['feature_bank_path']}\n"
            "Please run 01_build_feature_bank_v3.py first."
        )

    metadata = {}
    if os.path.exists(config.get("metadata_path", "")):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if metadata:
        logger.info(f"feature_bank_sources={list(metadata.get('sources', {}).keys())}")
        logger.info(f"feature_bank_dimension_match={metadata.get('dimension_match', 'N/A')}")
        logger.info(f"feature_bank_feature_dim_actual={metadata.get('feature_dim_actual', 'N/A')}")
        logger.info(f"preprocess_signature={metadata.get('preprocess_signature', None)}")

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    from PatchRes.ResSAM import ResSAM

    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        anomaly_threshold=config.get("beta_threshold", config["anomaly_threshold"]),
        region_coarse_threshold=config.get("beta_threshold", config["region_coarse_threshold"]),
        sam_model_type=config["sam_model_type"],
        sam_checkpoint=config["sam_checkpoint"],
    )
    model.load_feature_bank(config["feature_bank_path"])

    logger.info('ResSAM initialized and feature bank loaded')

    all_results = {}

    resized_h, resized_w = config["image_size"]

    for click_cfg in config["click_configs"]:
        cfg_name = click_cfg["name"]
        pos_clicks = click_cfg["pos_clicks"]
        neg_clicks = click_cfg["neg_clicks"]

        click_key = f"click_{cfg_name}"
        logger.info(f"Start click config: {cfg_name} (pos={pos_clicks}, neg={neg_clicks})")
        click_results = {}

        for category, data_dir in config["test_data_dirs"].items():
            if not os.path.exists(data_dir):
                logger.warning(f"Category dir not found, skip: category={category}, dir={data_dir}")
                continue

            category_key = f"{click_key}_{category}_v3"
            click_checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_{category_key}.json")

            checkpoint_parent = os.path.dirname(click_checkpoint_path)
            if checkpoint_parent:
                os.makedirs(checkpoint_parent, exist_ok=True)

            start_idx = 0
            results = []
            last_completed = 0

            checkpoint = None
            checkpoint_loaded_from = ""
            if os.path.exists(click_checkpoint_path):
                checkpoint = _load_checkpoint_if_valid(click_checkpoint_path)
                checkpoint_loaded_from = click_checkpoint_path if checkpoint is not None else ""
            else:
                legacy_path = _find_legacy_checkpoint(config.get("checkpoint_dir", ""), cfg_name, category)
                if legacy_path:
                    checkpoint = _load_checkpoint_if_valid(legacy_path)
                    checkpoint_loaded_from = legacy_path if checkpoint is not None else ""

            if checkpoint is not None:
                if checkpoint_loaded_from and checkpoint_loaded_from != click_checkpoint_path:
                    logger.info(
                        f"Legacy checkpoint found: legacy={checkpoint_loaded_from} -> new={click_checkpoint_path}"
                    )
                if checkpoint.get("completed", False):
                    logger.info(f"Checkpoint completed, skip: {click_checkpoint_path}")
                    click_results[category] = checkpoint.get("results", [])
                    continue
                start_idx = int(checkpoint.get("processed_count", 0) or 0)
                results = checkpoint.get("results", []) or []
                last_completed = start_idx
                logger.info(
                    f"Resume from checkpoint: click={cfg_name}, category={category}, start_idx={start_idx}, checkpoint={click_checkpoint_path}"
                )

            image_files = sorted(
                [
                    f
                    for f in os.listdir(data_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ]
            )

            max_images = config.get("max_images_per_category", None)
            if isinstance(max_images, int) and max_images > 0:
                image_files = image_files[:max_images]

            logger.info(f"Category start: click={cfg_name}, category={category}, num_images={len(image_files)}, start_idx={start_idx}")

            effective_interval = 5

            annotation_dir = config["annotation_dirs"].get(category, "")

            try:
                for i in tqdm(range(start_idx, len(image_files)), desc=f"Click {cfg_name} {category}"):
                    img_file = image_files[i]
                    img_path = os.path.join(data_dir, img_file)

                    xml_name = os.path.splitext(img_file)[0] + ".xml"
                    xml_path = os.path.join(annotation_dir, xml_name)
                    gt = parse_voc_xml(xml_path) if os.path.exists(xml_path) else None

                    img = load_image(img_path, config["image_size"])
                    img_h, img_w = img.shape

                    gt_boxes = []
                    gt_w = None
                    gt_h = None
                    if gt and gt.get("objects"):
                        gt_w = gt.get("width")
                        gt_h = gt.get("height")
                        gt_boxes = [[o["xmin"], o["ymin"], o["xmax"], o["ymax"]] for o in gt["objects"]]
                        if len(gt_boxes) > 1:
                            logger.warning(
                                f"image={img_file} | click={cfg_name} | category={category} | 多个 GT bbox (num_gt={len(gt_boxes)})，"
                                f"当前点击采样仅使用第一个 bbox（用于 Table 1 的 click 模拟）。"
                            )

                    pred_bboxes = []
                    anomaly_scores = []

                    pos_points = []
                    neg_points = []

                    if category == "normal_auc":
                        pos_points = [
                            (random.randint(0, img_w - 1), random.randint(0, img_h - 1))
                            for _ in range(max(0, pos_clicks))
                        ]
                        neg_points = [
                            (random.randint(0, img_w - 1), random.randint(0, img_h - 1))
                            for _ in range(max(0, neg_clicks))
                        ]

                        result = model.detect_click_guided(
                            img,
                            positive_points=pos_points,
                            negative_points=neg_points,
                            box=None,
                        )

                        pred_bboxes = [r["bbox"] for r in result.get("anomaly_regions", [])]
                        anomaly_scores = [
                            r.get("max_anomaly_score", 0.0)
                            for r in result.get("anomaly_regions", [])
                        ]

                    elif gt_boxes and gt_w and gt_h:
                        gt_box_resized = _scale_box_to_resized(gt_boxes[0], gt_w, gt_h, resized_h, resized_w)
                        pos_points, neg_points = _sample_clicks_from_gt_bbox(
                            gt_box_resized, pos_clicks, neg_clicks, img_w, img_h
                        )

                        result = model.detect_click_guided(
                            img,
                            positive_points=pos_points,
                            negative_points=neg_points,
                            box=None,
                        )

                        pred_bboxes = [r["bbox"] for r in result.get("anomaly_regions", [])]
                        anomaly_scores = [
                            r.get("max_anomaly_score", 0.0)
                            for r in result.get("anomaly_regions", [])
                        ]

                        pred_bboxes = [
                            _scale_box_to_original(b, gt_w, gt_h, resized_h, resized_w)
                            for b in pred_bboxes
                        ]

                    exclude_from_det_metrics = False
                    if category == "normal_auc" or not gt_boxes:
                        exclude_from_det_metrics = True

                    exclude_from_auc = False
                    if category != "normal_auc" and annotation_dir and gt is None:
                        exclude_from_auc = True

                    logger.info(
                        "image=%s | click=%s | category=%s | has_gt=%s | pos_points=%s | neg_points=%s | num_pred=%s | scores=%s | exclude_det=%s | exclude_auc=%s",
                        img_file,
                        cfg_name,
                        category,
                        bool(gt_boxes),
                        pos_points,
                        neg_points,
                        len(pred_bboxes),
                        anomaly_scores,
                        exclude_from_det_metrics,
                        exclude_from_auc,
                    )

                    results.append(
                        {
                            "click_config": cfg_name,
                            "category": category,
                            "image_name": img_file,
                            "pos_clicks": pos_clicks,
                            "neg_clicks": neg_clicks,
                            "pred_bboxes": pred_bboxes,
                            "anomaly_scores": anomaly_scores,
                            "gt_bboxes": gt_boxes,
                            "num_gt": len(gt_boxes),
                            "exclude_from_det_metrics": exclude_from_det_metrics,
                            "exclude_from_auc": exclude_from_auc,
                        }
                    )

                    last_completed = i + 1

                    if effective_interval > 0 and ((i + 1) % effective_interval == 0):
                        logger.info(f"Save checkpoint: click={cfg_name}, category={category}, processed={last_completed}, path={click_checkpoint_path}")
                        with open(click_checkpoint_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "processed_count": last_completed,
                                    "results": results,
                                    "timestamp": datetime.now().isoformat(),
                                    "completed": False,
                                },
                                f,
                                ensure_ascii=False,
                            )
            except KeyboardInterrupt:
                logger.warning(
                    f"KeyboardInterrupt: 保存断点并退出 | click={cfg_name} | category={category} | processed_count={last_completed} | path={click_checkpoint_path}"
                )
                with open(click_checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "processed_count": last_completed,
                            "results": results,
                            "timestamp": datetime.now().isoformat(),
                            "completed": False,
                        },
                        f,
                        ensure_ascii=False,
                    )
                raise

            with open(click_checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "processed_count": len(image_files),
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                        "completed": True,
                    },
                    f,
                    ensure_ascii=False,
                )

            logger.info(f"Category completed: click={cfg_name}, category={category}, processed={len(image_files)}, checkpoint={click_checkpoint_path}")

            click_results[category] = results

        all_results[click_key] = click_results

    output_path = os.path.join(config["output_dir"], "click_predictions_v3.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "version": "V3",
                    "timestamp": datetime.now().isoformat(),
                    "feature_bank_path": config.get("feature_bank_path", ""),
                    "feature_bank_metadata_path": config.get("metadata_path", ""),
                    "preprocess_signature": metadata.get("preprocess_signature", None),
                    "image_size_hw": list(config.get("image_size", (369, 369))),
                    "window_size": int(config.get("window_size", 50)),
                    "stride": int(config.get("stride", 5)),
                    "hidden_size": int(config.get("hidden_size", 30)),
                    "beta_threshold": float(config.get("beta_threshold", config.get("anomaly_threshold", 0.5))),
                    "click_configs": config.get("click_configs", []),
                },
                "results": all_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n结果保存至: {output_path}")
    logger.info(f"Results saved: {output_path}")
    return all_results


if __name__ == "__main__":
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    with torch.no_grad():
        run_click_guided(CONFIG)
