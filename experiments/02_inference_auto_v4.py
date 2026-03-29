"""
Res-SAM 复现实验 V4 - Step 2: Fully Automatic 模式推理

V3 改进（严格按论文对齐）：
- window_size = 50（论文默认值）
- 特征口径 f = [W_out, b]（论文 Eq.(2)-(3)）
- Region 级粗筛步骤（论文 Fully Automatic 关键流程）
- 特征维度 = 2*hidden_size + 1 = 61

论文对应：Table 2 - Fully Automatic Mode
"""

import sys
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib
import uuid
import logging
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

from experiments.resize_policy import RESIZE_POLICY_VOC_ANNOTATION, target_hw_for_preprocess
from experiments.dataset_layout import (
    DATASET_ENHANCED,
    apply_layout_to_config_02_03,
)


def _require_segment_anything() -> None:
    try:
        import importlib

        importlib.import_module("segment_anything")
    except Exception:
        print(
            "ERROR: segment-anything 未安装，无法运行 Res-SAM 推理。\n"
            "请先安装：pip install segment-anything\n"
            "或从 https://github.com/facebookresearch/segment-anything 安装。",
            flush=True,
        )
        raise SystemExit(1)


def _to_abs(base_dir: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p

    base_name = os.path.basename(base_dir)
    p_norm = os.path.normpath(p.replace("/", os.sep))
    if p_norm.startswith("." + os.sep):
        p_norm = p_norm[2:]
    if p_norm == base_name or p_norm.startswith(base_name + os.sep):
        p_norm = p_norm[len(base_name) + 1 :]

    return os.path.abspath(os.path.join(base_dir, p_norm))


# ============ 配置 ============
CONFIG = {
    # enhanced | open_source | real_world | synthetic（非 enhanced 需 data_paper/<mode>/）
    "dataset_mode": DATASET_ENHANCED,
    # Feature Bank 路径 (V4)
    "feature_bank_path": os.path.join(
        BASE_DIR, "outputs", "feature_banks_v4", "feature_bank_v4.pth"
    ),
    "metadata_path": os.path.join(
        BASE_DIR, "outputs", "feature_banks_v4", "metadata.json"
    ),
    # 测试数据
    "test_data_dirs": {
        "cavities": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_cavities"
        ),
        "utilities": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_utilities"
        ),
        "normal_auc": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_intact"
        ),
    },
    "annotation_dirs": {
        "cavities": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_cavities", "annotations", "VOC_XML_format"
        ),
        "utilities": os.path.join(
            BASE_DIR, "data", "GPR_data", "augmented_utilities", "annotations", "VOC_XML_format"
        ),
    },
    "output_dir": os.path.join(
        BASE_DIR, "outputs", "predictions_v4"
    ),
    "checkpoint_dir": os.path.join(
        BASE_DIR, "outputs", "checkpoints_v4"
    ),
    # 论文参数（V3 严格对齐）
    "window_size": 50,  # 论文默认值
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": 0.2,  # 论文 Eq.(9) 的单一预设阈值 β
    "anomaly_threshold": 0.2,  # 兼容旧字段：内部与 beta_threshold 保持一致
    "region_coarse_threshold": 0.2,  # 兼容旧字段：内部与 beta_threshold 保持一致
    # SAM 参数
    "sam_model_type": "vit_l",
    "sam_checkpoint": os.path.join(
        BASE_DIR, "sam", "sam_vit_l_0b3195.pth"
    ),
    "device": "auto",
    # 图像预处理
    # Resize: experiments/resize_policy.py
    # Default voc_annotation = VOC XML (height,width) when XML exists; else native file size.
    # Set resize_policy=fixed to force image_size for backwards-compatible runs.
    "resize_policy": RESIZE_POLICY_VOC_ANNOTATION,
    "image_size": (369, 369),
    # 推理参数
    # The paper does not specify a fixed maximum number of returned regions.
    # Keep this disabled by default for a closer paper-aligned run.
    # None means: do not cap the number of retained/final regions.
    "max_candidates_per_image": None,
    # The paper does not provide a fixed minimum coarse-region area threshold.
    # Keep this disabled by default for a closer paper-aligned run.
    # None means: do not drop coarse regions by area before feature discrimination.
    "min_region_area": None,
    "max_images_per_category": None,
    # 断点
    "checkpoint_interval": 50,
    # 随机种子
    "random_seed": 42,
    # 版本标识
    "version": "V4",
    "alignment_notes": "Enhanced-eval setup: fb_source=intact, eval=augmented_intact + augmented anomalies, feature f=[W_out,b]",
}


# 环境变量支持
_max_images_env = os.environ.get("MAX_IMAGES_PER_CATEGORY", "").strip()
if _max_images_env:
    try:
        _max_images_val = int(_max_images_env)
        if _max_images_val > 0:
            CONFIG["max_images_per_category"] = _max_images_val
    except Exception:
        pass


def parse_voc_xml(xml_path: str) -> dict:
    """解析 VOC XML 标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # 图像信息
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        # 边界框
        bboxes = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax])
        return {
            "width": width,
            "height": height,
            "bboxes": bboxes,
        }
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None


def compute_iou(box1: list, box2: list) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_image_with_orig_size(path: str, size: tuple) -> tuple:
    """加载图像并返回 (orig_w, orig_h, img_array)"""
    with Image.open(path) as im:
        orig_w, orig_h = im.size
        im_l = im.convert("L")
        if size:
            im_l = im_l.resize((size[1], size[0]), Image.BILINEAR)
        img_array = np.array(im_l, dtype=np.float32)
        img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
    return orig_w, orig_h, img_array


def run_inference(config: dict):
    """运行 Fully Automatic 推理（V3 严格对齐）"""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_inference_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
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

    logger.info(f"run_id={run_id}")

    print("=" * 60)
    print("Res-SAM V4: Fully Automatic Inference (Strict Paper Alignment)")
    print("=" * 60)
    print(f"日志文件: {log_file}")
    print(f"run_id: {run_id}")
    print(f"  window_size = {config['window_size']}")
    print(f"  stride = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    print(f"  Expected feature dim = {2*config['hidden_size'] + 1}")
    print(f"  beta_threshold = {config['beta_threshold']}")
    print("=" * 60)

    logger.info("=" * 60)
    logger.info("Res-SAM V4: Fully Automatic Inference (Strict Paper Alignment)")
    logger.info("=" * 60)
    logger.info(f"log_file={log_file}")
    logger.info(
        "window_size=%s, stride=%s, hidden_size=%s, beta_threshold=%s, resize_policy=%s, fixed_image_size_hw=%s",
        config.get("window_size"),
        config.get("stride"),
        config.get("hidden_size"),
        config.get("beta_threshold"),
        config.get("resize_policy"),
        config.get("image_size"),
    )

    config = dict(config)
    logger.info("dataset_mode=%s", config.get("dataset_mode"))
    print(f"dataset_mode={config.get('dataset_mode')}", flush=True)
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
            "Please run 01_build_feature_bank_v4.py first."
        )

    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    # 设置随机种子
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    # 导入 ResSAM
    from PatchRes.ResSAM import ResSAM
    
    # 初始化模型（V3 参数）
    print("\n初始化 ResSAM (V4)...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],  # 50
        stride=config["stride"],
        anomaly_threshold=config.get("beta_threshold", config["anomaly_threshold"]),
        region_coarse_threshold=config.get("beta_threshold", config["region_coarse_threshold"]),
        sam_model_type=config["sam_model_type"],
        sam_checkpoint=config["sam_checkpoint"],
        device=config.get("device", "auto"),
    )
    
    # 加载 Feature Bank
    print(f"加载 Feature Bank: {config['feature_bank_path']}")
    model.load_feature_bank(config["feature_bank_path"])

    try:
        config["beta_threshold"] = float(config.get("beta_threshold", 0.2))
        config["anomaly_threshold"] = float(config["beta_threshold"])
        config["region_coarse_threshold"] = float(config["beta_threshold"])
    except Exception:
        pass

    print(f"最终 beta_threshold = {config.get('beta_threshold')}")
    try:
        logger.info("Calibrated beta_threshold=%s", config.get("beta_threshold"))
    except Exception:
        pass
    
    # 加载 Feature Bank 元数据
    metadata = {}
    if os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Feature Bank 元数据: {metadata.get('feature_bank_shape', 'Unknown')}")
    
    # 处理每个类别
    all_results = {}
    
    for category, data_dir in config["test_data_dirs"].items():
        print(f"\n{'='*60}")
        print(f"处理类别: {category}")
        print(f"目录: {data_dir}")
        print("=" * 60)

        logger.info(
            "Category start: category=%s, dir=%s",
            category,
            data_dir,
        )
        
        if not os.path.exists(data_dir):
            print(f"警告: 目录不存在: {data_dir}")
            continue
        
        # 获取图像文件列表
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        if config["max_images_per_category"]:
            image_files = image_files[:config["max_images_per_category"]]
        
        print(f"找到 {len(image_files)} 张图像")
        logger.info("Category images: category=%s, num_images=%s", category, len(image_files))

        checkpoint_interval = int(config.get("checkpoint_interval", 50))
        effective_interval = checkpoint_interval
        if len(image_files) > 0 and len(image_files) <= checkpoint_interval:
            effective_interval = 1
        
        # 断点支持
        checkpoint_file = os.path.join(config["checkpoint_dir"], f"checkpoint_auto_{category}.json")
        start_idx = 0
        category_results = []
        last_completed = 0
        legacy_processed_files = set()

        category_success = 0
        category_failed = 0
        fail_reasons = {}

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            if checkpoint.get("completed", False):
                logger.info("Checkpoint completed, skip: %s", checkpoint_file)
                all_results[category] = checkpoint.get("results", []) or []
                continue

            if isinstance(checkpoint.get("processed_count", None), int):
                start_idx = int(checkpoint.get("processed_count", 0) or 0)
                category_results = checkpoint.get("results", []) or []
                last_completed = start_idx
                print(f"从断点继续，start_idx={start_idx}")
                logger.info(
                    "Resume from checkpoint: category=%s, start_idx=%s, checkpoint=%s",
                    category,
                    start_idx,
                    checkpoint_file,
                )
            else:
                legacy_processed_files = set(checkpoint.get("processed_files", []) or [])
                if legacy_processed_files:
                    print(f"从旧断点继续（processed_files），已处理 {len(legacy_processed_files)} 张图像")
                    logger.info(
                        "Resume from legacy checkpoint: category=%s, processed=%s, checkpoint=%s",
                        category,
                        len(legacy_processed_files),
                        checkpoint_file,
                    )
        
        # 处理每张图像
        try:
            for i in tqdm(
                range(start_idx, len(image_files)),
                desc=f"推理 {category}",
                disable=False,
                file=sys.stdout,
            ):
                img_file = image_files[i]
                if legacy_processed_files and img_file in legacy_processed_files:
                    last_completed = i + 1
                    continue

                img_path = os.path.join(data_dir, img_file)
                
                # 获取标注（如果存在）
                annotation_dir = config["annotation_dirs"].get(category)
                xml_file = os.path.splitext(img_file)[0] + ".xml"
                xml_path = os.path.join(annotation_dir, xml_file) if annotation_dir else None

                gt = parse_voc_xml(xml_path) if xml_path and os.path.exists(xml_path) else None
                target_hw = target_hw_for_preprocess(config, gt)
                
                try:
                    orig_w, orig_h, img = load_image_with_orig_size(img_path, target_hw)
                    proc_h, proc_w = int(img.shape[0]), int(img.shape[1])
                    
                    # 推理（V3 detect_automatic 包含 region 级粗筛）
                    result = model.detect_automatic(
                        img,
                        min_region_area=config["min_region_area"],
                        max_regions=config["max_candidates_per_image"],
                    )
                    
                    # 准备结果记录
                    record = {
                        "image_name": img_file,
                        "image_path": img_path,
                        # detect_automatic 返回 bbox 在 resize 坐标系下（与输入 img 对齐）
                        "pred_bboxes": [r["bbox"] for r in result["anomaly_regions"]],
                        "anomaly_scores": [r["max_anomaly_score"] for r in result["anomaly_regions"]],
                        "num_candidates": result["num_candidates"],
                        "num_coarse_discarded": result["num_coarse_discarded"],
                        "num_esn_fits": result["num_esn_fits"],
                    }

                    # 统一输出坐标系：pred_bboxes -> 原图/VOC 坐标系；pred_bboxes_resized -> resize 坐标系。
                    # 注意：若存在 VOC XML，评估时 gt_bboxes 的坐标系以 XML 的 width/height 为准，
                    # 因此这里缩放应优先对齐到 XML 声明的图像尺寸，避免“图像文件尺寸 != XML size”导致 IoU 漂移。
                    target_w = int(gt["width"]) if gt and gt.get("width") else int(orig_w)
                    target_h = int(gt["height"]) if gt and gt.get("height") else int(orig_h)

                    resized_w = proc_w
                    resized_h = proc_h
                    scale_x_img = target_w / resized_w
                    scale_y_img = target_h / resized_h
                    pred_bboxes_resized = record.get("pred_bboxes", [])
                    pred_bboxes_scaled = [
                        [
                            int(b[0] * scale_x_img),
                            int(b[1] * scale_y_img),
                            int(b[2] * scale_x_img),
                            int(b[3] * scale_y_img),
                        ]
                        for b in pred_bboxes_resized
                    ]
                    
                    record["pred_bboxes_resized"] = pred_bboxes_resized
                    record["pred_bboxes"] = pred_bboxes_scaled
                    
                    # 添加 GT 信息（如果存在）
                    if gt:
                        _has_gt = True
                        _num_gt = len(gt['bboxes'])
                        
                        # 同时提供 GT 的 resize 坐标系（便于与 pred_bboxes 直接对照）
                        inv_scale_x = proc_w / gt["width"]
                        inv_scale_y = proc_h / gt["height"]
                        
                        gt_boxes_resized = [
                            [
                                int(bbox[0] * inv_scale_x),
                                int(bbox[1] * inv_scale_y),
                                int(bbox[2] * inv_scale_x),
                                int(bbox[3] * inv_scale_y),
                            ]
                            for bbox in gt["bboxes"]
                        ]
                        
                        record.update({
                            'gt_bboxes': gt['bboxes'],  # 原图坐标系
                            'gt_bboxes_resized': gt_boxes_resized,  # resize 坐标系
                            'num_gt': int(_num_gt),
                            'gt_width': gt['width'],
                            'gt_height': gt['height'],
                            'exclude_from_det_metrics': False,
                            'exclude_from_auc': False,
                        })
                    else:
                        _has_gt = False
                        _num_gt = 0

                        if category == "normal_auc":
                            record.update({
                                "gt_bboxes": [],
                                "num_gt": 0,
                                "exclude_from_det_metrics": True,
                                "exclude_from_auc": False,
                            })
                        else:
                            record.update({
                                "gt_bboxes": [],
                                "num_gt": 0,
                                "exclude_from_det_metrics": True,
                                "exclude_from_auc": True,
                            })

                    category_results.append(record)
                    category_success += 1

                    logger.info(
                        "image=%s | category=%s | num_pred=%s | scores=%s | num_candidates=%s | num_coarse_discarded=%s | num_esn_fits=%s | has_gt=%s | exclude_det=%s | exclude_auc=%s",
                        img_file,
                        category,
                        len(record.get("pred_bboxes", []) or []),
                        record.get("anomaly_scores", []),
                        record.get("num_candidates", None),
                        record.get("num_coarse_discarded", None),
                        record.get("num_esn_fits", None),
                        bool(gt),
                        record.get("exclude_from_det_metrics", None),
                        record.get("exclude_from_auc", None),
                    )

                    last_completed = i + 1
                    if effective_interval > 0 and (last_completed % effective_interval == 0):
                        logger.info(
                            "Save checkpoint: category=%s, processed=%s, path=%s",
                            category,
                            last_completed,
                            checkpoint_file,
                        )
                        try:
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
                        except Exception:
                            logger.exception("Failed to write checkpoint: category=%s", category)

                except Exception as e:
                    category_failed += 1
                    reason = "unknown"
                    try:
                        err_s = str(e)
                        err_s_low = err_s.lower()
                        if "segment_anything" in err_s or "segment-anything" in err_s:
                            reason = "missing_segment_anything"
                        elif "cannot identify image file" in err_s_low or "unidentifiedimageerror" in err_s_low:
                            reason = "image_open_failed"
                        elif "no such file" in err_s_low or "not found" in err_s_low:
                            reason = "file_not_found"
                    except Exception:
                        reason = "unknown"
                    fail_reasons[reason] = int(fail_reasons.get(reason, 0)) + 1
                    print(f"处理图像 {img_file} 时出错: {e}")
                    logger.exception("Error processing image=%s | category=%s", img_file, category)
                    continue

        except KeyboardInterrupt:
            logger.warning(
                "KeyboardInterrupt: 保存断点并退出 | category=%s | processed_count=%s | path=%s",
                category,
                int(last_completed),
                checkpoint_file,
            )
            try:
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
            except Exception:
                pass
            raise
        
        all_results[category] = category_results
        print(f"类别 {category} 完成，处理了 {len(category_results)} 张图像")

        try:
            logger.info(
                "Category summary: category=%s, total=%s, success=%s, failed=%s, fail_reasons=%s",
                category,
                len(image_files),
                int(category_success),
                int(category_failed),
                dict(fail_reasons),
            )
        except Exception:
            pass

        # 标记该类别已完成（与 03 对齐：保留 checkpoint 以便下次直接跳过）
        try:
            completed_ok = True
            if int(category_success) == 0 and int(category_failed) > 0:
                completed_ok = False
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
            logger.info(
                "Checkpoint write: category=%s, processed=%s, completed=%s, path=%s",
                category,
                int(last_completed),
                bool(completed_ok),
                checkpoint_file,
            )
        except Exception:
            logger.exception("Failed to write completed checkpoint: category=%s", category)
    
    # 保存结果
    output_file = os.path.join(config["output_dir"], "auto_predictions_v4.json")
    
    # V4 输出格式：包含元数据
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
            "image_preprocess_note": "Per-image HxW follows resize_policy (VOC or native); fixed_image_size_hw only applies when resize_policy=fixed.",
            "window_size": int(config.get("window_size", 50)),
            "stride": int(config.get("stride", 5)),
            "hidden_size": int(config.get("hidden_size", 30)),
            "beta_threshold": float(config.get("beta_threshold", config.get("anomaly_threshold", 0.2))),
        },
        "results": all_results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存至: {output_file}")
    logger.info(f"Results saved: {output_file}")
    
    # 统计汇总
    total_images = sum(len(results) for results in all_results.values())
    total_detections = sum(len(r['pred_bboxes']) for results in all_results.values() for r in results)
    total_coarse_discarded = sum(r['num_coarse_discarded'] for results in all_results.values() for r in results)
    
    print(f"\n统计汇总:")
    print(f"  总图像数: {total_images}")
    print(f"  总检测数: {total_detections}")
    print(f"  Region 级粗筛丢弃: {total_coarse_discarded}")
    
    print("\n" + "=" * 60)
    print("Fully Automatic V4 推理完成!")
    print("=" * 60)
    
    return all_results


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id"):
            record.run_id = self._run_id
        return True


if __name__ == "__main__":
    _require_segment_anything()
    CONFIG = apply_layout_to_config_02_03(dict(CONFIG), BASE_DIR, "v4")
    with torch.no_grad():
        results = run_inference(CONFIG)
