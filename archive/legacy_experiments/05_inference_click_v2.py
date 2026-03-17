"""Res-SAM 复现实验 V2 - Click-guided 模式推理

功能：
- 模拟用户点击（从 GT bbox 内/外采样正/负点）
- 使用 ResSAM.detect_click_guided() 进行点击引导异常检测
- 支持断点继续
- 结果输出到 outputs/predictions_v2/click_predictions_v2.json

注意：
- 这是 V2 的 click-guided 入口脚本（V1 的 03_inference_click.py 走旧 PatchRes 路线）。
- 评估口径与 V2 自动模式一致：一对一 IoU 匹配（阈值 0.5）。

论文对应：Table 1
"""

import sys
import os
import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


# ============ 配置 ============
CONFIG = {
    # Feature Bank 路径（V2）
    "feature_bank_path": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "feature_banks_v2",
        "feature_bank.pth",
    ),

    # 测试数据
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
        "predictions_v2",
    ),
    "checkpoint_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "checkpoints_v2",
    ),

    # 论文参数（若需严格对齐，可在此处统一调整）
    "window_size": 30,
    "stride": 5,
    "hidden_size": 30,
    "anomaly_threshold": 0.5,

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

    # 随机种子
    "random_seed": 42,
}


def parse_voc_xml(xml_path: str) -> dict:
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


def compute_iou(box1: list, box2: list) -> float:
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


def match_tp_fp_fn(pred_bboxes: list, gt_bboxes: list, iou_thresh: float = 0.5) -> tuple:
    pred_bboxes = pred_bboxes or []
    gt_bboxes = gt_bboxes or []

    matched_gt = [False] * len(gt_bboxes)
    tp = 0
    fp = 0

    for pred in pred_bboxes:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_bboxes):
            if matched_gt[gt_idx]:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            matched_gt[best_gt_idx] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in matched_gt if not m)
    return tp, fp, fn


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


def _sample_clicks_from_gt_bbox(gt_box: list, pos_clicks: int, neg_clicks: int, img_w: int, img_h: int):
    """从 GT bbox 内部采样正点，从 bbox 外部采样负点。"""
    xmin, ymin, xmax, ymax = gt_box
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    pos_points = []
    neg_points = []

    # 正点：bbox 内随机采样
    for _ in range(max(0, pos_clicks)):
        if xmax <= xmin or ymax <= ymin:
            break
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)
        pos_points.append((x, y))

    # 负点：bbox 外随机采样
    attempts = 0
    while len(neg_points) < max(0, neg_clicks) and attempts < 200:
        x = random.randint(0, img_w - 1)
        y = random.randint(0, img_h - 1)
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            neg_points.append((x, y))
        attempts += 1

    return pos_points, neg_points


def run_click_guided(config: dict):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config = dict(config)
    config["feature_bank_path"] = _to_abs(base_dir, config.get("feature_bank_path", ""))
    config["output_dir"] = _to_abs(base_dir, config.get("output_dir", ""))
    config["checkpoint_dir"] = _to_abs(base_dir, config.get("checkpoint_dir", ""))
    config["sam_checkpoint"] = _to_abs(base_dir, config.get("sam_checkpoint", ""))
    config["test_data_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("test_data_dirs", {}).items()}
    config["annotation_dirs"] = {k: _to_abs(base_dir, v) for k, v in config.get("annotation_dirs", {}).items()}

    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(
            f"Feature bank not found: {config['feature_bank_path']}\n"
            "Please run 04_build_feature_bank_v2.py first."
        )

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    from PatchRes.ResSAM import ResSAM

    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        anomaly_threshold=config["anomaly_threshold"],
        sam_model_type=config["sam_model_type"],
        sam_checkpoint=config["sam_checkpoint"],
    )
    model.load_feature_bank(config["feature_bank_path"])

    all_results = {}

    for click_cfg in config["click_configs"]:
        cfg_name = click_cfg["name"]
        pos_clicks = click_cfg["pos_clicks"]
        neg_clicks = click_cfg["neg_clicks"]

        click_key = f"click_{cfg_name}"

        # click_key 维度的汇总结果（按类别分开存储）
        click_results = {}

        # 对每个类别跑一遍（click 配置固定）
        for category, data_dir in config["test_data_dirs"].items():
            if not os.path.exists(data_dir):
                continue

            category_key = f"{click_key}_{category}"
            click_checkpoint_path = os.path.join(
                config["checkpoint_dir"], f"checkpoint_{category_key}.json"
            )

            # 断点（每个 click_config + category 单独维护）
            start_idx = 0
            results = []
            last_completed = 0
            if os.path.exists(click_checkpoint_path):
                with open(click_checkpoint_path, "r") as f:
                    checkpoint = json.load(f)
                if checkpoint.get("completed", False):
                    click_results[category] = checkpoint.get("results", [])
                    continue
                start_idx = checkpoint.get("processed_count", 0)
                results = checkpoint.get("results", [])
                last_completed = start_idx

            image_files = sorted(
                [f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
            annotation_dir = config["annotation_dirs"].get(category, "")

            for i in tqdm(range(start_idx, len(image_files)), desc=f"Click {cfg_name} {category}"):
                img_file = image_files[i]
                img_path = os.path.join(data_dir, img_file)

                xml_name = os.path.splitext(img_file)[0] + ".xml"
                xml_path = os.path.join(annotation_dir, xml_name)
                gt = parse_voc_xml(xml_path) if os.path.exists(xml_path) else None

                img = load_image(img_path, config["image_size"])
                img_h, img_w = img.shape

                gt_boxes = []
                if gt and gt.get("objects"):
                    gt_boxes = [[o["xmin"], o["ymin"], o["xmax"], o["ymax"]] for o in gt["objects"]]

                # 只对“有 GT 的样本”模拟 click（更贴近 Table 1 的交互设定）
                # 没有 GT 的样本：不生成 click，跳过或记录空结果。
                pred_bboxes = []
                anomaly_scores = []

                if gt_boxes:
                    # 目前按第一个 GT 生成 click（若论文要求多目标交互，可扩展）
                    pos_points, neg_points = _sample_clicks_from_gt_bbox(
                        gt_boxes[0], pos_clicks, neg_clicks, img_w, img_h
                    )

                    result = model.detect_click_guided(
                        img,
                        positive_points=pos_points,
                        negative_points=neg_points,
                        box=None,
                    )

                    pred_bboxes = [r["bbox"] for r in result.get("anomaly_regions", [])]
                    anomaly_scores = [r.get("max_anomaly_score", 0.0) for r in result.get("anomaly_regions", [])]

                tp, fp, fn = match_tp_fp_fn(pred_bboxes, gt_boxes, iou_thresh=0.5)

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
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "num_gt": len(gt_boxes),
                    }
                )

                last_completed = i + 1

                if (i + 1) % config["checkpoint_interval"] == 0:
                    with open(click_checkpoint_path, "w") as f:
                        json.dump(
                            {
                                "processed_count": last_completed,
                                "results": results,
                                "timestamp": datetime.now().isoformat(),
                                "completed": False,
                            },
                            f,
                        )

            # 类别完成
            with open(click_checkpoint_path, "w") as f:
                json.dump(
                    {
                        "processed_count": len(image_files),
                        "results": results,
                        "timestamp": datetime.now().isoformat(),
                        "completed": True,
                    },
                    f,
                )

            click_results[category] = results

        all_results[click_key] = click_results

    output_path = os.path.join(config["output_dir"], "click_predictions_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果保存至: {output_path}")
    return all_results


if __name__ == "__main__":
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    with torch.no_grad():
        run_click_guided(CONFIG)
