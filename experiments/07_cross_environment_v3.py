"""Res-SAM 复现实验 V3 - Step 6: Cross-environment (Fig.6)

功能：
- 论文 Fig.6：不同环境 Feature Bank 与测试环境的交叉 AUC 矩阵
- 当前复现范围限制：仅有 open-source 数据集有 bbox 标注
- 本脚本实现：同一 open-source 数据体系内的跨类别实验
  - Feature Bank 来源：intact (non-target)
  - 测试集：cavities / utilities
- 输出 AUC 矩阵
- 支持断点继续

论文对应：Fig.6（部分复现，受标注限制）

注意：严格跨数据源（real-world / synthetic / open-source）需要对应标注，
当前仅实现 open-source 内部的跨类别验证。
"""

import sys
import os
import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from PIL import Image

CONFIG = {
    # Feature Bank 来源（行）
    "feature_bank_sources": {
        "intact": os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
            "feature_banks_v3",
            "feature_bank_v3.pth",
        ),
    },
    # 测试集（列）
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
        "normal_auc": os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "GPR_data",
            "intact",
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
        "normal_auc": "",
    },
    # V3 模型参数
    "hidden_size": 30,
    "window_size": 50,
    "stride": 5,
    "spectral_radius": 0.9,
    "connectivity": 0.1,
    "beta_threshold": 0.5,
    # 图像预处理
    "image_size": (369, 369),
    # SAM 配置
    "sam_model_type": "vit_b",
    "sam_checkpoint": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "sam",
        "sam_vit_b_01ec64.pth",
    ),
    # 输出
    "output_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "metrics_v3",
    ),
    "output_file": "cross_environment_v3.json",
    # 断点
    "checkpoint_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs",
        "checkpoints_v3",
    ),
    "checkpoint_file": "checkpoint_cross_environment_v3.json",
    # 随机种子
    "random_seed": 42,
    # 限制图片数
    "max_images_per_category": None,

    # 与 V3 推理（02_inference_auto_v3.py）保持一致的 detect_automatic 参数
    "min_region_area": 100,
    "max_candidates_per_image": 10,
}


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
            objects.append({
                "name": name,
                "xmin": int(bbox.find("xmin").text),
                "ymin": int(bbox.find("ymin").text),
                "xmax": int(bbox.find("xmax").text),
                "ymax": int(bbox.find("ymax").text),
            })
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


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def main():
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    print("=" * 60)
    print("Res-SAM V3: Cross-environment (Fig.6)")
    print("=" * 60)
    print("注意：当前仅实现 open-source 内部跨类别验证")
    print("严格跨数据源需要 real-world/synthetic 的 bbox 标注")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # 初始化模型
    from PatchRes.ResSAM import ResSAM

    # 收集所有测试图片
    all_images = []
    for category, data_dir in CONFIG["test_data_dirs"].items():
        if not os.path.exists(data_dir):
            continue
        anno_dir = CONFIG["annotation_dirs"].get(category, "")
        
        for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
            for img_file in os.listdir(data_dir):
                if img_file.endswith(ext):
                    xml_path = os.path.join(anno_dir, os.path.splitext(img_file)[0] + ".xml")
                    all_images.append({
                        "category": category,
                        "path": os.path.join(data_dir, img_file),
                        "name": img_file,
                        "xml_path": xml_path if os.path.exists(xml_path) else None,
                    })

    max_img = CONFIG.get("max_images_per_category")
    if max_img:
        by_cat = {}
        for item in all_images:
            cat = item["category"]
            if cat not in by_cat:
                by_cat[cat] = []
            if len(by_cat[cat]) < max_img:
                by_cat[cat].append(item)
        all_images = []
        for cat_items in by_cat.values():
            all_images.extend(cat_items)

    print(f"共 {len(all_images)} 张测试图片")

    # 对每个 Feature Bank 来源运行推理
    matrix = {}
    for fb_name, fb_path in CONFIG["feature_bank_sources"].items():
        checkpoint_path = os.path.join(
            CONFIG["checkpoint_dir"],
            f"checkpoint_cross_environment_v3_{fb_name}.json",
        )

        # 断点恢复（按 feature bank source 独立）
        start_idx = 0
        items = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            if ckpt.get("completed", False):
                print(f"已完成，跳过：{checkpoint_path}")
                matrix[fb_name] = ckpt.get("matrix", {})
                continue
            start_idx = int(ckpt.get("processed_count", 0))
            items = ckpt.get("items", [])

        if not os.path.exists(fb_path):
            print(f"Feature Bank 不存在: {fb_path}")
            continue

        print(f"\n使用 Feature Bank: {fb_name}")

        model = ResSAM(
            hidden_size=CONFIG["hidden_size"],
            window_size=CONFIG["window_size"],
            stride=CONFIG["stride"],
            anomaly_threshold=CONFIG["beta_threshold"],
            region_coarse_threshold=CONFIG["beta_threshold"],
            sam_model_type=CONFIG["sam_model_type"],
            sam_checkpoint=CONFIG["sam_checkpoint"],
        )
        model.load_feature_bank(fb_path)

        scores_list = []
        labels_list = []
        per_category_scores = {k: [] for k in CONFIG["test_data_dirs"].keys()}
        per_category_labels = {k: [] for k in CONFIG["test_data_dirs"].keys()}

        last_completed = start_idx
        try:
            for idx in tqdm(range(start_idx, len(all_images)), desc=f"Testing {fb_name}"):
                item = all_images[idx]
                img = load_image(item["path"], CONFIG["image_size"])

                # 检测
                result = model.detect_automatic(
                    img,
                    min_region_area=int(CONFIG.get("min_region_area", 100)),
                    max_regions=int(CONFIG.get("max_candidates_per_image", 10)),
                )
                regions = result.get("anomaly_regions", [])
                
                if regions:
                    max_score = max(r.get("max_anomaly_score", 0) for r in regions)
                else:
                    max_score = 0.0

                # GT
                gt = parse_voc_xml(item["xml_path"]) if item["xml_path"] else None
                has_gt = gt is not None and len(gt.get("objects", [])) > 0

                if item.get("category") == "normal_auc":
                    has_gt = False

                label = 1 if has_gt else 0
                scores_list.append(max_score)
                labels_list.append(label)
                if item["category"] in per_category_scores:
                    per_category_scores[item["category"]].append(max_score)
                    per_category_labels[item["category"]].append(label)

                items.append({
                    "feature_bank": fb_name,
                    "category": item["category"],
                    "image_name": item["name"],
                    "score": max_score,
                    "has_gt": has_gt,
                })

                last_completed = idx + 1

                if (idx + 1) % 50 == 0:
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "processed_count": last_completed,
                            "items": items,
                            "timestamp": datetime.now().isoformat(),
                            "completed": False,
                        }, f, ensure_ascii=False)
        except KeyboardInterrupt:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump({
                    "processed_count": int(last_completed),
                    "items": items,
                    "timestamp": datetime.now().isoformat(),
                    "completed": False,
                }, f, ensure_ascii=False)
            raise

        def _safe_auc(_labels: list, _scores: list) -> float:
            if len(_labels) == 0:
                return 0.5
            if len(set(_labels)) > 1:
                fpr, tpr, _ = roc_curve(_labels, _scores)
                return float(auc(fpr, tpr))
            return 0.5

        # 计算 AUC：overall + per-category
        fb_stats = {
            "overall": {
                "auc": _safe_auc(labels_list, scores_list),
                "num_samples": len(scores_list),
                "num_positive": int(sum(labels_list)),
            }
        }

        for cat in CONFIG["test_data_dirs"].keys():
            fb_stats[cat] = {
                "auc": _safe_auc(per_category_labels.get(cat, []), per_category_scores.get(cat, [])),
                "num_samples": len(per_category_scores.get(cat, [])),
                "num_positive": int(sum(per_category_labels.get(cat, []))),
            }

        matrix[fb_name] = fb_stats

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({
                "processed_count": len(all_images),
                "items": items,
                "matrix": fb_stats,
                "timestamp": datetime.now().isoformat(),
                "completed": True,
            }, f, ensure_ascii=False)

    # 输出
    out = {
        "matrix": matrix,
        "note": "当前仅实现 open-source 内部跨类别验证；严格跨数据源需要 real-world/synthetic 标注",
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(CONFIG["output_dir"], CONFIG["output_file"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nCross-environment AUC 矩阵:")
    for fb_name, stats in matrix.items():
        overall = stats.get("overall", {})
        print(f"  Feature Bank {fb_name}: overall AUC = {overall.get('auc', 0.5):.4f}")
        for cat in CONFIG["test_data_dirs"].keys():
            c = stats.get(cat, {})
            print(f"    - {cat}: AUC = {c.get('auc', 0.5):.4f} (n={c.get('num_samples', 0)})")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
