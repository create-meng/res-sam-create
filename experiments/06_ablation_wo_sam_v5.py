"""Res-SAM 复现实验 V3 - Step 5: w/o SAM Ablation (Fig.5)

功能：
- 对比 Res-SAM 与 w/o SAM 的 2D-ESN fitting counts
- w/o SAM：直接对整图滑动窗口提取 patches 并拟合（不使用 SAM 生成候选区域）
- Res-SAM：使用 SAM 生成候选区域，仅对区域内 patches 拟合
- 输出 fitting counts 对比与 reduction 百分比
- 支持断点继续

论文对应：Fig.5
"""

import sys
import os
import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

CONFIG = {
    # 数据目录（open-source annotated 子集）
    "test_data_dirs": {
        "cavities": os.path.join(
            BASE_DIR,
            "data",
            "GPR_data",
            "augmented_cavities",
        ),
        "utilities": os.path.join(
            BASE_DIR,
            "data",
            "GPR_data",
            "augmented_utilities",
        ),
    },
    # V3 模型参数
    "hidden_size": 30,
    "window_size": 50,
    "stride": 5,
    "spectral_radius": 0.9,
    "connectivity": 0.1,
    # 图像预处理
    "image_size": (369, 369),
    # SAM 配置
    "sam_model_type": "vit_b",
    "sam_checkpoint": os.path.join(
        BASE_DIR,
        "sam",
        "sam_vit_b_01ec64.pth",
    ),
    # Feature Bank
    "feature_bank_path": os.path.join(
        BASE_DIR,
        "outputs",
        "feature_banks_v5",
        "feature_bank_v5.pth",
    ),
    # 输出
    "output_dir": os.path.join(
        BASE_DIR,
        "outputs",
        "metrics_v5",
    ),
    "output_file": "ablation_wo_sam_v5.json",
    # 断点
    "checkpoint_dir": os.path.join(
        BASE_DIR,
        "outputs",
        "checkpoints_v5",
    ),
    "checkpoint_file": "checkpoint_ablation_wo_sam_v5.json",
    # 随机种子
    "random_seed": 42,
    # 限制图片数（用于快速验证）
    "max_images_per_category": None,

    # 与 V3 主推理（02_inference_auto_v3.py）保持一致的 detect_automatic 参数
    "min_region_area": 100,
    "max_candidates_per_image": 10,
}


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
        p_norm = p_norm[len(base_name) + 1:]

    return os.path.abspath(os.path.join(base_dir, p_norm))


_max_images_env = os.environ.get('MAX_IMAGES_PER_CATEGORY', '').strip()
if _max_images_env:
    try:
        _max_images_val = int(_max_images_env)
        if _max_images_val > 0:
            CONFIG['max_images_per_category'] = _max_images_val
    except Exception:
        pass


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


def count_patches_wo_sam(image: np.ndarray, window_size: int, stride: int) -> int:
    """计算 w/o SAM 模式下的 patch 数量（整图滑动窗口）"""
    h, w = image.shape
    count = 0
    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            count += 1
    return count


def count_patches_in_box(image: np.ndarray, box: list, window_size: int, stride: int) -> int:
    """在给定矩形框内做 centered patch dense 扫描并计数（用于 w/o SAM 的 click-guided：以 GT bbox 模拟用户矩形）。"""
    h, w = image.shape
    if not box or len(box) != 4:
        return 0
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    half_win = window_size // 2
    count = 0
    for y in range(y1 + half_win, y2 - half_win, stride):
        for x in range(x1 + half_win, x2 - half_win, stride):
            if y - half_win < 0 or y + half_win > h:
                continue
            if x - half_win < 0 or x + half_win > w:
                continue
            count += 1
    return count


def count_patches_with_sam(
    image: np.ndarray,
    candidate_regions: list,
    window_size: int,
    stride: int,
) -> int:
    """计算 Res-SAM 模式下的 patch 数量（仅候选区域内）"""
    h, w = image.shape
    half_win = window_size // 2
    count = 0
    
    for region in candidate_regions:
        bbox = region.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        
        # 区域内密集采样
        for y in range(y1 + half_win, y2 - half_win, stride):
            for x in range(x1 + half_win, x2 - half_win, stride):
                if y - half_win < 0 or y + half_win > h:
                    continue
                if x - half_win < 0 or x + half_win > w:
                    continue
                count += 1
    
    return count


def main():
    global CONFIG
    CONFIG = dict(CONFIG)
    CONFIG["sam_checkpoint"] = _to_abs(BASE_DIR, CONFIG.get("sam_checkpoint", ""))
    CONFIG["feature_bank_path"] = _to_abs(BASE_DIR, CONFIG.get("feature_bank_path", ""))
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG.get("output_dir", ""))
    CONFIG["checkpoint_dir"] = _to_abs(BASE_DIR, CONFIG.get("checkpoint_dir", ""))
    CONFIG["test_data_dirs"] = {k: _to_abs(BASE_DIR, v) for k, v in CONFIG.get("test_data_dirs", {}).items()}
 
    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    print("=" * 60)
    print("Res-SAM V3: w/o SAM Ablation (Fig.5)")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"])

    from PatchRes.ResSAM import ResSAM

    # 断点恢复
    start_idx = 0
    items = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt.get("completed", False):
            print(f"已完成，跳过：{checkpoint_path}")
            return
        start_idx = int(ckpt.get("processed_count", 0))
        items = ckpt.get("items", [])

    click_configs = [
        {"name": "5/5", "pos_clicks": 5, "neg_clicks": 5},
        {"name": "5/3", "pos_clicks": 5, "neg_clicks": 3},
        {"name": "3/1", "pos_clicks": 3, "neg_clicks": 1},
    ]

    # 收集所有图片（仅有 VOC 标注的样本参与 click-guided；automatic 仍可统计所有图片）
    all_images = []
    for category, data_dir in CONFIG["test_data_dirs"].items():
        if not os.path.exists(data_dir):
            continue
        anno_dir = os.path.join(data_dir, "annotations", "VOC_XML_format")
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
    
    # 限制图片数
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

    print(f"共 {len(all_images)} 张图片待处理")

    model = ResSAM(
        hidden_size=CONFIG["hidden_size"],
        window_size=CONFIG["window_size"],
        stride=CONFIG["stride"],
        anomaly_threshold=0.5,
        region_coarse_threshold=0.5,
        sam_model_type=CONFIG["sam_model_type"],
        sam_checkpoint=CONFIG["sam_checkpoint"],
    )
    if os.path.exists(CONFIG["feature_bank_path"]):
        model.load_feature_bank(CONFIG["feature_bank_path"])

    resized_h, resized_w = CONFIG["image_size"]

    last_completed = start_idx
    try:
        for idx in tqdm(range(start_idx, len(all_images)), desc="Counting patches"):
            item = all_images[idx]
            img_path = item["path"]
            img_name = item["name"]
            category = item["category"]
            xml_path = item.get("xml_path")

            img = load_image(img_path, CONFIG["image_size"])
            img_h, img_w = img.shape

            # Automatic: w/o SAM（整图 dense sliding window）
            wo_sam_auto = count_patches_wo_sam(img, CONFIG["window_size"], CONFIG["stride"])

            # Automatic: Res-SAM（真实执行的 2D-ESN fitting 次数）
            res_auto = None
            try:
                res_auto = model.detect_automatic(
                    img,
                    min_region_area=int(CONFIG.get("min_region_area", 100)),
                    max_regions=int(CONFIG.get("max_candidates_per_image", 10)),
                )
                with_sam_auto = int(res_auto.get("num_esn_fits", 0))
                num_candidates = int(res_auto.get("num_candidates", 0))
                num_coarse_discarded = int(res_auto.get("num_coarse_discarded", 0))
            except Exception:
                with_sam_auto = 0
                num_candidates = 0
                num_coarse_discarded = 0

            # Click-guided：需要 GT bbox 才能模拟点击
            click_records = {}
            gt = parse_voc_xml(xml_path) if xml_path else None
            gt_box_resized = None
            if gt and gt.get("objects"):
                gt_w = gt.get("width")
                gt_h = gt.get("height")
                gt_boxes = [[o["xmin"], o["ymin"], o["xmax"], o["ymax"]] for o in gt["objects"]]
                gt_box_resized = _scale_box_to_resized(gt_boxes[0], gt_w, gt_h, resized_h, resized_w)

            for cfg in click_configs:
                cfg_name = cfg["name"]
                pos_clicks = cfg["pos_clicks"]
                neg_clicks = cfg["neg_clicks"]

                if gt_box_resized is None:
                    click_records[cfg_name] = {
                        "wo_sam": 0,
                        "with_sam": 0,
                        "pos_clicks": pos_clicks,
                        "neg_clicks": neg_clicks,
                        "has_gt": False,
                    }
                    continue

                # w/o SAM click-guided：在 GT bbox（模拟用户矩形）内 dense centered patch 扫描
                wo_sam_click = count_patches_in_box(img, gt_box_resized, CONFIG["window_size"], CONFIG["stride"])

                # Res-SAM click-guided：运行 detect_click_guided，并读取 num_esn_fits
                pos_points, neg_points = _sample_clicks_from_gt_bbox(gt_box_resized, pos_clicks, neg_clicks, img_w, img_h)
                try:
                    res_click = model.detect_click_guided(
                        img,
                        positive_points=pos_points,
                        negative_points=neg_points,
                        box=None,
                    )
                    with_sam_click = int(res_click.get("num_esn_fits", 0))
                except Exception:
                    with_sam_click = 0

                click_records[cfg_name] = {
                    "wo_sam": int(wo_sam_click),
                    "with_sam": int(with_sam_click),
                    "pos_clicks": pos_clicks,
                    "neg_clicks": neg_clicks,
                    "has_gt": True,
                }

            items.append(
                {
                    "image_name": img_name,
                    "category": category,
                    "automatic": {
                        "wo_sam": int(wo_sam_auto),
                        "with_sam": int(with_sam_auto),
                        "num_candidates": int(num_candidates),
                        "num_coarse_discarded": int(num_coarse_discarded),
                    },
                    "click_guided": click_records,
                    "has_gt": bool(gt_box_resized is not None),
                }
            )

            last_completed = idx + 1

            if (idx + 1) % 50 == 0:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "processed_count": last_completed,
                            "items": items,
                            "timestamp": datetime.now().isoformat(),
                            "completed": False,
                        },
                        f,
                        ensure_ascii=False,
                    )
    except KeyboardInterrupt:
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "processed_count": int(last_completed),
                    "items": items,
                    "timestamp": datetime.now().isoformat(),
                    "completed": False,
                },
                f,
                ensure_ascii=False,
            )
        raise

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "processed_count": len(all_images),
                "items": items,
                "timestamp": datetime.now().isoformat(),
                "completed": True,
            },
            f,
            ensure_ascii=False,
        )

    def _reduction(wo: int, with_sam: int) -> float:
        return (float(wo - with_sam) / float(wo) * 100.0) if wo > 0 else 0.0

    summary = {
        "automatic": {"wo_sam": 0, "with_sam": 0, "reduction_percent": 0.0},
        "5/5": {"wo_sam": 0, "with_sam": 0, "reduction_percent": 0.0},
        "5/3": {"wo_sam": 0, "with_sam": 0, "reduction_percent": 0.0},
        "3/1": {"wo_sam": 0, "with_sam": 0, "reduction_percent": 0.0},
    }

    for it in items:
        a = it.get("automatic", {})
        summary["automatic"]["wo_sam"] += int(a.get("wo_sam", 0))
        summary["automatic"]["with_sam"] += int(a.get("with_sam", 0))

        clicks = it.get("click_guided", {})
        for k in ["5/5", "5/3", "3/1"]:
            if k not in clicks:
                continue
            if not clicks[k].get("has_gt", False):
                continue
            summary[k]["wo_sam"] += int(clicks[k].get("wo_sam", 0))
            summary[k]["with_sam"] += int(clicks[k].get("with_sam", 0))

    for k, v in summary.items():
        v["reduction_percent"] = _reduction(int(v.get("wo_sam", 0)), int(v.get("with_sam", 0)))

    out = {
        "total_images": len(items),
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(CONFIG["output_dir"], CONFIG["output_file"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n结果:")
    print(f"  Automatic | w/o SAM: {summary['automatic']['wo_sam']} | Res-SAM: {summary['automatic']['with_sam']} | Reduction: {summary['automatic']['reduction_percent']:.1f}%")
    print(f"  5/5      | w/o SAM: {summary['5/5']['wo_sam']} | Res-SAM: {summary['5/5']['with_sam']} | Reduction: {summary['5/5']['reduction_percent']:.1f}%")
    print(f"  5/3      | w/o SAM: {summary['5/3']['wo_sam']} | Res-SAM: {summary['5/3']['with_sam']} | Reduction: {summary['5/3']['reduction_percent']:.1f}%")
    print(f"  3/1      | w/o SAM: {summary['3/1']['wo_sam']} | Res-SAM: {summary['3/1']['with_sam']} | Reduction: {summary['3/1']['reduction_percent']:.1f}%")
    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
