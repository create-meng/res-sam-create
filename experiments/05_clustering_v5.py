"""Res-SAM 复现实验 V3 - Step 4: 异常聚类（区域级再拟合）

功能：
- 读取 V3 automatic 推理结果（final anomaly regions bboxes）
- 按论文描述：对最终 anomaly regions 再做一次 2D-ESN 拟合，提取区域级动态特征 f=[W_out,b]
- 使用 K-Means / Agglomerative / FCM 聚类
- 输出 Acc / ARI / NMI 指标
- 支持断点继续

论文对应：Table 3
"""

import sys
import os
import json
import random
import xml.etree.ElementTree as ET
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

from experiments.resize_policy import RESIZE_POLICY_VOC_ANNOTATION, target_hw_for_preprocess
from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_05

import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from PIL import Image


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    # V3 推理结果（final bboxes）
    "predictions_path": os.path.join(
        BASE_DIR,
        "outputs",
        "predictions_v5",
        "auto_predictions_v5.json",
    ),
    # 数据目录（open-source annotated 子集，按复现范围锁定）
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
    # VOC 标注（用于真实类别标签）
    "annotation_dirs": {
        "cavities": os.path.join(
            BASE_DIR,
            "data",
            "GPR_data",
            "augmented_cavities",
            "annotations",
            "VOC_XML_format",
        ),
        "utilities": os.path.join(
            BASE_DIR,
            "data",
            "GPR_data",
            "augmented_utilities",
            "annotations",
            "VOC_XML_format",
        ),
    },
    # V3 模型参数（必须与 feature bank/inference 对齐）
    "hidden_size": 30,
    "window_size": 50,
    "stride": 5,
    # ESN implementation defaults reused in clustering re-fit.
    # These are not currently treated as paper-explicit values unless later
    # evidence from the paper/supplement/official code confirms them.
    "spectral_radius": 0.9,
    "connectivity": 0.1,
    "resize_policy": RESIZE_POLICY_VOC_ANNOTATION,
    "image_size": (369, 369),
    # 聚类参数
    # Local binary anomaly-type setup for the current annotated subset
    # (cavities vs utilities). This is dataset-mapping-dependent, not a
    # universal paper constant.
    "n_clusters": 2,
    # 输出
    "output_dir": os.path.join(
        BASE_DIR,
        "outputs",
        "metrics_v5",
    ),
    "output_file": "clustering_v5.json",
    # 断点
    "checkpoint_dir": os.path.join(
        BASE_DIR,
        "outputs",
        "checkpoints_v5",
    ),
    "checkpoint_file": "checkpoint_clustering_v5.json",
    # 断点写盘频率（按用户要求固定为 50；不影响聚类口径，只影响恢复粒度）
    "checkpoint_every": 50,
    # 随机种子
    "random_seed": 42,
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
        p_norm = p_norm[len(base_name) + 1 :]

    return os.path.abspath(os.path.join(base_dir, p_norm))


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


def _find_image_path(data_dir: str, image_name: str) -> str:
    cand = os.path.join(data_dir, image_name)
    if os.path.exists(cand):
        return cand

    base = os.path.splitext(image_name)[0]
    for ext in [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]:
        cand = os.path.join(data_dir, base + ext)
        if os.path.exists(cand):
            return cand

    return ""


def _scale_box_to_resized(box: list, orig_w: int, orig_h: int, resized_h: int, resized_w: int) -> list:
    scale_x = resized_w / float(orig_w)
    scale_y = resized_h / float(orig_h)
    x1, y1, x2, y2 = box
    return [
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    ]


def _crop_region(img: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def _extract_region_crop(img: np.ndarray, bbox: list) -> np.ndarray:
    """Crop the final anomaly region for region-level re-fitting (paper Table 3)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def fcm_clustering(X: np.ndarray, n_clusters: int, max_iter: int = 100, m: float = 2.0) -> np.ndarray:
    n_samples = X.shape[0]
    if n_samples < n_clusters:
        return np.zeros(n_samples, dtype=int)

    U = np.random.rand(n_samples, n_clusters)
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        U_m = U ** m
        centers = (U_m.T @ X) / U_m.sum(axis=0, keepdims=True).T

        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - centers[k], axis=1)

        distances = np.maximum(distances, 1e-10)

        for k in range(n_clusters):
            U[:, k] = 1.0 / np.sum((distances[:, k:k + 1] / distances) ** (2 / (m - 1)), axis=1)

    return np.argmax(U, axis=1)


def compute_clustering_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    from scipy.optimize import linear_sum_assignment

    if len(true_labels) == 0:
        return {"Accuracy": 0.0, "ARI": 0.0, "NMI": 0.0}

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)

    true_uniques = np.unique(true_labels)
    pred_uniques = np.unique(pred_labels)

    true_map = {int(v): i for i, v in enumerate(true_uniques.tolist())}
    pred_map = {int(v): i for i, v in enumerate(pred_uniques.tolist())}

    n_true = len(true_uniques)
    n_pred = len(pred_uniques)

    contingency = np.zeros((n_true, n_pred), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        contingency[true_map[int(t)], pred_map[int(p)]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    accuracy = contingency[row_ind, col_ind].sum() / len(true_labels)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return {"Accuracy": float(accuracy), "ARI": float(ari), "NMI": float(nmi)}


def main():
    global CONFIG
    CONFIG = apply_layout_to_config_05(dict(CONFIG), BASE_DIR, "v5")
    print(f"dataset_mode={CONFIG.get('dataset_mode')} | predictions={CONFIG.get('predictions_path')}", flush=True)
    CONFIG["predictions_path"] = _to_abs(BASE_DIR, CONFIG.get("predictions_path", ""))
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG.get("output_dir", ""))
    CONFIG["checkpoint_dir"] = _to_abs(BASE_DIR, CONFIG.get("checkpoint_dir", ""))
    CONFIG["test_data_dirs"] = {k: _to_abs(BASE_DIR, v) for k, v in CONFIG.get("test_data_dirs", {}).items()}
    CONFIG["annotation_dirs"] = {k: _to_abs(BASE_DIR, v) for k, v in CONFIG.get("annotation_dirs", {}).items()}

    np.random.seed(CONFIG["random_seed"])
    random.seed(CONFIG["random_seed"])
    torch.manual_seed(CONFIG["random_seed"])

    print("=" * 60)
    print("Res-SAM V3: Anomaly Clustering (Region-level re-fitting)")
    print("=" * 60)

    if not os.path.exists(CONFIG["predictions_path"]):
        raise FileNotFoundError(
            f"Predictions not found: {CONFIG['predictions_path']}\n"
            "Please run 02_inference_auto_v5.py first."
        )

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"])

    with open(CONFIG["predictions_path"], "r", encoding="utf-8") as f:
        obj = json.load(f)

    # 兼容输出格式：旧版为 {category: [records...]}
    # 新版为 {"meta": {...}, "results": {category: [records...]}}
    predictions = obj.get("results", obj) if isinstance(obj, dict) else obj

    # 初始化 2D-ESN（用于区域级再拟合）
    from PatchRes.ESN_2D_nobatch import ESN_2D

    esn = ESN_2D(
        input_dim=1,
        n_reservoir=CONFIG["hidden_size"],
        alpha=5,
        spectral_radius=(CONFIG["spectral_radius"], CONFIG["spectral_radius"]),
        connectivity=CONFIG["connectivity"],
    )

    # 断点恢复
    start_pos = 0
    items = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt.get("completed", False):
            print(f"特征提取已完成，继续聚类与指标输出：{checkpoint_path}")
        start_pos = int(ckpt.get("processed_count", 0))
        items = ckpt.get("items", [])

    # 将 predictions 展平为列表，便于断点
    flat = []
    allowed_categories = set(CONFIG.get("test_data_dirs", {}).keys())
    for category, preds in predictions.items():
        if category not in allowed_categories:
            continue
        for r in preds:
            flat.append({"category": category, "record": r})

    checkpoint_every = int(CONFIG.get("checkpoint_every", 50))

    last_completed = start_pos

    try:
        for idx in tqdm(range(start_pos, len(flat)), desc="Extracting region features"):
            category = flat[idx]["category"]
            r = flat[idx]["record"]
            image_name = r.get("image_name", "")

            data_dir = CONFIG["test_data_dirs"].get(category, "")
            anno_dir = CONFIG["annotation_dirs"].get(category, "")

            img_path = _find_image_path(data_dir, image_name) if data_dir else ""
            xml_path = os.path.join(anno_dir, os.path.splitext(image_name)[0] + ".xml") if anno_dir else ""

            gt = parse_voc_xml(xml_path) if (xml_path and os.path.exists(xml_path)) else None
            if gt is None:
                # 无 GT 的不参与 Table 3（按当前复现范围锁定：仅 annotated 子集出数）
                items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "no_gt"})
            else:
                target_hw = target_hw_for_preprocess(CONFIG, gt)
                img = load_image(img_path, target_hw) if img_path else None
                if img is None:
                    items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "missing_image"})
                else:
                    proc_h, proc_w = int(img.shape[0]), int(img.shape[1])
                    gt_w = gt.get("width")
                    gt_h = gt.get("height")

                    # 取 V3 final anomaly regions：这里用推理输出的 pred_bboxes（可能多框）
                    pred_bboxes = r.get("pred_bboxes") or []
                    if not pred_bboxes:
                        items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "no_pred"})
                    else:
                        # Table 3 语义是对最终 anomaly regions 做区域级再拟合后聚类。
                        # 因此这里将每张图的所有 final bboxes 展开为多个 region 样本。
                        any_valid_region = False
                        for bbox_idx, pred_box_orig in enumerate(pred_bboxes):
                            pred_box_resized = _scale_box_to_resized(
                                pred_box_orig,
                                gt_w,
                                gt_h,
                                proc_h,
                                proc_w,
                            )

                            region_crop = _extract_region_crop(img, pred_box_resized)
                            if region_crop.size == 0:
                                continue

                            any_valid_region = True

                            # Paper Table 3 / Methods: reapply 2D-ESN to each final anomaly region directly.
                            region_tensor = torch.tensor(region_crop[None, ...], dtype=torch.float32)
                            with torch.no_grad():
                                feat = esn.forward(region_tensor).detach().cpu().numpy()[0]

                            items.append(
                                {
                                    "image_name": image_name,
                                    "category": category,
                                    "bbox_idx": int(bbox_idx),
                                    "pred_bbox_orig": [int(v) for v in pred_box_orig],
                                    "pred_bbox_resized": [int(v) for v in pred_box_resized],
                                    "region_shape_hw": [int(region_crop.shape[0]), int(region_crop.shape[1])],
                                    "refit_level": "region_direct",
                                    "skipped": False,
                                    "feature": feat.tolist(),
                                    "true_label": 0 if category == "cavities" else 1,
                                }
                            )

                        if not any_valid_region:
                            items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "empty_region"})

            last_completed = idx + 1

            if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
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
                "processed_count": len(flat),
                "items": items,
                "timestamp": datetime.now().isoformat(),
                "completed": True,
            },
            f,
            ensure_ascii=False,
        )

    # 聚类与指标
    feats = [np.array(it["feature"], dtype=np.float32) for it in items if not it.get("skipped") and it.get("feature")]
    labels = [int(it["true_label"]) for it in items if not it.get("skipped") and it.get("feature")]

    X = np.stack(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    y = np.array(labels, dtype=int) if labels else np.zeros((0,), dtype=int)

    metrics = {}
    if len(y) > 0:
        km = KMeans(n_clusters=CONFIG["n_clusters"], random_state=CONFIG["random_seed"], n_init=10)
        km_pred = km.fit_predict(X)
        metrics["KMeans"] = compute_clustering_metrics(y, km_pred)

        ac = AgglomerativeClustering(n_clusters=CONFIG["n_clusters"])
        ac_pred = ac.fit_predict(X)
        metrics["AgglomerativeClustering"] = compute_clustering_metrics(y, ac_pred)

        fcm_pred = fcm_clustering(X, CONFIG["n_clusters"])
        metrics["FuzzyCMeans"] = compute_clustering_metrics(y, fcm_pred)

    out = {
        "num_samples": int(len(y)),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(CONFIG["output_dir"], CONFIG["output_file"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n输出: {output_path}")


if __name__ == "__main__":
    main()
