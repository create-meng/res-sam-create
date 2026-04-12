"""
Res-SAM v10 - Step 4: anomaly categorization / clustering.

v10 clustering reuses the final anomaly regions from automatic inference,
re-fits each final region with 2D-ESN, and clusters the resulting
region-level dynamic features. As with the other v10 steps, preprocessing
uses a fixed 369x369 resize and the runtime seed is unified to 11.
"""

from __future__ import annotations

import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_05
from experiments.paper_constants import preflight_faiss_or_raise
from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "predictions_path": os.path.join(BASE_DIR, "outputs", "predictions_v10", "auto_predictions_v10.json"),
    "feature_bank_path": os.path.join(BASE_DIR, "outputs", "feature_banks_v10", "feature_bank_v10.pth"),
    "test_data_dirs": {
        "cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
    },
    "annotation_dirs": {
        "cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities", "annotations", "VOC_XML_format"),
        "utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities", "annotations", "VOC_XML_format"),
    },
    "hidden_size": 30,
    "window_size": 50,
    "stride": 5,
    "spectral_radius": 0.9,
    "connectivity": 0.1,
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "n_clusters": None,
    "output_dir": os.path.join(BASE_DIR, "outputs", "metrics_v10"),
    "output_file": "04_clustering_v10.json",
    "checkpoint_dir": os.path.join(BASE_DIR, "outputs", "checkpoints_v10"),
    "checkpoint_file": "checkpoint_04_clustering_v10.json",
    "checkpoint_every": 50,
    "random_seed": 11,
    "version": "v10",
    "alignment_notes": (
        "Paper-first mainline (v10): clustering re-fits each final anomaly region "
        "with the same 2D-ESN reservoir weights used by feature-bank/inference, "
        "then clusters the solved dynamic feature f=[W_out,b]."
    ),
}


def _to_abs(base_dir: str, path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def _load_esn_from_feature_bank_bundle(feature_bank_path: str, hidden_size: int, spectral_radius: float, connectivity: float):
    from PatchRes.ESN_2D_nobatch import ESN_2D

    esn = ESN_2D(
        input_dim=1,
        n_reservoir=hidden_size,
        alpha=5,
        spectral_radius=(spectral_radius, spectral_radius),
        connectivity=connectivity,
    )
    try:
        raw = torch.load(feature_bank_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(feature_bank_path, map_location="cpu")

    if isinstance(raw, dict) and raw.get("format") == "res_sam_fb_v2":
        state_dict = raw.get("esn_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Feature bank bundle missing esn_state_dict: {feature_bank_path}")
        esn.load_state_dict(state_dict, strict=True)
        return esn

    raise ValueError(
        "Clustering requires a feature bank bundle with ESN weights (format=res_sam_fb_v2). "
        f"Got incompatible file: {feature_bank_path}"
    )


def parse_voc_xml(xml_path: str):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)
        objects = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            objects.append(
                {
                    "name": obj.find("name").text,
                    "xmin": int(bbox.find("xmin").text),
                    "ymin": int(bbox.find("ymin").text),
                    "xmax": int(bbox.find("xmax").text),
                    "ymax": int(bbox.find("ymax").text),
                }
            )
        return {"width": img_width, "height": img_height, "objects": objects}
    except Exception:
        return None


def load_image(path: str, size_hw: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size_hw:
        img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr


def _find_image_path(data_dir: str, image_name: str) -> str:
    direct = os.path.join(data_dir, image_name)
    if os.path.exists(direct):
        return direct
    base = os.path.splitext(image_name)[0]
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"):
        cand = os.path.join(data_dir, base + ext)
        if os.path.exists(cand):
            return cand
    return ""


def _scale_box_to_resized(box: list[int], orig_w: int, orig_h: int, resized_h: int, resized_w: int) -> list[int]:
    scale_x = resized_w / float(orig_w)
    scale_y = resized_h / float(orig_h)
    x1, y1, x2, y2 = box
    return [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]


def _extract_region_crop(img: np.ndarray, bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0), dtype=img.dtype)
    return img[y1:y2, x1:x2]


def _region_crop_meets_esn_min_size(region_crop: np.ndarray, start_node: tuple[int, int] = (1, 1)) -> bool:
    if region_crop is None or region_crop.size == 0:
        return False
    if region_crop.ndim < 2:
        return False

    min_h = int(start_node[0]) + 1
    min_w = int(start_node[1]) + 1
    return int(region_crop.shape[0]) >= min_h and int(region_crop.shape[1]) >= min_w


def fcm_clustering(
    x: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    m: float = 2.0,
    tol: float = 1e-6,
    random_state: int | None = None,
) -> np.ndarray:
    n_samples = x.shape[0]
    if n_samples < n_clusters:
        return np.zeros((n_samples,), dtype=int)

    rng = np.random.default_rng(random_state)
    memberships = rng.random((n_samples, n_clusters), dtype=np.float64)
    memberships = memberships / memberships.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        memberships_prev = memberships.copy()
        memberships_m = memberships ** m
        denom = memberships_m.sum(axis=0, keepdims=True).T
        denom = np.maximum(denom, 1e-12)
        centers = (memberships_m.T @ x) / denom
        distances = np.zeros((n_samples, n_clusters), dtype=np.float64)
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(x - centers[k], axis=1)
        distances = np.maximum(distances, 1e-10)
        for k in range(n_clusters):
            memberships[:, k] = 1.0 / np.sum(
                (distances[:, k : k + 1] / distances) ** (2.0 / (m - 1.0)),
                axis=1,
            )
        if np.max(np.abs(memberships - memberships_prev)) < tol:
            break

    return np.argmax(memberships, axis=1)


def compute_clustering_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    from scipy.optimize import linear_sum_assignment

    if len(true_labels) == 0:
        return {"Accuracy": 0.0, "ARI": 0.0, "NMI": 0.0}

    true_labels = np.asarray(true_labels)
    pred_labels = np.asarray(pred_labels)
    true_uniques = np.unique(true_labels)
    pred_uniques = np.unique(pred_labels)
    true_map = {int(v): i for i, v in enumerate(true_uniques.tolist())}
    pred_map = {int(v): i for i, v in enumerate(pred_uniques.tolist())}

    contingency = np.zeros((len(true_uniques), len(pred_uniques)), dtype=np.int64)
    for true_v, pred_v in zip(true_labels, pred_labels):
        contingency[true_map[int(true_v)], pred_map[int(pred_v)]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    accuracy = contingency[row_ind, col_ind].sum() / len(true_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return {"Accuracy": float(accuracy), "ARI": float(ari), "NMI": float(nmi)}


def main() -> None:
    config = apply_layout_to_config_05(dict(CONFIG), BASE_DIR, "v10")
    config["predictions_path"] = _to_abs(BASE_DIR, config.get("predictions_path", ""))
    config["feature_bank_path"] = _to_abs(BASE_DIR, config.get("feature_bank_path", ""))
    config["output_dir"] = _to_abs(BASE_DIR, config.get("output_dir", ""))
    config["checkpoint_dir"] = _to_abs(BASE_DIR, config.get("checkpoint_dir", ""))
    config["test_data_dirs"] = {k: _to_abs(BASE_DIR, v) for k, v in config.get("test_data_dirs", {}).items()}
    config["annotation_dirs"] = {k: _to_abs(BASE_DIR, v) for k, v in config.get("annotation_dirs", {}).items()}

    np.random.seed(config["random_seed"])
    random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    if not os.path.exists(config["predictions_path"]):
        raise FileNotFoundError(
            f"Predictions not found: {config['predictions_path']}\n"
            "Please run 02_inference_auto_v10.py first."
        )
    if not os.path.exists(config["feature_bank_path"]):
        raise FileNotFoundError(
            f"Feature bank not found: {config['feature_bank_path']}\n"
            "Please run 01_build_feature_bank_v10.py first."
        )

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(config["checkpoint_dir"], config["checkpoint_file"])

    with open(config["predictions_path"], "r", encoding="utf-8") as f:
        obj = json.load(f)
    predictions = obj.get("results", obj) if isinstance(obj, dict) else obj

    esn = _load_esn_from_feature_bank_bundle(
        config["feature_bank_path"],
        hidden_size=int(config["hidden_size"]),
        spectral_radius=float(config["spectral_radius"]),
        connectivity=float(config["connectivity"]),
    )

    start_pos = 0
    items: list[dict] = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        start_pos = int(ckpt.get("processed_count", 0))
        items = ckpt.get("items", []) or []

    flat_records = []
    allowed_categories = set(config["test_data_dirs"].keys())
    for category, preds in predictions.items():
        if category not in allowed_categories:
            continue
        for record in preds:
            flat_records.append({"category": category, "record": record})

    checkpoint_every = int(config.get("checkpoint_every", 50) or 0)
    last_completed = start_pos

    try:
        for idx in tqdm(range(start_pos, len(flat_records)), desc="Extracting region features", file=sys.stdout):
            category = flat_records[idx]["category"]
            record = flat_records[idx]["record"]
            image_name = record.get("image_name", "")

            data_dir = config["test_data_dirs"].get(category, "")
            anno_dir = config["annotation_dirs"].get(category, "")
            img_path = _find_image_path(data_dir, image_name) if data_dir else ""
            xml_path = os.path.join(anno_dir, os.path.splitext(image_name)[0] + ".xml") if anno_dir else ""

            gt = parse_voc_xml(xml_path) if (xml_path and os.path.exists(xml_path)) else None
            if gt is None:
                items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "no_gt"})
                last_completed = idx + 1
                continue

            target_hw = target_hw_for_preprocess(config, gt)
            img = load_image(img_path, target_hw) if img_path else None
            if img is None:
                items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "missing_image"})
                last_completed = idx + 1
                continue

            proc_h, proc_w = int(img.shape[0]), int(img.shape[1])
            gt_w = int(gt["width"])
            gt_h = int(gt["height"])
            pred_bboxes = record.get("pred_bboxes") or []
            if not pred_bboxes:
                items.append({"image_name": image_name, "category": category, "skipped": True, "reason": "no_pred"})
                last_completed = idx + 1
                continue

            any_valid_region = False
            for bbox_idx, pred_box_orig in enumerate(pred_bboxes):
                pred_box_resized = _scale_box_to_resized(pred_box_orig, gt_w, gt_h, proc_h, proc_w)
                region_crop = _extract_region_crop(img, pred_box_resized)
                if region_crop.size == 0:
                    continue
                if not _region_crop_meets_esn_min_size(region_crop):
                    items.append(
                        {
                            "image_name": image_name,
                            "category": category,
                            "bbox_idx": int(bbox_idx),
                            "pred_bbox_orig": [int(v) for v in pred_box_orig],
                            "pred_bbox_resized": [int(v) for v in pred_box_resized],
                            "region_shape_hw": [int(region_crop.shape[0]), int(region_crop.shape[1])],
                            "refit_level": "region_direct",
                            "skipped": True,
                            "reason": "region_too_small_for_esn",
                        }
                    )
                    continue

                any_valid_region = True
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
                "processed_count": len(flat_records),
                "items": items,
                "timestamp": datetime.now().isoformat(),
                "completed": True,
            },
            f,
            ensure_ascii=False,
        )

    feats = [np.array(it["feature"], dtype=np.float32) for it in items if (not it.get("skipped")) and it.get("feature")]
    labels = [int(it["true_label"]) for it in items if (not it.get("skipped")) and it.get("feature")]

    x = np.stack(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)
    y = np.array(labels, dtype=int) if labels else np.zeros((0,), dtype=int)

    effective_n_clusters = int(len(np.unique(y))) if len(y) > 0 else 0
    if config.get("n_clusters") is not None:
        effective_n_clusters = int(config["n_clusters"])

    metrics: dict[str, dict[str, float]] = {}
    if len(y) > 0 and effective_n_clusters > 0:
        km = KMeans(n_clusters=effective_n_clusters, random_state=config["random_seed"], n_init=10)
        metrics["KMeans"] = compute_clustering_metrics(y, km.fit_predict(x))

        ac = AgglomerativeClustering(n_clusters=effective_n_clusters)
        metrics["AgglomerativeClustering"] = compute_clustering_metrics(y, ac.fit_predict(x))

        metrics["FuzzyCMeans"] = compute_clustering_metrics(
            y,
            fcm_clustering(x, effective_n_clusters, random_state=int(config["random_seed"])),
        )

    output_path = os.path.join(config["output_dir"], config["output_file"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "version": config["version"],
                    "alignment_notes": config["alignment_notes"],
                    "feature_bank_path": config["feature_bank_path"],
                    "n_clusters": int(effective_n_clusters),
                    "feature_dim": int(x.shape[1]) if len(x.shape) == 2 and x.shape[0] > 0 else 0,
                    "timestamp": datetime.now().isoformat(),
                },
                "num_samples": int(len(y)),
                "metrics": metrics,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved clustering metrics to: {output_path}")


if __name__ == "__main__":
    preflight_faiss_or_raise()
    main()

