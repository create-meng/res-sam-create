"""
Res-SAM v10 - 第 3 步：评估与可视化。

论文优先口径：
- 特征定义遵循 Eq.(3)/(8)：f=[W_out,b]
- 检测是否命中使用论文正文中的 IoU>0.5
- 保持主线固定阈值，不做后验阈值扫描选优
- 主指标保留论文正文可直接落地的 IoU 检测指标
- 额外提供区域级 AUC：以预测框为样本、IoU>0.5 为标签、anomaly_score 为分数
- 区域级 AUC 属于贴近论文任务语义的扩展评估，不宣称为论文原文显式公式
- 当前唯一评测数据映射为 augmented_intact / augmented_cavities / augmented_utilities
"""

# Paper-aligned rule: a detection is correct only when IoU > 0.5.
from __future__ import annotations
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_04
from experiments.paper_constants import EVAL_DETECTION_IOU_THRESHOLD

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "predictions_path": os.path.join(BASE_DIR, "outputs", "predictions_v10", "auto_predictions_v10.json"),
    "vis_output_dir": os.path.join(BASE_DIR, "outputs", "visualizations_v10"),
    "analysis_output": os.path.join(BASE_DIR, "outputs", "visualizations_v10", "03_evaluate_and_visualize_v10_report.md"),
    "paper_reference": {
        "dataset_name": "论文 Table 2 参考结果（仅作论文外部对照）",
        "dataset_note": (
            "当前评估固定使用 augmented_intact + augmented_cavities + augmented_utilities。"
            "论文数值仅作外部参考，不视为同数据集直接对齐。"
        ),
        "auto": {"Precision": 0.842, "Recall": 0.877, "F1": 0.859, "AUC": 0.832},
    },
    "version": "v10",
    "alignment_notes": (
        "Paper-first mainline (V10): Eq.(3)/(8) uses f=[W_out,b]; "
        "beta=p99(FB_dist)≈0.183; merge_all_anomaly_patches=True; "
        "fb_source=augmented_intact, eval=augmented_intact + current annotated anomaly sets"
    ),
}


def _to_abs(base_dir: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))


def _compute_iou(box1: list, box2: list) -> float:
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


def _match_tp_fp_fn(pred_bboxes: list, gt_bboxes: list, iou_thresh: float = EVAL_DETECTION_IOU_THRESHOLD) -> tuple[int, int, int]:
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
            iou = _compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou > iou_thresh:
            matched_gt[best_gt_idx] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for m in matched_gt if not m)
    return tp, fp, fn


def _collect_region_level_auc_samples(
    pred_bboxes: list,
    pred_scores: list,
    gt_bboxes: list,
    iou_thresh: float = EVAL_DETECTION_IOU_THRESHOLD,
) -> tuple[list[int], list[float]]:
    """
    Auxiliary region-level AUC samples.

    Each predicted anomaly box is one sample:
    - label 1 if it overlaps any GT box with IoU > threshold
    - label 0 otherwise
    - score is the box-level anomaly score emitted by inference
    """
    labels: list[int] = []
    scores: list[float] = []
    pred_bboxes = pred_bboxes or []
    pred_scores = pred_scores or []
    gt_bboxes = gt_bboxes or []
    for bbox, score in zip(pred_bboxes, pred_scores):
        best_iou = 0.0
        for gt in gt_bboxes:
            best_iou = max(best_iou, _compute_iou(bbox, gt))
        labels.append(1 if best_iou > iou_thresh else 0)
        scores.append(float(score if score is not None else 0.0))
    return labels, scores


def compute_metrics(results: dict) -> dict:
    auc_labels: list[int] = []
    auc_scores: list[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_coarse_discarded = 0
    per_category = {}

    for category, cat_results in results.items():
        tp = 0
        fp = 0
        fn = 0
        coarse_discarded = 0
        for record in cat_results:
            if record.get("exclude_from_det_metrics", False):
                continue
            rtp, rfp, rfn = _match_tp_fp_fn(
                record.get("pred_bboxes", []),
                record.get("gt_bboxes", []),
                iou_thresh=EVAL_DETECTION_IOU_THRESHOLD,
            )
            tp += rtp
            fp += rfp
            fn += rfn
            coarse_discarded += int(record.get("num_coarse_discarded", 0))

        per_category[category] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "coarse_discarded": coarse_discarded,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_coarse_discarded += coarse_discarded

        for record in cat_results:
            if record.get("exclude_from_det_metrics", False):
                continue
            labels, scores = _collect_region_level_auc_samples(
                record.get("pred_bboxes", []),
                record.get("anomaly_scores", []),
                record.get("gt_bboxes", []),
                iou_thresh=EVAL_DETECTION_IOU_THRESHOLD,
            )
            auc_labels.extend(labels)
            auc_scores.extend(scores)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    region_auc = None
    if len(auc_labels) > 0 and len(set(auc_labels)) > 1:
        fpr, tpr, _ = roc_curve(np.asarray(auc_labels, dtype=int), np.asarray(auc_scores, dtype=float))
        region_auc = float(auc(fpr, tpr))

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "region_auc": region_auc,
        "region_auc_num_samples": int(len(auc_labels)),
        "coarse_discarded": total_coarse_discarded,
        "per_category": per_category,
    }


def plot_region_auc_curve(results: dict, output_path: str) -> float | None:
    labels_all: list[int] = []
    scores_all: list[float] = []
    for _, cat_results in results.items():
        for record in cat_results:
            if record.get("exclude_from_det_metrics", False):
                continue
            labels, scores = _collect_region_level_auc_samples(
                record.get("pred_bboxes", []),
                record.get("anomaly_scores", []),
                record.get("gt_bboxes", []),
                iou_thresh=EVAL_DETECTION_IOU_THRESHOLD,
            )
            labels_all.extend(labels)
            scores_all.extend(scores)
    if len(labels_all) == 0 or len(set(labels_all)) < 2:
        return None
    fpr, tpr, _ = roc_curve(np.asarray(labels_all, dtype=int), np.asarray(scores_all, dtype=float))
    roc_auc = float(auc(fpr, tpr))
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Res-SAM v10 Region-level ROC")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return roc_auc


def generate_comparison_table(metrics: dict, paper_reference: dict) -> str:
    auto_baseline = paper_reference.get("auto", {})
    dataset_name = paper_reference.get("dataset_name", "Paper reference")
    dataset_note = paper_reference.get("dataset_note", "")
    lines = [
        "# Res-SAM v10 评估报告",
        "",
        "## 数据映射说明",
        f"- 论文参考结果：{dataset_name}",
        "- 本地评估数据：augmented_intact + augmented_cavities + augmented_utilities",
        "- 类别映射：cavities -> cavities，utilities -> pipes/utilities，normal_auc -> augmented_intact",
        "- 扩展指标：Region-level AUC 以预测框为样本、IoU>0.5 为正负标签、框分数为 anomaly_score",
        f"- 备注：{dataset_note}",
        "",
        "## 全自动模式对比",
        "",
        "| 指标 | 论文 | 本地 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for metric in ["Precision", "Recall", "F1", "AUC"]:
        paper_val = auto_baseline.get(metric, 0.0)
        local_val = metrics.get("region_auc", 0.0) if metric == "AUC" else metrics.get(metric.lower(), 0.0)
        lines.append(f"| {metric} | {paper_val:.3f} | {local_val:.3f} | {local_val - paper_val:+.3f} |")
    lines.extend(
        [
            "",
            "## 检测统计",
            f"- TP：{metrics.get('tp', 0)}",
            f"- FP：{metrics.get('fp', 0)}",
            f"- FN：{metrics.get('fn', 0)}",
            f"- Precision：{metrics.get('precision', 0.0):.3f}",
            f"- Recall：{metrics.get('recall', 0.0):.3f}",
            f"- F1：{metrics.get('f1', 0.0):.3f}",
            f"- Region-level AUC：{metrics.get('region_auc', 0.0) if metrics.get('region_auc', None) is not None else 'N/A'}",
            f"- Region-level AUC 样本数：{metrics.get('region_auc_num_samples', 0)}",
            f"- Region 级粗筛剔除数：{metrics.get('coarse_discarded', 0)}",
            "",
            "## 分类别汇总",
            "",
            "| 类别 | TP | FP | FN | Precision | Recall | F1 | 粗筛剔除数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for cat, cat_metrics in metrics.get("per_category", {}).items():
        tp = cat_metrics.get("tp", 0)
        fp = cat_metrics.get("fp", 0)
        fn = cat_metrics.get("fn", 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lines.append(
            f"| {cat} | {tp} | {fp} | {fn} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {cat_metrics.get('coarse_discarded', 0)} |"
        )
    lines.extend(["", f"*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"])
    return "\n".join(lines)


def main() -> None:
    print("=" * 60)
    print("Res-SAM v10：评估与可视化")
    print("dataset=single_annotated_dataset")
    print("=" * 60)
    os.makedirs(CONFIG["vis_output_dir"], exist_ok=True)
    report_parts = []

    if os.path.exists(CONFIG["predictions_path"]):
        with open(CONFIG["predictions_path"], "r", encoding="utf-8") as f:
            obj = json.load(f)
        results = obj.get("results", obj) if isinstance(obj, dict) else obj
        metrics = compute_metrics(results)
        print(
            f"Detection: Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )
        region_auc = plot_region_auc_curve(results, os.path.join(CONFIG["vis_output_dir"], "region_roc_curve_v10.png"))
        if region_auc is not None:
            print(f"Region-level AUC={region_auc:.4f}")
        report_parts.append(generate_comparison_table(metrics, CONFIG["paper_reference"]))
    else:
        print(f"警告：未找到预测结果文件：{CONFIG['predictions_path']}")

    report = "\n\n".join([p for p in report_parts if p])
    if report:
        os.makedirs(os.path.dirname(CONFIG["analysis_output"]), exist_ok=True)
        with open(CONFIG["analysis_output"], "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存到：{CONFIG['analysis_output']}")


if __name__ == "__main__":
    CONFIG = apply_layout_to_config_04(dict(CONFIG), BASE_DIR, "v10")
    CONFIG["predictions_path"] = _to_abs(BASE_DIR, CONFIG.get("predictions_path", ""))
    CONFIG["vis_output_dir"] = _to_abs(BASE_DIR, CONFIG.get("vis_output_dir", ""))
    CONFIG["analysis_output"] = _to_abs(BASE_DIR, CONFIG.get("analysis_output", ""))
    main()
