#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Res-SAM v21 评估脚本
评估全图 patch 分数图方法的效果（V21 优化版）

V21 新增评估内容：
- 方案1参数优化效果（top_k=1, nms=0.3, min_area=5000）
- 方案2局部对比度增强效果
- 多 IoU 阈值评估（0.1, 0.2, 0.3, 0.5）
- 图像级 AUC
"""
from __future__ import print_function
import json
import os
import sys

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

sys.path.insert(0, BASE_DIR)

try:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from PatchRes.logger import setup_global_logger, log_section, log_finish
except ImportError as e:
    print("Error: Missing required packages. Please install: pip install numpy scikit-learn")
    print("Error details:", str(e))
    sys.exit(1)


def compute_iou(box1, box2):
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


def evaluate_v21(pred_path, meta_path):
    """评估 v21 结果"""
    try:
        with open(pred_path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception as e:
        print("Error loading files: {}".format(str(e)))
        return None

    results = obj.get('results', {})
    pred_meta = obj.get('meta', {})

    # 多 IoU 阈值评估
    metrics_by_iou = {}
    for iou_thresh in [0.1, 0.2, 0.3, 0.5]:
        tp, fp, fn = 0, 0, 0
        tp_scores, fp_scores = [], []

        for cat, recs in results.items():
            if cat == 'normal_auc':
                continue
            for r in recs:
                if r.get('exclude_from_det_metrics'):
                    continue

                pred_bboxes = r.get('pred_bboxes', [])
                gt_bboxes   = r.get('gt_bboxes', [])
                scores      = r.get('anomaly_scores', [])

                matched_gt = set()
                for pred, score in zip(pred_bboxes, scores):
                    best_iou, best_gt_idx = 0.0, -1
                    for gt_idx, gt in enumerate(gt_bboxes):
                        if gt_idx in matched_gt:
                            continue
                        iou_val = compute_iou(pred, gt)
                        if iou_val > best_iou:
                            best_iou, best_gt_idx = iou_val, gt_idx

                    if best_iou >= iou_thresh:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                        tp_scores.append(score)
                    else:
                        fp += 1
                        fp_scores.append(score)

                fn += len(gt_bboxes) - len(matched_gt)

        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

        metrics_by_iou[iou_thresh] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'tp_scores': tp_scores, 'fp_scores': fp_scores,
        }

    # 图像级 AUC
    normal_scores, anomaly_scores = [], []
    for cat, recs in results.items():
        for r in recs:
            scores = r.get('anomaly_scores', [])
            img_score = max(scores) if scores else 0.0
            if cat == 'normal_auc':
                normal_scores.append(img_score)
            elif not r.get('exclude_from_det_metrics'):
                anomaly_scores.append(img_score)

    y_true  = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    y_score = normal_scores + anomaly_scores
    image_auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0

    # 总 pred 框数
    total_pred = sum(
        len(r.get('pred_bboxes', []))
        for cat, recs in results.items()
        for r in recs
        if not r.get('exclude_from_det_metrics')
    )

    return {
        'meta': meta,
        'pred_meta': pred_meta,
        'metrics_by_iou': metrics_by_iou,
        'image_auc': image_auc,
        'total_pred': total_pred,
        'normal_mean': float(np.mean(normal_scores)) if normal_scores else 0.0,
        'anomaly_mean': float(np.mean(anomaly_scores)) if anomaly_scores else 0.0,
    }


if __name__ == '__main__':
    logger = setup_global_logger(BASE_DIR, "03_evaluate_v21")
    log_section("v21 评估：全图 patch 分数图方法（方案1+方案2）", logger)

    # 从环境变量读取输出目录后缀
    output_suffix = os.environ.get("OUTPUT_SUFFIX", "").strip()

    if output_suffix:
        pred_path   = os.path.join(BASE_DIR, "outputs", f"predictions_v21{output_suffix}", "auto_predictions_v21.json")
        report_path = os.path.join(BASE_DIR, "outputs", f"predictions_v21{output_suffix}", "evaluation_report_v21.json")
    else:
        pred_path   = os.path.join(BASE_DIR, "outputs", "predictions_v21", "auto_predictions_v21.json")
        report_path = os.path.join(BASE_DIR, "outputs", "predictions_v21", "evaluation_report_v21.json")

    meta_path = os.path.join(BASE_DIR, "outputs", "feature_banks_v21", "metadata.json")

    if not os.path.exists(pred_path):
        logger.error(f"预测结果文件不存在: {pred_path}")
        logger.error("请先运行: python experiments/02_inference_auto_v21.py")
        sys.exit(1)

    if not os.path.exists(meta_path):
        logger.error(f"元数据文件不存在: {meta_path}")
        logger.error("请先运行: python experiments/01_build_feature_bank_v21.py")
        sys.exit(1)

    logger.info(f"读取预测结果: {pred_path}")
    logger.info(f"读取元数据:   {meta_path}")

    result = evaluate_v21(pred_path, meta_path)

    if result is None:
        logger.error("评估失败")
        sys.exit(1)

    log_section("v21 评估结果", logger)

    logger.info("Feature Bank 配置:")
    logger.info(f"  hidden_size:          {result['meta'].get('config', {}).get('hidden_size')}")
    logger.info(f"  beta_threshold:       {result['meta'].get('adaptive_beta', 0.0):.4f}")
    logger.info(f"  background_removal:   {result['meta'].get('config', {}).get('background_removal_method')}")

    logger.info("\nV21 推理配置:")
    pm = result['pred_meta']
    logger.info(f"  [方案1] top_k_per_image:       {pm.get('top_k_per_image')}")
    logger.info(f"  [方案1] nms_iou_threshold:     {pm.get('nms_iou_threshold')}")
    logger.info(f"  [方案1] min_bbox_area:         {pm.get('min_bbox_area')}")
    logger.info(f"  [方案2] use_local_contrast:    {pm.get('use_local_contrast')}")
    logger.info(f"  [方案2] local_contrast_window: {pm.get('local_contrast_window')}")
    logger.info(f"  [方案2] local_contrast_clip:   {pm.get('local_contrast_clip')}")
    logger.info(f"  score_map_smooth_sigma:        {pm.get('score_map_smooth_sigma')}")
    logger.info(f"  per_image_threshold_ratio:     {pm.get('per_image_threshold_ratio')}")

    logger.info(f"\n图像级分数:")
    logger.info(f"  正常图平均分: {result['normal_mean']:.4f}")
    logger.info(f"  异常图平均分: {result['anomaly_mean']:.4f}")
    logger.info(f"  图像级 AUC:   {result['image_auc']:.4f}")
    logger.info(f"  总 pred 框数: {result['total_pred']}")

    logger.info("\n多 IoU 阈值评估:")
    for iou_thresh in [0.1, 0.2, 0.3, 0.5]:
        m = result['metrics_by_iou'][iou_thresh]
        logger.info(f"  IoU={iou_thresh}: TP={m['tp']}, FP={m['fp']}, FN={m['fn']} | "
                    f"P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")

    # TP/FP 分数分析（IoU=0.5）
    m05 = result['metrics_by_iou'][0.5]
    if m05['tp_scores']:
        logger.info(f"\nTP框分数(IoU>0.5): mean={np.mean(m05['tp_scores']):.4f}, "
                    f"min={np.min(m05['tp_scores']):.4f}, max={np.max(m05['tp_scores']):.4f}")
    if m05['fp_scores']:
        logger.info(f"FP框分数(IoU>0.5): mean={np.mean(m05['fp_scores']):.4f}, "
                    f"min={np.min(m05['fp_scores']):.4f}, max={np.max(m05['fp_scores']):.4f}")

    # 保存评估报告
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    save_result = {
        'meta': result['meta'],
        'pred_meta': result['pred_meta'],
        'metrics_by_iou': {
            str(k): {
                'tp': v['tp'], 'fp': v['fp'], 'fn': v['fn'],
                'precision': v['precision'], 'recall': v['recall'], 'f1': v['f1'],
            }
            for k, v in result['metrics_by_iou'].items()
        },
        'image_auc':   result['image_auc'],
        'total_pred':  result['total_pred'],
        'normal_mean': result['normal_mean'],
        'anomaly_mean': result['anomaly_mean'],
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)

    logger.info(f"\n评估报告已保存: {report_path}")
    log_finish("03_evaluate_v21", logger)

    # 终端汇总输出
    print("\n" + "=" * 80)
    print("v21 评估完成！")
    print("=" * 80)
    print(f"主要指标 (IoU=0.5):")
    print(f"  TP={m05['tp']}, FP={m05['fp']}, FN={m05['fn']}")
    print(f"  Precision={m05['precision']:.4f}, Recall={m05['recall']:.4f}, F1={m05['f1']:.4f}")
    print(f"  图像级 AUC={result['image_auc']:.4f}")
    print(f"\n对比 V20 Baseline (IoU=0.5):")
    print(f"  V20 Baseline: TP=18, FP=63, F1=0.255, AUC=0.869")
    print(f"  V21 当前:     TP={m05['tp']}, FP={m05['fp']}, F1={m05['f1']:.3f}, AUC={result['image_auc']:.3f}")
    f1_delta = m05['f1'] - 0.255
    print(f"  F1 变化: {f1_delta:+.3f}")
    print("=" * 80)
