#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Res-SAM V26 评估脚本
评估 V26 主线实验效果

V26 说明：
- 当前继承 V25 基线
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


def _normalize_suffix(env_name, fallback=""):
    value = (os.getenv(env_name, fallback) or "").strip()
    if not value:
        return ""
    invalid_chars = set('/\\:')
    if any(ch in value for ch in invalid_chars):
        raise ValueError("{} 包含非法路径字符: {!r}".format(env_name, value))
    return value


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


def get_image_score_strategies(record):
    region_scores = record.get('anomaly_scores', []) or []
    region_max = max(region_scores) if region_scores else 0.0
    patch_max = float(record.get('max_patch_score', 0.0) or 0.0)
    patch_mean = float(record.get('mean_patch_score', 0.0) or 0.0)
    num_patches = int(record.get('num_patches', 0) or 0)
    num_anomaly_patches = int(record.get('num_anomaly_patches', 0) or 0)
    ratio = (float(num_anomaly_patches) / float(num_patches)) if num_patches > 0 else 0.0

    return {
        'region_max': region_max,
        'patch_mean': patch_mean,
        'patch_max': patch_max,
        'blend_region_patch': 0.5 * region_max + 0.5 * patch_max,
        'density_weighted_patch': patch_max * (1.0 + ratio),
    }


def evaluate_v26(pred_path, meta_path):
    """评估 v26 结果"""
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

    # 图像级 AUC（多种 image-level score 聚合方式）
    strategy_scores = {
        'region_max': {'normal': [], 'anomaly': []},
        'patch_mean': {'normal': [], 'anomaly': []},
        'patch_max': {'normal': [], 'anomaly': []},
        'blend_region_patch': {'normal': [], 'anomaly': []},
        'density_weighted_patch': {'normal': [], 'anomaly': []},
    }
    for cat, recs in results.items():
        for r in recs:
            img_scores = get_image_score_strategies(r)
            if cat == 'normal_auc':
                for key in strategy_scores:
                    strategy_scores[key]['normal'].append(img_scores[key])
            elif not r.get('exclude_from_det_metrics'):
                for key in strategy_scores:
                    strategy_scores[key]['anomaly'].append(img_scores[key])

    image_auc_by_strategy = {}
    for key, bucket in strategy_scores.items():
        normal_scores = bucket['normal']
        anomaly_scores = bucket['anomaly']
        y_true  = [0] * len(normal_scores) + [1] * len(anomaly_scores)
        y_score = normal_scores + anomaly_scores
        image_auc_by_strategy[key] = {
            'auc': roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0,
            'normal_mean': float(np.mean(normal_scores)) if normal_scores else 0.0,
            'anomaly_mean': float(np.mean(anomaly_scores)) if anomaly_scores else 0.0,
        }

    # 总 pred 框数
    total_pred = sum(
        len(r.get('pred_bboxes', []))
        for cat, recs in results.items()
        for r in recs
        if not r.get('exclude_from_det_metrics')
    )

    primary_strategy = 'patch_mean'

    return {
        'meta': meta,
        'pred_meta': pred_meta,
        'metrics_by_iou': metrics_by_iou,
        'primary_image_score_strategy': primary_strategy,
        'image_auc': image_auc_by_strategy[primary_strategy]['auc'],
        'image_auc_by_strategy': image_auc_by_strategy,
        'total_pred': total_pred,
        'normal_mean': image_auc_by_strategy[primary_strategy]['normal_mean'],
        'anomaly_mean': image_auc_by_strategy[primary_strategy]['anomaly_mean'],
    }


if __name__ == '__main__':
    logger = setup_global_logger(BASE_DIR, "03_evaluate_v26")
    log_section("V26 评估：主线版本", logger)

    output_suffix = _normalize_suffix("OUTPUT_SUFFIX", "")
    bank_suffix = _normalize_suffix("BANK_SUFFIX", "")
    pred_filename = f"auto_predictions_v26{output_suffix}.json" if output_suffix else "auto_predictions_v26.json"
    report_filename = f"evaluation_report_v26{output_suffix}.json" if output_suffix else "evaluation_report_v26.json"
    pred_path   = os.path.join(BASE_DIR, "outputs", "predictions_v26", pred_filename)
    report_path = os.path.join(BASE_DIR, "outputs", "predictions_v26", report_filename)
    if bank_suffix:
        meta_path = os.path.join(BASE_DIR, "outputs", f"feature_banks_v26{bank_suffix}", "metadata.json")
    else:
        meta_path = os.path.join(BASE_DIR, "outputs", "feature_banks_v26", "metadata.json")

    if not os.path.exists(pred_path):
        logger.error(f"预测结果文件不存在: {pred_path}")
        logger.error("请先运行: python experiments/02_inference_auto_v26.py")
        sys.exit(1)

    if not os.path.exists(meta_path):
        logger.error(f"元数据文件不存在: {meta_path}")
        logger.error("请先运行: python experiments/01_build_feature_bank_v26.py")
        sys.exit(1)

    logger.info(f"读取预测结果: {pred_path}")
    logger.info(f"读取元数据:   {meta_path}")

    result = evaluate_v26(pred_path, meta_path)

    if result is None:
        logger.error("评估失败")
        sys.exit(1)

    log_section("v26 评估结果", logger)

    logger.info("Feature Bank 配置:")
    logger.info(f"  hidden_size:          {result['meta'].get('config', {}).get('hidden_size')}")
    logger.info(f"  beta_threshold:       {result['meta'].get('adaptive_beta', 0.0):.4f}")
    logger.info(f"  background_removal:   {result['meta'].get('config', {}).get('background_removal_method')}")

    logger.info("\nV26 推理配置:")
    pm = result['pred_meta']
    logger.info(f"  top_k_per_image:       {pm.get('top_k_per_image')}")
    logger.info(f"  nms_iou_threshold:     {pm.get('nms_iou_threshold')}")
    logger.info(f"  min_bbox_area:         {pm.get('min_bbox_area')}")
    logger.info(f"  score_map_smooth_sigma: {pm.get('score_map_smooth_sigma')}")
    logger.info(f"  per_image_threshold_ratio: {pm.get('per_image_threshold_ratio')}")

    logger.info(f"\n图像级分数:")
    logger.info(f"  默认主口径: {result['primary_image_score_strategy']}")
    logger.info(f"  默认主口径 AUC: {result['image_auc']:.4f}")
    for key in ['region_max', 'patch_mean', 'patch_max', 'blend_region_patch', 'density_weighted_patch']:
        stat = result['image_auc_by_strategy'][key]
        logger.info(f"  {key}: AUC={stat['auc']:.4f}, normal_mean={stat['normal_mean']:.4f}, anomaly_mean={stat['anomaly_mean']:.4f}")
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
        'primary_image_score_strategy': result['primary_image_score_strategy'],
        'image_auc':   result['image_auc'],
        'image_auc_by_strategy': result['image_auc_by_strategy'],
        'total_pred':  result['total_pred'],
        'normal_mean': result['normal_mean'],
        'anomaly_mean': result['anomaly_mean'],
    }
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)

    logger.info(f"\n评估报告已保存: {report_path}")
    log_finish("03_evaluate_v26", logger)

    # 终端汇总输出
    print("\n" + "=" * 80)
    print("v26 评估完成！")
    print("=" * 80)
    print(f"主要指标 (IoU=0.5):")
    print(f"  TP={m05['tp']}, FP={m05['fp']}, FN={m05['fn']}")
    print(f"  Precision={m05['precision']:.4f}, Recall={m05['recall']:.4f}, F1={m05['f1']:.4f}")
    print(f"  图像级 AUC({result['primary_image_score_strategy']})={result['image_auc']:.4f}")
    print(f"  图像级 AUC(patch_mean)={result['image_auc_by_strategy']['patch_mean']['auc']:.4f}")
    print(f"  图像级 AUC(region_max)={result['image_auc_by_strategy']['region_max']['auc']:.4f}")
    print(f"\n对比 V23 基线 (IoU=0.5):")
    print(f"  V26 当前: TP={m05['tp']}, FP={m05['fp']}, F1={m05['f1']:.3f}, Precision={m05['precision']:.3f}, AUC(patch_mean)={result['image_auc_by_strategy']['patch_mean']['auc']:.3f}, AUC(region_max)={result['image_auc_by_strategy']['region_max']['auc']:.3f}")
    print("=" * 80)
