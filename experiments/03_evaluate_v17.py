#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V17 评估脚本
评估所有改进的综合效果（Pixel-level Heatmap + 后处理过滤）
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

def evaluate_v17(pred_path, meta_path):
    """评估 V17 结果"""
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
    
    # 多IoU阈值评估
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
                gt_bboxes = r.get('gt_bboxes', [])
                scores = r.get('anomaly_scores', [])
                
                matched_gt = set()
                for pred, score in zip(pred_bboxes, scores):
                    best_iou = 0.0
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(gt_bboxes):
                        if gt_idx in matched_gt:
                            continue
                        iou_val = compute_iou(pred, gt)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_thresh:
                        tp += 1
                        matched_gt.add(best_gt_idx)
                        tp_scores.append(score)
                    else:
                        fp += 1
                        fp_scores.append(score)
                
                fn += len(gt_bboxes) - len(matched_gt)
        
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        
        metrics_by_iou[iou_thresh] = {
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'tp_scores': tp_scores, 'fp_scores': fp_scores
        }
    
    # 图像级AUC
    normal_scores = []
    anomaly_scores = []
    for cat, recs in results.items():
        for r in recs:
            scores = r.get('anomaly_scores', [])
            img_score = max(scores) if scores else 0.0
            if cat == 'normal_auc':
                normal_scores.append(img_score)
            elif not r.get('exclude_from_det_metrics'):
                anomaly_scores.append(img_score)
    
    y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    y_score = normal_scores + anomaly_scores
    image_auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else 0.0
    
    # 粗筛统计
    coarse_total = 0
    coarse_discarded = 0
    for cat, recs in results.items():
        for r in recs:
            coarse_total += r.get('num_candidates', 0) + r.get('num_coarse_discarded', 0)
            coarse_discarded += r.get('num_coarse_discarded', 0)
    
    discard_rate = coarse_discarded / coarse_total * 100 if coarse_total > 0 else 0.0
    
    # 总pred框数
    total_pred = sum(len(r.get('pred_bboxes', [])) 
                     for cat, recs in results.items() 
                     for r in recs 
                     if not r.get('exclude_from_det_metrics'))
    
    return {
        'meta': meta,
        'pred_meta': pred_meta,
        'metrics_by_iou': metrics_by_iou,
        'image_auc': image_auc,
        'discard_rate': discard_rate,
        'total_pred': total_pred,
        'normal_mean': np.mean(normal_scores) if normal_scores else 0.0,
        'anomaly_mean': np.mean(anomaly_scores) if anomaly_scores else 0.0,
    }

if __name__ == '__main__':
    pred_path = os.path.join(BASE_DIR, 'outputs', 'predictions_v17', 'auto_predictions_v17.json')
    meta_path = os.path.join(BASE_DIR, 'outputs', 'feature_banks_v17', 'metadata.json')
    
    if not os.path.exists(pred_path):
        print(f"结果文件不存在: {pred_path}")
        print("请先运行 01_build_feature_bank_v17.py 和 02_inference_auto_v17.py")
        sys.exit(1)
    
    print("=" * 80)
    print("V17-2 评估结果（参数调整：让pred框变大）")
    print("=" * 80)
    
    res = evaluate_v17(pred_path, meta_path)
    if res is None:
        print("评估失败")
        sys.exit(1)
    
    # 打印配置
    print("\n配置:")
    print("  hidden_size = {}".format(res['meta']['config']['hidden_size']))
    print("  bg_method = {}".format(res['meta']['config']['background_removal_method']))
    print("  adaptive_beta = {:.4f}".format(res['meta']['adaptive_beta']))
    print("  beta_method = {}".format(res['meta'].get('beta_calibration_method', 'unknown')))
    print("  特征维度 = {}".format(res['meta']['feature_bank_shape'][1]))
    
    # V17-2 后处理参数
    print("\nV17-2 核心改进参数:")
    print("  use_pixel_heatmap = {}".format(res['pred_meta'].get('use_pixel_heatmap')))
    if res['pred_meta'].get('use_pixel_heatmap'):
        print("  heatmap_beta_normalized = {}".format(res['pred_meta'].get('heatmap_beta_normalized')))
        print("  heatmap_min_area = {}".format(res['pred_meta'].get('heatmap_min_area')))
    print("  confidence_percentile = {}".format(res['pred_meta'].get('confidence_percentile')))
    print("  nms_iou_threshold = {}".format(res['pred_meta'].get('nms_iou_threshold')))
    print("  top_k_per_image = {}".format(res['pred_meta'].get('top_k_per_image')))
    
    # 打印验证集信息
    if 'validation_set' in res['meta']:
        val_set = res['meta']['validation_set']
        print("\n验证集:")
        print("  正常图: {} 张".format(val_set['num_normal']))
        print("  异常图: {} 张".format(val_set['num_anomaly']))
        if 'stats' in val_set:
            print("  正常scores: mean={:.4f}, p95={:.4f}".format(
                val_set['stats']['normal_scores']['mean'],
                val_set['stats']['normal_scores']['p95']))
            print("  异常scores: mean={:.4f}, p10={:.4f}".format(
                val_set['stats']['anomaly_scores']['mean'],
                val_set['stats']['anomaly_scores']['p10']))
    
    # 打印检测指标
    print("\n检测指标:")
    print("  {:<10} {:<6} {:<6} {:<6} {:<10} {:<10} {:<10}".format(
        'IoU阈值', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'))
    print("  " + "-"*70)
    for iou_t in [0.1, 0.2, 0.3, 0.5]:
        m = res['metrics_by_iou'][iou_t]
        print("  >{:<9.1f} {:<6} {:<6} {:<6} {:<10.3f} {:<10.3f} {:<10.3f}".format(
            iou_t, m['tp'], m['fp'], m['fn'], m['precision'], m['recall'], m['f1']))
    
    # 打印AUC和其他指标
    print("\n图像级AUC: {:.3f}".format(res['image_auc']))
    print("粗筛丢弃率: {:.1f}%".format(res['discard_rate']))
    print("总pred框数: {}".format(res['total_pred']))
    print("正常图mean_score: {:.3f}".format(res['normal_mean']))
    print("异常图mean_score: {:.3f}".format(res['anomaly_mean']))
    
    # TP/FP分数分析
    m05 = res['metrics_by_iou'][0.5]
    if m05['tp_scores']:
        print("\nTP框分数(IoU>0.5): mean={:.3f}, min={:.3f}, max={:.3f}".format(
            np.mean(m05['tp_scores']), np.min(m05['tp_scores']), np.max(m05['tp_scores'])))
    else:
        print("\nTP框分数(IoU>0.5): 无TP框")
    
    if m05['fp_scores']:
        print("FP框分数(IoU>0.5): mean={:.3f}, min={:.3f}, max={:.3f}".format(
            np.mean(m05['fp_scores']), np.min(m05['fp_scores']), np.max(m05['fp_scores'])))
    
    # TP/FP分离度
    if m05['tp_scores'] and m05['fp_scores']:
        separation = np.mean(m05['tp_scores']) - np.mean(m05['fp_scores'])
        status = 'TP>FP ✓' if separation > 0 else 'TP<FP ✗ 评分方向反转'
        print("TP/FP分离度: {:.3f} ({})".format(separation, status))
    
    # 对比 V16 和 V17
    print("\n" + "="*80)
    print("对比 V16 和 V17:")
    print("  V16: beta=0.1399(p5), FP=170, TP=10, F1(0.5)=0.083, AUC=0.821")
    print("  V17: beta=0.175(p10), FP=48, TP=0, F1(0.5)=0.000, AUC=0.849")
    print("  V17-2: beta={:.3f}(p10), FP={}, TP={}, F1(0.5)={:.3f}, AUC={:.3f}".format(
        res['meta']['adaptive_beta'], m05['fp'], m05['tp'], 
        m05['f1'], res['image_auc']))
    
    # 计算改进
    v17_fp = 48
    v17_f1 = 0.000
    fp_change = (m05['fp'] - v17_fp)  # 正数表示FP增加（变差），负数表示FP减少（变好）
    f1_change = m05['f1'] - v17_f1  # 正数表示F1提升（变好）
    
    print("\nV17-2 vs V17 改进:")
    print("  FP 变化: {} ({} → {})".format(
        "+{}".format(fp_change) if fp_change > 0 else fp_change, v17_fp, m05['fp']))
    print("  F1 变化: {:.3f} → {:.3f} ({:+.3f})".format(v17_f1, m05['f1'], f1_change))
    print("  TP 变化: 0 → {}".format(m05['tp']))
    
    if m05['tp'] > 0:
        print("\n✓ 成功：V17-2 参数调整有效，TP 恢复！")
    else:
        print("\n✗ 失败：V17-2 参数调整无效，TP 仍为 0")
    
    # 预期目标检查（V17-2）
    print("\n预期目标（V17-2）:")
    print("  TP (IoU>0.5): 5-10 (实际: {})".format(m05['tp']))
    print("  FP: 80-120 (实际: {})".format(m05['fp']))
    print("  F1 (IoU>0.5): 0.08-0.12 (实际: {:.3f})".format(m05['f1']))
    
    goals_met = 0
    if 5 <= m05['tp'] <= 10:
        print("  ✓ TP 达标")
        goals_met += 1
    if 80 <= m05['fp'] <= 120:
        print("  ✓ FP 达标")
        goals_met += 1
    if 0.08 <= m05['f1'] <= 0.12:
        print("  ✓ F1 达标")
        goals_met += 1
    
    print("\n达标项: {}/3".format(goals_met))
    
    print("=" * 80)
