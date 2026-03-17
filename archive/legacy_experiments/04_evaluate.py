"""
Res-SAM 复现实验 - Step 3: 评估指标计算

功能：
- 计算 IoU > 0.5 的检测正确率
- 计算 Precision, Recall, F1
- 计算 AUC（基于异常分数）
- 生成评估报告

论文对应：Table 2 的指标
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt


def compute_iou(box1, box2):
    """
    计算两个 bbox 的 IoU
    
    Args:
        box1: (xmin, ymin, xmax, ymax) 或 dict
        box2: (xmin, ymin, xmax, ymax) 或 dict
    
    Returns:
        float: IoU 值
    """
    # 转换格式
    if isinstance(box1, dict):
        x1_1, y1_1, x2_1, y2_1 = box1['xmin'], box1['ymin'], box1['xmax'], box1['ymax']
    elif isinstance(box1, (list, tuple)):
        x1_1, y1_1, x2_1, y2_1 = box1
    else:
        return 0
    
    if isinstance(box2, dict):
        x1_2, y1_2, x2_2, y2_2 = box2['xmin'], box2['ymin'], box2['xmax'], box2['ymax']
    elif isinstance(box2, (list, tuple)):
        x1_2, y1_2, x2_2, y2_2 = box2
    else:
        return 0
    
    # 计算交集
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0
    
    return inter_area / union_area


def evaluate_detection(predictions, iou_threshold=0.5):
    """
    评估检测结果
    
    Args:
        predictions: 预测结果列表，每个元素包含 pred_bbox 和 ground_truth
        iou_threshold: IoU 阈值，默认 0.5
    
    Returns:
        dict: 包含 TP, FP, FN, Precision, Recall, F1
    """
    tp, fp, fn = 0, 0, 0
    
    for pred in predictions:
        gt = pred.get('ground_truth')
        pred_bbox = pred.get('pred_bbox')
        
        # 判断是否有目标
        has_gt = gt is not None and len(gt.get('objects', [])) > 0
        has_pred = pred_bbox is not None
        
        if has_gt and has_pred:
            # 有 GT 且有预测，计算 IoU
            gt_obj = gt['objects'][0]  # 取第一个目标
            iou = compute_iou(pred_bbox, gt_obj)
            
            if iou > iou_threshold:
                tp += 1
            else:
                fp += 1
        elif has_gt and not has_pred:
            # 有 GT 但无预测 → 漏检
            fn += 1
        elif not has_gt and has_pred:
            # 无 GT 但有预测 → 误检
            fp += 1
        # 无 GT 且无预测 → TN，不计入
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
    }


def compute_auc(predictions):
    """
    计算 AUC（基于异常分数）
    
    Args:
        predictions: 预测结果列表
    
    Returns:
        float: AUC 值
    """
    scores = []
    labels = []
    
    for pred in predictions:
        gt = pred.get('ground_truth')
        score = pred.get('anomaly_score', 0)
        
        # 判断是否有异常
        has_anomaly = gt is not None and len(gt.get('objects', [])) > 0
        
        scores.append(score)
        labels.append(1 if has_anomaly else 0)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    # 检查是否有正负样本
    if len(np.unique(labels)) < 2:
        return 0.5  # 只有一类样本
    
    try:
        auc_value = roc_auc_score(labels, scores)
    except Exception as e:
        print(f"Error computing AUC: {e}")
        auc_value = 0
    
    return auc_value


def plot_roc_curve(predictions, output_path):
    """绘制 ROC 曲线"""
    scores = []
    labels = []
    
    for pred in predictions:
        gt = pred.get('ground_truth')
        score = pred.get('anomaly_score', 0)
        has_anomaly = gt is not None and len(gt.get('objects', [])) > 0
        
        scores.append(score)
        labels.append(1 if has_anomaly else 0)
    
    scores = np.array(scores)
    labels = np.array(labels)
    
    if len(np.unique(labels)) < 2:
        print("Cannot plot ROC curve: only one class present")
        return
    
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("Res-SAM Evaluation Metrics")
    print("=" * 60)
    
    # 加载预测结果
    predictions_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'outputs', 'predictions', 'auto_predictions.json'
    )
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\n"
            "Please run 02_inference_auto.py first.")
    
    print(f"Loading predictions from: {predictions_path}")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        all_predictions = json.load(f)
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                              'outputs', 'metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估每个类别
    all_results = {}
    
    for category, predictions in all_predictions.items():
        print(f"\n{'='*60}")
        print(f"Evaluating category: {category}")
        print(f"Number of samples: {len(predictions)}")
        print("=" * 60)
        
        # 检测评估
        det_metrics = evaluate_detection(predictions, iou_threshold=0.5)
        
        # AUC 计算
        auc_value = compute_auc(predictions)
        
        # 合并结果
        results = {
            **det_metrics,
            'AUC': auc_value,
            'num_samples': len(predictions),
        }
        all_results[category] = results
        
        # 打印结果
        print(f"\nDetection Metrics (IoU > 0.5):")
        print(f"  TP: {det_metrics['TP']}")
        print(f"  FP: {det_metrics['FP']}")
        print(f"  FN: {det_metrics['FN']}")
        print(f"  Precision: {det_metrics['Precision']:.4f}")
        print(f"  Recall: {det_metrics['Recall']:.4f}")
        print(f"  F1: {det_metrics['F1']:.4f}")
        print(f"\nAUC: {auc_value:.4f}")
        
        # 绘制 ROC 曲线
        roc_path = os.path.join(output_dir, f'roc_{category}.png')
        plot_roc_curve(predictions, roc_path)
    
    # 计算总体结果
    print(f"\n{'='*60}")
    print("Overall Results")
    print("=" * 60)
    
    total_tp = sum(r['TP'] for r in all_results.values())
    total_fp = sum(r['FP'] for r in all_results.values())
    total_fn = sum(r['FN'] for r in all_results.values())
    total_samples = sum(r['num_samples'] for r in all_results.values())
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0
    
    # 计算总体 AUC
    all_preds_combined = []
    for preds in all_predictions.values():
        all_preds_combined.extend(preds)
    overall_auc = compute_auc(all_preds_combined)
    
    overall_results = {
        'TP': total_tp,
        'FP': total_fp,
        'FN': total_fn,
        'Precision': overall_precision,
        'Recall': overall_recall,
        'F1': overall_f1,
        'AUC': overall_auc,
        'num_samples': total_samples,
    }
    all_results['overall'] = overall_results
    
    print(f"\nOverall Detection Metrics:")
    print(f"  Total samples: {total_samples}")
    print(f"  TP: {total_tp}")
    print(f"  FP: {total_fp}")
    print(f"  FN: {total_fn}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1: {overall_f1:.4f}")
    print(f"  AUC: {overall_auc:.4f}")
    
    # 保存结果
    output_path = os.path.join(output_dir, 'evaluation_results.json')
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 生成表格格式（Table 2）
    print(f"\n{'='*60}")
    print("Table 2 Reproduction: Fully Automatic Mode")
    print("=" * 60)
    print(f"{'Category':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 68)
    for category, results in all_results.items():
        if category != 'overall':
            print(f"{category:<20} {results['Precision']:<12.4f} {results['Recall']:<12.4f} "
                  f"{results['F1']:<12.4f} {results['AUC']:<12.4f}")
    print("-" * 68)
    print(f"{'Overall':<20} {overall_results['Precision']:<12.4f} {overall_results['Recall']:<12.4f} "
          f"{overall_results['F1']:<12.4f} {overall_results['AUC']:<12.4f}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    results = main()
