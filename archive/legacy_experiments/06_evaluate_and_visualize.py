"""
Res-SAM 复现实验 V2 - Step 3: 评估和可视化

功能：
- 计算完整评估指标 (Precision, Recall, F1, AUC)
- 生成可视化结果
- 阈值优化分析
- 与论文结果对比

论文对应：Table 2, Fig.3
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import cv2
from PIL import Image

# ============ 配置 ============
CONFIG = {
    # 结果路径
    'predictions_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'outputs', 'predictions_v2', 'auto_predictions.json'),
    
    # 可视化输出
    'vis_output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'outputs', 'visualizations_v2'),
    
    # 分析输出
    'analysis_output': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'experiments', 'ANALYSIS_V2.md'),
    
    # 论文基准结果 (Table 2 - Real-world dataset)
    'paper_baseline': {
        'Precision': 0.880,
        'Recall': 0.955,
        'F1': 0.916,
        'AUC': 0.872,
    },
}


def _compute_iou(box1: list, box2: list) -> float:
    """计算两个 bbox 的 IoU"""
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


def _match_tp_fp_fn(pred_bboxes: list, gt_bboxes: list, iou_thresh: float = 0.5) -> tuple:
    """一对一 IoU 匹配，返回 (tp, fp, fn)。"""
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

        if best_gt_idx >= 0 and best_iou >= iou_thresh:
            matched_gt[best_gt_idx] = True
            tp += 1
        else:
            fp += 1

    fn = sum(1 for m in matched_gt if not m)
    return tp, fp, fn


def compute_metrics(results: dict) -> dict:
    """计算完整评估指标"""
    all_tp = []
    all_fp = []
    all_fn = []
    all_scores = []
    all_labels = []
    
    for category, cat_results in results.items():
        # 统一按一对一 IoU 匹配口径重新统计，避免依赖推理阶段写入的计数口径
        tp = 0
        fp = 0
        fn = 0
        for r in cat_results:
            _tp, _fp, _fn = _match_tp_fp_fn(r.get('pred_bboxes', []), r.get('gt_bboxes', []), iou_thresh=0.5)
            tp += _tp
            fp += _fp
            fn += _fn
        
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        
        # 收集分数和标签 (用于 AUC)
        for r in cat_results:
            if r['anomaly_scores']:
                max_score = max(r['anomaly_scores'])
                all_scores.append(max_score)
                all_labels.append(1 if r['num_gt'] > 0 else 0)
    
    # 总体指标
    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_fn = sum(all_fn)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # AUC 计算
    if len(set(all_labels)) > 1:
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
    else:
        roc_auc = 0.5
    
    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'per_category': {
            cat: {
                'tp': sum(r['tp'] for r in res),
                'fp': sum(r['fp'] for r in res),
                'fn': sum(r['fn'] for r in res),
            }
            for cat, res in results.items()
        }
    }


def plot_roc_curve(results: dict, output_path: str):
    """绘制 ROC 曲线"""
    all_scores = []
    all_labels = []
    
    for category, cat_results in results.items():
        for r in cat_results:
            if r['anomaly_scores']:
                max_score = max(r['anomaly_scores'])
                all_scores.append(max_score)
                all_labels.append(1 if r['num_gt'] > 0 else 0)
    
    if len(set(all_labels)) < 2:
        print("无法绘制 ROC 曲线: 标签单一")
        return
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Res-SAM ROC Curve (Fully Automatic Mode)', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲线保存至: {output_path}")


def plot_threshold_analysis(results: dict, output_path: str):
    """阈值优化分析"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions = []
    recalls = []
    f1s = []
    
    for thresh in thresholds:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        
        for category, cat_results in results.items():
            for r in cat_results:
                pred_bboxes = r.get('pred_bboxes', [])
                pred_scores = r.get('anomaly_scores', [])
                gt_bboxes = r.get('gt_bboxes', [])

                # 根据阈值过滤预测框
                filtered_pred = [
                    b for b, s in zip(pred_bboxes, pred_scores)
                    if s is not None and s > thresh
                ]

                _tp, _fp, _fn = _match_tp_fp_fn(filtered_pred, gt_bboxes, iou_thresh=0.5)
                tp_total += _tp
                fp_total += _fp
                fn_total += _fn
        
        prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
    plt.plot(thresholds, recalls, 'g-', lw=2, label='Recall')
    plt.plot(thresholds, f1s, 'r-', lw=2, label='F1-score')
    
    # 标记最佳 F1 点
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    plt.axvline(x=best_thresh, color='gray', linestyle='--', alpha=0.7)
    plt.scatter([best_thresh], [best_f1], color='red', s=100, zorder=5)
    plt.annotate(f'Best: thresh={best_thresh:.2f}, F1={best_f1:.3f}',
                xy=(best_thresh, best_f1), xytext=(best_thresh + 0.1, best_f1 - 0.1),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.xlabel('Anomaly Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold Optimization Analysis', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"阈值分析图保存至: {output_path}")
    
    return best_thresh


def generate_comparison_table(metrics: dict, paper_baseline: dict) -> str:
    """生成与论文的对比表格"""
    md = """# Res-SAM V2 复现结果分析

## 一、评估指标对比

### Table: 复现结果 vs 论文基准

| 指标 | 论文 (Real-world) | 复现 V2 | 差距 |
|------|-------------------|---------|------|
"""
    
    for metric in ['Precision', 'Recall', 'F1', 'AUC']:
        paper_val = paper_baseline.get(metric, 0)
        our_val = metrics.get(metric.lower(), 0)
        diff = our_val - paper_val
        diff_str = f"+{diff:.3f}" if diff > 0 else f"{diff:.3f}"
        md += f"| {metric} | {paper_val:.3f} | {our_val:.3f} | {diff_str} |\n"
    
    md += """
---

## 二、详细统计

### 总体统计
"""
    md += f"""
- **TP (True Positive)**: {metrics['tp']}
- **FP (False Positive)**: {metrics['fp']}
- **FN (False Negative)**: {metrics['fn']}

### 各类别统计

| 类别 | TP | FP | FN |
|------|----|----|----|
"""
    
    for cat, cat_metrics in metrics.get('per_category', {}).items():
        md += f"| {cat} | {cat_metrics['tp']} | {cat_metrics['fp']} | {cat_metrics['fn']} |\n"
    
    md += """
---

## 三、改进说明

### V2 版本改进

1. **集成 SAM 模块**
   - 使用 SAM 生成候选异常区域
   - 符合论文原始流程

2. **修正 Feature Bank 构建**
   - 使用正确的 2D-ESN 特征提取
   - 记录数据来源用于环境一致性检查

3. **阈值优化**
   - 支持 ROC 曲线分析
   - 自动寻找最优阈值

### 待优化项

1. **SAM 模型权重**
   - 需要下载 SAM 预训练权重
   - 主线使用 vit_l；显存不足时可改用 vit_b

2. **环境一致性**
   - Feature Bank 应与测试数据来自同一环境
   - 论文 Fig.6 强调此点

3. **数据增强**
   - 可考虑对 Feature Bank 进行数据增强
   - 提高泛化能力

---

## 四、下一步行动

- [ ] 下载 SAM 模型权重
- [ ] 运行完整的推理流程
- [ ] 对比不同阈值的效果
- [ ] 分析 SAM 在 GPR 图像上的分割效果

---

*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return md


def visualize_sample_results(
    results: dict,
    data_dirs: dict,
    output_dir: str,
    num_samples: int = 5,
):
    """可视化样本结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    for category, cat_results in results.items():
        cat_dir = os.path.join(output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)
        
        # 选择有检测结果的样本
        samples_with_detection = [r for r in cat_results if r['pred_bboxes']]
        samples_without_detection = [r for r in cat_results if not r['pred_bboxes'] and r['num_gt'] > 0]
        
        # 可视化
        for i, sample in enumerate(samples_with_detection[:num_samples]):
            img_path = os.path.join(data_dirs.get(category, ''), sample['image_name'])
            if not os.path.exists(img_path):
                continue
            
            # 加载图像
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            
            # 转换为彩色
            vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # 绘制预测 bbox
            for j, bbox in enumerate(sample['pred_bboxes']):
                color = [(0, 0, 255), (0, 165, 255), (0, 255, 255)][j % 3]
                cv2.rectangle(vis_img, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             color, 2)
                
                score = sample['anomaly_scores'][j] if j < len(sample['anomaly_scores']) else 0
                cv2.putText(vis_img, f"#{j+1} {score:.3f}", 
                           (int(bbox[0]), int(bbox[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 保存
            output_path = os.path.join(cat_dir, f"sample_{i+1}.jpg")
            cv2.imwrite(output_path, vis_img)
        
        print(f"可视化保存至: {cat_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("Res-SAM V2: 评估和可视化")
    print("=" * 60)
    
    # 检查结果文件
    if not os.path.exists(CONFIG['predictions_path']):
        print(f"警告: 预测结果文件不存在: {CONFIG['predictions_path']}")
        print("请先运行 05_inference_auto_v2.py")
        return
    
    # 加载结果
    with open(CONFIG['predictions_path'], 'r') as f:
        results = json.load(f)
    
    print(f"加载了 {len(results)} 个类别的结果")
    
    # 创建输出目录
    os.makedirs(CONFIG['vis_output_dir'], exist_ok=True)
    
    # 计算指标
    print("\n计算评估指标...")
    metrics = compute_metrics(results)
    
    print(f"\n总体指标:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    # 绘制 ROC 曲线
    print("\n绘制 ROC 曲线...")
    roc_path = os.path.join(CONFIG['vis_output_dir'], 'roc_curve.png')
    plot_roc_curve(results, roc_path)
    
    # 阈值分析
    print("\n阈值优化分析...")
    thresh_path = os.path.join(CONFIG['vis_output_dir'], 'threshold_analysis.png')
    best_thresh = plot_threshold_analysis(results, thresh_path)
    print(f"最佳阈值: {best_thresh:.2f}")
    
    # 生成对比报告
    print("\n生成分析报告...")
    report = generate_comparison_table(metrics, CONFIG['paper_baseline'])
    
    with open(CONFIG['analysis_output'], 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"分析报告保存至: {CONFIG['analysis_output']}")
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
