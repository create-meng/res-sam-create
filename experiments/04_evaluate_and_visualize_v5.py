"""
Res-SAM 复现实验 V5 - Step 3: 评估和可视化

V3 改进（严格按论文对齐）：
- window_size = 50（论文默认值）
- 特征口径 f = [W_out, b]（论文 Eq.(2)-(3)）
- Region 级粗筛统计
- 严格一对一 IoU 匹配评估

论文对应：Table 2, Fig.3
"""

import sys
import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
try:
    import cv2
except Exception:
    cv2 = None
from PIL import Image

# ============ 配置 ============
CONFIG = {
    # 结果路径 (V5)
    'predictions_path': os.path.join(
        BASE_DIR,
        'outputs',
        'predictions_v5',
        'auto_predictions_v5.json',
    ),

    # Click-guided 结果路径 (V5 / Table 1)
    'click_predictions_path': os.path.join(
        BASE_DIR,
        'outputs',
        'predictions_v5',
        'click_predictions_v5.json',
    ),
    
    # 可视化输出
    'vis_output_dir': os.path.join(
        BASE_DIR,
        'outputs',
        'visualizations_v5',
    ),
    
    # 分析输出
    'analysis_output': os.path.join(
        BASE_DIR,
        'outputs',
        'visualizations_v5',
        '04_evaluate_and_visualize_v5_report.md',
    ),
    
    # Paper reference selected for this repo's local dataset mapping.
    # Local rule:
    # - augmented_intact -> normal
    # - augmented_cavities -> cavities
    # - augmented_utilities -> pipes/utilities
    # The raw cavities/Utilities folders are not used here because they
    # do not provide the required annotation files for this evaluation.
    'paper_reference': {
        'dataset_name': 'Open-source (paper Table 1/2)',
        'dataset_note': 'Local evaluation uses augmented_intact + augmented_cavities + augmented_utilities with a fully enhanced evaluation protocol; only the Feature Bank source differs in V5.',
        'auto': {
            'Precision': 0.842,
            'Recall': 0.877,
            'F1': 0.859,
            'AUC': 0.832,
        },
        'click': {
            '5/5': {'AUC': 0.823, 'F1': 0.852},
            '5/3': {'AUC': 0.827, 'F1': 0.863},
            '3/1': {'AUC': 0.832, 'F1': 0.850},
        },
    },

    'version': 'V5',
    'alignment_notes': 'Enhanced-eval setup: fb_source=augmented_intact, eval=augmented_intact + augmented anomalies, feature f=[W_out,b]',
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
    """计算完整评估指标（V3 包含粗筛统计）"""
    all_tp = []
    all_fp = []
    all_fn = []
    all_scores = []
    all_labels = []
    all_coarse_discarded = []
    per_category = {}
    
    for category, cat_results in results.items():
        # 统一按一对一 IoU 匹配口径重新统计
        tp = 0
        fp = 0
        fn = 0
        coarse_discarded = 0
        
        for r in cat_results:
            if r.get('exclude_from_det_metrics', False):
                continue
            _tp, _fp, _fn = _match_tp_fp_fn(r.get('pred_bboxes', []), r.get('gt_bboxes', []), iou_thresh=0.5)
            tp += _tp
            fp += _fp
            fn += _fn
            coarse_discarded += r.get('num_coarse_discarded', 0)
        
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_coarse_discarded.append(coarse_discarded)
        
        per_category[category] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'coarse_discarded': coarse_discarded,
        }

        # 收集分数和标签 (用于 AUC)
        # 注意：必须覆盖所有样本；无检测结果时记为 0 分，避免 AUC 仅在“有预测”的子集上计算。
        for r in cat_results:
            if r.get('exclude_from_auc', False):
                continue
            if r.get('anomaly_scores'):
                max_score = max(r['anomaly_scores'])
            else:
                max_score = 0.0
            all_scores.append(float(max_score))
            all_labels.append(1 if r.get('num_gt', 0) > 0 else 0)
    
    # 总体指标
    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_fn = sum(all_fn)
    total_coarse_discarded = sum(all_coarse_discarded)
    
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
        'coarse_discarded': total_coarse_discarded,
        'per_category': per_category,
    }


def plot_roc_curve(results: dict, output_path: str):
    """绘制 ROC 曲线"""
    all_scores = []
    all_labels = []
    
    for category, cat_results in results.items():
        for r in cat_results:
            if r.get('exclude_from_auc', False):
                continue
            scores = r.get('anomaly_scores') or []
            max_score = max(scores) if scores else 0.0
            all_scores.append(float(max_score))
            all_labels.append(1 if r.get('num_gt', 0) > 0 else 0)
    
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
    plt.title('Res-SAM V5 ROC Curve (Fully Automatic Mode)', fontsize=14)
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
                if r.get('exclude_from_det_metrics', False):
                    continue
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
    
    plt.xlabel('Anomaly Threshold (β)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('V3 Threshold Optimization Analysis', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"阈值分析图保存至: {output_path}")
    
    return best_thresh


def generate_comparison_table(metrics: dict, paper_reference: dict) -> str:
    """生成中文对比报告。"""
    auto_baseline = paper_reference.get('auto', {})
    dataset_name = paper_reference.get('dataset_name', 'Paper reference')
    dataset_note = paper_reference.get('dataset_note', '')

    lines = [
        '# Res-SAM V5 评估报告',
        '',
        '## 数据集对应关系',
        f'- 论文参考结果: {dataset_name}',
        f'- 本地评估数据: augmented_intact + augmented_cavities + augmented_utilities',
        f'- 类别映射: cavities -> cavities, utilities -> pipes/utilities, normal_auc -> augmented_intact',
        f'- 说明: {dataset_note}',
        '',
        '## 全自动模式直接对比',
        '',
        '| 指标 | 论文 | 本地 | 差值 |',
        '|---|---:|---:|---:|',
    ]

    for metric in ['Precision', 'Recall', 'F1', 'AUC']:
        paper_val = auto_baseline.get(metric, 0.0)
        our_val = metrics.get(metric.lower(), 0.0)
        diff = our_val - paper_val
        metric_cn = {
            'Precision': 'Precision',
            'Recall': 'Recall',
            'F1': 'F1',
            'AUC': 'AUC',
        }[metric]
        lines.append(f'| {metric_cn} | {paper_val:.3f} | {our_val:.3f} | {diff:+.3f} |')

    lines.extend([
        '',
        '## 全自动模式统计',
        '',
        f'- TP: {metrics["tp"]}',
        f'- FP: {metrics["fp"]}',
        f'- FN: {metrics["fn"]}',
        f'- Region 级粗筛丢弃数: {metrics["coarse_discarded"]}',
        '',
        '## 分类别结果',
        '',
        '| 类别 | TP | FP | FN | Precision | Recall | F1 | 粗筛丢弃数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ])

    for cat, cat_metrics in metrics.get('per_category', {}).items():
        tp = cat_metrics.get('tp', 0)
        fp = cat_metrics.get('fp', 0)
        fn = cat_metrics.get('fn', 0)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        lines.append(
            f'| {cat} | {tp} | {fp} | {fn} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {cat_metrics.get("coarse_discarded", 0)} |'
        )

    lines.extend([
        '',
        f'*生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*',
        '*版本: V5（论文对齐评估脚本）*',
    ])

    return '\n'.join(lines)


def generate_click_summary(click_metrics_map: dict, paper_reference: dict) -> str:
    """生成中文 click-guided 对比报告。"""
    click_baseline = paper_reference.get('click', {})
    lines = [
        '## Click-guided 模式直接对比',
        '',
        '| 点击配置 | 论文 AUC | 本地 AUC | 差值 | 论文 F1 | 本地 F1 | 差值 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]

    def _sort_key(name: str):
        part = name.replace('click_', '')
        a, b = part.split('/')
        return (-int(a), -int(b))

    for click_key in sorted(click_metrics_map.keys(), key=_sort_key):
        label = click_key.replace('click_', '')
        local_metrics = click_metrics_map[click_key]
        paper_metrics = click_baseline.get(label, {})
        paper_auc = paper_metrics.get('AUC', 0.0)
        paper_f1 = paper_metrics.get('F1', 0.0)
        local_auc = local_metrics.get('auc', 0.0)
        local_f1 = local_metrics.get('f1', 0.0)
        lines.append(
            f'| {label} | {paper_auc:.3f} | {local_auc:.3f} | {local_auc - paper_auc:+.3f} | {paper_f1:.3f} | {local_f1:.3f} | {local_f1 - paper_f1:+.3f} |'
        )

    return '\n'.join(lines) + '\n'


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
            
            # 添加粗筛信息
            coarse_discarded = sample.get('num_coarse_discarded', 0)
            cv2.putText(vis_img, f"Coarse discarded: {coarse_discarded}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # 保存
            output_path = os.path.join(cat_dir, f"sample_{i+1}.jpg")
            cv2.imwrite(output_path, vis_img)
        
        print(f"可视化保存至: {cat_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("Res-SAM V5: 评估和可视化 (Strict Paper Alignment)")
    print("=" * 60)
    print(f"  window_size = 50")
    print(f"  feature = [W_out, b] (dim=61)")
    print(f"  region-level coarse filtering enabled")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(CONFIG['vis_output_dir'], exist_ok=True)

    report_parts = []

    # ========== Fully automatic (Table 2 / Fig.3) ==========
    if os.path.exists(CONFIG['predictions_path']):
        # 加载结果
        with open(CONFIG['predictions_path'], 'r') as f:
            obj = json.load(f)

        # 兼容输出格式：旧版为 {category: [records...]}
        # 新版为 {"meta": {...}, "results": {category: [records...]}}
        results = obj.get('results', obj) if isinstance(obj, dict) else obj

        print(f"加载了 {len(results)} 个类别的结果")

        # 计算指标
        print("\n计算评估指标...")
        metrics = compute_metrics(results)

        print(f"\nV3 总体指标:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Region 级粗筛丢弃: {metrics['coarse_discarded']}")

        # 绘制 ROC 曲线
        print("\n绘制 ROC 曲线...")
        roc_path = os.path.join(CONFIG['vis_output_dir'], 'roc_curve_v5.png')
        plot_roc_curve(results, roc_path)

        # 阈值分析
        print("\n阈值优化分析...")
        thresh_path = os.path.join(CONFIG['vis_output_dir'], 'threshold_analysis_v5.png')
        best_thresh = plot_threshold_analysis(results, thresh_path)
        print(f"最佳阈值: {best_thresh:.2f}")

        report_parts.append(generate_comparison_table(metrics, CONFIG['paper_reference']))
    else:
        print(f"警告: 预测结果文件不存在: {CONFIG['predictions_path']}")
        print("跳过 automatic 评估：请先运行 02_inference_auto_v5.py")

    # ========== Click-guided (Table 1) ==========
    click_path = CONFIG.get('click_predictions_path', '')
    if click_path and os.path.exists(click_path):
        with open(click_path, 'r') as f:
            click_obj = json.load(f)

        # 兼容输出格式：旧版为 {click_key: {category: [records...]}}
        # 新版为 {"meta": {...}, "results": {click_key: {category: [records...]}}}
        click_results_all = click_obj.get('results', click_obj) if isinstance(click_obj, dict) else click_obj

        click_keys = sorted(list(click_results_all.keys()))
        print(f"\n加载 click-guided 结果: {len(click_keys)} 个 click 配置")

        click_metrics_map = {}
        for click_key in click_keys:
            click_results = click_results_all.get(click_key, {})
            click_metrics_map[click_key] = compute_metrics(click_results)

        report_parts.append(generate_click_summary(click_metrics_map, CONFIG['paper_reference']))
    else:
        print(f"\n警告: click-guided 结果文件不存在: {click_path}")
        print("跳过 click 评估：请先运行 03_inference_click_v5.py")

    # 写出分析报告
    report = "\n\n".join([p for p in report_parts if p])
    if report:
        print("\n生成分析报告...")
        os.makedirs(os.path.dirname(CONFIG['analysis_output']), exist_ok=True)
        with open(CONFIG['analysis_output'], 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"分析报告保存至: {CONFIG['analysis_output']}")
    
    print("\n" + "=" * 60)
    print("V3 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    CONFIG = dict(CONFIG)
    CONFIG['predictions_path'] = _to_abs(BASE_DIR, CONFIG.get('predictions_path', ''))
    CONFIG['click_predictions_path'] = _to_abs(BASE_DIR, CONFIG.get('click_predictions_path', ''))
    CONFIG['vis_output_dir'] = _to_abs(BASE_DIR, CONFIG.get('vis_output_dir', ''))
    CONFIG['analysis_output'] = _to_abs(BASE_DIR, CONFIG.get('analysis_output', ''))
    main()
