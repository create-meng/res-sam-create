"""
Res-SAM 复现实验 - Step 5: 异常聚类

功能：
- 对检测到的异常区域提取特征
- 使用 K-Means / Agglomerative / FCM 聚类
- 计算 Accuracy / ARI / NMI 指标

论文对应：Table 3
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import json
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from PatchRes.PatchRes import PatchRes
from PatchRes.functions import jpg_to_tensor
import xml.etree.ElementTree as ET

# ============ 配置 ============
CONFIG = {
    # 数据路径
    'feature_bank_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'outputs', 'feature_banks', 'features.pth'),
    'predictions_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'outputs', 'predictions', 'auto_predictions.json'),
    'test_data_dirs': {
        'cavities': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'GPR_data', 'augmented_cavities'),
        'utilities': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'data', 'GPR_data', 'augmented_utilities'),
    },
    'output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'outputs', 'metrics'),
    
    # 论文参数（必须与 01_build_feature_bank.py 和 02_inference_auto.py 一致）
    'window_size': 50,
    'stride': 10,  # 增大 stride 减少内存占用
    'hidden_size': 30,
    'anomaly_threshold': 0.1,
    
    # 聚类参数
    'n_clusters': 2,  # cavities vs utilities
    
    # 图像预处理
    'image_size': (256, 256),
}


def parse_voc_xml(xml_path):
    """解析 VOC XML 标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            objects.append({
                'name': name,
                'xmin': int(bbox.find('xmin').text),
                'ymin': int(bbox.find('ymin').text),
                'xmax': int(bbox.find('xmax').text),
                'ymax': int(bbox.find('ymax').text),
            })
        return {'objects': objects}
    except:
        return None


def extract_anomaly_features(patch_res, img_tensor, bbox):
    """
    从异常区域提取特征
    """
    if bbox is None or bbox == (0, 0, 0, 0):
        return None
    
    # img_tensor 从 main 传进来时是 [1, 1, 256, 256]
    # PatchRes.extractor 期望 [Batch, H, W]
    try:
        # 降维到 [1, 256, 256]
        if img_tensor.dim() == 4:
            input_tensor = img_tensor.squeeze(1) # [1, 256, 256]
        else:
            input_tensor = img_tensor
            
        # 提取特征
        # 对于 256x256, window=50, stride=10, 会得到 441 个 patch
        # 每个 patch 特征维度是 hidden_size * 2 = 60
        features = patch_res.extractor(input_tensor) # [441, 60]
        
        if features is not None and features.numel() > 0:
            # 取该图像所有 patch 的平均特征作为代表
            return features.mean(dim=0).cpu().numpy()
    except Exception as e:
        if not hasattr(extract_anomaly_features, "error_shown"):
            print(f"\nExtraction error example: {e}")
            extract_anomaly_features.error_shown = True
        return None
    
    return None


def fcm_clustering(X, n_clusters, max_iter=100, m=2.0):
    """
    Fuzzy C-Means 聚类实现
    
    Args:
        X: 数据矩阵 [n_samples, n_features]
        n_clusters: 聚类数
        max_iter: 最大迭代次数
        m: 模糊指数
    
    Returns:
        labels: 硬聚类标签
    """
    n_samples = X.shape[0]
    
    if n_samples < n_clusters:
        return np.zeros(n_samples, dtype=int)
    
    # 初始化隶属度矩阵
    U = np.random.rand(n_samples, n_clusters)
    U = U / U.sum(axis=1, keepdims=True)
    
    for _ in range(max_iter):
        # 计算聚类中心
        U_m = U ** m
        centers = (U_m.T @ X) / U_m.sum(axis=0, keepdims=True).T
        
        # 更新隶属度
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - centers[k], axis=1)
        
        # 避免除零
        distances = np.maximum(distances, 1e-10)
        
        for k in range(n_clusters):
            U[:, k] = 1.0 / np.sum((distances[:, k:k+1] / distances) ** (2/(m-1)), axis=1)
    
    # 转换为硬聚类
    labels = np.argmax(U, axis=1)
    return labels


def compute_clustering_metrics(true_labels, pred_labels):
    """
    计算聚类评估指标
    
    Args:
        true_labels: 真实标签
        pred_labels: 预测标签
    
    Returns:
        dict: 包含 Acc, ARI, NMI
    """
    # Accuracy（需要匹配标签）
    # 由于聚类标签可能与真实标签不一致，需要找到最佳匹配
    from scipy.optimize import linear_sum_assignment
    
    n_clusters = len(np.unique(true_labels))
    n_pred_clusters = len(np.unique(pred_labels))
    
    # 构建混淆矩阵
    contingency = np.zeros((n_clusters, n_pred_clusters))
    for i, t in enumerate(true_labels):
        contingency[t, pred_labels[i]] += 1
    
    # 匈牙利算法找最佳匹配
    row_ind, col_ind = linear_sum_assignment(-contingency)
    accuracy = contingency[row_ind, col_ind].sum() / len(true_labels)
    
    # ARI
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # NMI
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    return {
        'Accuracy': accuracy,
        'ARI': ari,
        'NMI': nmi,
    }


def main():
    """主函数"""
    print("=" * 60)
    print("Res-SAM Anomaly Clustering")
    print("=" * 60)
    
    # 检查依赖
    if not os.path.exists(CONFIG['feature_bank_path']):
        raise FileNotFoundError(
            f"Feature bank not found: {CONFIG['feature_bank_path']}\n"
            "Please run 01_build_feature_bank.py first.")
    
    if not os.path.exists(CONFIG['predictions_path']):
        raise FileNotFoundError(
            f"Predictions not found: {CONFIG['predictions_path']}\n"
            "Please run 02_inference_auto.py first.")
    
    # 加载预测结果
    print(f"\nLoading predictions from: {CONFIG['predictions_path']}")
    try:
        with open(CONFIG['predictions_path'], 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return
    
    # 统计总预测数
    total_preds = sum(len(p) for p in predictions.values())
    print(f"Total images with predictions: {total_preds}")
    
    # 初始化 PatchRes
    print("\nInitializing PatchRes...")
    patch_res = PatchRes(
        hidden_size=CONFIG['hidden_size'],
        stride=CONFIG['stride'],
        window_size=[CONFIG['window_size'], CONFIG['window_size']],
        anomaly_threshold=CONFIG['anomaly_threshold'],
        features=CONFIG['feature_bank_path']
    )
    patch_res.fit(0)
    
    # 收集异常区域的特征和标签
    features_list = []
    labels_list = []  # 0: cavities, 1: utilities
    
    print("\nExtracting anomaly features...")
    
    for category_idx, (category, preds) in enumerate(predictions.items()):
        print(f"\nCategory: {category}")
        data_dir = CONFIG['test_data_dirs'].get(category)
        
        if data_dir is None or not os.path.exists(data_dir):
            print(f"Data directory not found for {category}: {data_dir}")
            continue
        
        count = 0
        skipped_no_bbox = 0
        for pred in tqdm(preds, desc=f"Extracting {category}"):
            # 只有检测到异常的才参与聚类
            if pred.get('pred_bbox') is None:
                skipped_no_bbox += 1
                continue
            
            # 加载图像
            img_path = None
            img_name_base = pred['image_name'].rsplit('.', 1)[0]
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                candidate = os.path.join(data_dir, img_name_base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            try:
                img_tensor = jpg_to_tensor(img_path, to_one_channel=True, 
                                           size=CONFIG['image_size'])
                if img_tensor is None:
                    continue
                
                # 标准化
                img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)
                img_tensor = img_tensor.unsqueeze(0) # [1, 1, 256, 256]
                
                # 提取特征
                bbox = tuple(pred['pred_bbox'])
                features = extract_anomaly_features(patch_res, img_tensor, bbox)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(category_idx)
                    count += 1
            except Exception:
                continue
        
        print(f"Summary for {category}:")
        print(f"  - Total predictions: {len(preds)}")
        print(f"  - Skipped (no anomaly detected): {skipped_no_bbox}")
        print(f"  - Successfully extracted features: {count}")
    
    if len(features_list) == 0:
        print("No anomaly features extracted!")
        return
    
    # 转换为数组
    X = np.array(features_list)
    true_labels = np.array(labels_list)
    
    print(f"\nExtracted {len(X)} anomaly features")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Class distribution: cavities={sum(true_labels==0)}, utilities={sum(true_labels==1)}")
    
    # 聚类
    n_clusters = CONFIG['n_clusters']
    results = {}
    
    # 1. K-Means
    print(f"\n{'='*60}")
    print("K-Means Clustering")
    print("=" * 60)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    kmeans_metrics = compute_clustering_metrics(true_labels, kmeans_labels)
    results['K-Means'] = kmeans_metrics
    print(f"Accuracy: {kmeans_metrics['Accuracy']:.4f}")
    print(f"ARI: {kmeans_metrics['ARI']:.4f}")
    print(f"NMI: {kmeans_metrics['NMI']:.4f}")
    
    # 2. Agglomerative Clustering (AC)
    print(f"\n{'='*60}")
    print("Agglomerative Clustering")
    print("=" * 60)
    ac = AgglomerativeClustering(n_clusters=n_clusters)
    ac_labels = ac.fit_predict(X)
    ac_metrics = compute_clustering_metrics(true_labels, ac_labels)
    results['AC'] = ac_metrics
    print(f"Accuracy: {ac_metrics['Accuracy']:.4f}")
    print(f"ARI: {ac_metrics['ARI']:.4f}")
    print(f"NMI: {ac_metrics['NMI']:.4f}")
    
    # 3. FCM
    print(f"\n{'='*60}")
    print("Fuzzy C-Means Clustering")
    print("=" * 60)
    fcm_labels = fcm_clustering(X, n_clusters)
    fcm_metrics = compute_clustering_metrics(true_labels, fcm_labels)
    results['FCM'] = fcm_metrics
    print(f"Accuracy: {fcm_metrics['Accuracy']:.4f}")
    print(f"ARI: {fcm_metrics['ARI']:.4f}")
    print(f"NMI: {fcm_metrics['NMI']:.4f}")
    
    # 保存结果
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    output_path = os.path.join(CONFIG['output_dir'], 'clustering_results.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成表格格式（Table 3）
    print(f"\n{'='*60}")
    print("Table 3 Reproduction: Anomaly Clustering")
    print("=" * 60)
    print(f"{'Method':<20} {'Accuracy':<12} {'ARI':<12} {'NMI':<12}")
    print("-" * 56)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['Accuracy']:<12.4f} {metrics['ARI']:<12.4f} {metrics['NMI']:<12.4f}")
    
    print("\n" + "=" * 60)
    print("Clustering Complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    with torch.no_grad():
        results = main()
