"""
Res-SAM 复现实验 - Step 1: Feature Bank 构建

功能：
- 从 normal 数据（intact/）随机抽取 20 张图片
- 使用 2D-ESN 拟合每个 patch，提取动态特征
- 构建特征银行并保存

论文参数：
- window_size = 50
- stride = 5
- hidden_size = 30
- init_normal_samples = 20
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PatchRes.PatchRes import PatchRes
from PatchRes.functions import random_select_images_in_one_folder

# ============ 配置 ============
CONFIG = {
    # 数据路径
    'normal_data_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'data', 'GPR_data', 'intact'),
    'output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'outputs', 'feature_banks'),
    'output_file': 'features.pth',
    
    # 论文参数（调整以适应内存限制）
    'window_size': 50,
    'stride': 10,  # 增大 stride 减少内存占用
    'hidden_size': 30,
    'num_normal_samples': 20,
    
    # 图像预处理 - 统一 resize 到 256x256
    'image_size': (256, 256),
    
    # 随机种子（可复现）
    'random_seed': 42,
}

def build_feature_bank(config):
    """构建特征银行"""
    
    # 设置随机种子
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    print("=" * 60)
    print("Res-SAM Feature Bank Construction")
    print("=" * 60)
    print(f"Normal data directory: {config['normal_data_dir']}")
    print(f"Output file: {os.path.join(config['output_dir'], config['output_file'])}")
    print(f"Parameters:")
    print(f"  - window_size: {config['window_size']}")
    print(f"  - stride: {config['stride']}")
    print(f"  - hidden_size: {config['hidden_size']}")
    print(f"  - num_normal_samples: {config['num_normal_samples']}")
    print("=" * 60)
    
    # 检查数据目录
    if not os.path.exists(config['normal_data_dir']):
        raise FileNotFoundError(f"Normal data directory not found: {config['normal_data_dir']}")
    
    # 统计图片数量（支持大小写后缀）
    image_files = [f for f in os.listdir(config['normal_data_dir']) 
                   if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    print(f"\nFound {len(image_files)} images in normal data directory.")
    
    if len(image_files) < config['num_normal_samples']:
        print(f"Warning: Only {len(image_files)} images available, using all.")
        config['num_normal_samples'] = len(image_files)
    
    # 初始化 PatchRes
    print("\nInitializing PatchRes...")
    patch_res = PatchRes(
        hidden_size=config['hidden_size'],
        stride=config['stride'],
        window_size=[config['window_size'], config['window_size']],
        anomaly_threshold=0.1,  # 构建特征银行时不需要此参数
    )
    
    # 加载 normal 数据
    print(f"\nLoading {config['num_normal_samples']} normal images...")
    normal_data = random_select_images_in_one_folder(
        data_folder=config['normal_data_dir'],
        num=config['num_normal_samples'],
        rand_select=True,
        size=config['image_size'],
        to_one_channel=True,
    )
    
    print(f"Loaded normal data shape: {normal_data.shape}")
    
    # 提取特征
    print("\nExtracting features (2D-ESN fitting)...")
    features_list = []
    
    for i in tqdm(range(normal_data.shape[0]), desc="Processing images"):
        # 每张图片提取特征
        features = patch_res.fit(normal_data[i].unsqueeze(0))
        features_list.append(features)
    
    # 合并所有特征
    print("\nConcatenating features...")
    all_features = torch.cat(features_list, dim=0)
    
    print(f"Total features shape: {all_features.shape}")
    print(f"Feature dimension: {all_features.shape[1]}")
    print(f"Number of feature vectors: {all_features.shape[0]}")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 保存特征银行
    output_path = os.path.join(config['output_dir'], config['output_file'])
    print(f"\nSaving feature bank to: {output_path}")
    torch.save(all_features, output_path)
    
    # 验证保存
    loaded_features = torch.load(output_path)
    print(f"Verification - Loaded features shape: {loaded_features.shape}")
    
    print("\n" + "=" * 60)
    print("Feature Bank Construction Complete!")
    print("=" * 60)
    
    return all_features


if __name__ == "__main__":
    with torch.no_grad():
        features = build_feature_bank(CONFIG)
