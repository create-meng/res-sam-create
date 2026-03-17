"""
Res-SAM 复现实验 V2 - Step 1: Feature Bank 构建

改进：
- 使用 ResSAM 模块
- 支持多种数据来源
- 记录环境信息用于一致性检查
- 支持断点继续

论文参数：
- window_size = 50
- stride = 5
- hidden_size = 30
- init_normal_samples = 20
"""

import sys
import os
import json
import hashlib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# ============ 配置 ============
CONFIG = {
    # 数据路径 - 支持多个来源
    'normal_data_sources': {
        'intact': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'data', 'GPR_data', 'intact'),
        # 可以添加更多来源
        # 'intact_aug': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
        #                            'data', 'GPR_data', 'augmented_intact'),
    },
    
    'output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'outputs', 'feature_banks_v2'),
    'output_file': 'feature_bank.pth',
    'metadata_file': 'metadata.json',
    
    # 论文参数
    'window_size': 30,  # 降低以匹配 SAM 候选区域
    'stride': 5,  # 论文原始值
    'hidden_size': 30,
    'num_normal_samples': 20,  # 每个来源的样本数
    
    # 图像预处理 - 使用论文原始尺寸
    'image_size': (369, 369),
    
    # 随机种子
    'random_seed': 42,
    
    # 断点文件
    'checkpoint_file': 'checkpoint.json',
}


def get_image_hash(path: str) -> str:
    """获取图像文件哈希"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def load_normal_images(
    data_dir: str, 
    num_samples: int, 
    size: tuple = (256, 256),
    random_select: bool = True,
) -> tuple:
    """
    加载正常样本图像
    
    Returns:
    --------
    tuple: (images, paths, hashes)
    """
    image_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    
    if random_select:
        np.random.shuffle(image_files)
    
    selected_files = image_files[:min(num_samples, len(image_files))]
    
    images = []
    paths = []
    hashes = []
    
    for img_file in selected_files:
        img_path = os.path.join(data_dir, img_file)
        
        try:
            # 加载并预处理
            img = Image.open(img_path).convert('L')  # 灰度
            img = img.resize((size[1], size[0]), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32)
            
            # 标准化
            img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
            
            images.append(img_array)
            paths.append(img_path)
            hashes.append(get_image_hash(img_path))
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    return np.array(images), paths, hashes


def build_feature_bank(config: dict, resume: bool = True):
    """
    构建 Feature Bank
    
    Parameters:
    -----------
    config : dict
        配置字典
    resume : bool
        是否支持断点继续
    """
    # 设置随机种子
    np.random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    print("=" * 60)
    print("Res-SAM V2: Feature Bank Construction")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 检查断点
    checkpoint_path = os.path.join(config['output_dir'], config['checkpoint_file'])
    processed_sources = {}
    
    if resume and os.path.exists(checkpoint_path):
        print(f"\n发现断点文件: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        processed_sources = checkpoint.get('processed_sources', {})
        print(f"已处理来源: {list(processed_sources.keys())}")
    
    # 导入 ResSAM
    from PatchRes.ResSAM import ResSAM
    
    # 初始化模型
    print("\n初始化 ResSAM...")
    model = ResSAM(
        hidden_size=config['hidden_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        anomaly_threshold=0.5,  # Feature Bank 构建时不需要
    )
    
    # 处理每个数据来源
    all_images = []
    all_metadata = {
        'config': config,
        'sources': {},
        'creation_time': datetime.now().isoformat(),
    }
    
    for source_name, source_dir in config['normal_data_sources'].items():
        if source_name in processed_sources:
            print(f"\n跳过已处理来源: {source_name}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理来源: {source_name}")
        print(f"目录: {source_dir}")
        print("=" * 60)
        
        if not os.path.exists(source_dir):
            print(f"警告: 目录不存在: {source_dir}")
            continue
        
        # 加载图像
        images, paths, hashes = load_normal_images(
            source_dir,
            config['num_normal_samples'],
            config['image_size'],
        )
        
        print(f"加载了 {len(images)} 张图像")
        
        # 记录元数据
        all_metadata['sources'][source_name] = {
            'directory': source_dir,
            'num_images': len(images),
            'image_paths': paths,
            'image_hashes': hashes,
        }
        
        all_images.append(images)
        
        # 更新断点
        processed_sources[source_name] = {
            'num_images': len(images),
            'completed': True,
        }
        
        # 保存断点
        with open(checkpoint_path, 'w') as f:
            json.dump({
                'processed_sources': processed_sources,
                'timestamp': datetime.now().isoformat(),
            }, f, indent=2)
    
    # 合并所有图像
    if len(all_images) == 0:
        print("没有新数据需要处理")
        return None
    
    all_images = np.concatenate(all_images, axis=0)
    print(f"\n总图像数: {len(all_images)}")
    
    # 构建 Feature Bank
    print("\n构建 Feature Bank...")
    source_info = ", ".join(config['normal_data_sources'].keys())
    feature_bank = model.build_feature_bank(all_images, source_info=source_info)
    
    # 保存 Feature Bank
    output_path = os.path.join(config['output_dir'], config['output_file'])
    torch.save(feature_bank, output_path)
    print(f"\nFeature Bank 保存至: {output_path}")
    print(f"形状: {feature_bank.shape}")
    
    # 保存元数据
    all_metadata['feature_bank_shape'] = list(feature_bank.shape)
    all_metadata['feature_bank_path'] = output_path
    
    metadata_path = os.path.join(config['output_dir'], config['metadata_file'])
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据保存至: {metadata_path}")
    
    # 清除断点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("断点文件已清理")
    
    print("\n" + "=" * 60)
    print("Feature Bank 构建完成!")
    print("=" * 60)
    
    return feature_bank


if __name__ == "__main__":
    with torch.no_grad():
        feature_bank = build_feature_bank(CONFIG)
