"""
Res-SAM 复现实验 V4 - Step 1: Feature Bank 构建

V3 改进（严格按论文对齐）：
- window_size = 50（论文默认值）
- 特征口径 f = [W_out, b]（论文 Eq.(2)-(3)）
- 特征维度 = 2*hidden_size + 1 = 61

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

# ============ 配置 ============
CONFIG = {
    # 数据路径 - 支持多个来源
    'normal_data_sources': {
        'intact': os.path.join(BASE_DIR, 'data', 'GPR_data', 'intact'),
    },
    
    'output_dir': os.path.join(BASE_DIR, 'outputs', 'feature_banks_v4'),
    'output_file': 'feature_bank_v4.pth',
    'metadata_file': 'metadata.json',
    
    # 论文参数（V3 严格对齐）
    'window_size': 50,  # 论文默认值
    'stride': 5,
    'hidden_size': 30,
    'num_normal_samples': 20,  # 每个来源的样本数
    
    # 图像预处理 - 使用论文原始尺寸
    'image_size': (369, 369),

    'device': 'auto',
    
    # 随机种子
    'random_seed': 42,
    
    # 断点文件
    'checkpoint_file': 'checkpoint.json',
    
    # 版本标识
    'version': 'V4',
    'alignment_notes': 'Enhanced-eval setup with paper-style clean feature bank: source=intact, eval=augmented_*, feature f=[W_out,b]',
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
    构建 Feature Bank（V3 严格对齐论文）
    
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
    print("Res-SAM V4: Feature Bank Construction (Strict Paper Alignment)")
    print("=" * 60)
    print(f"  window_size = {config['window_size']}")
    print(f"  stride = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    print(f"  Expected feature dim = {2*config['hidden_size'] + 1}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 目标输出文件
    output_path = os.path.join(config['output_dir'], config['output_file'])
    
    # 检查断点
    checkpoint_path = os.path.join(config['output_dir'], config['checkpoint_file'])
    processed_sources = {}

    # 若 feature bank 目标文件缺失但 checkpoint 存在，说明之前运行中断在“来源完成”阶段，
    # 可能导致后续 all_images 为空直接退出，从而不生成 feature bank 文件。
    # 这种情况下应清理 checkpoint 并强制重建。
    if (not os.path.exists(output_path)) and os.path.exists(checkpoint_path):
        print(
            f"\n发现断点文件但 feature bank 目标文件缺失，将清理断点并强制重建：\n"
            f"  checkpoint: {checkpoint_path}\n"
            f"  missing: {output_path}"
        )
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass
        processed_sources = {}
    
    if resume and os.path.exists(checkpoint_path):
        print(f"\n发现断点文件: {checkpoint_path}")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        processed_sources = checkpoint.get('processed_sources', {})
        print(f"已处理来源: {list(processed_sources.keys())}")
    
    # 导入 ResSAM
    from PatchRes.ResSAM import ResSAM
    
    # 初始化模型（V3 参数）
    print("\n初始化 ResSAM (V4)...")
    model = ResSAM(
        hidden_size=config['hidden_size'],
        window_size=config['window_size'],  # 50
        stride=config['stride'],
        anomaly_threshold=config.get('beta_threshold', config.get('anomaly_threshold', 0.5)),  # Feature Bank 构建阶段不使用，仅保持口径一致
        region_coarse_threshold=config.get('beta_threshold', config.get('region_coarse_threshold', 0.5)),  # Feature Bank 构建阶段不使用，仅保持口径一致
        device=config.get('device', 'cuda'),
    )
    
    # 处理每个数据来源
    all_images = []
    all_metadata = {
        'config': config,
        'preprocess_signature': {
            'color_mode': 'L',
            'resize': {'enabled': True, 'size_hw': list(config.get('image_size', (369, 369)))},
            'normalize': {'enabled': True, 'method': '(x-mean)/(std+1e-8)', 'per_image': True},
        },
        'sources': {},
        'creation_time': datetime.now().isoformat(),
        'version': 'V4',
        'alignment_notes': config['alignment_notes'],
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

    # 为保证跨设备复现稳定性，保存前强制转为 CPU tensor（避免 .pth 绑定 CUDA 设备）
    feature_bank_to_save = feature_bank.detach().cpu()
    
    # 验证特征维度
    expected_dim = 2 * config['hidden_size'] + 1
    actual_dim = feature_bank.shape[1]
    print(f"\n特征维度验证:")
    print(f"  期望维度 (2*hidden_size + 1): {expected_dim}")
    print(f"  实际维度: {actual_dim}")
    if actual_dim == expected_dim:
        print("  ✓ 维度匹配，符合论文 Eq.(2)-(3)")
    else:
        print(f"  ✗ 维度不匹配！期望 {expected_dim}，实际 {actual_dim}")
    
    # 保存 Feature Bank
    torch.save(feature_bank_to_save, output_path)
    print(f"\nFeature Bank V4 保存至: {output_path}")
    print(f"形状: {feature_bank_to_save.shape}")
    
    # 保存元数据
    all_metadata['feature_bank_shape'] = list(feature_bank_to_save.shape)
    all_metadata['feature_bank_path'] = output_path
    all_metadata['feature_dim_expected'] = expected_dim
    all_metadata['feature_dim_actual'] = actual_dim
    all_metadata['dimension_match'] = (actual_dim == expected_dim)
    
    metadata_path = os.path.join(config['output_dir'], config['metadata_file'])
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据保存至: {metadata_path}")
    
    # 清除断点文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("断点文件已清理")
    
    print("\n" + "=" * 60)
    print("Feature Bank V4 构建完成!")
    print("=" * 60)
    
    return feature_bank_to_save


if __name__ == "__main__":
    CONFIG = dict(CONFIG)
    CONFIG['output_dir'] = _to_abs(BASE_DIR, CONFIG.get('output_dir', ''))
    CONFIG['normal_data_sources'] = {k: _to_abs(BASE_DIR, v) for k, v in CONFIG.get('normal_data_sources', {}).items()}
    with torch.no_grad():
        feature_bank = build_feature_bank(CONFIG)
