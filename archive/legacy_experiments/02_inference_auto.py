"""
Res-SAM 复现实验 - Step 2: Fully Automatic 模式批量推理

功能：
- 加载 Feature Bank
- 对测试集进行自动异常检测
- 输出每张图的预测 bbox 和异常分数
- 支持断点继续（自动保存进度）

论文对应：Table 2
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PatchRes.PatchRes import PatchRes
from PatchRes.functions import random_select_images_in_one_folder, jpg_to_tensor
import json
import hashlib


def get_file_hash(filepath):
    """获取文件 MD5 哈希，用于断点校验"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:8]

# ============ 配置 ============
CONFIG = {
    # 数据路径
    'feature_bank_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'outputs', 'feature_banks', 'features.pth'),
    'test_data_dirs': {
        'cavities': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'GPR_data', 'augmented_cavities'),
        'utilities': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'data', 'GPR_data', 'augmented_utilities'),
    },
    'annotation_dirs': {
        'cavities': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'GPR_data', 'augmented_cavities', 
                                 'annotations', 'VOC_XML_format'),
        'utilities': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'data', 'GPR_data', 'augmented_utilities', 
                                  'annotations', 'VOC_XML_format'),
    },
    'output_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               'outputs', 'predictions'),
    'checkpoint_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'outputs', 'checkpoints'),
    
    # 论文参数（调整以适应内存限制）
    'window_size': 50,
    'stride': 10,  # 增大 stride 减少内存占用
    'hidden_size': 30,
    'anomaly_threshold': 0.1,
    
    # 图像预处理 - 统一 resize 到 256x256
    'image_size': (256, 256),
}


def parse_voc_xml(xml_path):
    """解析 VOC XML 标注文件，返回 bbox"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像尺寸
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        # 获取所有目标
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
        
        return {
            'width': img_width,
            'height': img_height,
            'objects': objects
        }
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None


def load_test_images(data_dir, size=(256, 256)):
    """加载测试图片"""
    image_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    
    images = []
    paths = []
    for img_file in image_files:
        img_path = os.path.join(data_dir, img_file)
        try:
            img_tensor = jpg_to_tensor(img_path, to_one_channel=True, size=size)
            # jpg_to_tensor 返回 [C, H, W]，检查 H 和 W
            if img_tensor is not None and img_tensor.shape[1] >= 50 and img_tensor.shape[2] >= 50:
                # 标准化
                img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)
                images.append(img_tensor)
                paths.append(img_path)
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    
    return images, paths


def run_inference(patch_res, images, paths, category_name, annotation_dir, config, checkpoint_dir=None):
    """对一组图片运行推理，支持断点继续"""
    
    # 断点文件路径
    checkpoint_file = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_{category_name}.json')
    
    # 尝试加载断点
    start_idx = 0
    results = []
    
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"\n发现断点文件: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            start_idx = checkpoint.get('processed_count', 0)
            results = checkpoint.get('results', [])
            print(f"从第 {start_idx} 张图片继续...")
        except Exception as e:
            print(f"加载断点失败: {e}，从头开始")
            start_idx = 0
            results = []
    
    # 保存进度的函数
    def save_checkpoint(idx, results_list):
        if checkpoint_file:
            checkpoint = {
                'processed_count': idx,
                'results': results_list,
                'total_images': len(images),
                'timestamp': str(np.datetime64('now'))
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    # 从 start_idx 开始处理
    for i in range(start_idx, len(images)):
        img = images[i]
        img_path = paths[i]
        
        img_name = os.path.basename(img_path)
        xml_name = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(annotation_dir, xml_name)
        
        # 显示进度
        print(f"\r[{i+1}/{len(images)}] Processing: {img_name[:30]:<30}", end='', flush=True)
        
        try:
            # 获取 ground truth
            gt = parse_voc_xml(xml_path) if os.path.exists(xml_path) else None
            
            # img shape: [C, H, W] -> 需要转换为 [1, H, W] 用于 predict
            img_2d = img.squeeze(0)  # [H, W]
            
            # 推理 - pixel_seg 模式获取异常分数图
            ps_mask, tile_scores, _, _, _ = patch_res.predict(
                img_2d.unsqueeze(0), mode="pixel_seg")
            ps_mask = ps_mask.squeeze(0)
            
            # 推理 - OD 模式获取 bbox
            od_mask, _, _, _, _ = patch_res.predict(
                img_2d.unsqueeze(0), mode="OD")
            
            # 计算最大异常分数
            anomaly_score = float(np.max(ps_mask))
            
            # 从 OD 结果提取 bbox
            from PatchRes.functions import generate_masks
            tile_scores_np = tile_scores if isinstance(tile_scores, np.ndarray) else tile_scores.detach().cpu().numpy()
            bbox = generate_masks(
                img_2d.unsqueeze(0).numpy(),
                tile_scores_np, 
                mode="OD", 
                window_size=[config['window_size'], config['window_size']],
                anomaly_threshold=config['anomaly_threshold'],
                stride=config['stride'],
                return_box=True
            )
            
            result = {
                'image_path': img_path,
                'image_name': img_name,
                'category': category_name,
                'pred_bbox': bbox if bbox != (0, 0, 0, 0) else None,
                'anomaly_score': anomaly_score,
                'ground_truth': gt,
            }
            results.append(result)
            
            # 每处理 10 张保存一次断点
            if (i + 1) % 10 == 0:
                save_checkpoint(i + 1, results)
                
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            # 保存当前进度
            save_checkpoint(i, results)
            continue
    
    # 处理完成，删除断点文件
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n断点文件已清理")
    
    print()  # 换行
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("Res-SAM Fully Automatic Inference")
    print("=" * 60)
    
    # 检查 Feature Bank
    if not os.path.exists(CONFIG['feature_bank_path']):
        raise FileNotFoundError(
            f"Feature bank not found: {CONFIG['feature_bank_path']}\n"
            "Please run 01_build_feature_bank.py first.")
    
    print(f"Feature bank: {CONFIG['feature_bank_path']}")
    
    # 初始化 PatchRes 并加载 Feature Bank
    print("\nInitializing PatchRes...")
    patch_res = PatchRes(
        hidden_size=CONFIG['hidden_size'],
        stride=CONFIG['stride'],
        window_size=[CONFIG['window_size'], CONFIG['window_size']],
        anomaly_threshold=CONFIG['anomaly_threshold'],
        features=CONFIG['feature_bank_path']
    )
    patch_res.fit(0)  # 加载预置特征
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    all_results = {}
    
    # 遍历每个测试集
    for category, data_dir in CONFIG['test_data_dirs'].items():
        print(f"\n{'='*60}")
        print(f"Processing category: {category}")
        print(f"Data directory: {data_dir}")
        print("=" * 60)
        
        if not os.path.exists(data_dir):
            print(f"Warning: Directory not found: {data_dir}")
            continue
        
        # 加载图片
        images, paths = load_test_images(data_dir, CONFIG['image_size'])
        print(f"Loaded {len(images)} images")
        
        # 推理
        annotation_dir = CONFIG['annotation_dirs'].get(category, '')
        results = run_inference(patch_res, images, paths, category, 
                               annotation_dir, CONFIG, CONFIG['checkpoint_dir'])
        
        all_results[category] = results
        
        # 统计
        detected = sum(1 for r in results if r['pred_bbox'] is not None)
        print(f"\nDetected anomalies: {detected}/{len(results)}")
    
    # 保存结果
    output_path = os.path.join(CONFIG['output_dir'], 'auto_predictions.json')
    print(f"\nSaving results to: {output_path}")
    
    # 转换为可序列化格式
    serializable_results = {}
    for category, results in all_results.items():
        serializable_results[category] = []
        for r in results:
            serializable_results[category].append({
                'image_name': r['image_name'],
                'pred_bbox': list(r['pred_bbox']) if r['pred_bbox'] else None,
                'anomaly_score': r['anomaly_score'],
                'ground_truth': r['ground_truth'],
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Fully Automatic Inference Complete!")
    print(f"Total images processed: {sum(len(r) for r in all_results.values())}")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    with torch.no_grad():
        results = main()
