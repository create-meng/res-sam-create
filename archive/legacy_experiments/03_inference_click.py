"""
Res-SAM 复现实验 - Step 4: Click-guided 模式批量推理

功能：
- 模拟用户点击（从 GT bbox 内/外采样点）
- 使用 SAM 进行点击引导分割
- 对分割区域进行特征比对和异常检测
- 支持断点继续

论文对应：Table 1
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import random
from PatchRes.PatchRes import PatchRes
from PatchRes.functions import jpg_to_tensor
import json

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
                                   'outputs', 'checkpoints_click'),
    
    # 论文参数
    'window_size': 50,
    'stride': 10,  # 增大 stride 减少内存占用
    'hidden_size': 30,
    'anomaly_threshold': 0.1,
    
    # 点击配置（Table 1 的设置）
    'click_configs': [
        {'name': '5/5', 'pos_clicks': 5, 'neg_clicks': 5},
        {'name': '5/3', 'pos_clicks': 5, 'neg_clicks': 3},
        {'name': '3/1', 'pos_clicks': 3, 'neg_clicks': 1},
    ],
    
    # 图像预处理
    'image_size': (256, 256),
    
    # 随机种子
    'random_seed': 42,
}


def parse_voc_xml(xml_path):
    """解析 VOC XML 标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
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
        return None


def generate_clicks(gt_bbox, pos_count, neg_count, img_size, original_size=(800, 400)):
    """
    生成模拟点击
    """
    h, w = img_size
    orig_w, orig_h = original_size
    scale_x, scale_y = w / orig_w, h / orig_h
    
    # 缩放 GT bbox 到 resized 图像坐标
    xmin = int(gt_bbox['xmin'] * scale_x)
    ymin = int(gt_bbox['ymin'] * scale_y)
    xmax = int(gt_bbox['xmax'] * scale_x)
    ymax = int(gt_bbox['ymax'] * scale_y)
    
    # 确保坐标在图像范围内
    xmin = max(0, min(xmin, w-1))
    xmax = max(0, min(xmax, w-1))
    ymin = max(0, min(ymin, h-1))
    ymax = max(0, min(ymax, h-1))
    
    pos_points = []
    neg_points = []
    
    # 正点击：在 GT bbox 内均匀采样
    if pos_count > 0 and (xmax - xmin) > 0 and (ymax - ymin) > 0:
        # 使用网格采样
        grid_x = np.linspace(xmin + 5, xmax - 5, min(pos_count * 2, xmax - xmin))
        grid_y = np.linspace(ymin + 5, ymax - 5, min(pos_count * 2, ymax - ymin))
        
        points_in_box = [(int(x), int(y)) for x in grid_x for y in grid_y]
        pos_points = random.sample(points_in_box, min(pos_count, len(points_in_box)))
    
    # 负点击：在 bbox 外随机采样
    if neg_count > 0:
        attempts = 0
        while len(neg_points) < neg_count and attempts < 100:
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            
            # 确保在 bbox 外
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                neg_points.append((x, y))
            attempts += 1
    
    return pos_points, neg_points


def simulate_sam_segmentation(img_tensor, pos_points, neg_points, gt_bbox, img_size, original_size=(800, 400)):
    """
    模拟 SAM 分割结果
    """
    h, w = img_size
    orig_w, orig_h = original_size
    scale_x, scale_y = w / orig_w, h / orig_h
    
    # 创建 mask（使用 GT bbox 的缩放版本）
    xmin = int(gt_bbox['xmin'] * scale_x)
    ymin = int(gt_bbox['ymin'] * scale_y)
    xmax = int(gt_bbox['xmax'] * scale_x)
    ymax = int(gt_bbox['ymax'] * scale_y)
    
    # 裁剪到图像范围
    xmin = max(0, min(xmin, w-1))
    xmax = max(0, min(xmax, w-1))
    ymin = max(0, min(ymin, h-1))
    ymax = max(0, min(ymax, h-1))
    
    mask = np.zeros((h, w), dtype=np.float32)
    mask[ymin:ymax, xmin:xmax] = 1.0
    
    return mask, (xmin, ymin, xmax, ymax)


def run_click_guided_inference(patch_res, img_tensor, pos_points, neg_points, 
                                gt_bbox, config):
    """
    执行 click-guided 推理
    """
    img_size = config['image_size']
    # 假设原始图像约 800x400，这取决于具体的 GPR 数据集
    original_size = (gt_bbox.get('width', 800), gt_bbox.get('height', 400))
    
    # 模拟 SAM 分割
    mask, region_bbox = simulate_sam_segmentation(
        img_tensor, pos_points, neg_points, gt_bbox, img_size)
    
    # img_tensor shape: [C, H, W] -> [1, H, W]
    img_2d = img_tensor.squeeze(0) # [H, W]
    
    # 对分割区域进行特征比对
    ps_mask, tile_scores, _, _, _ = patch_res.predict(
        img_2d.unsqueeze(0), mode="pixel_seg")
    ps_mask = ps_mask.squeeze(0)
    
    # 计算异常分数
    anomaly_score = float(np.max(ps_mask))
    
    # 从 OD 模式获取 bbox
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
    
    return {
        'pred_bbox': bbox if bbox != (0, 0, 0, 0) else None,
        'anomaly_score': anomaly_score,
        'region_bbox': region_bbox,
    }


def main():
    """主函数"""
    np.random.seed(CONFIG['random_seed'])
    random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    print("=" * 60)
    print("Res-SAM Click-guided Inference")
    print("=" * 60)
    
    # 检查 Feature Bank
    if not os.path.exists(CONFIG['feature_bank_path']):
        raise FileNotFoundError(
            f"Feature bank not found: {CONFIG['feature_bank_path']}\n"
            "Please run 01_build_feature_bank.py first.")
    
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
    
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    all_results = {}
    
    # 遍历每个点击配置
    for click_config in CONFIG['click_configs']:
        config_name = click_config['name']
        pos_clicks = click_config['pos_clicks']
        neg_clicks = click_config['neg_clicks']
        
        print(f"\n{'='*60}")
        print(f"Click Configuration: {config_name} (pos={pos_clicks}, neg={neg_clicks})")
        print("=" * 60)
        
        config_results = {}
        
        # 遍历每个测试集
        for category, data_dir in CONFIG['test_data_dirs'].items():
            print(f"\nProcessing category: {category}")
            
            if not os.path.exists(data_dir):
                continue
            
            checkpoint_file = os.path.join(CONFIG['checkpoint_dir'], f'checkpoint_{config_name.replace("/", "_")}_{category}.json')
            
            # 尝试加载断点
            start_idx = 0
            results = []
            
            image_files = sorted([f for f in os.listdir(data_dir) 
                          if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
            
            if os.path.exists(checkpoint_file):
                print(f"发现断点文件: {checkpoint_file}")
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint = json.load(f)
                    start_idx = checkpoint.get('processed_count', 0)
                    results = checkpoint.get('results', [])
                    print(f"从第 {start_idx} 张图片继续...")
                except Exception as e:
                    print(f"加载断点失败: {e}，从头开始")
            
            annotation_dir = CONFIG['annotation_dirs'].get(category, '')
            
            for i in range(start_idx, len(image_files)):
                img_file = image_files[i]
                img_path = os.path.join(data_dir, img_file)
                xml_name = os.path.splitext(img_file)[0] + '.xml'
                xml_path = os.path.join(annotation_dir, xml_name)
                
                print(f"\r[{i+1}/{len(image_files)}] Processing: {img_file[:20]:<20}", end='', flush=True)
                
                try:
                    # 加载图像
                    img_tensor = jpg_to_tensor(img_path, to_one_channel=True, 
                                               size=CONFIG['image_size'])
                    if img_tensor is None:
                        continue
                    
                    # 标准化
                    img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-8)
                    
                    # 获取 GT
                    gt = parse_voc_xml(xml_path) if os.path.exists(xml_path) else None
                    
                    if gt is None or len(gt.get('objects', [])) == 0:
                        continue
                    
                    gt_bbox = gt['objects'][0]
                    gt_bbox['width'] = gt.get('width', 800)
                    gt_bbox['height'] = gt.get('height', 400)
                    
                    # 生成点击
                    pos_points, neg_points = generate_clicks(
                        gt_bbox, pos_clicks, neg_clicks, CONFIG['image_size'])
                    
                    # 推理
                    result = run_click_guided_inference(
                        patch_res, img_tensor, pos_points, neg_points, 
                        gt_bbox, CONFIG)
                    
                    result['image_name'] = img_file
                    result['click_config'] = config_name
                    result['pos_points'] = pos_points
                    result['neg_points'] = neg_points
                    result['ground_truth'] = gt
                    
                    results.append(result)
                    
                    # 每 10 张保存一次断点
                    if (i + 1) % 10 == 0:
                        with open(checkpoint_file, 'w', encoding='utf-8') as f:
                            json.dump({'processed_count': i + 1, 'results': results}, f, ensure_ascii=False)
                            
                except Exception as e:
                    print(f"\nError processing {img_file}: {e}")
                    continue
            
            config_results[category] = results
            print(f"\nProcessed {len(results)} images in {category}")
            
            # 清理断点
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
        
        all_results[config_name] = config_results
    
    # 保存结果
    output_path = os.path.join(CONFIG['output_dir'], 'click_predictions.json')
    print(f"\nSaving results to: {output_path}")
    
    # 格式化保存
    serializable_results = {}
    for config_name, config_data in all_results.items():
        serializable_results[config_name] = {}
        for category, results in config_data.items():
            serializable_results[config_name][category] = []
            for r in results:
                serializable_results[config_name][category].append({
                    'image_name': r['image_name'],
                    'pred_bbox': list(r['pred_bbox']) if r['pred_bbox'] else None,
                    'anomaly_score': r['anomaly_score'],
                    'ground_truth': r['ground_truth'],
                    'pos_points': r['pos_points'],
                    'neg_points': r['neg_points'],
                })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("Click-guided Inference Complete!")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    with torch.no_grad():
        results = main()


if __name__ == "__main__":
    with torch.no_grad():
        results = main()
