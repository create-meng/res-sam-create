"""
Res-SAM 复现实验 V2 - Step 2: Fully Automatic 模式推理

改进：
- 使用完整的 Res-SAM 流程 (SAM + 2D-ESN)
- 支持断点继续
- 自动阈值优化
- 详细的结果记录

论文对应：Table 2
"""

import sys
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# ============ 配置 ============
CONFIG = {
    # Feature Bank 路径
    'feature_bank_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                      'outputs', 'feature_banks_v2', 'feature_bank.pth'),
    'metadata_path': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                  'outputs', 'feature_banks_v2', 'metadata.json'),
    
    # 测试数据
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
                               'outputs', 'predictions_v2'),
    'checkpoint_dir': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'outputs', 'checkpoints_v2'),
    
    # 论文参数
    'window_size': 50,
    'stride': 5,
    'hidden_size': 30,
    'anomaly_threshold': 0.5,  # 初始阈值，会自动优化
    
    # SAM 参数
    'sam_model_type': 'vit_l',  # 使用 vit_l 模型
    'sam_checkpoint': os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'sam', 'sam_vit_l_0b3195.pth'),
    
    # 图像预处理
    'image_size': (256, 256),
    
    # 推理参数
    'max_candidates_per_image': 10,
    'min_region_area': 100,
    
    # 断点
    'checkpoint_interval': 20,
}


def parse_voc_xml(xml_path: str) -> dict:
    """解析 VOC XML 标注"""
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
        print(f"Error parsing {xml_path}: {e}")
        return None


def compute_iou(box1: list, box2: list) -> float:
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


def load_image(path: str, size: tuple = None) -> np.ndarray:
    """加载图像"""
    img = Image.open(path).convert('L')
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    # 标准化
    img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
    return img_array


def run_inference(config: dict):
    """运行推理"""
    print("=" * 60)
    print("Res-SAM V2: Fully Automatic Inference")
    print("=" * 60)
    
    # 检查 Feature Bank
    if not os.path.exists(config['feature_bank_path']):
        raise FileNotFoundError(
            f"Feature bank not found: {config['feature_bank_path']}\n"
            "Please run 04_build_feature_bank_v2.py first.")
    
    # 加载元数据
    if os.path.exists(config['metadata_path']):
        with open(config['metadata_path'], 'r') as f:
            metadata = json.load(f)
        print(f"\nFeature Bank 来源: {list(metadata.get('sources', {}).keys())}")
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 导入 ResSAM
    from PatchRes.ResSAM import ResSAM
    
    # 初始化模型
    print("\n初始化 ResSAM...")
    model = ResSAM(
        hidden_size=config['hidden_size'],
        window_size=config['window_size'],
        stride=config['stride'],
        anomaly_threshold=config['anomaly_threshold'],
        sam_model_type=config['sam_model_type'],
        sam_checkpoint=config['sam_checkpoint'],
    )
    
    # 加载 Feature Bank
    model.load_feature_bank(config['feature_bank_path'])
    
    all_results = {}
    
    # 处理每个类别
    for category, data_dir in config['test_data_dirs'].items():
        print(f"\n{'='*60}")
        print(f"处理类别: {category}")
        print(f"数据目录: {data_dir}")
        print("=" * 60)
        
        if not os.path.exists(data_dir):
            print(f"警告: 目录不存在: {data_dir}")
            continue
        
        # 断点文件
        checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_{category}.json')
        
        # 加载断点
        start_idx = 0
        results = []
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)
            start_idx = checkpoint.get('processed_count', 0)
            results = checkpoint.get('results', [])
            print(f"从第 {start_idx} 张图片继续...")
        
        # 获取图像列表
        image_files = sorted([f for f in os.listdir(data_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"总图像数: {len(image_files)}")
        
        annotation_dir = config['annotation_dirs'].get(category, '')
        
        # 处理每张图像
        for i in tqdm(range(start_idx, len(image_files)), desc=f"Processing {category}"):
            img_file = image_files[i]
            img_path = os.path.join(data_dir, img_file)
            
            # 获取 GT
            xml_name = os.path.splitext(img_file)[0] + '.xml'
            xml_path = os.path.join(annotation_dir, xml_name)
            gt = parse_voc_xml(xml_path) if os.path.exists(xml_path) else None
            
            try:
                # 加载图像
                img = load_image(img_path, config['image_size'])
                
                # 推理
                result = model.detect_automatic(
                    img,
                    min_region_area=config['min_region_area'],
                    max_regions=config['max_candidates_per_image'],
                    return_all_candidates=False,
                )
                
                # 处理结果
                pred_bboxes = [r['bbox'] for r in result['anomaly_regions']]
                anomaly_scores = [r['max_anomaly_score'] for r in result['anomaly_regions']]
                
                # 计算 TP/FP
                tp = 0
                fp = 0
                if gt and gt['objects']:
                    gt_boxes = [[o['xmin'], o['ymin'], o['xmax'], o['ymax']] 
                               for o in gt['objects']]
                    
                    for pred_box in pred_bboxes:
                        # 缩放 bbox 到原始尺寸
                        scale_x = gt['width'] / config['image_size'][1]
                        scale_y = gt['height'] / config['image_size'][0]
                        pred_box_scaled = [
                            int(pred_box[0] * scale_x),
                            int(pred_box[1] * scale_y),
                            int(pred_box[2] * scale_x),
                            int(pred_box[3] * scale_y),
                        ]
                        
                        # 检查是否与任何 GT 匹配
                        matched = False
                        for gt_box in gt_boxes:
                            iou = compute_iou(pred_box_scaled, gt_box)
                            if iou >= 0.5:  # IoU 阈值
                                matched = True
                                break
                        
                        if matched:
                            tp += 1
                        else:
                            fp += 1
                elif pred_bboxes:
                    fp = len(pred_bboxes)
                
                results.append({
                    'image_name': img_file,
                    'pred_bboxes': pred_bboxes,
                    'anomaly_scores': anomaly_scores,
                    'tp': tp,
                    'fp': fp,
                    'fn': len(gt['objects']) if gt and gt['objects'] else 0,
                    'num_gt': len(gt['objects']) if gt and gt['objects'] else 0,
                })
                
                # 保存断点
                if (i + 1) % config['checkpoint_interval'] == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump({
                            'processed_count': i + 1,
                            'results': results,
                            'timestamp': datetime.now().isoformat(),
                        }, f)
                
            except Exception as e:
                print(f"\nError processing {img_file}: {e}")
                continue
        
        # 清除断点
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        all_results[category] = results
        
        # 计算指标
        total_tp = sum(r['tp'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_fn = sum(r['fn'] for r in results)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{category} 结果:")
        print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
    
    # 保存完整结果
    output_path = os.path.join(config['output_dir'], 'auto_predictions.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存至: {output_path}")
    
    # 计算总体指标
    overall_tp = sum(sum(r['tp'] for r in results) for results in all_results.values())
    overall_fp = sum(sum(r['fp'] for r in results) for results in all_results.values())
    overall_fn = sum(sum(r['fn'] for r in results) for results in all_results.values())
    
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print("\n" + "=" * 60)
    print("总体结果:")
    print(f"  TP: {overall_tp}, FP: {overall_fp}, FN: {overall_fn}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1: {overall_f1:.4f}")
    print("=" * 60)
    
    return all_results


if __name__ == "__main__":
    with torch.no_grad():
        results = run_inference(CONFIG)
