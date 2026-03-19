"""
Res-SAM 复现实验 V3 - Step 2: Fully Automatic 模式推理

V3 改进（严格按论文对齐）：
- window_size = 50（论文默认值）
- 特征口径 f = [W_out, b]（论文 Eq.(2)-(3)）
- Region 级粗筛步骤（论文 Fully Automatic 关键流程）
- 特征维度 = 2*hidden_size + 1 = 61

论文对应：Table 2 - Fully Automatic Mode
"""

import sys
import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import hashlib
import uuid
import atexit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

class _TeeStream:
    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        for s in self._streams:
            try:
                if hasattr(s, "isatty") and s.isatty():
                    return True
            except Exception:
                pass
        return False

# ============ 配置 ============
CONFIG = {
    # Feature Bank 路径 (V3)
    "feature_bank_path": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs", "feature_banks_v3", "feature_bank_v3.pth"
    ),
    "metadata_path": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs", "feature_banks_v3", "metadata.json"
    ),
    
    # 测试数据
    "test_data_dirs": {
        "cavities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", "GPR_data", "augmented_cavities"
        ),
        "utilities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", "GPR_data", "augmented_utilities"
        ),
        "normal_auc": os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", "GPR_data", "intact"
        ),
    },
    "annotation_dirs": {
        "cavities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", "GPR_data", "augmented_cavities", 
            "annotations", "VOC_XML_format"
        ),
        "utilities": os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "data", "GPR_data", "augmented_utilities", 
            "annotations", "VOC_XML_format"
        ),
    },
    
    "output_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs", "predictions_v3"
    ),
    "checkpoint_dir": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "outputs", "checkpoints_v3"
    ),
    
    # 论文参数（V3 严格对齐）
    "window_size": 50,  # 论文默认值
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": 0.5,  # 论文 Eq.(9) 的单一预设阈值 β
    "anomaly_threshold": 0.5,  # 兼容旧字段：内部与 beta_threshold 保持一致
    "region_coarse_threshold": 0.5,  # 兼容旧字段：内部与 beta_threshold 保持一致
    
    # SAM 参数
    "sam_model_type": "vit_b",
    "sam_checkpoint": os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "sam", "sam_vit_b_01ec64.pth"
    ),
    
    # 图像预处理
    "image_size": (369, 369),
    
    # 推理参数
    "max_candidates_per_image": 10,
    "min_region_area": 100,
    "max_images_per_category": None,
    
    # 断点
    "checkpoint_interval": 20,
    
    # 随机种子
    "random_seed": 42,
    
    # 版本标识
    "version": "V3",
    "alignment_notes": "Strictly aligned with paper: window_size=50, region-level coarse filtering, feature f=[W_out,b]",
}


# 环境变量支持
_max_images_env = os.environ.get("MAX_IMAGES_PER_CATEGORY", "").strip()
if _max_images_env:
    try:
        _max_images_val = int(_max_images_env)
        if _max_images_val > 0:
            CONFIG["max_images_per_category"] = _max_images_val
    except Exception:
        pass


def parse_voc_xml(xml_path: str) -> dict:
    """解析 VOC XML 标注文件"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 图像信息
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 边界框
        bboxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            bboxes.append([xmin, ymin, xmax, ymax])
        
        return {
            'width': width,
            'height': height,
            'bboxes': bboxes,
        }
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
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
    """加载并预处理图像"""
    img = Image.open(path).convert('L')
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
    return img_array


def run_inference(config: dict):
    """运行 Fully Automatic 推理（V3 严格对齐）"""
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, "outputs", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"auto_inference_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_fp = open(log_file, "w", encoding="utf-8")
    tee = _TeeStream(original_stdout, log_fp)
    sys.stdout = tee
    sys.stderr = tee

    def _restore_streams():
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        except Exception:
            pass
        try:
            log_fp.close()
        except Exception:
            pass

    atexit.register(_restore_streams)

    print("=" * 60)
    print("Res-SAM V3: Fully Automatic Inference (Strict Paper Alignment)")
    print("=" * 60)
    print(f"日志文件: {log_file}")
    print(f"run_id: {run_id}")
    print(f"  window_size = {config['window_size']}")
    print(f"  stride = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    print(f"  Expected feature dim = {2*config['hidden_size'] + 1}")
    print(f"  beta_threshold = {config['beta_threshold']}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    # 设置随机种子
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    
    # 导入 ResSAM
    from PatchRes.ResSAM import ResSAM
    
    # 初始化模型（V3 参数）
    print("\n初始化 ResSAM (V3)...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],  # 50
        stride=config["stride"],
        anomaly_threshold=config.get("beta_threshold", config["anomaly_threshold"]),
        region_coarse_threshold=config.get("beta_threshold", config["region_coarse_threshold"]),
        sam_model_type=config["sam_model_type"],
        sam_checkpoint=config["sam_checkpoint"],
    )
    
    # 加载 Feature Bank
    print(f"加载 Feature Bank: {config['feature_bank_path']}")
    model.load_feature_bank(config["feature_bank_path"])
    
    # 加载 Feature Bank 元数据
    metadata = {}
    if os.path.exists(config["metadata_path"]):
        with open(config["metadata_path"], "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Feature Bank 元数据: {metadata.get('feature_bank_shape', 'Unknown')}")
    
    # 处理每个类别
    all_results = {}
    
    for category, data_dir in config["test_data_dirs"].items():
        print(f"\n{'='*60}")
        print(f"处理类别: {category}")
        print(f"目录: {data_dir}")
        print("=" * 60)
        
        if not os.path.exists(data_dir):
            print(f"警告: 目录不存在: {data_dir}")
            continue
        
        # 获取图像文件列表
        image_files = [f for f in os.listdir(data_dir) 
                      if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        
        if config["max_images_per_category"]:
            image_files = image_files[:config["max_images_per_category"]]
        
        print(f"找到 {len(image_files)} 张图像")

        checkpoint_interval = int(config.get("checkpoint_interval", 20))
        effective_interval = checkpoint_interval
        if len(image_files) > 0 and len(image_files) <= checkpoint_interval:
            effective_interval = 1
            print(
                f"短任务断点策略: num_images={len(image_files)} <= checkpoint_interval={checkpoint_interval}，"
                f"将 effective_checkpoint_interval=1（每张保存一次断点）"
            )
        
        # 断点支持
        checkpoint_file = os.path.join(config["checkpoint_dir"], f"checkpoint_auto_{category}.json")
        processed_files = set()
        
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            processed_files = set(checkpoint.get("processed_files", []))
            print(f"从断点继续，已处理 {len(processed_files)} 张图像")
        
        category_results = []
        
        # 处理每张图像
        for i, img_file in enumerate(
            tqdm(
                image_files,
                desc=f"推理 {category}",
                disable=(not sys.stdout.isatty()),
                file=sys.stdout,
            )
        ):
            if img_file in processed_files:
                continue
            
            img_path = os.path.join(data_dir, img_file)
            
            # 获取标注（如果存在）
            annotation_dir = config["annotation_dirs"].get(category)
            xml_file = os.path.splitext(img_file)[0] + ".xml"
            xml_path = os.path.join(annotation_dir, xml_file) if annotation_dir else None
            
            try:
                # 读取原图尺寸（用于将预测 bbox 从 resize 坐标系缩放回原图坐标系，避免评估/可视化口径漂移）
                with Image.open(img_path) as _im:
                    orig_w, orig_h = _im.size

                # 加载图像
                img = load_image(img_path, config["image_size"])
                
                # 推理（V3 detect_automatic 包含 region 级粗筛）
                result = model.detect_automatic(
                    img,
                    min_region_area=config["min_region_area"],
                    max_regions=config["max_candidates_per_image"],
                )
                
                # 解析标注
                gt = parse_voc_xml(xml_path) if xml_path and os.path.exists(xml_path) else None
                
                # 准备结果记录
                record = {
                    "image_name": img_file,
                    "image_path": img_path,
                    # detect_automatic 返回 bbox 在 resize 坐标系下（与输入 img 对齐）
                    "pred_bboxes": [r["bbox"] for r in result["anomaly_regions"]],
                    "anomaly_scores": [r["max_anomaly_score"] for r in result["anomaly_regions"]],
                    "num_candidates": result["num_candidates"],
                    "num_coarse_discarded": result["num_coarse_discarded"],
                    "num_esn_fits": result["num_esn_fits"],
                }

                # 统一输出坐标系：pred_bboxes -> 原图/VOC 坐标系；pred_bboxes_resized -> resize 坐标系。
                # 注意：若存在 VOC XML，评估时 gt_bboxes 的坐标系以 XML 的 width/height 为准，
                # 因此这里缩放应优先对齐到 XML 声明的图像尺寸，避免“图像文件尺寸 != XML size”导致 IoU 漂移。
                target_w = int(gt["width"]) if gt and gt.get("width") else int(orig_w)
                target_h = int(gt["height"]) if gt and gt.get("height") else int(orig_h)

                resized_w = config["image_size"][1]
                resized_h = config["image_size"][0]
                scale_x_img = target_w / resized_w
                scale_y_img = target_h / resized_h
                pred_bboxes_resized = record.get("pred_bboxes", [])
                pred_bboxes_scaled = [
                    [
                        int(b[0] * scale_x_img),
                        int(b[1] * scale_y_img),
                        int(b[2] * scale_x_img),
                        int(b[3] * scale_y_img),
                    ]
                    for b in pred_bboxes_resized
                ]
                
                record["pred_bboxes_resized"] = pred_bboxes_resized
                record["pred_bboxes"] = pred_bboxes_scaled
                
                # 添加 GT 信息（如果存在）
                if gt:
                    _has_gt = True
                    _num_gt = len(gt['bboxes'])
                    
                    # 同时提供 GT 的 resize 坐标系（便于与 pred_bboxes 直接对照）
                    inv_scale_x = config["image_size"][1] / gt["width"]
                    inv_scale_y = config["image_size"][0] / gt["height"]
                    
                    gt_boxes_resized = [
                        [
                            int(bbox[0] * inv_scale_x),
                            int(bbox[1] * inv_scale_y),
                            int(bbox[2] * inv_scale_x),
                            int(bbox[3] * inv_scale_y),
                        ]
                        for bbox in gt["bboxes"]
                    ]
                    
                    record.update({
                        'gt_bboxes': gt['bboxes'],  # 原图坐标系
                        'gt_bboxes_resized': gt_boxes_resized,  # resize 坐标系
                        'num_gt': int(_num_gt),
                        'gt_width': gt['width'],
                        'gt_height': gt['height'],
                        'exclude_from_det_metrics': False,
                        'exclude_from_auc': False,
                    })
                
                else:
                    _has_gt = False
                    _num_gt = 0
                    
                    # normal_auc 类别：无 GT，仅用于 AUC 计算
                    if category == "normal_auc":
                        record.update({
                            "gt_bboxes": [],
                            "num_gt": 0,
                            "exclude_from_det_metrics": True,  # 不计入 TP/FP/FN
                            "exclude_from_auc": False,  # 参与 ROC/AUC
                        })
                    else:
                        # 其他类别但无标注：完全排除
                        record.update({
                            "gt_bboxes": [],
                            "num_gt": 0,
                            "exclude_from_det_metrics": True,
                            "exclude_from_auc": True,
                        })
                
                category_results.append(record)
                
                # 更新断点
                processed_files.add(img_file)
                if effective_interval > 0 and ((i + 1) % effective_interval == 0):
                    with open(checkpoint_file, "w") as f:
                        json.dump({"processed_files": list(processed_files)}, f)
                
            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {e}")
                continue
        
        all_results[category] = category_results
        print(f"类别 {category} 完成，处理了 {len(category_results)} 张图像")
        
        # 清理断点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
    
    # 保存结果
    output_file = os.path.join(config["output_dir"], "auto_predictions_v3.json")
    
    # V3 输出格式：包含元数据
    output_data = {
        "meta": {
            "version": config["version"],
            "alignment_notes": config["alignment_notes"],
            "creation_time": datetime.now().isoformat(),
            "feature_bank_path": config["feature_bank_path"],
            "feature_bank_metadata_path": config.get("metadata_path", ""),
            "preprocess_signature": metadata.get("preprocess_signature", None),
            "image_size_hw": list(config.get("image_size", (369, 369))),
            "window_size": int(config.get("window_size", 50)),
            "stride": int(config.get("stride", 5)),
            "hidden_size": int(config.get("hidden_size", 30)),
            "beta_threshold": float(config.get("beta_threshold", config.get("anomaly_threshold", 0.5))),
        },
        "results": all_results,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果保存至: {output_file}")
    
    # 统计汇总
    total_images = sum(len(results) for results in all_results.values())
    total_detections = sum(len(r['pred_bboxes']) for results in all_results.values() for r in results)
    total_coarse_discarded = sum(r['num_coarse_discarded'] for results in all_results.values() for r in results)
    
    print(f"\n统计汇总:")
    print(f"  总图像数: {total_images}")
    print(f"  总检测数: {total_detections}")
    print(f"  Region 级粗筛丢弃: {total_coarse_discarded}")
    
    print("\n" + "=" * 60)
    print("Fully Automatic V3 推理完成!")
    print("=" * 60)
    
    _final_results = all_results
    _restore_streams()
    return _final_results


if __name__ == "__main__":
    with torch.no_grad():
        results = run_inference(CONFIG)