"""
SAM (Segment Anything Model) 集成模块

功能：
- 加载 SAM 模型
- 当前主线使用 automatic mask generation
- 生成候选异常区域

论文对应：
- Fig.2: SAM 用于定位候选区域

当前仓库主线说明：
- click-guided 相关接口仍保留在底层模块中，但已不属于当前 experiments 主线。
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import cv2
import os
import logging

_logger = logging.getLogger(__name__)
# 统计：多 crop 层且无任何 mask 通过滤时，原库会在 NMS 处崩溃；本补丁跳过该步。
_sam_amg_skip_nms_empty_crop_count = 0

# SamAutomaticMaskGenerator：与作者仓库 zhouxr6066/Res-SAM 的 sam/sam.py 显式参数一致，
# 仅传入作者在 sam.py 里写出的 8 个关键字；其余全部使用当前已安装 segment_anything
# 版本的构造函数默认值（与作者在官方仓库中的写法一致）。
# 参考：https://github.com/zhouxr6066/Res-SAM/blob/main/sam/sam.py
# V17-3: 降低阈值，让SAM生成更多候选区域
_SAM_AUTOMATIC_MASK_GENERATOR_KWARGS = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.80,  # V17-3: 从0.95降到0.80，生成更多候选
    "stability_score_thresh": 0.85,  # V17-3: 从0.95降到0.85，生成更多候选
    "crop_n_layers": 1,
    "box_nms_thresh": 0.7,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 1000,
    "output_mode": "binary_mask",
}


def _patch_sam_automatic_mask_generator_generate_masks() -> None:
    """
    segment_anything SamAutomaticMaskGenerator._generate_masks can call
    torchvision box_area(data['crop_boxes']) when len(crop_boxes) > 1,
    even if no masks survived filtering. Then crop_boxes may be an empty
    1D tensor (shape [0]) instead of [0,4], causing IndexError in box_area.
    See: https://github.com/facebookresearch/segment-anything/issues/614
    """
    try:
        from segment_anything import SamAutomaticMaskGenerator
    except ImportError:
        return
    if getattr(SamAutomaticMaskGenerator._generate_masks, "_res_sam_patched", False):
        return

    from torchvision.ops.boxes import batched_nms, box_area

    from segment_anything.utils.amg import MaskData, generate_crop_boxes

    def _generate_masks_fixed(self, image: np.ndarray) -> Any:
        global _sam_amg_skip_nms_empty_crop_count
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        if len(crop_boxes) > 1:
            if len(data["crop_boxes"]) > 0:
                scores = 1 / box_area(data["crop_boxes"])
                scores = scores.to(data["boxes"].device)
                keep_by_nms = batched_nms(
                    data["boxes"].float(),
                    scores,
                    torch.zeros_like(data["boxes"][:, 0]),
                    iou_threshold=self.crop_nms_thresh,
                )
                data.filter(keep_by_nms)
            else:
                _sam_amg_skip_nms_empty_crop_count += 1
                _logger.warning(
                    "[SAM_AMG_PATCH] skip crop-level NMS: multi-layer crops (%d) but no boxes "
                    "survived filtering (would trigger segment-anything#614 crash in unpatched "
                    "torchvision box_area path). Total skip count=%d",
                    len(crop_boxes),
                    _sam_amg_skip_nms_empty_crop_count,
                )

        data.to_numpy()
        return data

    _generate_masks_fixed._res_sam_patched = True
    SamAutomaticMaskGenerator._generate_masks = _generate_masks_fixed


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 grayscale with minimal peak memory."""
    if image.dtype == np.uint8:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[-1] == 1:
            return image[:, :, 0]

    if image.ndim == 3:
        # If RGB-like, convert to gray first (uint8 path if possible).
        if image.dtype == np.uint8 and image.shape[-1] >= 3:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        else:
            # Float RGB -> gray float
            image = cv2.cvtColor(image[:, :, :3].astype(np.float32), cv2.COLOR_RGB2GRAY)

    img = image.astype(np.float32, copy=False)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx - mn < 1e-8:
        return np.zeros(img.shape, dtype=np.uint8)
    img_u8 = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)
    return img_u8


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 RGB with minimal peak memory."""
    if image.ndim == 3 and image.dtype == np.uint8 and image.shape[-1] == 3:
        return image
    gray_u8 = _to_uint8_gray(image)
    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)


class SAMIntegration:
    """SAM 模型集成类"""
    
    def __init__(
        self, 
        model_type: str = "vit_l",
        checkpoint_path: str = None,
        device: str = "cuda",
    ):
        """
        初始化 SAM 模型
        
        Parameters:
        -----------
        model_type : str
            SAM 模型类型: "vit_h", "vit_l", "vit_b"
        checkpoint_path : str
            模型权重路径，如果为 None 则自动下载
        device : str
            运行设备
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        
        # 延迟加载模型
        self._sam = None
        self._mask_generator = None
    
    def _load_sam(self):
        """延迟加载 SAM 模型"""
        if self._sam is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment-anything 未安装。请运行: pip install segment-anything\n"
                "或从 https://github.com/facebookresearch/segment-anything 安装"
            )

        _patch_sam_automatic_mask_generator_generate_masks()
        
        # 确定模型路径
        if self.checkpoint_path is None:
            # 默认路径
            import os
            default_paths = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth", 
                "vit_b": "sam_vit_b_01ec64.pth",
            }
            # 检查常见位置
            possible_dirs = [
                os.path.expanduser("~/.cache/torch/hub/checkpoints/"),
                os.path.expanduser("~/models/"),
                "./models/",
                "./",
            ]
            for dir_path in possible_dirs:
                path = os.path.join(dir_path, default_paths.get(self.model_type, "sam_vit_l_0b3195.pth"))
                if os.path.exists(path):
                    self.checkpoint_path = path
                    break
            
            if self.checkpoint_path is None:
                raise FileNotFoundError(
                    f"SAM 模型权重未找到。请下载 {self.model_type} 模型权重:\n"
                    f"  - vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n"
                    f"  - vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                    f"  - vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                    f"并放置在 ./models/ 目录下或指定 checkpoint_path"
                )
        
        print(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")

        # 避免 Windows 上 segment-anything 内部 build_sam.py 的 torch.load 触发进程崩溃（exit code 3221225477）。
        # 改为：先构建空模型，再手动 torch.load state_dict 并 load_state_dict。
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception:
            pass

        self._sam = sam_model_registry[self.model_type](checkpoint=None)
        try:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        self._sam.load_state_dict(state_dict, strict=True)
        self._sam.to(device=self.device)
        
        # 创建 predictor (用于 click-guided 模式)
        
        # 与官方仓库 sam/sam.py 中 SamAutomaticMaskGenerator(...) 一致，见 _SAM_AUTOMATIC_MASK_GENERATOR_KWARGS。
        self._mask_generator = SamAutomaticMaskGenerator(
            self._sam,
            **_SAM_AUTOMATIC_MASK_GENERATOR_KWARGS,
        )
        
        print(f"SAM model loaded successfully on {self.device}")
    
    def generate_masks_automatic(
        self, 
        image: np.ndarray,
        min_area_ratio: Optional[float] = None,
        max_area_ratio: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        自动生成所有可能的 mask (Fully Automatic 模式)
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像 (H, W, C) 或 (H, W)，灰度图会自动转为 RGB
        min_area_ratio : float
            最小区域面积比例（相对于图像面积）。
            论文未给出这类固定比例阈值，因此这里只把它保留为可选工程过滤参数。
        max_area_ratio : float
            最大区域面积比例（相对于图像面积）。
            同上，默认不启用，避免把工程经验阈值误写成论文口径。
            
        Returns:
        --------
        List[Dict]
            候选区域列表，每个包含:
            - 'bbox': [x1, y1, x2, y2]
            - 'mask': 二值 mask
            - 'area': 区域面积
            - 'stability_score': 稳定性分数
        """
        self._load_sam()

        # 统一转换为 uint8 RGB（避免构造 (H,W,3) float32 的中间数组造成内存峰值）
        image = _to_uint8_rgb(image)
        
        # 生成 masks；与官方 sam/sam.py 中 predict_mask 一致：仅使用 area < 2e4 的 mask
        masks = self._mask_generator.generate(image)
        masks = [m for m in masks if m.get("area", 0) < 2e4]
        
        # 面积过滤保持可选。
        # 论文没有给出固定的 candidate-region 面积比例阈值，因此仅在调用方显式提供时启用。
        img_area = image.shape[0] * image.shape[1]
        min_area = int(img_area * min_area_ratio) if min_area_ratio is not None else None
        max_area = int(img_area * max_area_ratio) if max_area_ratio is not None else None
        
        candidate_regions = []
        for mask_info in masks:
            area = mask_info['area']
            area_ok = True
            if min_area is not None:
                area_ok = area_ok and (area >= min_area)
            if max_area is not None:
                area_ok = area_ok and (area <= max_area)
            if area_ok:
                # 转换 bbox 格式: SAM 使用 [x, y, w, h] -> 转为 [x1, y1, x2, y2]
                x, y, w, h = mask_info['bbox']
                bbox = [int(x), int(y), int(x + w), int(y + h)]
                
                candidate_regions.append({
                    'bbox': bbox,
                    'mask': mask_info['segmentation'],
                    'area': area,
                    'stability_score': mask_info['stability_score'],
                    'predicted_iou': mask_info['predicted_iou'],
                })
        
        # 按稳定性分数排序
        candidate_regions.sort(key=lambda x: x['stability_score'], reverse=True)
        
        return candidate_regions
    
    def _removed_click_guided_masks(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[int, int]] = None,
        negative_points: List[Tuple[int, int]] = None,
        box: List[int] = None,
        image_already_prepared: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        基于 click 提示生成 mask (Click-guided 模式)
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        positive_points : List[Tuple[int, int]]
            正样本点 (异常区域内的点)
        negative_points : List[Tuple[int, int]]
            负样本点 (非异常区域内的点)
        box : List[int]
            边界框 [x1, y1, x2, y2]
            
        Returns:
        --------
        List[Dict]
            候选区域列表
        """
        raise NotImplementedError(
            "Click-guided mode has been removed from this repository. Use generate_masks_automatic() instead."
        )

        # 统一转换为 uint8 RGB（避免构造 (H,W,3) float32 的中间数组造成内存峰值）
        image = _to_uint8_rgb(image)
        
        # 设置图像
        # Prefer content-keyed cache reuse to avoid recomputing the same image
        # embedding across different click configurations.
        self.prepare_image(image, force=False)
        
        # 准备输入
        point_coords = None
        point_labels = None
        box_np = None
        
        if positive_points or negative_points:
            points = []
            labels = []
            if positive_points:
                points.extend(positive_points)
                labels.extend([1] * len(positive_points))  # 1 = 前景
            if negative_points:
                points.extend(negative_points)
                labels.extend([0] * len(negative_points))  # 0 = 背景
            point_coords = np.array(points)
            point_labels = np.array(labels)
        
        if box:
            box_np = np.array(box)
        
        # 预测（对齐作者开源仓库：multimask_output=True，取 scores 最大者）
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
            multimask_output=True,
        )

        if masks is None or scores is None or len(masks) == 0:
            return []
        try:
            best_idx = int(np.asarray(scores).argmax())
        except Exception:
            best_idx = 0
        best_mask = masks[best_idx]
        best_score = float(np.asarray(scores)[best_idx]) if len(np.asarray(scores).shape) > 0 else float(scores)

        # 转换为候选区域格式（click-guided 仅保留主候选区）
        rows = np.any(best_mask, axis=1)
        cols = np.any(best_mask, axis=0)
        if not (rows.any() and cols.any()):
            return []
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        # bbox 统一为半开区间 [x1, y1, x2, y2) 以匹配 numpy 切片 image[y1:y2, x1:x2]
        bbox = [int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1]
        return [
            {
                "bbox": bbox,
                "mask": best_mask,
                "area": int(np.sum(best_mask)),
                "score": best_score,
            }
        ]

    def extract_region(
        self,
        image: np.ndarray,
        bbox: List[int],
        padding: int = 5,
    ) -> np.ndarray:
        """
        从图像中提取指定区域
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        bbox : List[int]
            [x1, y1, x2, y2]
        padding : int
            边界填充
            
        Returns:
        --------
        np.ndarray
            提取的区域
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        return image[y1:y2, x1:x2]
    
    def visualize_regions(
        self,
        image: np.ndarray,
        regions: List[Dict[str, Any]],
        output_path: str = None,
        title: str = "SAM Candidate Regions",
    ) -> np.ndarray:
        """
        可视化候选区域
        
        Parameters:
        -----------
        image : np.ndarray
            原始图像
        regions : List[Dict]
            候选区域列表
        output_path : str
            保存路径
        title : str
            图像标题
            
        Returns:
        --------
        np.ndarray
            可视化图像
        """
        # 处理灰度图
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # 确保是 uint8
        if vis_image.dtype != np.uint8:
            vis_image = ((vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8) * 255).astype(np.uint8)
        
        # 绘制每个区域
        colors = [
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 红色
            (255, 0, 0),    # 蓝色
            (0, 255, 255),  # 黄色
            (255, 0, 255),  # 紫色
        ]
        
        for i, region in enumerate(regions[:5]):  # 最多显示 5 个
            color = colors[i % len(colors)]
            bbox = region['bbox']
            
            # 绘制 bbox
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 绘制 mask (半透明叠加)
            if 'mask' in region:
                mask = region['mask'].astype(np.uint8) * 255
                colored_mask = np.zeros_like(vis_image)
                colored_mask[:, :] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            
            # 添加标签
            score = region.get('stability_score', region.get('score', 0))
            label = f"#{i+1} s={score:.2f}"
            cv2.putText(vis_image, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 添加标题
        cv2.putText(vis_image, title, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 保存
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image


def check_sam_installation():
    """检查 SAM 安装状态"""
    try:
        import segment_anything
        print("✓ segment-anything 已安装")
        return True
    except ImportError:
        print("✗ segment-anything 未安装")
        print("  安装命令: pip install segment-anything")
        print("  或: pip install git+https://github.com/facebookresearch/segment-anything.git")
        return False


def download_sam_weights(model_type: str = "vit_l", save_dir: str = "./models/"):
    """下载 SAM 模型权重"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    
    if model_type not in urls:
        raise ValueError(f"Unknown model type: {model_type}")
    
    import urllib.request
    save_path = os.path.join(save_dir, urls[model_type].split('/')[-1])
    
    print(f"Downloading SAM {model_type} weights to {save_path}...")
    urllib.request.urlretrieve(urls[model_type], save_path)
    print(f"Download complete: {save_path}")
    
    return save_path


if __name__ == "__main__":
    # 测试 SAM 集成
    print("Testing SAM Integration...")
    
    # 检查安装
    if not check_sam_installation():
        exit(1)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    
    # 测试初始化
    try:
        sam = SAMIntegration(model_type="vit_l")
        print("✓ SAMIntegration 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
