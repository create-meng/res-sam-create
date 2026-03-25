"""
SAM (Segment Anything Model) 集成模块

功能：
- 加载 SAM 模型
- 支持 automatic mask generation 和 click-guided 模式
- 生成候选异常区域

论文对应：
- Fig.2: SAM 用于定位候选区域
- Table 1: click-guided vs automatic 模式对比
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import cv2
import os
import hashlib
from collections import OrderedDict


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
        model_type: str = "vit_b",
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
        self._predictor = None
        self._mask_generator = None
        self._prepared_image_ref = None
        self._prepared_image_key = None
        self._image_embedding_cache = OrderedDict()
    
    def _load_sam(self):
        """延迟加载 SAM 模型"""
        if self._sam is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "segment-anything 未安装。请运行: pip install segment-anything\n"
                "或从 https://github.com/facebookresearch/segment-anything 安装"
            )
        
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
                path = os.path.join(dir_path, default_paths.get(self.model_type, "sam_vit_b_01ec64.pth"))
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
        self._predictor = SamPredictor(self._sam)
        
        # 创建 mask generator (用于 automatic 模式)
        # 参数调整：生成更大更完整的区域
        self._mask_generator = SamAutomaticMaskGenerator(
            self._sam,
            points_per_side=32,       # 增加采样点
            points_per_batch=64,
            pred_iou_thresh=0.75,     # 降低阈值保留更多区域
            stability_score_thresh=0.80,  # 降低稳定性阈值
            stability_score_offset=1.0,
            crop_n_layers=1,          # 启用裁剪层增加覆盖
            crop_overlap_ratio=0.3,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=50,  # 降低最小区域
        )
        
        print(f"SAM model loaded successfully on {self.device}")
    
    def generate_masks_automatic(
        self, 
        image: np.ndarray,
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        自动生成所有可能的 mask (Fully Automatic 模式)
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像 (H, W, C) 或 (H, W)，灰度图会自动转为 RGB
        min_area_ratio : float
            最小区域面积比例（相对于图像面积）
        max_area_ratio : float  
            最大区域面积比例（相对于图像面积）
            
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
        
        # 生成 masks
        masks = self._mask_generator.generate(image)
        
        # 过滤和排序
        img_area = image.shape[0] * image.shape[1]
        min_area = int(img_area * min_area_ratio)
        max_area = int(img_area * max_area_ratio)
        
        candidate_regions = []
        for mask_info in masks:
            area = mask_info['area']
            if min_area <= area <= max_area:
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
    
    def generate_masks_from_clicks(
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
        self._load_sam()

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
        
        # 预测
        masks, scores, logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_np,
            multimask_output=True,  # 输出多个 mask
        )
        
        # 转换为候选区域格式
        candidate_regions = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            # 计算 bbox
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                y_min, y_max = np.where(rows)[0][[0, -1]]
                x_min, x_max = np.where(cols)[0][[0, -1]]
                # bbox 统一为半开区间 [x1, y1, x2, y2) 以匹配 numpy 切片 image[y1:y2, x1:x2]
                bbox = [int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1]
                
                candidate_regions.append({
                    'bbox': bbox,
                    'mask': mask,
                    'area': int(np.sum(mask)),
                    'score': float(score),
                })
        
        # 按分数排序
        candidate_regions.sort(key=lambda x: x['score'], reverse=True)
        
        return candidate_regions

    def prepare_image(self, image: np.ndarray, force: bool = False) -> np.ndarray:
        self._load_sam()
        image_rgb = _to_uint8_rgb(image)
        cache_key = self._make_image_cache_key(image_rgb)
        if (not force) and (self._prepared_image_key == cache_key):
            return image_rgb

        cached_state = None if force else self._image_embedding_cache.get(cache_key)
        if cached_state is not None:
            self._restore_predictor_state(cached_state)
        else:
            self._predictor.set_image(image_rgb)
            self._store_predictor_state(cache_key)
        self._prepared_image_ref = image_rgb
        self._prepared_image_key = cache_key
        return image_rgb

    def clear_prepared_image(self) -> None:
        self._prepared_image_ref = None
        self._prepared_image_key = None

    def _make_image_cache_key(self, image_rgb: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(image_rgb)
        digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()
        return f"{contiguous.shape}|{contiguous.dtype}|{digest}"

    def _cache_capacity(self) -> int:
        try:
            env_value = int((os.environ.get("RES_SAM_SAM_CACHE_SIZE", "") or "").strip())
            if env_value > 0:
                return env_value
        except Exception:
            pass
        return 256

    def _store_predictor_state(self, cache_key: str) -> None:
        features = getattr(self._predictor, "features", None)
        if features is None:
            return
        state = {
            "features": features.detach().clone(),
            "input_size": tuple(getattr(self._predictor, "input_size", ())),
            "original_size": tuple(getattr(self._predictor, "original_size", ())),
        }
        self._image_embedding_cache[cache_key] = state
        self._image_embedding_cache.move_to_end(cache_key)
        while len(self._image_embedding_cache) > self._cache_capacity():
            self._image_embedding_cache.popitem(last=False)

    def _restore_predictor_state(self, state: Dict[str, Any]) -> None:
        self._predictor.reset_image()
        self._predictor.features = state["features"]
        self._predictor.input_size = tuple(state["input_size"])
        self._predictor.original_size = tuple(state["original_size"])
        self._predictor.is_image_set = True
    
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


def download_sam_weights(model_type: str = "vit_b", save_dir: str = "./models/"):
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
        sam = SAMIntegration(model_type="vit_b")
        print("✓ SAMIntegration 初始化成功")
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
