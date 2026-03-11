"""
Res-SAM: Reservoir-enhanced Segment Anything Model

论文复现核心模块，整合 SAM 和 2D-ESN。

工作流程：
1. SAM 生成候选异常区域
2. 2D-ESN 提取候选区域的动态特征
3. 与 Feature Bank 比较计算异常分数
4. 输出最终异常区域和分类

论文对应：
- Fig.2: 整体框架
- Table 2: Fully Automatic 模式
- Table 1: Click-guided 模式
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
from PIL import Image
import cv2

from .ESN_2D_nobatch import ESN_2D
from .common import NearestNeighbourScorer
from .sam_integration import SAMIntegration


class ResSAM:
    """
    Res-SAM: 结合 SAM 和 Reservoir Computing 的 GPR 异常检测框架
    
    Parameters:
    -----------
    hidden_size : int
        2D-ESN 隐藏层大小 (论文默认 30)
    window_size : int
        滑动窗口大小 (论文默认 50)
    stride : int
        滑动窗口步长 (论文默认 5)
    spectral_radius : float
        ESN 谱半径 (论文默认 0.9)
    connectivity : float
        ESN 连接率 (论文默认 0.1)
    anomaly_threshold : float
        异常判定阈值
    sam_model_type : str
        SAM 模型类型: "vit_b", "vit_l", "vit_h"
    sam_checkpoint : str
        SAM 权重路径
    device : str
        运行设备
    """
    
    def __init__(
        self,
        hidden_size: int = 30,
        window_size: int = 50,
        stride: int = 5,
        spectral_radius: float = 0.9,
        connectivity: float = 0.1,
        anomaly_threshold: float = 0.5,
        sam_model_type: str = "vit_b",
        sam_checkpoint: str = None,
        device: str = "cuda",
    ):
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.stride = stride
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.anomaly_threshold = anomaly_threshold
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 初始化 2D-ESN
        self.esn = ESN_2D(
            input_dim=1,
            n_reservoir=hidden_size,
            alpha=5,
            spectral_radius=(spectral_radius, spectral_radius),
            connectivity=connectivity,
        )
        
        # 初始化异常评分器
        self.anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours=1)
        
        # 初始化 SAM
        self.sam = SAMIntegration(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint,
            device=self.device,
        )
        
        # Feature Bank
        self.feature_bank = None
        self.feature_bank_source = None  # 记录来源
    
    def build_feature_bank(
        self,
        normal_images: np.ndarray,
        source_info: str = "unknown",
    ) -> torch.Tensor:
        """
        构建 Feature Bank
        
        Parameters:
        -----------
        normal_images : np.ndarray
            正常样本图像 [N, H, W] 或 [N, C, H, W]
        source_info : str
            数据来源信息（用于环境一致性检查）
            
        Returns:
        --------
        torch.Tensor
            Feature Bank [num_patches, hidden_size^2]
        """
        print(f"Building Feature Bank from {len(normal_images)} normal images...")
        
        # 确保图像格式正确
        if len(normal_images.shape) == 4:
            # [N, C, H, W] -> [N, H, W]
            normal_images = normal_images.squeeze(1)
        
        # 提取特征
        all_features = []
        
        for i, img in enumerate(normal_images):
            # 滑动窗口提取 patches
            patches = self._extract_patches(img)
            
            # 2D-ESN 拟合
            features = self._fit_patches(patches)
            all_features.append(features)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(normal_images)} images")
        
        # 合并特征
        self.feature_bank = torch.cat(all_features, dim=0)
        self.feature_bank_source = source_info
        
        # 训练异常评分器
        self.anomaly_scorer.fit(detection_features=self.feature_bank)
        
        print(f"Feature Bank built: shape={self.feature_bank.shape}, source={source_info}")
        
        return self.feature_bank
    
    def load_feature_bank(self, path: str):
        """加载预存的 Feature Bank"""
        self.feature_bank = torch.load(path)
        self.anomaly_scorer.fit(detection_features=self.feature_bank)
        print(f"Feature Bank loaded: shape={self.feature_bank.shape}")
    
    def save_feature_bank(self, path: str):
        """保存 Feature Bank"""
        torch.save(self.feature_bank, path)
        print(f"Feature Bank saved to {path}")
    
    def _extract_patches(self, image: np.ndarray) -> torch.Tensor:
        """
        滑动窗口提取 patches
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像 [H, W]
            
        Returns:
        --------
        torch.Tensor
            patches [num_patches, 1, window_size, window_size]
        """
        h, w = image.shape
        patches = []
        
        for i in range(0, h - self.window_size + 1, self.stride):
            for j in range(0, w - self.window_size + 1, self.stride):
                patch = image[i:i+self.window_size, j:j+self.window_size]
                patches.append(patch)
        
        # 转换为 tensor [N, 1, H, W]
        patches_tensor = torch.tensor(np.array(patches), dtype=torch.float32).unsqueeze(1)
        
        return patches_tensor
    
    def _fit_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        2D-ESN 拟合 patches 提取特征
        
        Parameters:
        -----------
        patches : torch.Tensor
            [num_patches, 1, window_size, window_size]
            
        Returns:
        --------
        torch.Tensor
            特征 [num_patches, hidden_size^2]
        """
        with torch.no_grad():
            features = self.esn.forward(patches)
        return features
    
    def detect_automatic(
        self,
        image: np.ndarray,
        min_region_area: int = 500,
        max_regions: int = 10,
        return_all_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Fully Automatic 模式异常检测
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像 [H, W] 或 [H, W, C]
        min_region_area : int
            最小区域面积
        max_regions : int
            最大返回区域数
        return_all_candidates : bool
            是否返回所有候选区域
            
        Returns:
        --------
        Dict
            - 'anomaly_regions': 异常区域列表
            - 'anomaly_scores': 异常分数
            - 'all_candidates': 所有候选区域 (如果 return_all_candidates=True)
        """
        # 确保灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Step 1: SAM 生成候选区域
        print("Step 1: SAM generating candidate regions...")
        candidate_regions = self.sam.generate_masks_automatic(
            image,
            min_area_ratio=0.005,
            max_area_ratio=0.5,
        )
        
        print(f"  Found {len(candidate_regions)} candidate regions")
        
        # Step 2: 分析每个候选区域
        print("Step 2: Analyzing candidate regions with 2D-ESN...")
        anomaly_regions = []
        anomaly_scores = []
        
        for i, region in enumerate(candidate_regions[:max_regions]):
            bbox = region['bbox']
            mask = region['mask']
            
            # 提取区域
            x1, y1, x2, y2 = bbox
            region_img = image[y1:y2, x1:x2]
            
            # 跳过过小区域
            if region_img.size < min_region_area:
                continue
            
            # 提取特征并计算异常分数
            patches = self._extract_patches(region_img)
            if len(patches) == 0:
                continue
            
            features = self._fit_patches(patches)
            
            # 计算异常分数 (与 Feature Bank 的最小距离)
            features_np = features.numpy()
            scores = self.anomaly_scorer.predict(features_np)[0]
            
            # 归一化分数
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # 平均异常分数
            avg_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            
            anomaly_regions.append({
                'bbox': bbox,
                'mask': mask,
                'avg_anomaly_score': avg_score,
                'max_anomaly_score': max_score,
                'stability_score': region.get('stability_score', 0),
            })
            anomaly_scores.append(max_score)
        
        # Step 3: 过滤异常区域
        print("Step 3: Filtering anomaly regions...")
        filtered_regions = [
            r for r in anomaly_regions 
            if r['max_anomaly_score'] > self.anomaly_threshold
        ]
        
        # 按异常分数排序
        filtered_regions.sort(key=lambda x: x['max_anomaly_score'], reverse=True)
        
        print(f"  Detected {len(filtered_regions)} anomaly regions")
        
        result = {
            'anomaly_regions': filtered_regions[:max_regions],
            'anomaly_scores': anomaly_scores,
            'num_candidates': len(candidate_regions),
        }
        
        if return_all_candidates:
            result['all_candidates'] = candidate_regions
        
        return result
    
    def detect_click_guided(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[int, int]] = None,
        negative_points: List[Tuple[int, int]] = None,
        box: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Click-guided 模式异常检测
        
        Parameters:
        -----------
        image : np.ndarray
            输入图像
        positive_points : List[Tuple[int, int]]
            正样本点 (异常区域内)
        negative_points : List[Tuple[int, int]]
            负样本点 (非异常区域)
        box : List[int]
            边界框 [x1, y1, x2, y2]
            
        Returns:
        --------
        Dict
            检测结果
        """
        # 确保灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Step 1: SAM 基于 click 生成候选区域
        print("Step 1: SAM generating regions from clicks...")
        candidate_regions = self.sam.generate_masks_from_clicks(
            image,
            positive_points=positive_points,
            negative_points=negative_points,
            box=box,
        )
        
        print(f"  Found {len(candidate_regions)} candidate regions")
        
        # Step 2: 分析候选区域
        anomaly_regions = []
        
        for region in candidate_regions:
            bbox = region['bbox']
            mask = region['mask']
            
            # 提取区域
            x1, y1, x2, y2 = bbox
            region_img = image[y1:y2, x1:x2]
            
            if region_img.size < 100:
                continue
            
            # 提取特征
            patches = self._extract_patches(region_img)
            if len(patches) == 0:
                continue
            
            features = self._fit_patches(patches)
            
            # 计算异常分数
            features_np = features.numpy()
            scores = self.anomaly_scorer.predict(features_np)[0]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            avg_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            
            anomaly_regions.append({
                'bbox': bbox,
                'mask': mask,
                'avg_anomaly_score': avg_score,
                'max_anomaly_score': max_score,
                'sam_score': region.get('score', 0),
            })
        
        # 过滤
        filtered_regions = [
            r for r in anomaly_regions 
            if r['max_anomaly_score'] > self.anomaly_threshold
        ]
        filtered_regions.sort(key=lambda x: x['max_anomaly_score'], reverse=True)
        
        return {
            'anomaly_regions': filtered_regions,
            'num_candidates': len(candidate_regions),
        }
    
    def visualize_detection(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        output_path: str = None,
        show_scores: bool = True,
    ) -> np.ndarray:
        """
        可视化检测结果
        
        Parameters:
        -----------
        image : np.ndarray
            原始图像
        result : Dict
            detect_automatic 或 detect_click_guided 的返回结果
        output_path : str
            保存路径
        show_scores : bool
            是否显示分数
            
        Returns:
        --------
        np.ndarray
            可视化图像
        """
        # 转换为彩色
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        if vis_image.dtype != np.uint8:
            vis_image = ((vis_image - vis_image.min()) / (vis_image.max() - vis_image.min() + 1e-8) * 255).astype(np.uint8)
        
        # 绘制异常区域
        colors = [
            (0, 0, 255),    # 红色 - 最高优先级
            (0, 165, 255),  # 橙色
            (0, 255, 255),  # 黄色
            (255, 0, 0),    # 蓝色
            (255, 0, 255),  # 紫色
        ]
        
        for i, region in enumerate(result.get('anomaly_regions', [])):
            color = colors[i % len(colors)]
            bbox = region['bbox']
            
            # 绘制 bbox
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 绘制 mask (半透明)
            if 'mask' in region and region['mask'] is not None:
                mask = region['mask'].astype(np.uint8) * 128
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                vis_image = cv2.addWeighted(vis_image, 0.7, colored_mask, 0.3, 0)
            
            # 显示分数
            if show_scores:
                score = region.get('max_anomaly_score', 0)
                label = f"#{i+1} score={score:.3f}"
                cv2.putText(vis_image, label, (bbox[0], bbox[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 添加统计信息
        info_text = f"Anomalies: {len(result.get('anomaly_regions', []))}"
        cv2.putText(vis_image, info_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image


def load_image(path: str, size: Tuple[int, int] = None) -> np.ndarray:
    """
    加载图像
    
    Parameters:
    -----------
    path : str
        图像路径
    size : Tuple[int, int]
        目标大小 (H, W)
        
    Returns:
    --------
    np.ndarray
        图像数组 [H, W]
    """
    img = Image.open(path).convert('L')  # 灰度
    if size:
        img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(img)


if __name__ == "__main__":
    print("Res-SAM Module Test")
    
    # 测试初始化
    model = ResSAM(
        hidden_size=30,
        window_size=50,
        stride=5,
        anomaly_threshold=0.5,
    )
    print("✓ ResSAM 初始化成功")
