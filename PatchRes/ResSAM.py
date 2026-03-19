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
from tqdm import tqdm
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
        window_size: int = 50,  # 论文默认值
        stride: int = 5,
        spectral_radius: float = 0.9,
        connectivity: float = 0.1,
        anomaly_threshold: float = 0.5,
        region_coarse_threshold: float = 0.5,  # 论文 Eq.(9) 要求单一 β 阈值，与 anomaly_threshold 保持一致
        sam_model_type: str = "vit_b",
        sam_checkpoint: str = None,
        device: str = "cuda",
    ):
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.stride = stride
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        # 论文 Eq.(9) 给出单一预设阈值 β。这里保留旧参数名作为兼容入口，但内部统一为 beta_threshold。
        self.beta_threshold = float(anomaly_threshold)
        self.anomaly_threshold = self.beta_threshold
        self.region_coarse_threshold = self.beta_threshold
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
        
        # NearestNeighbors 搜索器 (用于 patch 级别异常分数)
        self.nn_searcher = None
        
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

        for i, img in enumerate(tqdm(normal_images, desc="Building Feature Bank", unit="img")):
            # 滑动窗口提取 patches
            patches = self._extract_patches(img)
            
            # 2D-ESN 拟合
            features = self._fit_patches(patches)
            all_features.append(features)
        
        # 合并特征
        self.feature_bank = torch.cat(all_features, dim=0)
        self.feature_bank_source = source_info
        
        # 训练异常评分器 - NearestNeighbourScorer.fit 期望 List[np.ndarray]
        # torch.Tensor 需要转换为 numpy 以便 np.concatenate
        feature_bank_np = self.feature_bank.detach().cpu().numpy()
        self.anomaly_scorer.fit([feature_bank_np])
        
        print(f"Feature Bank built: shape={self.feature_bank.shape}, source={source_info}")
        
        return self.feature_bank
    
    def load_feature_bank(self, path: str):
        """加载预存的 Feature Bank"""
        # 显式指定 map_location，避免 feature bank 在保存时绑定到 CUDA，导致 CPU 环境加载失败。
        self.feature_bank = torch.load(path, map_location=self.device)
        # NearestNeighbourScorer.fit 期望 List[np.ndarray]
        feature_bank_np = self.feature_bank.detach().cpu().numpy()
        self.anomaly_scorer.fit([feature_bank_np])
        
        # 构建 NearestNeighbors 搜索器用于 patch 级别异常分数
        from sklearn.neighbors import NearestNeighbors
        self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1)
        self.nn_searcher.fit(feature_bank_np)
        
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
        2D-ESN 拟合 patches 提取特征（论文 Eq.(2)-(3)）
        
        Parameters:
        -----------
        patches : torch.Tensor
            [num_patches, 1, window_size, window_size]
            
        Returns:
        --------
        torch.Tensor
            特征 [num_patches, 2*hidden_size + 1]
            论文定义的动态特征 f = [W_out, b]
        """
        # ESN_2D.forward 期望输入 [batch_size, height, width]
        # 需要去掉 channel 维度
        patches_2d = patches.squeeze(1)  # [num_patches, window_size, window_size]
        if hasattr(self, "_fitting_count") and isinstance(self._fitting_count, int):
            self._fitting_count += int(patches_2d.shape[0])

        batch_size = 32
        feats = []
        with torch.no_grad():
            for start in range(0, int(patches_2d.shape[0]), batch_size):
                end = min(start + batch_size, int(patches_2d.shape[0]))
                feats.append(self.esn.forward(patches_2d[start:end]))
        return torch.cat(feats, dim=0) if feats else torch.zeros((0, 2 * self.hidden_size + 1), dtype=torch.float32)
    
    def detect_automatic(
        self,
        image: np.ndarray,
        min_region_area: int = 500,
        max_regions: int = 10,
        return_all_candidates: bool = False,
    ) -> Dict[str, Any]:
        """
        Fully Automatic 模式异常检测（论文 Page 10-11）
        
        论文流程：
        1. SAM 将整幅 B-scan 分割为 coarse regions
        2. 对每个 coarse region 做 2D-ESN 拟合并与 Feature Bank 比较
           - discard normal-like regions
           - retain potential anomaly regions
        3. 对 retained region 取最小外接矩形作为 candidate region
        4. 对 candidate region 内每个点提取 centered patch
        5. 每个 patch 独立 2D-ESN 拟合 → 特征 f*
        6. f* 与 Feature Bank 比较 → 每个 patch 一个异常分数（Eq.(7)-(9)）
        7. 合并异常 patches → 计算最小包围框 → 最终 bbox
        
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
            - 'num_candidates': 候选区域数量
            - 'num_coarse_discarded': 粗筛阶段丢弃的区域数
            - 'all_candidates': 所有候选区域 (如果 return_all_candidates=True)
        """
        # 确保灰度图
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        self._fitting_count = 0
        
        img_h, img_w = image.shape
        
        # Step 1: SAM 生成 coarse regions
        print("Step 1: SAM generating coarse regions...")
        coarse_regions = self.sam.generate_masks_automatic(
            image,
            min_area_ratio=0.005,
            max_area_ratio=0.5,
        )
        
        print(f"  Found {len(coarse_regions)} coarse regions")
        
        # 过滤有效区域：尺寸必须 >= window_size
        valid_coarse_regions = []
        for region in coarse_regions:
            bbox = region['bbox']
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w >= self.window_size and h >= self.window_size:
                valid_coarse_regions.append(region)
        
        print(f"  Valid coarse regions (>={self.window_size}x{self.window_size}): {len(valid_coarse_regions)}")
        
        # Step 2: Region 级粗筛（论文 Fully Automatic 关键步骤）
        # 对每个 coarse region 做 2D-ESN 拟合并与 Feature Bank 比较
        # discard normal-like; retain potential anomaly regions
        print("Step 2: Region-level coarse filtering...")
        debug_coarse = bool(os.environ.get("RES_SAM_DEBUG_COARSE", "").strip())
        retained_regions = []
        num_discarded = 0
        
        half_win = self.window_size // 2
        
        for region_idx, region in enumerate(valid_coarse_regions):
            bbox = region['bbox']
            mask = region['mask']
            x1, y1, x2, y2 = bbox
            
            # 跳过过小区域
            region_area = (x2 - x1) * (y2 - y1)
            if region_area < min_region_area:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=area_lt_min "
                        f"bbox={bbox} area={region_area} min_region_area={min_region_area}")
                num_discarded += 1
                continue
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=empty_crop "
                        f"bbox={bbox} crop_shape={getattr(crop, 'shape', None)}")
                num_discarded += 1
                continue

            if mask is not None:
                try:
                    crop_mask = mask[y1:y2, x1:x2]
                    if crop_mask is not None and crop_mask.shape == crop.shape:
                        crop = crop * crop_mask.astype(crop.dtype)
                except Exception:
                    pass

            try:
                crop_rs = cv2.resize(
                    crop.astype(np.float32),
                    (int(self.window_size), int(self.window_size)),
                    interpolation=cv2.INTER_LINEAR,
                )
            except Exception:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=resize_failed "
                        f"bbox={bbox} crop_shape={getattr(crop, 'shape', None)} window_size={self.window_size}")
                num_discarded += 1
                continue

            patches_tensor = torch.tensor(np.array([crop_rs]), dtype=torch.float32).unsqueeze(1)
            features = self._fit_patches(patches_tensor)

            # 计算与 Feature Bank 的最近邻距离（region-level score）
            features_np = features.detach().cpu().numpy()
            if self.nn_searcher is None:
                raise RuntimeError(
                    "Feature bank searcher not initialized. Call load_feature_bank() or build_feature_bank() first."
                )
            distances, _ = self.nn_searcher.kneighbors(features_np)

            region_max_score = float(distances.flatten()[0])
            region_mean_score = float(distances.flatten()[0])
            
            # 粗筛阈值：保留异常分数高于阈值的 region
            if region_max_score > self.beta_threshold:
                if debug_coarse:
                    print(
                        f"  [coarse][retain][{region_idx}] bbox={bbox} area={region_area} "
                        f"score={region_max_score:.6f} beta={self.beta_threshold:.6f}")
                retained_regions.append({
                    'bbox': bbox,
                    'mask': mask,
                    'coarse_max_score': region_max_score,
                    'coarse_mean_score': region_mean_score,
                    'stability_score': region.get('stability_score', 0),
                })
            else:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=score_le_beta bbox={bbox} area={region_area} "
                        f"score={region_max_score:.6f} beta={self.beta_threshold:.6f}")
                num_discarded += 1
        
        print(f"  Retained {len(retained_regions)} regions after coarse filtering (discarded {num_discarded})")
        
        # 按粗筛分数排序（高的优先）
        retained_regions.sort(key=lambda r: r['coarse_max_score'], reverse=True)
        
        # Step 3: 对 retained regions 进行精细 patch 级分析
        print("Step 3: Fine-grained patch analysis...")
        anomaly_regions = []
        
        from tqdm import tqdm
        region_iter = tqdm(enumerate(retained_regions[:max_regions]), 
                          total=min(len(retained_regions), max_regions),
                          desc="  Fine analysis", leave=False)
        
        for i, region in region_iter:
            bbox = region['bbox']
            mask = region['mask']
            x1, y1, x2, y2 = bbox
            
            # 收集 region 内所有 patches（密集采样）
            all_patches = []
            patch_positions = []
            
            for y in range(y1 + half_win, y2 - half_win, self.stride):
                for x in range(x1 + half_win, x2 - half_win, self.stride):
                    # 检查是否在 mask 内
                    if mask is not None:
                        try:
                            if not bool(mask[y, x]):
                                continue
                        except Exception:
                            pass
                    
                    # 提取 centered patch
                    patch_y1 = y - half_win
                    patch_y2 = y + half_win
                    patch_x1 = x - half_win
                    patch_x2 = x + half_win
                    
                    if patch_y1 < 0 or patch_y2 > img_h or patch_x1 < 0 or patch_x2 > img_w:
                        continue
                    
                    patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
                    if patch.shape == (self.window_size, self.window_size):
                        all_patches.append(patch)
                        patch_positions.append((x, y))
            
            if len(all_patches) == 0:
                continue
            
            # 批量处理所有 patches
            patches_tensor = torch.tensor(np.array(all_patches), dtype=torch.float32).unsqueeze(1)
            features = self._fit_patches(patches_tensor)
            
            # 计算每个 patch 与 Feature Bank 的最近邻距离（论文 Eq.(7)）
            features_np = features.detach().cpu().numpy()
            if self.nn_searcher is None:
                raise RuntimeError(
                    "Feature bank searcher not initialized. Call load_feature_bank() or build_feature_bank() first."
                )
            distances, _ = self.nn_searcher.kneighbors(features_np)
            scores = distances.flatten()
            
            # 更新进度条
            region_iter.set_postfix({
                'patches': len(all_patches),
                'max_score': f'{scores.max():.2f}'
            })
            
            # Step 4: 合并异常 patches 计算最终 bbox（论文 Eq.(9)）
            # 阈值 β 判别异常 patch
            anomaly_positions = [
                pos for pos, s in zip(patch_positions, scores)
                if s > self.beta_threshold
            ]
            
            if len(anomaly_positions) > 0:
                # 计算异常 patches 的最小包围框
                anomaly_x = [p[0] for p in anomaly_positions]
                anomaly_y = [p[1] for p in anomaly_positions]
                
                final_x1 = max(0, min(anomaly_x) - half_win)
                final_y1 = max(0, min(anomaly_y) - half_win)
                final_x2 = min(img_w, max(anomaly_x) + half_win)
                final_y2 = min(img_h, max(anomaly_y) + half_win)
                
                final_bbox = [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]
                max_score = float(max(scores))
                avg_score = float(np.mean(scores))
                
                anomaly_regions.append({
                    'bbox': final_bbox,
                    'mask': mask,
                    'avg_anomaly_score': avg_score,
                    'max_anomaly_score': max_score,
                    'coarse_max_score': region['coarse_max_score'],
                    'stability_score': region.get('stability_score', 0),
                    'num_anomaly_patches': len(anomaly_positions),
                })
        
        # 按异常分数排序
        anomaly_regions.sort(key=lambda x: x['max_anomaly_score'], reverse=True)
        
        print(f"  Detected {len(anomaly_regions)} anomaly regions")
        
        result = {
            'anomaly_regions': anomaly_regions[:max_regions],
            'num_candidates': len(retained_regions),
            'num_coarse_discarded': num_discarded,
            'num_esn_fits': int(getattr(self, "_fitting_count", 0)),
        }
        
        if return_all_candidates:
            result['all_candidates'] = retained_regions
        
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

        self._fitting_count = 0
        
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

        img_h, img_w = image.shape
        half_win = self.window_size // 2
        
        for region in candidate_regions:
            bbox = region['bbox']
            mask = region['mask']

            # 候选框必须能容纳 patch
            x1, y1, x2, y2 = bbox
            if (x2 - x1) < self.window_size or (y2 - y1) < self.window_size:
                continue

            # 论文 Page 11：candidate region 内每个点为中心提取 patch（越界排除）
            # 实现上按 stride 采样中心点以控制计算量（stride=5 与论文默认一致）。
            all_patches = []
            patch_positions = []

            for y in range(y1 + half_win, y2 - half_win, self.stride):
                for x in range(x1 + half_win, x2 - half_win, self.stride):
                    if mask is not None:
                        try:
                            if not bool(mask[y, x]):
                                continue
                        except Exception:
                            pass

                    patch_y1 = y - half_win
                    patch_y2 = y + half_win
                    patch_x1 = x - half_win
                    patch_x2 = x + half_win

                    if patch_y1 < 0 or patch_y2 > img_h or patch_x1 < 0 or patch_x2 > img_w:
                        continue

                    patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
                    if patch.shape == (self.window_size, self.window_size):
                        all_patches.append(patch)
                        patch_positions.append((x, y))

            if len(all_patches) == 0:
                continue

            patches_tensor = torch.tensor(np.array(all_patches), dtype=torch.float32).unsqueeze(1)
            features = self._fit_patches(patches_tensor)

            # 论文 Eq.(7)-(8)：s(f*) = min_{f in M} L2(f*, f)
            features_np = features.detach().cpu().numpy()
            if self.nn_searcher is None:
                raise RuntimeError("Feature bank searcher not initialized. Call load_feature_bank() or build_feature_bank() first.")
            distances, _ = self.nn_searcher.kneighbors(features_np)
            scores = distances.flatten()

            # 论文 Eq.(9)：s(f*) > β 判为 abnormal
            anomaly_positions = [
                pos for pos, s in zip(patch_positions, scores)
                if s > self.beta_threshold
            ]

            if len(anomaly_positions) == 0:
                continue

            anomaly_x = [p[0] for p in anomaly_positions]
            anomaly_y = [p[1] for p in anomaly_positions]

            final_x1 = max(0, min(anomaly_x) - half_win)
            final_y1 = max(0, min(anomaly_y) - half_win)
            final_x2 = min(img_w, max(anomaly_x) + half_win)
            final_y2 = min(img_h, max(anomaly_y) + half_win)

            final_bbox = [int(final_x1), int(final_y1), int(final_x2), int(final_y2)]
            avg_score = float(np.mean(scores))
            max_score = float(np.max(scores))

            anomaly_regions.append({
                'bbox': final_bbox,
                'mask': mask,
                'avg_anomaly_score': avg_score,
                'max_anomaly_score': max_score,
                'sam_score': region.get('score', 0),
                'num_anomaly_patches': len(anomaly_positions),
            })
        
        # 过滤
        filtered_regions = [
            r for r in anomaly_regions 
            if r['max_anomaly_score'] > self.beta_threshold
        ]
        filtered_regions.sort(key=lambda x: x['max_anomaly_score'], reverse=True)
        
        return {
            'anomaly_regions': filtered_regions,
            'num_candidates': len(candidate_regions),
            'num_esn_fits': int(getattr(self, "_fitting_count", 0)),
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
