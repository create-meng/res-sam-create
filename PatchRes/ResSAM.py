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

当前仓库主线说明：
- 当前 experiments 主线只保留 fully automatic + clustering。
- click-guided 方案已从当前仓库主线移除。
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union, Sequence
import logging
import os
import sys
from PIL import Image
import cv2

from .ESN_2D_nobatch import ESN_2D
from .common import NearestNeighbourScorer
from .sam_integration import SAMIntegration

# Feature bank on-disk bundle: tensor + 2D-ESN weights (author minimal code only saves tensors).
FEATURE_BANK_BUNDLE_FORMAT = "res_sam_fb_v2"

def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("1", "true", "yes", "y", "on")


def _strict_faiss_default() -> bool:
    """
    Default is STRICT: require faiss unless explicitly allowing sklearn fallback.

    - RES_SAM_ALLOW_SKLEARN_KNN=1: allow sklearn fallback (non-official path)
    - RES_SAM_REQUIRE_FAISS=1: force strict even if other configs are present
    """
    if _env_flag("RES_SAM_REQUIRE_FAISS"):
        return True
    if _env_flag("RES_SAM_ALLOW_SKLEARN_KNN"):
        return False
    return True


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
        ESN 谱半径。当前实现默认 0.9。
        该值属于实现级 ESN 超参数，现阶段未在已核对的论文正文段落中看到明确给值，
        因此不要将其误记为论文明示参数。
    connectivity : float
        ESN 连接率。当前实现默认 0.1。
        该值同样属于实现级 ESN 超参数，现阶段未在已核对的论文正文段落中看到明确给值。
    beta_threshold : float
        论文 Eq.(9) 的单一阈值 β（与最近邻距离 ``s(f*)`` 比较）。
        与作者仓库 ``PatchRes`` 里 historically 命名的 ``anomaly_threshold`` 是同一物理量；本类对外推荐关键字 ``beta_threshold``。
    sam_model_type : str
        SAM 模型类型: "vit_l"（本仓库主线与作者公开仓库默认）、"vit_b"（显存紧张时可选）、"vit_h"
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
        beta_threshold: float = 0.1,
        sam_model_type: str = "vit_l",
        sam_checkpoint: str = None,
        device: str = "auto",
        *,
        anomaly_threshold: Optional[float] = None,
        region_coarse_threshold: Optional[float] = None,
        feature_with_bias: bool = True,
        automatic_fine_use_mask: bool = False,
        automatic_coarse_use_patch_aggregation: bool = False,
        merge_all_anomaly_patches: bool = False,
        coarse_beta_threshold: Optional[float] = None,
    ):
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.stride = stride
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.feature_with_bias = bool(feature_with_bias)
        self.automatic_fine_use_mask = bool(automatic_fine_use_mask)
        self.automatic_coarse_use_patch_aggregation = bool(automatic_coarse_use_patch_aggregation)
        # V10: 将所有异常patch合并为一个最小外接矩形，而不是按连通分量分割成多个小框
        # 论文原文："overlapping anomaly patches are consolidated into the smallest possible rectangular box"
        # 论文未明确要求按连通分量分割，合并成一个框同样符合论文描述
        self.merge_all_anomaly_patches = bool(merge_all_anomaly_patches)
        # V11: 粗筛和细筛可以用不同的beta
        # coarse_beta_threshold=None 时退化为与细筛相同的beta（向后兼容）
        self.coarse_beta_threshold = float(coarse_beta_threshold) if coarse_beta_threshold is not None else None
        # Eq.(9) β：单一阈值。推荐只传 beta_threshold；anomaly_threshold / region_coarse_threshold 仅为旧调用兼容。
        if anomaly_threshold is not None and region_coarse_threshold is not None:
            if float(anomaly_threshold) != float(region_coarse_threshold):
                raise ValueError(
                    "anomaly_threshold and region_coarse_threshold disagree; use beta_threshold only (single β)."
                )
        if anomaly_threshold is not None:
            _beta = float(anomaly_threshold)
        elif region_coarse_threshold is not None:
            _beta = float(region_coarse_threshold)
        else:
            _beta = float(beta_threshold)
        self.beta_threshold = _beta
        self.anomaly_threshold = self.beta_threshold  # 与 PatchRes 命名兼容，即 β
        self.region_coarse_threshold = self.beta_threshold
        _device_in = (device if device is not None else "").strip().lower() if isinstance(device, str) else device
        if _device_in in {"", None, "auto"}:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif _device_in in {"cuda", "cpu"}:
            self.device = str(_device_in)
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 初始化 2D-ESN
        try:
            _esn_device = torch.device(self.device)
        except Exception:
            _esn_device = torch.device("cpu")

        self.esn = ESN_2D(
            input_dim=1,
            n_reservoir=hidden_size,
            alpha=5,
            spectral_radius=(spectral_radius, spectral_radius),
            connectivity=connectivity,
            device_override=_esn_device,
        )
        
        # 初始化异常评分器
        self.anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours=1)
        
        # NearestNeighbors 搜索器 (用于 patch 级别异常分数)
        self.nn_searcher = None
        self.nn_backend = "sklearn"
        
        # 初始化 SAM
        self.sam = SAMIntegration(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint,
            device=self.device,
        )
        
        # Feature Bank
        self.feature_bank = None
        self.feature_bank_source = None  # 记录来源

        self._beta_calibrated = False

    def _calibrate_beta_threshold(self, feature_bank_np: np.ndarray) -> float:
        logger = logging.getLogger(__name__)

        if feature_bank_np is None or (not hasattr(feature_bank_np, "shape")) or len(feature_bank_np.shape) != 2:
            raise ValueError("Invalid feature bank array for beta calibration")
        if int(feature_bank_np.shape[0]) < 3:
            raise ValueError("Feature bank too small for beta calibration")

        quantile = 0.995

        k = 2
        nn_dists = None

        try:
            import faiss  # type: ignore

            xb = np.ascontiguousarray(feature_bank_np.astype(np.float32, copy=False))
            index = faiss.IndexFlatL2(int(xb.shape[1]))
            index.add(xb)
            dists, _ = index.search(xb, k)
            nn_dists = np.sqrt(np.maximum(dists[:, 1], 0.0))
        except Exception:
            if _strict_faiss_default():
                raise RuntimeError(
                    "Failed to calibrate beta threshold because faiss is unavailable (strict-faiss default). "
                    "Set RES_SAM_ALLOW_SKLEARN_KNN=1 to permit sklearn fallback."
                )
            try:
                from sklearn.neighbors import NearestNeighbors

                nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", n_jobs=-1)
                nn.fit(feature_bank_np)
                dists, _ = nn.kneighbors(feature_bank_np, n_neighbors=k)
                nn_dists = dists[:, 1]
            except Exception as e:
                raise RuntimeError(f"Failed to calibrate beta threshold (sklearn fallback): {e}")

        nn_dists = np.asarray(nn_dists, dtype=np.float64)
        nn_dists = nn_dists[np.isfinite(nn_dists)]
        if nn_dists.size == 0:
            raise RuntimeError("Beta calibration failed: empty nearest-neighbour distance distribution")

        beta = float(np.quantile(nn_dists, quantile))
        if not np.isfinite(beta):
            raise RuntimeError("Beta calibration failed: non-finite beta")

        self.beta_threshold = beta
        self.anomaly_threshold = beta
        self.region_coarse_threshold = beta
        self._beta_calibrated = True

        try:
            logger.info(
                "[BETA] calibrated beta_threshold=%s quantile=%s nn_dist_stats={min:%s,p50:%s,p90:%s,p99:%s,max:%s}",
                beta,
                quantile,
                float(np.min(nn_dists)),
                float(np.quantile(nn_dists, 0.5)),
                float(np.quantile(nn_dists, 0.9)),
                float(np.quantile(nn_dists, 0.99)),
                float(np.max(nn_dists)),
            )
        except Exception:
            pass

        return beta

    def build_feature_bank(
        self,
        normal_images: Union[np.ndarray, Sequence[np.ndarray]],
        source_info: str = "unknown",
    ) -> torch.Tensor:
        """
        构建 Feature Bank
        
        Parameters:
        -----------
        normal_images : np.ndarray | sequence of np.ndarray
            正常样本图像 [N, H, W] / [N, C, H, W]，或与之一致的二维图像数组列表（可变量化 H、W）
        source_info : str
            数据来源信息（用于环境一致性检查）
            
        Returns:
        --------
        torch.Tensor
            Feature Bank [num_patches, hidden_size^2]
        """
        if isinstance(normal_images, np.ndarray):
            arr = normal_images
            if len(arr.shape) == 4:
                arr = arr.squeeze(1)
            iter_images: List[np.ndarray] = [arr[i] for i in range(len(arr))]
        else:
            iter_images = [np.asarray(x) for x in normal_images]

        print(f"Building Feature Bank from {len(iter_images)} normal images...")

        # 提取特征
        all_features = []

        for i, img in enumerate(tqdm(iter_images, desc="Building Feature Bank", unit="img")):
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

        self._init_nn_searcher(feature_bank_np)
        
        print(f"Feature Bank built: shape={self.feature_bank.shape}, source={source_info}")
        
        return self.feature_bank

    def load_feature_bank(self, path: str):
        """加载预存的 Feature Bank（支持 v2 捆绑包：特征 + ESN state_dict；兼容旧版仅张量）。"""
        # 先加载到 CPU，再把张量迁到目标设备；ESN 权重通过 load_state_dict 写入当前模块。
        try:
            raw = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(path, map_location="cpu")

        if isinstance(raw, dict) and raw.get("format") == FEATURE_BANK_BUNDLE_FORMAT:
            fb = raw.get("feature_bank")
            if not isinstance(fb, torch.Tensor):
                raise ValueError("Invalid feature bank bundle: missing feature_bank tensor")
            self.feature_bank = fb.to(self.device)
            esd = raw.get("esn_state_dict")
            if not isinstance(esd, dict):
                raise ValueError("Invalid feature bank bundle: missing esn_state_dict")
            try:
                self.esn.load_state_dict(esd, strict=True)
            except RuntimeError as e:
                # Compatibility fallback for old/invalid bundles where ESN fixed reservoir
                # weights were not serialized (e.g. W_in/W_res_1/W_res_2 missing).
                missing_keys = [k for k in ("W_in", "W_res_1", "W_res_2") if k not in esd]
                if not missing_keys:
                    raise
                logger_fb = logging.getLogger(__name__)
                msg = (
                    "[FeatureBank] Bundle ESN state_dict missing keys %s; "
                    "fallback to current-runtime ESN (legacy behavior). "
                    "For deterministic cross-platform inference, rebuild feature bank with latest code."
                    % str(missing_keys)
                )
                print(msg, flush=True)
                try:
                    logger_fb.warning(msg)
                    logger_fb.warning("Original load_state_dict error: %s", str(e))
                except Exception:
                    pass
        elif isinstance(raw, torch.Tensor):
            self.feature_bank = raw.to(self.device)
            logger_fb = logging.getLogger(__name__)
            msg = (
                "[FeatureBank] Legacy tensor-only file (no ESN weights in file). "
                "Inference relies on ESN init matching build-time RNG; re-run 01 to emit %s bundle."
                % FEATURE_BANK_BUNDLE_FORMAT
            )
            print(msg, flush=True)
            try:
                logger_fb.warning(msg)
            except Exception:
                pass
        else:
            raise ValueError(
                "Unrecognized feature bank file: expected Tensor or bundle with format=%s"
                % FEATURE_BANK_BUNDLE_FORMAT
            )

        # NearestNeighbourScorer.fit 期望 List[np.ndarray]
        feature_bank_np = np.ascontiguousarray(self.feature_bank.detach().cpu().numpy(), dtype=np.float32)
        self.anomaly_scorer.fit([feature_bank_np])
         
        self._init_nn_searcher(feature_bank_np)
        try:
            logger = logging.getLogger(__name__)
            nn_backend = str(getattr(self, "nn_backend", "unknown"))
            msg = (
                "[INIT] device=%s nn_backend=%s feature_bank_shape=%s feature_dim=%s"
                % (
                    str(getattr(self, "device", "unknown")),
                    nn_backend,
                    str(tuple(getattr(self.feature_bank, "shape", ()))),
                    str(int(feature_bank_np.shape[1]) if (hasattr(feature_bank_np, "shape") and len(feature_bank_np.shape) > 1) else "unknown"),
                )
            )
            print(msg, flush=True)
            logger.info(msg)
        except Exception:
            pass
        
        print(f"Feature Bank loaded: shape={self.feature_bank.shape}")

    def _init_nn_searcher(self, feature_bank_np: np.ndarray):
        strict_faiss = _strict_faiss_default()
        backend_env = (os.environ.get("RES_SAM_KNN_BACKEND", "") or "").strip().lower()
        backend = backend_env
        if backend and backend not in {"sklearn", "faiss_cpu", "faiss_gpu"}:
            backend = ""

        if backend == "sklearn":
            if strict_faiss:
                raise RuntimeError(
                    "Strict-faiss default: do not use RES_SAM_KNN_BACKEND=sklearn. "
                    "Set RES_SAM_ALLOW_SKLEARN_KNN=1 if you really want sklearn fallback."
                )
            from sklearn.neighbors import NearestNeighbors
            xb = np.ascontiguousarray(feature_bank_np, dtype=np.float32)
            self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
            self.nn_searcher.fit(xb)
            self.nn_backend = "sklearn"
            return

        # Default acceleration: if backend is not explicitly specified, prefer Faiss.
        # If CUDA is available, try faiss_gpu first then fall back to faiss_cpu/sklearn.
        if not backend:
            backend = "faiss_gpu" if (self.device == "cuda" and torch.cuda.is_available()) else "faiss_cpu"

        if backend == "sklearn":
            from sklearn.neighbors import NearestNeighbors
            xb = np.ascontiguousarray(feature_bank_np, dtype=np.float32)
            self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
            self.nn_searcher.fit(xb)
            self.nn_backend = "sklearn"
            return

        try:
            import faiss
        except Exception:
            if strict_faiss:
                raise ImportError(
                    "Strict-faiss default but faiss could not be imported in ResSAM._init_nn_searcher. "
                    "Fix faiss install, or set RES_SAM_ALLOW_SKLEARN_KNN=1 to permit sklearn fallback."
                ) from None
            backend = "sklearn"
            from sklearn.neighbors import NearestNeighbors
            xb = np.ascontiguousarray(feature_bank_np, dtype=np.float32)
            self.nn_searcher = NearestNeighbors(n_neighbors=1, algorithm="brute", n_jobs=1)
            self.nn_searcher.fit(xb)
            self.nn_backend = "sklearn"
            return

        xb = np.ascontiguousarray(feature_bank_np.astype(np.float32, copy=False))
        index = faiss.IndexFlatL2(int(xb.shape[1]))
        if backend == "faiss_gpu":
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
            except Exception:
                backend = "faiss_cpu"
        index.add(xb)
        self.nn_searcher = index
        self.nn_backend = backend
    
    def save_feature_bank(self, path: str):
        """保存 Feature Bank：v2 捆绑包含 2D-ESN 权重，推理时与建库阶段同一水库（对齐官方仅存向量的不足）。"""
        bundle = {
            "format": FEATURE_BANK_BUNDLE_FORMAT,
            "feature_bank": self.feature_bank.detach().cpu(),
            "esn_state_dict": {k: v.detach().cpu() for k, v in self.esn.state_dict().items()},
        }
        torch.save(bundle, path)
        print(f"Feature Bank bundle ({FEATURE_BANK_BUNDLE_FORMAT}) saved to {path}")
    
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
        win = int(self.window_size)
        stride = int(self.stride)

        if h < win or w < win:
            return torch.zeros((0, 1, win, win), dtype=torch.float32)

        try:
            view = np.lib.stride_tricks.sliding_window_view(image, (win, win))
            patches_np = view[::stride, ::stride].reshape(-1, win, win)
        except Exception:
            patches = []
            for i in range(0, h - win + 1, stride):
                for j in range(0, w - win + 1, stride):
                    patches.append(image[i : i + win, j : j + win])
            patches_np = np.asarray(patches)

        patches_np = np.ascontiguousarray(patches_np, dtype=np.float32)
        return torch.from_numpy(patches_np).unsqueeze(1)
    
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
            特征张量。论文主线默认形状为 [num_patches, 2*hidden_size+1]，
            对应 Eq.(3)/(8) 中真实求解的 f=[W_out,b]。
            若显式关闭 feature_with_bias，则退化为官方兼容口径 [W_out]。
        """
        patches_2d = patches.squeeze(1)  # [num_patches, window_size, window_size]
        if self.device == "cuda" and torch.cuda.is_available() and patches_2d.device.type != "cuda":
            patches_2d = patches_2d.to(device="cuda", dtype=torch.float32, non_blocking=True)
        elif self.device == "cpu" and patches_2d.device.type != "cpu":
            patches_2d = patches_2d.to(device="cpu", dtype=torch.float32)
        if hasattr(self, "_fitting_count") and isinstance(self._fitting_count, int):
            self._fitting_count += int(patches_2d.shape[0])

        # Default batch sizing: larger batch on GPU for throughput; keep CPU default small.
        batch_size = 32
        if self.device == "cuda" and torch.cuda.is_available():
            batch_size = 128
            try:
                total_mem_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
                if total_mem_gb >= 14:
                    batch_size = 512
                elif total_mem_gb >= 10:
                    batch_size = 256
            except Exception:
                batch_size = 128
        try:
            env_bs = os.environ.get("RES_SAM_ESN_BATCH", "").strip()
            if env_bs:
                batch_size = max(1, int(env_bs))
        except Exception:
            batch_size = batch_size
        feats = []
        with torch.inference_mode():
            for start in range(0, int(patches_2d.shape[0]), batch_size):
                end = min(start + batch_size, int(patches_2d.shape[0]))
                feats.append(self.esn.forward(patches_2d[start:end]))
        if feats:
            features = torch.cat(feats, dim=0)
        else:
            base_dim = 2 * self.hidden_size + 1
            features = torch.zeros((0, base_dim), dtype=torch.float32)
        # Paper-first mainline: ESN already solves f=[W_out, b]. Only trim the
        # last bias term when an explicit official-compatibility fallback is requested.
        if (not self.feature_with_bias) and int(features.shape[1]) == (2 * self.hidden_size + 1):
            features = features[:, :-1]
        return features

    def _patch_list_to_tensor(self, patches_list: List[np.ndarray]) -> torch.Tensor:
        if isinstance(patches_list, np.ndarray):
            arr = patches_list
            if arr.size == 0 or arr.shape[0] == 0:
                return torch.zeros((0, 1, int(self.window_size), int(self.window_size)), dtype=torch.float32)
        else:
            if not patches_list:
                return torch.zeros((0, 1, int(self.window_size), int(self.window_size)), dtype=torch.float32)
            arr = np.stack(patches_list, axis=0)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return torch.from_numpy(arr).unsqueeze(1)

    def _score_features_against_bank(self, features_np: np.ndarray) -> np.ndarray:
        if self.nn_searcher is None:
            raise RuntimeError(
                "Feature bank searcher not initialized. Call load_feature_bank() or build_feature_bank() first."
            )

        if features_np is None or getattr(features_np, "size", 0) == 0:
            return np.zeros((0,), dtype=np.float32)

        if getattr(self, "nn_backend", "sklearn") == "sklearn":
            distances, _ = self.nn_searcher.kneighbors(features_np)
            return distances.reshape(-1).astype(np.float32, copy=False)

        import numpy as _np

        xq = _np.ascontiguousarray(features_np.astype(_np.float32, copy=False))
        dists, _ = self.nn_searcher.search(xq, 1)
        return _np.sqrt(_np.maximum(dists.reshape(-1), 0.0)).astype(_np.float32, copy=False)

    def _collect_candidate_patches(
        self,
        image: np.ndarray,
        bbox: List[int],
        mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        img_h, img_w = image.shape
        win = int(self.window_size)
        half_win = win // 2
        x1, y1, x2, y2 = bbox

        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, win, win), dtype=np.float32), []

        # Paper wording is "each point in the candidate region serves as the center"
        # and only patches extending beyond image boundaries are excluded.
        y_centers = np.arange(y1, y2, 1, dtype=np.int32)
        x_centers = np.arange(x1, x2, 1, dtype=np.int32)
        if y_centers.size == 0 or x_centers.size == 0:
            return np.zeros((0, win, win), dtype=np.float32), []

        yy, xx = np.meshgrid(y_centers, x_centers, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)
        y_tops = yy - half_win
        x_lefts = xx - half_win

        valid = (y_tops >= 0) & ((y_tops + win) <= img_h) & (x_lefts >= 0) & ((x_lefts + win) <= img_w)
        if not np.any(valid):
            return np.zeros((0, win, win), dtype=np.float32), []

        y_tops = y_tops[valid]
        x_lefts = x_lefts[valid]
        yy = yy[valid]
        xx = xx[valid]

        view = np.lib.stride_tricks.sliding_window_view(image, (win, win))
        patches_np = np.ascontiguousarray(view[y_tops, x_lefts], dtype=np.float32)
        patch_positions = list(zip(xx.tolist(), yy.tolist()))
        return patches_np, patch_positions

    def _extract_region_feature(self, image: np.ndarray, bbox: List[int], mask: Optional[np.ndarray]) -> torch.Tensor:
        """
        Fit a retained SAM coarse region itself for automatic coarse filtering.

        Paper order:
        1. fit each coarse region with 2D-ESN;
        2. retain anomaly-like regions;
        3. then convert retained regions to enclosing rectangles.

        We therefore fit the irregular SAM mask directly through the masked
        2D-ESN path instead of approximating it by a zero-filled rectangle.
        """
        base_dim = 2 * self.hidden_size + (1 if self.feature_with_bias else 0)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return torch.zeros((0, base_dim), dtype=torch.float32)

        crop = np.array(image[y1:y2, x1:x2], dtype=np.float32, copy=True)
        if crop.size == 0:
            return torch.zeros((0, base_dim), dtype=torch.float32)

        if mask is None:
            crop = np.ascontiguousarray(crop, dtype=np.float32)
            crop_tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0)
            return self._fit_patches(crop_tensor)

        mask_crop = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
        if mask_crop.shape != crop.shape or (not np.any(mask_crop)):
            return torch.zeros((0, base_dim), dtype=torch.float32)

        crop_tensor = torch.from_numpy(np.ascontiguousarray(crop, dtype=np.float32)).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask_crop, dtype=np.bool_)).unsqueeze(0)
        if self.device == "cuda" and torch.cuda.is_available():
            crop_tensor = crop_tensor.to(device="cuda", dtype=torch.float32, non_blocking=True)
            mask_tensor = mask_tensor.to(device="cuda", non_blocking=True)
        elif self.device == "cpu":
            crop_tensor = crop_tensor.to(device="cpu", dtype=torch.float32)
            mask_tensor = mask_tensor.to(device="cpu")

        if hasattr(self, "_fitting_count") and isinstance(self._fitting_count, int):
            self._fitting_count += 1

        with torch.inference_mode():
            features = self.esn.forward_masked(crop_tensor, mask_tensor)
        if (not self.feature_with_bias) and int(features.shape[1]) == (2 * self.hidden_size + 1):
            features = features[:, :-1]
        return features

    def _merge_overlapping_anomaly_patches(
        self,
        image_shape: Tuple[int, int],
        anomaly_positions: List[Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        """
        Merge anomaly patches by overlap connectivity within one candidate region.

        Paper text specifies that overlapping anomaly-relevant patches are
        merged. Therefore we first convert each anomaly-centered patch into its
        patch box, group boxes by overlap connectivity, and then take the
        smallest enclosing rectangle for each connected overlap group.
        """
        if not anomaly_positions:
            return []

        img_h, img_w = image_shape
        half_win = int(self.window_size) // 2
        patch_boxes: List[Tuple[int, int, int, int]] = []
        for cx, cy in anomaly_positions:
            x1 = max(0, int(cx) - half_win)
            y1 = max(0, int(cy) - half_win)
            x2 = min(img_w, int(cx) + half_win)
            y2 = min(img_h, int(cy) + half_win)
            if x2 <= x1 or y2 <= y1:
                continue
            patch_boxes.append((x1, y1, x2, y2))

        if not patch_boxes:
            return []

        def _boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
            return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])

        n = len(patch_boxes)
        visited = [False] * n
        merged_regions: List[Dict[str, Any]] = []

        for start_idx in range(n):
            if visited[start_idx]:
                continue

            stack = [start_idx]
            visited[start_idx] = True
            component_indices: List[int] = []

            while stack:
                idx = stack.pop()
                component_indices.append(idx)
                box_i = patch_boxes[idx]
                for j in range(n):
                    if visited[j]:
                        continue
                    if _boxes_overlap(box_i, patch_boxes[j]):
                        visited[j] = True
                        stack.append(j)

            component_boxes = [patch_boxes[idx] for idx in component_indices]
            x1_all = min(box[0] for box in component_boxes)
            y1_all = min(box[1] for box in component_boxes)
            x2_all = max(box[2] for box in component_boxes)
            y2_all = max(box[3] for box in component_boxes)
            if x2_all <= x1_all or y2_all <= y1_all:
                continue
            merged_regions.append(
                {
                    "bbox": [int(x1_all), int(y1_all), int(x2_all), int(y2_all)],
                    "num_anomaly_patches": int(len(component_boxes)),
                }
            )

        return merged_regions
    
    def detect_automatic(
        self,
        image: np.ndarray,
        min_region_area: Optional[int] = None,
        max_regions: Optional[int] = None,
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
            最小区域面积。
            这是可选工程保护阈值；若为 None，则不按面积额外丢弃 coarse region，
            更接近论文只按特征判别保留/丢弃区域的描述。
        max_regions : int
            最大返回区域数。
            这是可选工程截断；若为 None，则不对 retained/final regions 施加固定上限。
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
        coarse_regions = self.sam.generate_masks_automatic(image)
        
        print(f"  Found {len(coarse_regions)} coarse regions")
        
        # Step 2: Region 级粗筛（论文 Fully Automatic 关键步骤）
        # 对每个 coarse region 做 2D-ESN 拟合并与 Feature Bank 比较
        # discard normal-like; retain potential anomaly regions
        print("Step 2: Region-level coarse filtering...")
        debug_coarse = bool(os.environ.get("RES_SAM_DEBUG_COARSE", "").strip())
        retained_regions = []
        num_discarded = 0
        
        for region_idx, region in enumerate(coarse_regions):
            bbox = region['bbox']
            mask = region['mask']
            x1, y1, x2, y2 = bbox
            
            # 跳过过小区域
            region_area = (x2 - x1) * (y2 - y1)
            if min_region_area is not None and region_area < min_region_area:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=area_lt_min "
                        f"bbox={bbox} area={region_area} min_region_area={min_region_area}")
                num_discarded += 1
                continue
            
            if self.automatic_coarse_use_patch_aggregation:
                patches_np, patch_positions = self._collect_candidate_patches(image, bbox, mask)
                if patches_np.size == 0 or len(patch_positions) == 0:
                    if debug_coarse:
                        print(
                            f"  [coarse][discard][{region_idx}] reason=no_valid_region_patches "
                            f"bbox={bbox}")
                    num_discarded += 1
                    continue
                patches_tensor = self._patch_list_to_tensor(patches_np)
                features = self._fit_patches(patches_tensor)
                features_np = features.detach().cpu().numpy()
                region_scores = self._score_features_against_bank(features_np)
                region_num_features = int(region_scores.shape[0])
            else:
                # Paper-first automatic mode: discriminate the SAM region itself
                # before converting it to a candidate rectangle.
                region_feature = self._extract_region_feature(image, bbox, mask)
                region_feature_np = region_feature.detach().cpu().numpy()
                region_scores = self._score_features_against_bank(region_feature_np)
                if region_scores.size == 0:
                    if debug_coarse:
                        print(
                            f"  [coarse][discard][{region_idx}] reason=no_region_score "
                            f"bbox={bbox}")
                    num_discarded += 1
                    continue
                region_num_features = int(region_scores.shape[0])

            if region_scores.size == 0:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=no_region_score "
                        f"bbox={bbox}")
                num_discarded += 1
                continue
            region_max_score = float(np.max(region_scores))
            region_mean_score = float(np.mean(region_scores))

            # Retain coarse region when its region-level feature exceeds beta.
            # V11: 粗筛用独立的 coarse_beta_threshold（若设置），否则退化为 beta_threshold
            _coarse_beta = self.coarse_beta_threshold if self.coarse_beta_threshold is not None else self.beta_threshold
            if region_max_score > _coarse_beta:
                if debug_coarse:
                    print(
                        f"  [coarse][retain][{region_idx}] bbox={bbox} area={region_area} "
                        f"max_score={region_max_score:.6f} mean_score={region_mean_score:.6f} "
                        f"num_region_features={region_num_features} coarse_beta={_coarse_beta:.6f}")
                retained_regions.append({
                    'bbox': bbox,
                    'mask': mask,
                    'coarse_max_score': region_max_score,
                    'coarse_mean_score': region_mean_score,
                    'num_region_features': region_num_features,
                    'stability_score': region.get('stability_score', 0),
                })
            else:
                if debug_coarse:
                    print(
                        f"  [coarse][discard][{region_idx}] reason=score_le_beta bbox={bbox} area={region_area} "
                        f"max_score={region_max_score:.6f} mean_score={region_mean_score:.6f} "
                        f"num_region_features={region_num_features} coarse_beta={_coarse_beta:.6f}")
                num_discarded += 1
        print(f"  Retained {len(retained_regions)} regions after coarse filtering (discarded {num_discarded})")
        
        # 按粗筛分数排序（高的优先）
        retained_regions.sort(key=lambda r: r['coarse_max_score'], reverse=True)
        
        # Step 3: 对 retained regions 进行精细 patch 级分析
        print("Step 3: Fine-grained patch analysis...")
        anomaly_regions = []
        
        from tqdm import tqdm
        regions_for_fine = retained_regions if max_regions is None else retained_regions[:max_regions]
        region_iter = tqdm(
            enumerate(regions_for_fine),
            total=len(regions_for_fine),
            desc="  Fine analysis",
            leave=False,
        )
        
        for i, region in region_iter:
            bbox = region['bbox']
            mask = region['mask']

            fine_mask = mask if self.automatic_fine_use_mask else None
            patches_np, patch_positions = self._collect_candidate_patches(
                image, bbox, fine_mask
            )
            if patches_np.size == 0 or len(patch_positions) == 0:
                continue
            patches_tensor = self._patch_list_to_tensor(patches_np)
            features = self._fit_patches(patches_tensor)
            features_np = features.detach().cpu().numpy()
            scores = self._score_features_against_bank(features_np)

            if len(patch_positions) == 0:
                continue
            
            region_iter.set_postfix({
                'patches': len(patch_positions),
                'max_score': f'{scores.max():.2f}'
            })
            
            # Step 4: 合并异常 patches 计算最终 bbox（论文 Eq.(9)）
            # 阈值 β 判别异常 patch
            anomaly_positions = [
                pos for pos, s in zip(patch_positions, scores)
                if s > self.beta_threshold
            ]
            
            if len(anomaly_positions) > 0:
                if self.merge_all_anomaly_patches:
                    # V10: 所有异常patch合并为一个最小外接矩形
                    # 论文："overlapping anomaly patches are consolidated into the smallest possible rectangular box"
                    half_win = int(self.window_size) // 2
                    all_x1 = min(max(0, int(cx) - half_win) for cx, cy in anomaly_positions)
                    all_y1 = min(max(0, int(cy) - half_win) for cx, cy in anomaly_positions)
                    all_x2 = max(min(img_w, int(cx) + half_win) for cx, cy in anomaly_positions)
                    all_y2 = max(min(img_h, int(cy) + half_win) for cx, cy in anomaly_positions)
                    if all_x2 > all_x1 and all_y2 > all_y1:
                        all_scores = [float(s) for s in scores if s > self.beta_threshold]
                        anomaly_regions.append({
                            'bbox': [int(all_x1), int(all_y1), int(all_x2), int(all_y2)],
                            'mask': mask,
                            'avg_anomaly_score': float(np.mean(all_scores)),
                            'max_anomaly_score': float(np.max(all_scores)),
                            'coarse_max_score': region['coarse_max_score'],
                            'stability_score': region.get('stability_score', 0),
                            'num_anomaly_patches': int(len(anomaly_positions)),
                        })
                else:
                    # 原有逻辑：按连通分量分割
                    component_regions = self._merge_overlapping_anomaly_patches(
                        (img_h, img_w),
                        anomaly_positions,
                    )
                    
                    for component in component_regions:
                        final_bbox = component["bbox"]
                        x1, y1, x2, y2 = final_bbox
                        component_scores = [
                            float(s)
                            for (px, py), s in zip(patch_positions, scores)
                            if (s > self.beta_threshold) and (x1 <= px < x2) and (y1 <= py < y2)
                        ]
                        if not component_scores:
                            continue

                        anomaly_regions.append({
                            'bbox': final_bbox,
                            'mask': mask,
                            'avg_anomaly_score': float(np.mean(component_scores)),
                            'max_anomaly_score': float(np.max(component_scores)),
                            'coarse_max_score': region['coarse_max_score'],
                            'stability_score': region.get('stability_score', 0),
                            'num_anomaly_patches': int(component['num_anomaly_patches']),
                        })
        
        # 按异常分数排序
        anomaly_regions.sort(key=lambda x: x['max_anomaly_score'], reverse=True)
        
        print(f"  Detected {len(anomaly_regions)} anomaly regions")
        
        result = {
            'anomaly_regions': anomaly_regions if max_regions is None else anomaly_regions[:max_regions],
            'num_candidates': len(retained_regions),
            'num_coarse_discarded': num_discarded,
            'num_esn_fits': int(getattr(self, "_fitting_count", 0)),
        }
        
        if return_all_candidates:
            result['all_candidates'] = retained_regions
        
        return result
    
    def _removed_click_guided_mode(
        self,
        image: np.ndarray,
        positive_points: List[Tuple[int, int]] = None,
        negative_points: List[Tuple[int, int]] = None,
        box: List[int] = None,
        image_already_prepared: bool = False,
    ) -> Dict[str, Any]:
        """
        Click-guided 模式已移除。
        
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
        raise NotImplementedError(
            "Click-guided mode has been removed from this repository. Use detect_automatic() instead."
        )
    
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
            detect_automatic 的返回结果
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
        beta_threshold=0.1,
    )
    print("✓ ResSAM 初始化成功")
