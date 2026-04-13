"""
Res-SAM v13 - Step 1: build the feature bank.

V13 相对 V12 的改进：
1. PatchCore coreset 采样：从所有正常图的 patch 特征中贪心选出最具代表性的子集
   - 解决 Feature Bank 覆盖不足导致正常图被误判的问题
   - 比随机采样/分组采样更系统，保证特征多样性
2. GPR 背景去除预处理（可选）：在建库前对图像做水平背景滤波
   - GPR B-scan 的水平条带是背景噪声，去掉后特征更纯净
3. 保留 V12 的按原始图分组采样作为 coreset 的输入来源
"""

import sys
import os
import json
import hashlib
import re
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess
from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_01
from experiments.paper_constants import DEFAULT_BETA_THRESHOLD, preflight_faiss_or_raise


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "normal_data_sources": {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "output_dir": os.path.join(BASE_DIR, "outputs", "feature_banks_v13"),
    "output_file": "feature_bank_v13.pth",
    "metadata_file": "metadata.json",
    # ESN 参数（保持论文默认）
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    # V13：用全量正常图做 coreset 采样，不再限制 num_normal_samples
    # coreset 会从所有图的 patch 特征中选出最具代表性的子集
    "num_normal_samples": None,   # None = 使用全量
    "coreset_ratio": 0.01,        # 保留 1% 的 patch 作为 coreset（约 9000 个 patch）
    "coreset_min_size": 5000,     # coreset 最小 patch 数
    "coreset_max_size": 50000,    # coreset 最大 patch 数（防止内存溢出）
    # GPR 背景去除（可选）
    "gpr_background_removal": True,   # 水平背景滤波，去除 B-scan 水平条带噪声
    "background_removal_method": "row_mean",  # row_mean: 减去每行均值
    # 其他参数
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "device": "auto",
    "random_seed": 11,
    "checkpoint_file": "checkpoint.json",
    "version": "v13",
    "feature_with_bias": True,
    "alignment_notes": (
        "v13: PatchCore coreset采样建库 + GPR背景去除预处理 + merge_all推理; "
        "fb_source=augmented_intact全量, coreset_ratio=0.01"
    ),
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
        p_norm = p_norm[len(base_name) + 1:]
    return os.path.abspath(os.path.join(base_dir, p_norm))


def get_image_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def remove_gpr_background(img_array: np.ndarray, method: str = "row_mean") -> np.ndarray:
    """
    GPR B-scan 背景去除。
    GPR 图像中水平条带是背景噪声（每行的均值反映背景强度），
    减去每行均值后，异常双曲线特征更突出，正常区域更接近零。

    method:
        "row_mean": 减去每行均值（最常用）
        "row_median": 减去每行中位数（对异常更鲁棒）
    """
    if method == "row_mean":
        bg = img_array.mean(axis=1, keepdims=True)
    elif method == "row_median":
        bg = np.median(img_array, axis=1, keepdims=True)
    else:
        return img_array
    result = img_array - bg
    # 重新归一化
    std = result.std()
    if std > 1e-8:
        result = result / std
    return result.astype(np.float32)


def load_all_normal_images(data_dir: str, config: dict) -> tuple:
    """加载全量正常图像"""
    image_files = sorted([f for f in os.listdir(data_dir)
                          if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    target_hw = target_hw_for_preprocess(config, None)
    images, paths, hashes = [], [], []
    print(f"  加载 {len(image_files)} 张正常图像...")
    for img_file in tqdm(image_files, desc="  加载图像"):
        img_path = os.path.join(data_dir, img_file)
        try:
            img = Image.open(img_path).convert("L")
            if target_hw:
                img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            # 全局归一化
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            # GPR 背景去除
            if config.get("gpr_background_removal", False):
                arr = remove_gpr_background(arr, config.get("background_removal_method", "row_mean"))
            images.append(arr)
            paths.append(img_path)
            hashes.append(get_image_hash(img_path))
        except Exception as e:
            print(f"  警告: 加载失败 {img_file}: {e}")
    return images, paths, hashes


def greedy_coreset_sampling(features: np.ndarray, target_size: int, seed: int = 11) -> np.ndarray:
    """
    PatchCore 贪心 coreset 采样。
    从 features 中选出 target_size 个最具代表性的样本，
    使得选出的子集能最好地覆盖整个特征空间。

    算法：
    1. 随机选一个起始点
    2. 每次选距离当前 coreset 最远的点加入
    3. 重复直到达到 target_size

    这保证了 coreset 的多样性，避免重复覆盖相同区域。
    """
    n = features.shape[0]
    if target_size >= n:
        return np.arange(n)

    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(0, n))]
    # 每个点到当前 coreset 的最小距离
    min_dists = np.full(n, np.inf, dtype=np.float32)

    print(f"  Coreset 采样: {n} → {target_size} patches...")
    for i in tqdm(range(1, target_size), desc="  Coreset"):
        last = features[selected[-1]]
        # 更新最小距离（只需和最新加入的点比较）
        dists = np.sum((features - last) ** 2, axis=1).astype(np.float32)
        min_dists = np.minimum(min_dists, dists)
        # 选距离最大的点
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists[next_idx] = 0.0  # 已选，距离置0

    return np.array(selected, dtype=np.int64)


def build_feature_bank(config: dict):
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("=" * 60)
    print("Res-SAM v13：Feature Bank 构建（PatchCore Coreset）")
    print("=" * 60)
    print(f"  window_size = {config['window_size']}")
    print(f"  stride = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    print(f"  coreset_ratio = {config['coreset_ratio']}")
    print(f"  gpr_background_removal = {config.get('gpr_background_removal', False)}")
    print("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], config["output_file"])

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM (v13)...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        feature_with_bias=bool(config.get("feature_with_bias", True)),
        device=config.get("device", "auto"),
    )

    all_metadata = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "creation_time": datetime.now().isoformat(),
        "version": "v13",
        "alignment_notes": config.get("alignment_notes", ""),
        "sources": {},
    }

    all_features_list = []
    all_paths = []

    for source_name, source_dir in config["normal_data_sources"].items():
        if not os.path.exists(source_dir):
            print(f"警告: 目录不存在: {source_dir}")
            continue

        images, paths, hashes = load_all_normal_images(source_dir, config)
        all_metadata["sources"][source_name] = {
            "directory": source_dir,
            "num_images": len(images),
            "image_hashes": hashes[:20],  # 只记录前20个hash
        }

        print(f"\n提取 {len(images)} 张图像的 patch 特征...")
        for img in tqdm(images, desc="  提取特征"):
            patches = model._extract_patches(img)
            if patches.shape[0] == 0:
                continue
            feats = model._fit_patches(patches)
            all_features_list.append(feats.detach().cpu().numpy())
        all_paths.extend(paths)

    if not all_features_list:
        raise RuntimeError("没有提取到任何特征")

    all_features = np.concatenate(all_features_list, axis=0).astype(np.float32)
    print(f"\n全量特征: {all_features.shape}")

    # PatchCore coreset 采样
    target_size = max(
        config["coreset_min_size"],
        min(config["coreset_max_size"],
            int(all_features.shape[0] * config["coreset_ratio"]))
    )
    print(f"Coreset 目标大小: {target_size}")

    coreset_indices = greedy_coreset_sampling(all_features, target_size, seed=config["random_seed"])
    coreset_features = all_features[coreset_indices]
    print(f"Coreset 特征: {coreset_features.shape}")

    # 写入 Feature Bank
    model.feature_bank = torch.from_numpy(coreset_features).to(model.device)
    model.anomaly_scorer.fit([coreset_features])
    model._init_nn_searcher(coreset_features)

    model.save_feature_bank(output_path)
    print(f"\nFeature Bank v13 保存至: {output_path}")
    print(f"形状: {coreset_features.shape}")

    all_metadata["feature_bank_shape"] = list(coreset_features.shape)
    all_metadata["feature_bank_path"] = output_path
    all_metadata["total_patches_before_coreset"] = int(all_features.shape[0])
    all_metadata["coreset_size"] = int(coreset_features.shape[0])
    all_metadata["coreset_ratio_actual"] = float(coreset_features.shape[0] / all_features.shape[0])

    metadata_path = os.path.join(config["output_dir"], config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据保存至: {metadata_path}")

    return torch.from_numpy(coreset_features)


if __name__ == "__main__":
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_01(dict(CONFIG), BASE_DIR, "v13")
    CONFIG["version"] = "v13"
    CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v13")
    CONFIG["output_file"] = "feature_bank_v13.pth"
    CONFIG["normal_data_sources"] = {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact")
    }
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG["output_dir"])
    CONFIG["normal_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                      for k, v in CONFIG["normal_data_sources"].items()}
    with torch.no_grad():
        build_feature_bank(CONFIG)
