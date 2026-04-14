"""
Res-SAM v14 - Step 1: build the feature bank.

V14 相对 V13 的改进：
1. 自适应 beta 校准（方案1）：
   - 建库完成后，对 Feature Bank 所有 patch 计算 1-NN 距离分布
   - 取 p95 作为自适应 beta，写入 metadata
   - 解决 V13 粗筛丢弃率=0% 的问题（背景去除后固定 beta=0.1 变成全通）
   - 推理脚本从 metadata 读取 adaptive_beta，不再使用固定值

继承 V13：
- 小样本分组采样（20张，按原始图ID分组）
- PatchCore coreset（patch内部贪心去重，ratio=0.5）
- GPR row_mean 背景去除
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
    "output_dir": os.path.join(BASE_DIR, "outputs", "feature_banks_v14"),
    "output_file": "feature_bank_v14.pth",
    "metadata_file": "metadata.json",
    # ESN 参数
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    # 小样本分组采样（继承 V13）
    "num_normal_samples": 20,
    "grouped_sampling": True,
    # PatchCore coreset（继承 V13）
    "coreset_ratio": 0.5,
    "coreset_min_size": 500,
    "coreset_max_size": 5000,
    # GPR 背景去除（继承 V13）
    "gpr_background_removal": True,
    "background_removal_method": "row_mean",
    # V14 新增：自适应 beta 校准
    "adaptive_beta_percentile": 95,   # 取 1-NN 距离分布的 p95 作为 beta
    "adaptive_beta_scale": 1.0,       # 可选缩放因子，1.0 = 直接用 p95
    # 其他参数
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "device": "auto",
    "random_seed": 11,
    "checkpoint_file": "checkpoint.json",
    "version": "v14",
    "feature_with_bias": True,
    "alignment_notes": (
        "v14: 分组采样(20张) + coreset + GPR背景去除 + 自适应beta(p95 of 1-NN dists) + top-1框; "
        "解决V13粗筛丢弃率=0%问题"
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
    """GPR B-scan 背景去除"""
    if method == "row_mean":
        bg = img_array.mean(axis=1, keepdims=True)
    elif method == "row_median":
        bg = np.median(img_array, axis=1, keepdims=True)
    else:
        return img_array
    result = img_array - bg
    std = result.std()
    if std > 1e-8:
        result = result / std
    return result.astype(np.float32)


def _get_original_id(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    m = re.match(r'^(.+?)_aug_\d+$', name)
    return m.group(1) if m else name


def load_normal_images(data_dir: str, config: dict) -> tuple:
    all_files = sorted([f for f in os.listdir(data_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    num_samples = config.get("num_normal_samples", 20) or len(all_files)
    rng = np.random.RandomState(config.get("random_seed", 11))

    if config.get("grouped_sampling", False):
        groups: dict = {}
        for f in all_files:
            gid = _get_original_id(f)
            groups.setdefault(gid, []).append(f)
        group_ids = sorted(groups.keys())
        selected = [rng.choice(groups[gid]) for gid in group_ids]
        if len(selected) < num_samples:
            remaining = [f for f in all_files if f not in selected]
            rng.shuffle(remaining)
            selected += remaining[:num_samples - len(selected)]
        selected = selected[:num_samples]
        print(f"  分组采样: {len(group_ids)} 个原始图组 → 选 {len(selected)} 张")
    else:
        indices = rng.choice(len(all_files), size=min(num_samples, len(all_files)), replace=False)
        selected = [all_files[i] for i in sorted(indices)]
        print(f"  随机采样: {len(selected)} 张")

    target_hw = target_hw_for_preprocess(config, None)
    images, paths, hashes = [], [], []
    for img_file in tqdm(selected, desc="  加载图像"):
        img_path = os.path.join(data_dir, img_file)
        try:
            img = Image.open(img_path).convert("L")
            if target_hw:
                img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            if config.get("gpr_background_removal", False):
                arr = remove_gpr_background(arr, config.get("background_removal_method", "row_mean"))
            images.append(arr)
            paths.append(img_path)
            hashes.append(get_image_hash(img_path))
        except Exception as e:
            print(f"  警告: 加载失败 {img_file}: {e}")
    return images, paths, hashes


def greedy_coreset_sampling(features: np.ndarray, target_size: int, seed: int = 11) -> np.ndarray:
    n = features.shape[0]
    if target_size >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(0, n))]
    min_dists = np.full(n, np.inf, dtype=np.float32)
    print(f"  Coreset 采样: {n} → {target_size} patches...")
    for _ in tqdm(range(1, target_size), desc="  Coreset"):
        last = features[selected[-1]]
        dists = np.sum((features - last) ** 2, axis=1).astype(np.float32)
        min_dists = np.minimum(min_dists, dists)
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists[next_idx] = 0.0
    return np.array(selected, dtype=np.int64)


def compute_adaptive_beta(features: np.ndarray, percentile: int = 95, scale: float = 1.0) -> float:
    """
    V14 核心：自适应 beta 校准。

    对 Feature Bank 中每个 patch，计算它到其余所有 patch 的最小 L2 距离（1-NN 距离）。
    取这些距离的 p{percentile} 作为 beta 阈值。

    原理：
    - Feature Bank 内部的 1-NN 距离反映了"正常 patch 之间的典型距离"
    - 推理时，异常 patch 到 Feature Bank 的距离应该显著大于这个值
    - 用 p95 作为阈值，意味着只有距离超过 95% 的正常 patch 间距的 patch 才被判为异常
    - 这个阈值会随 Feature Bank 的实际分布自动调整，不受背景去除等预处理影响
    """
    n = features.shape[0]
    print(f"  计算 Feature Bank 1-NN 距离分布（n={n}）...")

    # 分批计算避免内存溢出
    batch_size = 256
    nn_dists = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(0, n, batch_size), desc="  1-NN 距离"):
        batch = features[i:i + batch_size]  # (B, D)
        # 计算 batch 中每个点到所有点的距离
        diffs = batch[:, None, :] - features[None, :, :]  # (B, N, D)
        dists = np.sum(diffs ** 2, axis=2)  # (B, N)
        # 排除自身（距离=0）
        for j in range(len(batch)):
            global_idx = i + j
            dists[j, global_idx] = np.inf
        nn_dists[i:i + batch_size] = dists.min(axis=1)

    nn_dists_sqrt = np.sqrt(nn_dists)  # 转为 L2 距离
    beta = float(np.percentile(nn_dists_sqrt, percentile)) * scale

    print(f"  1-NN 距离统计: mean={nn_dists_sqrt.mean():.4f} "
          f"p50={np.percentile(nn_dists_sqrt, 50):.4f} "
          f"p90={np.percentile(nn_dists_sqrt, 90):.4f} "
          f"p95={np.percentile(nn_dists_sqrt, 95):.4f} "
          f"p99={np.percentile(nn_dists_sqrt, 99):.4f}")
    print(f"  自适应 beta (p{percentile} × {scale}) = {beta:.4f}  "
          f"（V13 固定值 = {DEFAULT_BETA_THRESHOLD}）")
    return beta


def build_feature_bank(config: dict):
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("=" * 60)
    print("Res-SAM v14：Feature Bank 构建")
    print("=" * 60)
    print(f"  window_size = {config['window_size']}")
    print(f"  stride      = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    print(f"  coreset_ratio = {config['coreset_ratio']}")
    print(f"  gpr_background_removal = {config.get('gpr_background_removal')}")
    print(f"  adaptive_beta_percentile = {config.get('adaptive_beta_percentile')}")
    print("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], config["output_file"])

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM (v14)...")
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
        "version": "v14",
        "alignment_notes": config.get("alignment_notes", ""),
        "sources": {},
    }

    all_features_list = []

    for source_name, source_dir in config["normal_data_sources"].items():
        if not os.path.exists(source_dir):
            print(f"警告: 目录不存在: {source_dir}")
            continue

        images, paths, hashes = load_normal_images(source_dir, config)
        all_metadata["sources"][source_name] = {
            "directory": source_dir,
            "num_images": len(images),
            "image_hashes": hashes[:20],
        }

        print(f"\n提取 {len(images)} 张图像的 patch 特征...")
        for img in tqdm(images, desc="  提取特征"):
            patches = model._extract_patches(img)
            if patches.shape[0] == 0:
                continue
            feats = model._fit_patches(patches)
            all_features_list.append(feats.detach().cpu().numpy())

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
    coreset_indices = greedy_coreset_sampling(all_features, target_size, seed=config["random_seed"])
    coreset_features = all_features[coreset_indices]
    print(f"Coreset 特征: {coreset_features.shape}")

    # V14 核心：自适应 beta 校准
    print("\n--- 自适应 beta 校准 ---")
    adaptive_beta = compute_adaptive_beta(
        coreset_features,
        percentile=config.get("adaptive_beta_percentile", 95),
        scale=config.get("adaptive_beta_scale", 1.0),
    )

    # 写入 Feature Bank
    model.feature_bank = torch.from_numpy(coreset_features).to(model.device)
    model.anomaly_scorer.fit([coreset_features])
    model._init_nn_searcher(coreset_features)
    model.save_feature_bank(output_path)
    print(f"\nFeature Bank v14 保存至: {output_path}")

    all_metadata["feature_bank_shape"] = list(coreset_features.shape)
    all_metadata["feature_bank_path"] = output_path
    all_metadata["total_patches_before_coreset"] = int(all_features.shape[0])
    all_metadata["coreset_size"] = int(coreset_features.shape[0])
    all_metadata["coreset_ratio_actual"] = float(coreset_features.shape[0] / all_features.shape[0])
    # V14 新增：写入自适应 beta
    all_metadata["adaptive_beta"] = float(adaptive_beta)
    all_metadata["adaptive_beta_percentile"] = int(config.get("adaptive_beta_percentile", 95))
    all_metadata["fixed_beta_v13"] = float(DEFAULT_BETA_THRESHOLD)

    metadata_path = os.path.join(config["output_dir"], config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据保存至: {metadata_path}")
    print(f"\n自适应 beta = {adaptive_beta:.4f}（推理脚本将自动读取）")

    return torch.from_numpy(coreset_features)


if __name__ == "__main__":
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_01(dict(CONFIG), BASE_DIR, "v14")
    CONFIG["version"] = "v14"
    CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v14")
    CONFIG["output_file"] = "feature_bank_v14.pth"
    CONFIG["normal_data_sources"] = {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact")
    }
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG["output_dir"])
    CONFIG["normal_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                      for k, v in CONFIG["normal_data_sources"].items()}
    with torch.no_grad():
        build_feature_bank(CONFIG)
