"""
Res-SAM v15 - Step 1: build the feature bank.

V15 相对 V14 的改进：
1. 方案5：ESN hidden_size 30 → 100
   - 特征维度从 61 增加到 201（含 bias）
   - V14 诊断：TP 框 score 系统性低于 FP 框，ESN 判别能力不足
   - 更高维度的 reservoir 有更强的特征表达能力，有望分离 TP/FP 分布

2. 方案3：GPR 行列双向背景去除（row_mean + column_mean）
   - V13/V14 只做行均值去除（水平条带）
   - GPR B-scan 还有垂直方向的直达波（列方向强信号）
   - 行列双向去除后特征更纯净

继承 V13/V14：
- 小样本分组采样（20张，按原始图ID分组）
- PatchCore coreset（patch内部贪心去重）
- 自适应 beta 校准（p95 of 1-NN dists）

放弃（V14 诊断证明无效）：
- top-1 框过滤（TP score 低于 FP，top-1 永远不是 TP）
- score 阈值过滤（TP/FP 分布完全混叠）
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
    "output_dir": os.path.join(BASE_DIR, "outputs", "feature_banks_v15"),
    "output_file": "feature_bank_v15.pth",
    "metadata_file": "metadata.json",
    # V15 方案5：hidden_size 30 → 100
    "window_size": 50,
    "stride": 5,
    "hidden_size": 100,          # ← 核心改动，V13/V14 = 30
    # 小样本分组采样（继承）
    "num_normal_samples": 20,
    "grouped_sampling": True,
    # PatchCore coreset（继承，max_size 随 hidden_size 增大而增大）
    "coreset_ratio": 0.5,
    "coreset_min_size": 500,
    "coreset_max_size": 5000,
    # V15 方案3：行列双向背景去除
    "gpr_background_removal": True,
    "background_removal_method": "both",  # ← 核心改动，V13/V14 = "row_mean"
    # 自适应 beta（继承 V14）
    "adaptive_beta_percentile": 95,
    "adaptive_beta_scale": 1.0,
    # 其他
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "device": "auto",
    "random_seed": 11,
    "version": "v15",
    "feature_with_bias": True,
    "alignment_notes": (
        "v15: hidden_size=100(↑from 30) + 行列双向背景去除 + 自适应beta + merge_all; "
        "解决V14 TP/FP score混叠问题"
    ),
}


def _to_abs(base_dir: str, p: str) -> str:
    if not p: return p
    if os.path.isabs(p): return p
    base_name = os.path.basename(base_dir)
    p_norm = os.path.normpath(p.replace("/", os.sep))
    if p_norm.startswith("." + os.sep): p_norm = p_norm[2:]
    if p_norm == base_name or p_norm.startswith(base_name + os.sep):
        p_norm = p_norm[len(base_name) + 1:]
    return os.path.abspath(os.path.join(base_dir, p_norm))


def get_image_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def remove_gpr_background(img_array: np.ndarray, method: str = "both") -> np.ndarray:
    """
    GPR B-scan 背景去除。

    method:
        "row_mean"  : 减去每行均值（去水平条带）
        "col_mean"  : 减去每列均值（去垂直直达波）
        "both"      : 先减行均值，再减列均值（V15 新增）
        "row_median": 减去每行中位数
    """
    result = img_array.copy().astype(np.float32)
    if method in ("row_mean", "both"):
        result = result - result.mean(axis=1, keepdims=True)
    if method in ("col_mean", "both"):
        result = result - result.mean(axis=0, keepdims=True)
    if method == "row_median":
        result = img_array - np.median(img_array, axis=1, keepdims=True)
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
                arr = remove_gpr_background(arr, config.get("background_removal_method", "both"))
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
    """计算 Feature Bank 1-NN 距离的 p{percentile} 作为自适应 beta"""
    n = features.shape[0]
    print(f"  计算 1-NN 距离分布（n={n}）...")
    batch_size = 256
    nn_dists = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(0, n, batch_size), desc="  1-NN 距离"):
        batch = features[i:i + batch_size]
        diffs = batch[:, None, :] - features[None, :, :]
        dists = np.sum(diffs ** 2, axis=2)
        for j in range(len(batch)):
            dists[j, i + j] = np.inf
        nn_dists[i:i + batch_size] = dists.min(axis=1)
    nn_dists_sqrt = np.sqrt(nn_dists)
    beta = float(np.percentile(nn_dists_sqrt, percentile)) * scale
    print(f"  1-NN 距离: mean={nn_dists_sqrt.mean():.4f} "
          f"p50={np.percentile(nn_dists_sqrt,50):.4f} "
          f"p90={np.percentile(nn_dists_sqrt,90):.4f} "
          f"p95={np.percentile(nn_dists_sqrt,95):.4f} "
          f"p99={np.percentile(nn_dists_sqrt,99):.4f}")
    print(f"  自适应 beta = {beta:.4f}  (V13/V14 固定值 = {DEFAULT_BETA_THRESHOLD})")
    return beta


def build_feature_bank(config: dict):
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("=" * 60)
    print("Res-SAM v15：Feature Bank 构建")
    print("=" * 60)
    print(f"  hidden_size              = {config['hidden_size']}  (V14=30)")
    print(f"  window_size              = {config['window_size']}")
    print(f"  stride                   = {config['stride']}")
    print(f"  background_removal       = {config.get('background_removal_method')}  (V14=row_mean)")
    print(f"  adaptive_beta_percentile = {config.get('adaptive_beta_percentile')}")
    print("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], config["output_file"])

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM (v15)...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        feature_with_bias=bool(config.get("feature_with_bias", True)),
        device=config.get("device", "auto"),
    )
    feat_dim = config["hidden_size"] + (1 if config.get("feature_with_bias") else 0)
    print(f"  ESN 特征维度: {feat_dim}  (V14={30 + 1})")

    all_metadata = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "creation_time": datetime.now().isoformat(),
        "version": "v15",
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

    target_size = max(
        config["coreset_min_size"],
        min(config["coreset_max_size"],
            int(all_features.shape[0] * config["coreset_ratio"]))
    )
    coreset_indices = greedy_coreset_sampling(all_features, target_size, seed=config["random_seed"])
    coreset_features = all_features[coreset_indices]
    print(f"Coreset 特征: {coreset_features.shape}")

    print("\n--- 自适应 beta 校准 ---")
    adaptive_beta = compute_adaptive_beta(
        coreset_features,
        percentile=config.get("adaptive_beta_percentile", 95),
        scale=config.get("adaptive_beta_scale", 1.0),
    )

    model.feature_bank = torch.from_numpy(coreset_features).to(model.device)
    model.anomaly_scorer.fit([coreset_features])
    model._init_nn_searcher(coreset_features)
    model.save_feature_bank(output_path)
    print(f"\nFeature Bank v15 保存至: {output_path}")

    all_metadata.update({
        "feature_bank_shape": list(coreset_features.shape),
        "feature_bank_path": output_path,
        "total_patches_before_coreset": int(all_features.shape[0]),
        "coreset_size": int(coreset_features.shape[0]),
        "coreset_ratio_actual": float(coreset_features.shape[0] / all_features.shape[0]),
        "adaptive_beta": float(adaptive_beta),
        "adaptive_beta_percentile": int(config.get("adaptive_beta_percentile", 95)),
        "fixed_beta_legacy": float(DEFAULT_BETA_THRESHOLD),
    })

    metadata_path = os.path.join(config["output_dir"], config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据保存至: {metadata_path}")
    print(f"自适应 beta = {adaptive_beta:.4f}")

    return torch.from_numpy(coreset_features)


if __name__ == "__main__":
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_01(dict(CONFIG), BASE_DIR, "v15")
    CONFIG["version"] = "v15"
    CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v15")
    CONFIG["output_file"] = "feature_bank_v15.pth"
    CONFIG["normal_data_sources"] = {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact")
    }
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG["output_dir"])
    CONFIG["normal_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                      for k, v in CONFIG["normal_data_sources"].items()}

    # ── 环境变量覆盖（用于消融实验，不改代码直接切换配置）──────────────
    # RES_SAM_HIDDEN_SIZE : ESN 隐层大小，例如 30 / 100
    # RES_SAM_BG_METHOD   : 背景去除方式，例如 row_mean / both / none
    # MAX_IMAGES_PER_CATEGORY : 每类最多处理图像数（已有）
    _hs = os.environ.get("RES_SAM_HIDDEN_SIZE", "").strip()
    if _hs:
        CONFIG["hidden_size"] = int(_hs)
        print(f"[ENV] hidden_size = {CONFIG['hidden_size']}")

    _bg = os.environ.get("RES_SAM_BG_METHOD", "").strip()
    if _bg:
        CONFIG["background_removal_method"] = _bg
        CONFIG["gpr_background_removal"] = (_bg.lower() != "none")
        print(f"[ENV] background_removal_method = {CONFIG['background_removal_method']}")

    # 根据配置动态调整输出路径，避免不同消融实验互相覆盖
    hs  = CONFIG["hidden_size"]
    bgm = CONFIG["background_removal_method"] if CONFIG.get("gpr_background_removal") else "nobg"
    suffix = f"_hs{hs}_{bgm}"
    CONFIG["output_dir"]  = os.path.join(BASE_DIR, "outputs", f"feature_banks_v15{suffix}")
    CONFIG["output_file"] = f"feature_bank_v15{suffix}.pth"
    CONFIG["output_dir"]  = _to_abs(BASE_DIR, CONFIG["output_dir"])
    print(f"[ENV] output_dir = {CONFIG['output_dir']}")
    # ────────────────────────────────────────────────────────────────────

    with torch.no_grad():
        build_feature_bank(CONFIG)
