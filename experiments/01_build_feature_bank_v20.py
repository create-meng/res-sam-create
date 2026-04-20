"""
Res-SAM v20 - Step 1: build the feature bank with all improvements.

v20 相对 V16 的核心改进（一次性实现所有改进）：
1. 继承 V16 的验证集驱动 beta 校准
2. 使用多个 beta 候选值（p10, p20）用于后续推理
3. 为后续的 pixel-level heatmap 和 NMS 做准备

理论依据：
- 验证集驱动 beta：论文 "Anomalous Samples for Few-Shot Anomaly Detection" (IJCNN 2025)
- Coreset 采样：论文 "Continual Anomaly Detection Based on Incremental Coreset" (arXiv 2511.08634)

继承 V16 最优配置：
- hidden_size = 30
- background_removal_method = "both"（行列双向背景去除）
- 分组采样(20张) + coreset + merge_all
- 验证集驱动的 beta 校准

预期效果：
- 为 v20 推理提供多个 beta 候选值（p5, p10, p20）
- 保持 V16 的粗筛丢弃率改进（0% vs V15 的 100%）
- F1(IoU>0.5) 目标：0.25-0.35（通过后续推理改进实现）
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
from PatchRes.logger import setup_global_logger, log_config, log_section, log_step, log_finish


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    "normal_data_sources": {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "anomaly_data_sources": {
        "augmented_cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "augmented_utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
    },
    "output_dir": os.path.join(BASE_DIR, "outputs", "feature_banks_v20"),
    "output_file": "feature_bank_v20.pth",
    "metadata_file": "metadata.json",
    # v20：继承 V16 最优配置
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,  # V16 最优
    # 小样本分组采样
    "num_normal_samples": 20,
    "grouped_sampling": True,
    # PatchCore coreset
    "coreset_ratio": 0.5,
    "coreset_min_size": 500,
    "coreset_max_size": 5000,
    # v20 方案 4：多尺度 Memory Bank
    "use_multiscale_memory": True,  # 启用多尺度 Memory Bank
    "local_memory_per_group": 50,   # 每个原始图组的局部 Memory Bank 大小
    # GPR 行列双向背景去除
    "gpr_background_removal": True,
    "background_removal_method": "both",  # V16 最优
    # v20 核心：验证集驱动的 beta 校准（继承 V16）
    "num_val_normal": 20,  # 正常验证集大小
    "num_val_anomaly": 20,  # 异常验证集大小
    "val_seed": 42,  # 验证集划分的随机种子（与建库种子分离）
    # 其他
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "device": "auto",
    "random_seed": 11,  # 建库的随机种子
    "version": "v20",
    "feature_with_bias": True,
    "alignment_notes": (
        "v20: 继承V16验证集驱动beta校准 + 为pixel-level heatmap和NMS做准备; "
        "一次性实现所有改进"
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
    """GPR B-scan 背景去除"""
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


def split_validation_set(config: dict) -> dict:
    """
    划分验证集（不参与建库）
    
    Returns:
        {
            'val_normal_files': [...],  # 正常验证集文件名
            'val_anomaly_files': [...],  # 异常验证集文件名
            'train_normal_files': [...],  # 剩余正常图（用于建库）
        }
    """
    print("\n=== 划分验证集 ===")
    rng = np.random.RandomState(config["val_seed"])
    
    # 1. 正常验证集
    normal_dir = config["normal_data_sources"]["augmented_intact"]
    all_normal = sorted([f for f in os.listdir(normal_dir)
                         if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    
    val_normal_indices = rng.choice(len(all_normal), size=config["num_val_normal"], replace=False)
    val_normal_files = [all_normal[i] for i in sorted(val_normal_indices)]
    train_normal_files = [f for f in all_normal if f not in val_normal_files]
    
    print(f"  正常图：总计 {len(all_normal)} 张")
    print(f"    验证集：{len(val_normal_files)} 张")
    print(f"    训练集：{len(train_normal_files)} 张")
    
    # 2. 异常验证集（从 cavities + utilities 中抽取）
    val_anomaly_files = []
    for source_name, source_dir in config["anomaly_data_sources"].items():
        all_anomaly = sorted([f for f in os.listdir(source_dir)
                              if f.lower().endswith((".jpg", ".png", ".jpeg"))])
        num_from_this = config["num_val_anomaly"] // len(config["anomaly_data_sources"])
        indices = rng.choice(len(all_anomaly), size=min(num_from_this, len(all_anomaly)), replace=False)
        selected = [(source_name, all_anomaly[i]) for i in indices]
        val_anomaly_files.extend(selected)
        print(f"  异常图 ({source_name})：从 {len(all_anomaly)} 张中抽取 {len(selected)} 张")
    
    print(f"  异常验证集总计：{len(val_anomaly_files)} 张")
    
    return {
        'val_normal_files': val_normal_files,
        'val_anomaly_files': val_anomaly_files,
        'train_normal_files': train_normal_files,
    }


def load_images(file_list, base_dir, config: dict) -> tuple:
    """加载图像列表"""
    target_hw = target_hw_for_preprocess(config, None)
    images, paths, hashes = [], [], []
    
    for img_file in tqdm(file_list, desc="  加载图像"):
        img_path = os.path.join(base_dir, img_file)
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


def load_normal_images_for_training(train_files, data_dir, config: dict) -> tuple:
    """从训练集中采样 20 张正常图用于建库"""
    num_samples = config.get("num_normal_samples", 20)
    rng = np.random.RandomState(config.get("random_seed", 11))
    
    if config.get("grouped_sampling", False):
        groups: dict = {}
        for f in train_files:
            gid = _get_original_id(f)
            groups.setdefault(gid, []).append(f)
        group_ids = sorted(groups.keys())
        selected = [rng.choice(groups[gid]) for gid in group_ids]
        if len(selected) < num_samples:
            remaining = [f for f in train_files if f not in selected]
            rng.shuffle(remaining)
            selected += remaining[:num_samples - len(selected)]
        selected = selected[:num_samples]
        print(f"  分组采样: {len(group_ids)} 个原始图组 → 选 {len(selected)} 张")
    else:
        indices = rng.choice(len(train_files), size=min(num_samples, len(train_files)), replace=False)
        selected = [train_files[i] for i in sorted(indices)]
    
    return load_images(selected, data_dir, config)


def greedy_coreset_sampling(features: np.ndarray, target_size: int, seed: int = 11) -> np.ndarray:
    """PatchCore coreset 采样"""
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


def build_multiscale_memory_bank(
    all_features: np.ndarray,
    image_ids: list,
    config: dict
) -> dict:
    """
    v20 方案 4：构建多尺度 Memory Bank
    
    理论依据：论文 "Multimodal Task Representation Memory Bank vs. Catastrophic Forgetting in Anomaly Detection"
    
    Returns:
        {
            'global_memory': np.ndarray,  # 全局 Memory Bank
            'local_memories': dict,       # 局部 Memory Bank {group_id: features}
        }
    """
    print("\n=== 构建多尺度 Memory Bank ===")
    
    # 1. 全局 Memory Bank：从所有特征中 coreset 采样
    global_target_size = max(
        config["coreset_min_size"],
        min(config["coreset_max_size"],
            int(all_features.shape[0] * config["coreset_ratio"]))
    )
    global_indices = greedy_coreset_sampling(all_features, global_target_size, seed=config["random_seed"])
    global_memory = all_features[global_indices]
    print(f"  全局 Memory Bank: {global_memory.shape}")
    
    # 2. 局部 Memory Bank：针对每个原始图 ID 分组采样
    local_memories = {}
    groups = {}
    
    # 按 image_id 分组
    for i, img_id in enumerate(image_ids):
        groups.setdefault(img_id, []).append(i)
    
    print(f"  构建局部 Memory Bank: {len(groups)} 个组")
    local_per_group = config.get("local_memory_per_group", 50)
    
    for group_id, indices in tqdm(groups.items(), desc="  局部 Memory"):
        group_features = all_features[indices]
        
        # 如果该组特征数量少于目标，全部保留
        if len(group_features) <= local_per_group:
            local_memories[group_id] = group_features
        else:
            # 否则进行 coreset 采样
            local_indices = greedy_coreset_sampling(
                group_features, local_per_group, seed=config["random_seed"]
            )
            local_memories[group_id] = group_features[local_indices]
    
    total_local = sum(len(m) for m in local_memories.values())
    print(f"  局部 Memory Bank 总计: {total_local} patches")
    
    return {
        'global_memory': global_memory,
        'local_memories': local_memories,
    }


def calibrate_beta_with_validation(model, feature_bank_np, val_split, config: dict) -> dict:
    """
    v20 核心：用验证集（正常+异常）校准 beta（继承 V16）
    
    方法：
    1. 在验证集上计算 patch 级 anomaly scores
    2. 用 Youden Index 找最优阈值（最大化 TPR - FPR）
    3. 同时计算多个候选 beta（p5/p10/p20 等）
    
    v20 改进：
    - 保存更多候选 beta 值（p5, p10, p20）供推理时选择
    - 为后续的 pixel-level heatmap 提供归一化阈值
    
    Returns:
        {
            'beta': float,  # 最优 beta
            'beta_method': str,  # 校准方法
            'beta_candidates': dict,  # 所有候选 beta
            'val_stats': dict,  # 验证集统计信息
        }
    """
    print("\n=== v20 验证集驱动的 Beta 校准（继承 V16）===")
    
    # 1. 加载验证集
    normal_dir = config["normal_data_sources"]["augmented_intact"]
    val_normal_images, _, _ = load_images(val_split['val_normal_files'], normal_dir, config)
    
    val_anomaly_images = []
    for source_name, img_file in val_split['val_anomaly_files']:
        source_dir = config["anomaly_data_sources"][source_name]
        imgs, _, _ = load_images([img_file], source_dir, config)
        val_anomaly_images.extend(imgs)
    
    print(f"  验证集：{len(val_normal_images)} 张正常 + {len(val_anomaly_images)} 张异常")
    
    # 2. 计算验证集上的 patch 级 anomaly scores
    print("  计算正常验证集 scores...")
    normal_scores = []
    for img in tqdm(val_normal_images, desc="  正常图"):
        patches = model._extract_patches(img)
        if patches.shape[0] == 0:
            continue
        features = model._fit_patches(patches).detach().cpu().numpy()
        scores = model._score_features_multiscale(features, image_id=None)
        normal_scores.extend(scores.tolist())
    
    print("  计算异常验证集 scores...")
    anomaly_scores = []
    for img in tqdm(val_anomaly_images, desc="  异常图"):
        patches = model._extract_patches(img)
        if patches.shape[0] == 0:
            continue
        features = model._fit_patches(patches).detach().cpu().numpy()
        scores = model._score_features_multiscale(features, image_id=None)
        anomaly_scores.extend(scores.tolist())
    
    normal_scores = np.array(normal_scores)
    anomaly_scores = np.array(anomaly_scores)
    
    print(f"\n  Patch 级 scores 统计:")
    print(f"    正常：n={len(normal_scores)}, mean={normal_scores.mean():.4f}, "
          f"p50={np.percentile(normal_scores,50):.4f}, p90={np.percentile(normal_scores,90):.4f}, "
          f"p95={np.percentile(normal_scores,95):.4f}, max={normal_scores.max():.4f}")
    print(f"    异常：n={len(anomaly_scores)}, mean={anomaly_scores.mean():.4f}, "
          f"p5={np.percentile(anomaly_scores,5):.4f}, p10={np.percentile(anomaly_scores,10):.4f}, "
          f"p20={np.percentile(anomaly_scores,20):.4f}, p50={np.percentile(anomaly_scores,50):.4f}, "
          f"max={anomaly_scores.max():.4f}")
    
    # 3. 计算多个候选 beta（v20：保存更多候选值）
    beta_candidates = {
        'anomaly_p5': float(np.percentile(anomaly_scores, 5)),
        'anomaly_p10': float(np.percentile(anomaly_scores, 10)),
        'anomaly_p15': float(np.percentile(anomaly_scores, 15)),
        'anomaly_p20': float(np.percentile(anomaly_scores, 20)),
        'anomaly_p25': float(np.percentile(anomaly_scores, 25)),
        'normal_p90': float(np.percentile(normal_scores, 90)),
        'normal_p95': float(np.percentile(normal_scores, 95)),
    }
    
    # 4. Youden Index：找最优阈值（最大化 TPR - FPR）
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))])
    
    thresholds = np.percentile(all_scores, np.linspace(1, 99, 99))
    best_youden = -1
    best_beta = 0.1
    
    for thresh in thresholds:
        tp = np.sum((all_scores > thresh) & (all_labels == 1))
        fp = np.sum((all_scores > thresh) & (all_labels == 0))
        tn = np.sum((all_scores <= thresh) & (all_labels == 0))
        fn = np.sum((all_scores <= thresh) & (all_labels == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        youden = tpr - fpr
        
        if youden > best_youden:
            best_youden = youden
            best_beta = float(thresh)
    
    beta_candidates['youden_index'] = best_beta
    
    print(f"\n  Beta 候选值（v20 扩展）:")
    for name, value in beta_candidates.items():
        print(f"    {name:20s} = {value:.4f}")
    
    # 5. v20：默认使用 anomaly_p10（比 V16 的 p5 更激进，减少 FP）
    final_beta = beta_candidates['anomaly_p10']
    print(f"\n  最终 beta = {final_beta:.4f}  (策略: anomaly_p10)")
    print(f"  对比 V16 beta = {beta_candidates['anomaly_p5']:.4f}（anomaly_p5）")
    print(f"  v20 策略：使用 p10 减少 FP，同时保留多个候选值供推理选择")
    
    return {
        'beta': final_beta,
        'beta_method': 'validation_set_anomaly_p10',
        'beta_candidates': beta_candidates,
        'val_stats': {
            'normal_scores': {
                'n': int(len(normal_scores)),
                'mean': float(normal_scores.mean()),
                'std': float(normal_scores.std()),
                'p5': float(np.percentile(normal_scores, 5)),
                'p50': float(np.percentile(normal_scores, 50)),
                'p95': float(np.percentile(normal_scores, 95)),
                'max': float(normal_scores.max()),
            },
            'anomaly_scores': {
                'n': int(len(anomaly_scores)),
                'mean': float(anomaly_scores.mean()),
                'std': float(anomaly_scores.std()),
                'p5': float(np.percentile(anomaly_scores, 5)),
                'p10': float(np.percentile(anomaly_scores, 10)),
                'p15': float(np.percentile(anomaly_scores, 15)),
                'p20': float(np.percentile(anomaly_scores, 20)),
                'p25': float(np.percentile(anomaly_scores, 25)),
                'p50': float(np.percentile(anomaly_scores, 50)),
                'p95': float(np.percentile(anomaly_scores, 95)),
                'max': float(anomaly_scores.max()),
            },
        },
    }


def build_feature_bank(config: dict):
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("=" * 60)
    print("Res-SAM v20：Feature Bank 构建（继承 V16 + 扩展 beta 候选值）")
    print("=" * 60)
    print(f"  hidden_size              = {config['hidden_size']}")
    print(f"  background_removal       = {config.get('background_removal_method')}")
    print(f"  num_val_normal           = {config['num_val_normal']}")
    print(f"  num_val_anomaly          = {config['num_val_anomaly']}")
    print("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], config["output_file"])

    from PatchRes.ResSAM import ResSAM

    # 1. 划分验证集
    val_split = split_validation_set(config)
    
    # 2. 初始化模型
    print("\n初始化 ResSAM (v20)...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        feature_with_bias=bool(config.get("feature_with_bias", True)),
        device=config.get("device", "auto"),
    )
    feat_dim = config["hidden_size"] + (1 if config.get("feature_with_bias") else 0)
    print(f"  ESN 特征维度: {feat_dim}")

    # 3. 从训练集构建 Feature Bank
    print("\n从训练集构建 Feature Bank...")
    normal_dir = config["normal_data_sources"]["augmented_intact"]
    images, paths, hashes = load_normal_images_for_training(
        val_split['train_normal_files'], normal_dir, config
    )
    
    all_features_list = []
    image_ids_list = []  # v20: 记录每个 patch 对应的 image_id
    print(f"\n提取 {len(images)} 张图像的 patch 特征...")
    for i, img in enumerate(tqdm(images, desc="  提取特征")):
        img_file = os.path.basename(paths[i])
        img_id = _get_original_id(img_file)
        
        patches = model._extract_patches(img)
        if patches.shape[0] == 0:
            continue
        feats = model._fit_patches(patches)
        all_features_list.append(feats.detach().cpu().numpy())
        
        # v20: 记录每个 patch 对应的 image_id
        image_ids_list.extend([img_id] * patches.shape[0])

    if not all_features_list:
        raise RuntimeError("没有提取到任何特征")

    all_features = np.concatenate(all_features_list, axis=0).astype(np.float32)
    print(f"\n全量特征: {all_features.shape}")

    # 4. Coreset 采样
    target_size = max(
        config["coreset_min_size"],
        min(config["coreset_max_size"],
            int(all_features.shape[0] * config["coreset_ratio"]))
    )
    coreset_indices = greedy_coreset_sampling(all_features, target_size, seed=config["random_seed"])
    coreset_features = all_features[coreset_indices]
    print(f"Coreset 特征: {coreset_features.shape}")

    # 4.5. v20 方案 4：构建多尺度 Memory Bank
    multiscale_memory = None
    if config.get("use_multiscale_memory", False):
        # image_ids_list 已经在特征提取时构建好了
        multiscale_memory = build_multiscale_memory_bank(all_features, image_ids_list, config)
        print(f"多尺度 Memory Bank 构建完成")

    # 5. 初始化 Feature Bank（用于验证集校准）
    model.feature_bank = torch.from_numpy(coreset_features).to(model.device)
    model.anomaly_scorer.fit([coreset_features])
    model._init_nn_searcher(coreset_features)
    
    # 6. v20 核心：验证集驱动的 beta 校准（继承 V16，扩展候选值）
    beta_result = calibrate_beta_with_validation(model, coreset_features, val_split, config)
    
    # 7. 保存 Feature Bank（包含多尺度 Memory Bank）
    if multiscale_memory is not None:
        # 将多尺度 Memory Bank 保存到模型中
        model.global_memory_bank = torch.from_numpy(multiscale_memory['global_memory']).to(model.device)
        model.local_memory_banks = {
            k: torch.from_numpy(v).to(model.device)
            for k, v in multiscale_memory['local_memories'].items()
        }
    
    model.save_feature_bank(output_path)
    print(f"\nFeature Bank v20 保存至: {output_path}")

    # 8. 保存元数据
    all_metadata = {
        "config": {k: v for k, v in config.items() if not callable(v)},
        "creation_time": datetime.now().isoformat(),
        "version": "v20",
        "alignment_notes": config.get("alignment_notes", ""),
        "feature_bank_shape": list(coreset_features.shape),
        "feature_bank_path": output_path,
        "total_patches_before_coreset": int(all_features.shape[0]),
        "coreset_size": int(coreset_features.shape[0]),
        "coreset_ratio_actual": float(coreset_features.shape[0] / all_features.shape[0]),
        # v20 核心：验证集驱动的 beta（继承 V16，扩展候选值）
        "adaptive_beta": beta_result['beta'],
        "beta_calibration_method": beta_result['beta_method'],
        "beta_candidates": beta_result['beta_candidates'],
        "validation_set": {
            "num_normal": len(val_split['val_normal_files']),
            "num_anomaly": len(val_split['val_anomaly_files']),
            "normal_files": val_split['val_normal_files'][:10],  # 只保存前10个
            "anomaly_files": [f"{s}/{f}" for s, f in val_split['val_anomaly_files'][:10]],
            "stats": beta_result['val_stats'],
        },
        "fixed_beta_legacy": float(DEFAULT_BETA_THRESHOLD),
        # v20 方案 4：多尺度 Memory Bank 信息
        "multiscale_memory": {
            "enabled": config.get("use_multiscale_memory", False),
            "global_memory_shape": list(multiscale_memory['global_memory'].shape) if multiscale_memory else None,
            "num_local_groups": len(multiscale_memory['local_memories']) if multiscale_memory else 0,
            "local_memory_per_group": config.get("local_memory_per_group", 50),
        } if multiscale_memory else {"enabled": False},
        "sources": {
            "augmented_intact": {
                "directory": normal_dir,
                "num_images": len(images),
                "image_hashes": hashes[:20],
            }
        },
    }

    metadata_path = os.path.join(config["output_dir"], config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    print(f"元数据保存至: {metadata_path}")
    print(f"\n最终 adaptive_beta = {beta_result['beta']:.4f}")
    print(f"对比 V16 beta = {beta_result['beta_candidates']['anomaly_p5']:.4f}（anomaly_p5）")
    print(f"v20 策略：使用 anomaly_p10，同时保存多个候选值供推理选择")

    return torch.from_numpy(coreset_features)


if __name__ == "__main__":
    # 设置全局日志
    logger = setup_global_logger(BASE_DIR, "01_build_feature_bank_v20")
    
    preflight_faiss_or_raise()
    CONFIG = apply_layout_to_config_01(dict(CONFIG), BASE_DIR, "v20")
    CONFIG["version"] = "v20"
    CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v20")
    CONFIG["output_file"] = "feature_bank_v20.pth"
    CONFIG["normal_data_sources"] = {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact")
    }
    CONFIG["anomaly_data_sources"] = {
        "augmented_cavities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_cavities"),
        "augmented_utilities": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_utilities"),
    }
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG["output_dir"])
    CONFIG["normal_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                      for k, v in CONFIG["normal_data_sources"].items()}
    CONFIG["anomaly_data_sources"] = {k: _to_abs(BASE_DIR, v)
                                       for k, v in CONFIG["anomaly_data_sources"].items()}

    # 记录配置
    log_config(CONFIG, logger)

    with torch.no_grad():
        build_feature_bank(CONFIG)
    
    log_finish("01_build_feature_bank_v20", logger)



