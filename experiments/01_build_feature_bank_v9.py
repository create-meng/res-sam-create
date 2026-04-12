"""
Res-SAM V9 - Step 1: build the feature bank.

Current V9 mainline policy:
- follow the paper where formulas and procedure are explicit;
- when the paper is silent, fall back to the author public-code defaults.

For preprocessing, V9 uses a fixed resize to 369x369 for all images,
matching the default loading convention in the official repository.
The V9 runtime seed is unified to 11.
"""

import sys
import os
import json
import hashlib
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
import numpy as np
from PIL import Image

from experiments.resize_policy import RESIZE_POLICY_FIXED, target_hw_for_preprocess
from experiments.dataset_layout import DATASET_ENHANCED, apply_layout_to_config_01
from experiments.paper_constants import DEFAULT_BETA_THRESHOLD, preflight_faiss_or_raise


CONFIG = {
    "dataset_mode": DATASET_ENHANCED,
    # 当前唯一保留的 normal 源（augmented_intact）
    "normal_data_sources": {
        "augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact"),
    },
    "output_dir": os.path.join(BASE_DIR, "outputs", "feature_banks_v9"),
    "output_file": "feature_bank_v9.pth",
    "metadata_file": "metadata.json",
    # 论文优先参数
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "num_normal_samples": 20,
    "resize_policy": RESIZE_POLICY_FIXED,
    "image_size": (369, 369),
    "device": "auto",
    "random_seed": 11,
    "checkpoint_file": "checkpoint.json",
    "version": "V9",
    "feature_with_bias": True,
    "alignment_notes": "Paper-first mainline (V9): solve true f=[W_out,b]; fb_source=augmented_intact, eval=augmented_*",
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
        p_norm = p_norm[len(base_name) + 1 :]
    return os.path.abspath(os.path.join(base_dir, p_norm))


def get_image_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def load_normal_images(
    data_dir: str,
    num_samples: int,
    config: dict,
    random_select: bool = True,
):
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if len(image_files) == 0:
        raise ValueError(f"No images found in {data_dir}")
    if random_select:
        np.random.shuffle(image_files)
    selected_files = image_files[: min(num_samples, len(image_files))]

    target_hw = target_hw_for_preprocess(config, None)
    images, paths, hashes = [], [], []
    for img_file in selected_files:
        img_path = os.path.join(data_dir, img_file)
        try:
            img = Image.open(img_path).convert("L")
            if target_hw:
                img = img.resize((target_hw[1], target_hw[0]), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32)
            img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
            images.append(img_array)
            paths.append(img_path)
            hashes.append(get_image_hash(img_path))
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            continue
    if target_hw:
        return np.array(images), paths, hashes
    return images, paths, hashes


def build_feature_bank(config: dict, resume: bool = True):
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])

    print("=" * 60)
    print("Res-SAM V9：Feature Bank 构建")
    print("=" * 60)
    print(f"  window_size = {config['window_size']}")
    print(f"  stride = {config['stride']}")
    print(f"  hidden_size = {config['hidden_size']}")
    expected_dim = 2 * config["hidden_size"] + (1 if bool(config.get("feature_with_bias", False)) else 0)
    print(f"  Expected feature dim = {expected_dim}")
    print("=" * 60)

    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], config["output_file"])

    checkpoint_path = os.path.join(config["output_dir"], config["checkpoint_file"])
    processed_sources = {}
    if (not os.path.exists(output_path)) and os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

    if resume and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)
        processed_sources = checkpoint.get("processed_sources", {})

    from PatchRes.ResSAM import ResSAM

    print("\n初始化 ResSAM (V9)...")
    model = ResSAM(
        hidden_size=config["hidden_size"],
        window_size=config["window_size"],
        stride=config["stride"],
        beta_threshold=float(config.get("beta_threshold", DEFAULT_BETA_THRESHOLD)),
        feature_with_bias=bool(config.get("feature_with_bias", False)),
        device=config.get("device", "auto"),
    )

    all_images = []
    all_metadata = {
        "config": config,
        "preprocess_signature": {
            "color_mode": "L",
            "resize": {
                "policy": config.get("resize_policy"),
                "fixed_hw": list(config["image_size"]) if config.get("image_size") else None,
                "note": "V9 mainline uses fixed resize; all images are resized to image_size to match the official 369x369 default.",
            },
            "normalize": {"enabled": True, "method": "(x-mean)/(std+1e-8)", "per_image": True},
        },
        "sources": {},
        "creation_time": datetime.now().isoformat(),
        "version": "V9",
        "alignment_notes": config.get("alignment_notes", ""),
    }

    for source_name, source_dir in config["normal_data_sources"].items():
        if source_name in processed_sources:
            continue
        if not os.path.exists(source_dir):
            print(f"警告: 目录不存在: {source_dir}")
            continue
        images, paths, hashes = load_normal_images(source_dir, config["num_normal_samples"], config)
        all_metadata["sources"][source_name] = {
            "directory": source_dir,
            "num_images": len(images),
            "image_paths": paths,
            "image_hashes": hashes,
        }
        if isinstance(images, list):
            all_images.extend(images)
        else:
            all_images.extend([images[i] for i in range(len(images))])

        processed_sources[source_name] = {"num_images": len(images), "completed": True}
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"processed_sources": processed_sources, "timestamp": datetime.now().isoformat()}, f, indent=2)

    if len(all_images) == 0:
        print("没有新数据需要处理")
        return None

    print(f"\n总图像数: {len(all_images)}")
    print("\n构建 Feature Bank...")
    source_info = ", ".join(config["normal_data_sources"].keys())
    feature_bank = model.build_feature_bank(all_images, source_info=source_info)

    expected_dim = 2 * int(config["hidden_size"])
    if bool(config.get("feature_with_bias", False)):
        expected_dim += 1
    actual_dim = int(feature_bank.shape[1])
    print(f"\n特征维度验证: expected={expected_dim}, actual={actual_dim}")

    model.save_feature_bank(output_path)
    feature_bank_to_save = feature_bank.detach().cpu()
    print(f"\nFeature Bank V9 保存至: {output_path}")
    print(f"形状: {feature_bank_to_save.shape}")

    all_metadata["feature_bank_bundle_format"] = "res_sam_fb_v2"
    all_metadata["feature_bank_shape"] = list(feature_bank_to_save.shape)
    all_metadata["feature_bank_path"] = output_path
    all_metadata["feature_dim_expected"] = expected_dim
    all_metadata["feature_dim_actual"] = actual_dim
    all_metadata["dimension_match"] = (actual_dim == expected_dim)

    metadata_path = os.path.join(config["output_dir"], config["metadata_file"])
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except Exception:
            pass

    return feature_bank_to_save


if __name__ == "__main__":
    preflight_faiss_or_raise()
    # 复用当前唯一数据布局，并固定输出到 v9 Feature Bank 目录。
    CONFIG = apply_layout_to_config_01(dict(CONFIG), BASE_DIR, "v9")
    CONFIG["version"] = "V9"
    CONFIG["output_dir"] = os.path.join(BASE_DIR, "outputs", "feature_banks_v9")
    CONFIG["output_file"] = "feature_bank_v9.pth"
    CONFIG["metadata_file"] = "metadata.json"
    CONFIG["normal_data_sources"] = {"augmented_intact": os.path.join(BASE_DIR, "data", "GPR_data", "augmented_intact")}
    CONFIG["output_dir"] = _to_abs(BASE_DIR, CONFIG.get("output_dir", ""))
    CONFIG["normal_data_sources"] = {k: _to_abs(BASE_DIR, v) for k, v in CONFIG.get("normal_data_sources", {}).items()}
    with torch.no_grad():
        build_feature_bank(CONFIG)


