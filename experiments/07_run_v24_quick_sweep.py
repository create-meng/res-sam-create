"""
V24 quick sweep runner.

用途：
- 一次性跑完 v24 的完整开关覆盖测试
- 每类默认只跑 30 张样本
- 自动调用 02_inference_auto_v24.py 和 03_evaluate_v24.py

可选环境变量：
- BUILD_BANK=0          跳过 01_build_feature_bank_v24.py
- BANK_SUFFIX=_b1       指定 feature bank 后缀
- MAX_IMAGES_PER_CATEGORY=30
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _env_str(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip()


COMMON_ENV = {
    "RES_SAM_ALLOW_SKLEARN_KNN": _env_str("RES_SAM_ALLOW_SKLEARN_KNN", "1"),
    "RES_SAM_SKIP_FAISS_PREFLIGHT": _env_str("RES_SAM_SKIP_FAISS_PREFLIGHT", "1"),
    "BANK_SUFFIX": _env_str("BANK_SUFFIX", ""),
    "MAX_IMAGES_PER_CATEGORY": _env_str("MAX_IMAGES_PER_CATEGORY", "30"),
    "SECONDARY_FILTER_MIN_AREA_ORIG": _env_str("SECONDARY_FILTER_MIN_AREA_ORIG", "3500"),
    "SECONDARY_FILTER_MIN_MEAN_PATCH": _env_str("SECONDARY_FILTER_MIN_MEAN_PATCH", "0.27"),
    "NEIGHBORHOOD_KERNEL_SIZE": _env_str("NEIGHBORHOOD_KERNEL_SIZE", "3"),
    "NEIGHBORHOOD_MIN_SUPPORT_RATIO": _env_str("NEIGHBORHOOD_MIN_SUPPORT_RATIO", "0.45"),
    "NEIGHBORHOOD_SCORE_MARGIN": _env_str("NEIGHBORHOOD_SCORE_MARGIN", "0.00"),
    "BOX_REFINE_THRESHOLD_RATIO": _env_str("BOX_REFINE_THRESHOLD_RATIO", "0.88"),
    "BOX_REFINE_MIN_PIXELS": _env_str("BOX_REFINE_MIN_PIXELS", "25"),
    "BOX_REFINE_EXPAND_PIXELS": _env_str("BOX_REFINE_EXPAND_PIXELS", "4"),
    "SHIFT_ROBUST_RADIUS": _env_str("SHIFT_ROBUST_RADIUS", "1"),
    "SHIFT_ROBUST_REDUCE": _env_str("SHIFT_ROBUST_REDUCE", "median"),
}


CASES = [
    ("base", 0, 0, 0, 0),
    ("secf", 1, 0, 0, 0),
    ("neigh", 0, 1, 0, 0),
    ("refine", 0, 0, 1, 0),
    ("shift", 0, 0, 0, 1),
    ("secf_neigh", 1, 1, 0, 0),
    ("secf_refine", 1, 0, 1, 0),
    ("secf_shift", 1, 0, 0, 1),
    ("neigh_refine", 0, 1, 1, 0),
    ("neigh_shift", 0, 1, 0, 1),
    ("refine_shift", 0, 0, 1, 1),
    ("secf_neigh_refine", 1, 1, 1, 0),
    ("secf_neigh_shift", 1, 1, 0, 1),
    ("secf_refine_shift", 1, 0, 1, 1),
    ("neigh_refine_shift", 0, 1, 1, 1),
    ("all4", 1, 1, 1, 1),
]


def run_cmd(script_rel: str, env_extra: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(COMMON_ENV)
    env.update(env_extra)
    script_path = BASE_DIR / script_rel
    cmd = [PYTHON, str(script_path)]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)


def main() -> int:
    build_bank = _env_str("BUILD_BANK", "1") == "1"

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"BANK_SUFFIX={COMMON_ENV['BANK_SUFFIX']!r}")
    print(f"MAX_IMAGES_PER_CATEGORY={COMMON_ENV['MAX_IMAGES_PER_CATEGORY']}")
    print(f"TOTAL_CASES={len(CASES)}")

    if build_bank:
        print("\n=== BUILD FEATURE BANK ===")
        run_cmd("experiments/01_build_feature_bank_v24.py", {})

    for name, secf, neigh, refine, shift in CASES:
        print(f"\n=== CASE: {name} ===")
        case_env = {
            "OUTPUT_SUFFIX": f"_{name}",
            "USE_SECONDARY_FILTER": str(secf),
            "USE_NEIGHBORHOOD_CONSISTENCY": str(neigh),
            "USE_BOX_REFINEMENT": str(refine),
            "USE_SHIFT_ROBUST_AGGREGATION": str(shift),
        }
        run_cmd("experiments/02_inference_auto_v24.py", case_env)
        run_cmd("experiments/03_evaluate_v24.py", case_env)

    print("\nAll v24 quick sweep cases completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
