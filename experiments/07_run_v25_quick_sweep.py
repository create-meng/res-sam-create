"""
V25 quick sweep runner.

用途：
- 一次性跑完 v25 的完整开关覆盖测试
- 每类默认只跑 30 张样本
- 自动调用 01 / 02 / 03

可选环境变量：
- BUILD_BANK=auto|0|1   auto: 有 bank 就跳过 01，否则创建
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


def env_str(name: str, default: str) -> str:
    return (os.getenv(name, default) or default).strip()


def build_common_env() -> dict[str, str]:
    return {
        "BANK_SUFFIX": env_str("BANK_SUFFIX", ""),
        "MAX_IMAGES_PER_CATEGORY": env_str("MAX_IMAGES_PER_CATEGORY", "30"),
        "NEIGHBORHOOD_FUSION_KERNEL_SIZE": env_str("NEIGHBORHOOD_FUSION_KERNEL_SIZE", "5"),
        "NEIGHBORHOOD_FUSION_WEIGHT": env_str("NEIGHBORHOOD_FUSION_WEIGHT", "0.35"),
        "NEIGHBORHOOD_FUSION_MODE": env_str("NEIGHBORHOOD_FUSION_MODE", "mean"),
    }


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v25{bank_suffix}" / f"feature_bank_v25{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v25" / "feature_bank_v25.pth"


ALL_CASES = [
    ("base", 0),
    ("nfuse", 1),
]


def run_cmd(script_rel: str, common_env: dict[str, str], env_extra: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(common_env)
    env.update(env_extra)
    script_path = BASE_DIR / script_rel
    cmd = [PYTHON, str(script_path)]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)


def resolve_build_bank(common_env: dict[str, str], default_mode: str) -> bool:
    mode = env_str("BUILD_BANK", default_mode).lower()
    if mode == "1":
        return True
    if mode == "0":
        return False
    if mode == "auto":
        return not bank_file_path(common_env["BANK_SUFFIX"]).exists()
    raise ValueError(f"Unsupported BUILD_BANK={mode!r}, expected auto/0/1")


def run_sweep(cases: list[tuple[str, int]], *,
              build_bank_default: str, title: str) -> int:
    common_env = build_common_env()
    build_bank = resolve_build_bank(common_env, build_bank_default)
    case_indices_raw = env_str("V25_CASE_INDICES", "")
    if case_indices_raw:
        indices = [int(part.strip()) for part in case_indices_raw.split(",") if part.strip()]
        cases = [cases[i] for i in indices]
    title = env_str("V25_SWEEP_TITLE", title)

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"TITLE={title}")
    print(f"BANK_SUFFIX={common_env['BANK_SUFFIX']!r}")
    print(f"BANK_PATH={bank_file_path(common_env['BANK_SUFFIX'])}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
    print(f"TOTAL_CASES={len(cases)}")
    print(f"BUILD_BANK={int(build_bank)}")
    print(f"BUILD_BANK_MODE={env_str('BUILD_BANK', build_bank_default).lower()}")

    if build_bank:
        print("\n=== BUILD FEATURE BANK ===")
        run_cmd("experiments/01_build_feature_bank_v25.py", common_env, {})
    else:
        print("\n=== SKIP FEATURE BANK BUILD: existing bank detected or BUILD_BANK=0 ===")

    for name, nfuse in cases:
        print(f"\n=== CASE: {name} ===")
        case_env = {
            "OUTPUT_SUFFIX": f"_{name}",
            "USE_NEIGHBORHOOD_SCORE_FUSION": str(nfuse),
        }
        run_cmd("experiments/02_inference_auto_v25.py", common_env, case_env)
        run_cmd("experiments/03_evaluate_v25.py", common_env, case_env)

    print(f"\n{title} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_sweep(
            ALL_CASES,
            build_bank_default="auto",
            title="V25 quick sweep",
        )
    )
