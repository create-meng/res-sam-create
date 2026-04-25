"""
V27 quick sweep runner.

用途：
- 运行 v27 固定主线
- 每类默认只跑 30 张样本
- 自动按 01 / 02 / 03 顺序完成建库、推理、评估
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
        "BANK_ROOT_SUFFIX": env_str("BANK_SUFFIX", ""),
        "MAX_IMAGES_PER_CATEGORY": env_str("MAX_IMAGES_PER_CATEGORY", "30"),
        "SECONDARY_FILTER_MIN_MEAN_PATCH": env_str("SECONDARY_FILTER_MIN_MEAN_PATCH", "0.275"),
        "SECONDARY_FILTER_BOX_SCORE_MIN": env_str("SECONDARY_FILTER_BOX_SCORE_MIN", "0.0"),
    }


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v27{bank_suffix}" / f"feature_bank_v27{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v27" / "feature_bank_v27.pth"


def run_cmd(script_rel: str, common_env: dict[str, str], env_extra: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(common_env)
    env.update(env_extra)
    script_path = BASE_DIR / script_rel
    cmd = [PYTHON, str(script_path)]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)


def resolve_build_bank(mode: str, bank_suffix: str) -> bool:
    lower = mode.lower()
    if lower == "1":
        return True
    if lower == "0":
        return False
    if lower == "auto":
        return not bank_file_path(bank_suffix).exists()
    raise ValueError(f"Unsupported BUILD_BANK={mode!r}, expected auto/0/1")


def run_sweep(*, build_bank_default: str, title: str) -> int:
    common_env = build_common_env()
    title = env_str("V27_SWEEP_TITLE", title)
    build_bank_mode = env_str("BUILD_BANK", build_bank_default).lower()
    bank_suffix = common_env["BANK_ROOT_SUFFIX"]
    case_build_bank = resolve_build_bank(build_bank_mode, bank_suffix)

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"TITLE={title}")
    print(f"BANK_ROOT_SUFFIX={common_env['BANK_ROOT_SUFFIX']!r}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
    print(f"SECONDARY_FILTER_MIN_MEAN_PATCH={common_env['SECONDARY_FILTER_MIN_MEAN_PATCH']}")
    print(f"SECONDARY_FILTER_BOX_SCORE_MIN={common_env['SECONDARY_FILTER_BOX_SCORE_MIN']}")
    print("PIPELINE=base_only")
    print(f"BUILD_BANK_MODE={build_bank_mode}")
    print(f"BANK_PATH={bank_file_path(bank_suffix)}")
    print(f"BUILD_BANK={int(case_build_bank)}")

    case_env = {
        "BANK_SUFFIX": bank_suffix,
        "OUTPUT_SUFFIX": "",
    }

    if case_build_bank:
        print("\n=== BUILD FEATURE BANK ===")
        run_cmd("experiments/01_build_feature_bank_v27.py", common_env, case_env)
    else:
        print("\n=== SKIP FEATURE BANK BUILD: existing bank detected or BUILD_BANK=0 ===")

    print("\n=== RUN FIXED V27 PIPELINE ===")
    run_cmd("experiments/02_inference_auto_v27.py", common_env, case_env)
    run_cmd("experiments/03_evaluate_v27.py", common_env, case_env)

    print(f"\n{title} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_sweep(
            build_bank_default="auto",
            title="V27 fixed-base quick sweep",
        )
    )

