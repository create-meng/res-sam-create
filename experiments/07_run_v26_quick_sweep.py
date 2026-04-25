"""
V26 quick sweep runner.

用途：
- 一次性跑完 v26 的完整新方案覆盖测试
- 每类默认只跑 30 张样本
- 自动按 case 处理 01 / 02 / 03
- bank 侧方案和推理侧方案统一由环境变量透传
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
        "SOFTPATCH_KEEP_RATIO": env_str("SOFTPATCH_KEEP_RATIO", "0.85"),
        "SOFTPATCH_KNN_K": env_str("SOFTPATCH_KNN_K", "5"),
        "SOFTPATCH_SCORE_POWER": env_str("SOFTPATCH_SCORE_POWER", "1.0"),
        "PATCHCORE_TOPK": env_str("PATCHCORE_TOPK", "3"),
        "PATCHCORE_REWEIGHT_LAMBDA": env_str("PATCHCORE_REWEIGHT_LAMBDA", "0.35"),
        "THRESHOLD_NORMAL_STD_MULT": env_str("THRESHOLD_NORMAL_STD_MULT", "2.5"),
        "THRESHOLD_IMAGE_STD_MULT": env_str("THRESHOLD_IMAGE_STD_MULT", "1.0"),
        "THRESHOLD_MIN_FLOOR_DELTA": env_str("THRESHOLD_MIN_FLOOR_DELTA", "0.0"),
    }


def build_case(name: str, bank_tag: str, **env_overrides: str) -> dict[str, object]:
    return {
        "name": name,
        "bank_tag": bank_tag,
        "env": env_overrides,
    }


ALL_CASES = [
    build_case("base", "base"),
    build_case("softpatch", "softpatch", USE_SOFTPATCH_BANK_FILTER="1"),
    build_case("patchcore", "base", USE_PATCHCORE_TOPK_AGG="1"),
    build_case("threshold", "base", USE_THRESHOLD_LEARNER="1"),
    build_case(
        "softpatch_patchcore",
        "softpatch",
        USE_SOFTPATCH_BANK_FILTER="1",
        USE_PATCHCORE_TOPK_AGG="1",
    ),
    build_case(
        "softpatch_threshold",
        "softpatch",
        USE_SOFTPATCH_BANK_FILTER="1",
        USE_THRESHOLD_LEARNER="1",
    ),
    build_case(
        "patchcore_threshold",
        "base",
        USE_PATCHCORE_TOPK_AGG="1",
        USE_THRESHOLD_LEARNER="1",
    ),
    build_case(
        "all3",
        "softpatch",
        USE_SOFTPATCH_BANK_FILTER="1",
        USE_PATCHCORE_TOPK_AGG="1",
        USE_THRESHOLD_LEARNER="1",
    ),
]


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v26{bank_suffix}" / f"feature_bank_v26{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v26" / "feature_bank_v26.pth"


def case_output_suffix(case_name: str) -> str:
    return f"_{case_name}"


def case_bank_suffix(root_suffix: str, bank_tag: str) -> str:
    suffix = root_suffix or ""
    if bank_tag == "base":
        return suffix
    return f"{suffix}_{bank_tag}" if suffix else f"_{bank_tag}"


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


def run_sweep(cases: list[dict[str, object]], *,
              build_bank_default: str, title: str) -> int:
    common_env = build_common_env()
    case_indices_raw = env_str("V26_CASE_INDICES", "")
    if case_indices_raw:
        indices = [int(part.strip()) for part in case_indices_raw.split(",") if part.strip()]
        cases = [cases[i] for i in indices]
    title = env_str("V26_SWEEP_TITLE", title)
    build_bank_mode = env_str("BUILD_BANK", build_bank_default).lower()

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"TITLE={title}")
    print(f"BANK_ROOT_SUFFIX={common_env['BANK_ROOT_SUFFIX']!r}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
    print(f"SECONDARY_FILTER_MIN_MEAN_PATCH={common_env['SECONDARY_FILTER_MIN_MEAN_PATCH']}")
    print(f"SECONDARY_FILTER_BOX_SCORE_MIN={common_env['SECONDARY_FILTER_BOX_SCORE_MIN']}")
    print(f"SOFTPATCH_KEEP_RATIO={common_env['SOFTPATCH_KEEP_RATIO']}")
    print(f"SOFTPATCH_KNN_K={common_env['SOFTPATCH_KNN_K']}")
    print(f"SOFTPATCH_SCORE_POWER={common_env['SOFTPATCH_SCORE_POWER']}")
    print(f"PATCHCORE_TOPK={common_env['PATCHCORE_TOPK']}")
    print(f"PATCHCORE_REWEIGHT_LAMBDA={common_env['PATCHCORE_REWEIGHT_LAMBDA']}")
    print(f"THRESHOLD_NORMAL_STD_MULT={common_env['THRESHOLD_NORMAL_STD_MULT']}")
    print(f"THRESHOLD_IMAGE_STD_MULT={common_env['THRESHOLD_IMAGE_STD_MULT']}")
    print(f"THRESHOLD_MIN_FLOOR_DELTA={common_env['THRESHOLD_MIN_FLOOR_DELTA']}")
    print(f"TOTAL_CASES={len(cases)}")
    print(f"BUILD_BANK_MODE={build_bank_mode}")

    for case in cases:
        case_name = str(case["name"])
        bank_tag = str(case["bank_tag"])
        env_extra = dict(case["env"])
        bank_suffix = case_bank_suffix(common_env["BANK_ROOT_SUFFIX"], bank_tag)
        output_suffix = case_output_suffix(case_name)
        case_build_bank = resolve_build_bank(build_bank_mode, bank_suffix)

        print(f"\n=== CASE: {case_name} ===")
        print(f"BANK_SUFFIX={bank_suffix!r}")
        print(f"BANK_PATH={bank_file_path(bank_suffix)}")
        print(f"OUTPUT_SUFFIX={output_suffix}")
        print(f"BUILD_BANK={int(case_build_bank)}")

        case_env = {
            "BANK_SUFFIX": bank_suffix,
            "OUTPUT_SUFFIX": output_suffix,
            **env_extra,
        }

        if case_build_bank:
            print("\n=== BUILD FEATURE BANK ===")
            run_cmd("experiments/01_build_feature_bank_v26.py", common_env, case_env)
        else:
            print("\n=== SKIP FEATURE BANK BUILD: existing bank detected or BUILD_BANK=0 ===")

        run_cmd("experiments/02_inference_auto_v26.py", common_env, case_env)
        run_cmd("experiments/03_evaluate_v26.py", common_env, case_env)

    print(f"\n{title} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_sweep(
            ALL_CASES,
            build_bank_default="auto",
            title="V26 quick sweep",
        )
    )
