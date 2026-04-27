"""
V30 quick sweep runner.

用途：
- 只比较新的结构性 utilities 机制
- 不做参数搜索，只比较机制
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
        "UTILITIES_ASPECT_RATIO_THRESHOLD": env_str("UTILITIES_ASPECT_RATIO_THRESHOLD", "5.5"),
        "UTILITIES_SCORE_THRESHOLD_P80": env_str("UTILITIES_SCORE_THRESHOLD_P80", "0.20"),
        "UTILITIES_SECONDARY_FILTER_MIN_AREA_ORIG": env_str("UTILITIES_SECONDARY_FILTER_MIN_AREA_ORIG", "1200"),
        "UTILITIES_SECONDARY_FILTER_MIN_MEAN_PATCH": env_str("UTILITIES_SECONDARY_FILTER_MIN_MEAN_PATCH", "0.18"),
        "UTILITIES_SECONDARY_FILTER_BOX_SCORE_MIN": env_str("UTILITIES_SECONDARY_FILTER_BOX_SCORE_MIN", "0.0"),
        "UTILITIES_SMALL_TARGET_BLEND_ALPHA": env_str("UTILITIES_SMALL_TARGET_BLEND_ALPHA", "0.35"),
        "UTILITIES_SMALL_TARGET_SIGMA": env_str("UTILITIES_SMALL_TARGET_SIGMA", "1.2"),
        "UTILITIES_SMALL_TARGET_PERCENTILE": env_str("UTILITIES_SMALL_TARGET_PERCENTILE", "85.0"),
        "UTILITIES_REFINER_PERCENTILE": env_str("UTILITIES_REFINER_PERCENTILE", "90.0"),
        "UTILITIES_REFINER_EXPAND_PIXELS": env_str("UTILITIES_REFINER_EXPAND_PIXELS", "8"),
        "UTILITIES_REFINER_MIN_PEAK_RATIO": env_str("UTILITIES_REFINER_MIN_PEAK_RATIO", "1.10"),
        "UTILITIES_REFINER_MAX_CORE_AREA_RATIO": env_str("UTILITIES_REFINER_MAX_CORE_AREA_RATIO", "0.55"),
        "UTILITIES_REFINER_MIN_CORE_PIXELS": env_str("UTILITIES_REFINER_MIN_CORE_PIXELS", "24"),
        "UTILITIES_DIRECTIONAL_ALPHA": env_str("UTILITIES_DIRECTIONAL_ALPHA", "0.28"),
        "UTILITIES_DIRECTIONAL_SIGMA": env_str("UTILITIES_DIRECTIONAL_SIGMA", "1.4"),
        "UTILITIES_DIRECTIONAL_PERCENTILE": env_str("UTILITIES_DIRECTIONAL_PERCENTILE", "82.0"),
        "UTILITIES_CFAR_WINDOW": env_str("UTILITIES_CFAR_WINDOW", "25"),
        "UTILITIES_CFAR_GUARD": env_str("UTILITIES_CFAR_GUARD", "5"),
        "UTILITIES_CFAR_K": env_str("UTILITIES_CFAR_K", "1.35"),
        "UTILITIES_CFAR_FLOOR_RATIO": env_str("UTILITIES_CFAR_FLOOR_RATIO", "0.88"),
        "UTILITIES_NEIGHBOR_EXPAND_PIXELS": env_str("UTILITIES_NEIGHBOR_EXPAND_PIXELS", "18"),
        "UTILITIES_NEIGHBOR_MIN_CONTRAST_RATIO": env_str("UTILITIES_NEIGHBOR_MIN_CONTRAST_RATIO", "1.08"),
        "UTILITIES_NEIGHBOR_MIN_INNER_DELTA": env_str("UTILITIES_NEIGHBOR_MIN_INNER_DELTA", "0.015"),
        "UTILITIES_POSITION_BORDER_MARGIN_RATIO": env_str("UTILITIES_POSITION_BORDER_MARGIN_RATIO", "0.04"),
        "UTILITIES_POSITION_MIN_CENTER_RATIO": env_str("UTILITIES_POSITION_MIN_CENTER_RATIO", "0.06"),
        "UTILITIES_POSITION_MAX_CENTER_RATIO": env_str("UTILITIES_POSITION_MAX_CENTER_RATIO", "0.94"),
        "UTILITIES_PROPOSAL_SIGMA": env_str("UTILITIES_PROPOSAL_SIGMA", "1.2"),
        "UTILITIES_PROPOSAL_PERCENTILE": env_str("UTILITIES_PROPOSAL_PERCENTILE", "84.0"),
        "UTILITIES_PROPOSAL_DILATE": env_str("UTILITIES_PROPOSAL_DILATE", "2"),
        "UTILITIES_PROPOSAL_BLEND_ALPHA": env_str("UTILITIES_PROPOSAL_BLEND_ALPHA", "0.30"),
        "UTILITIES_TEMPLATE_BLEND_ALPHA": env_str("UTILITIES_TEMPLATE_BLEND_ALPHA", "0.22"),
        "UTILITIES_TEMPLATE_RESIDUAL_PERCENTILE": env_str("UTILITIES_TEMPLATE_RESIDUAL_PERCENTILE", "90.0"),
        "UTILITIES_MICRO_SCALE_STRIDES": env_str("UTILITIES_MICRO_SCALE_STRIDES", "2,3"),
        "UTILITIES_MICRO_SCALE_BLEND_ALPHA": env_str("UTILITIES_MICRO_SCALE_BLEND_ALPHA", "0.30"),
        "UTILITIES_MICRO_SCALE_USE_MAX_FUSION": env_str("UTILITIES_MICRO_SCALE_USE_MAX_FUSION", "1"),
    }


def build_case(name: str, **env_overrides: str) -> dict[str, object]:
    return {"name": name, "env": env_overrides}


ALL_CASES = [
    build_case(
        "base",
        USE_UTILITIES_CATEGORY_FILTER="0",
        USE_UTILITIES_SMALL_TARGET_ENHANCE="0",
        USE_UTILITIES_BOX_REFINER="0",
        USE_UTILITIES_DIRECTIONAL_PRIOR="0",
        USE_UTILITIES_LOCAL_CFAR="0",
        USE_UTILITIES_POSITION_NEIGHBOR_GATE="0",
        USE_UTILITIES_PROPOSAL_PRIOR="0",
        USE_UTILITIES_NORMAL_TEMPLATE="0",
        USE_UTILITIES_MICRO_SCALE="0",
    ),
    build_case(
        "utilities_proposal",
        USE_UTILITIES_CATEGORY_FILTER="0",
        USE_UTILITIES_SMALL_TARGET_ENHANCE="0",
        USE_UTILITIES_BOX_REFINER="0",
        USE_UTILITIES_DIRECTIONAL_PRIOR="0",
        USE_UTILITIES_LOCAL_CFAR="0",
        USE_UTILITIES_POSITION_NEIGHBOR_GATE="0",
        USE_UTILITIES_PROPOSAL_PRIOR="1",
        USE_UTILITIES_NORMAL_TEMPLATE="0",
        USE_UTILITIES_MICRO_SCALE="0",
    ),
    build_case(
        "utilities_template",
        USE_UTILITIES_CATEGORY_FILTER="0",
        USE_UTILITIES_SMALL_TARGET_ENHANCE="0",
        USE_UTILITIES_BOX_REFINER="0",
        USE_UTILITIES_DIRECTIONAL_PRIOR="0",
        USE_UTILITIES_LOCAL_CFAR="0",
        USE_UTILITIES_POSITION_NEIGHBOR_GATE="0",
        USE_UTILITIES_PROPOSAL_PRIOR="1",
        USE_UTILITIES_NORMAL_TEMPLATE="1",
        USE_UTILITIES_MICRO_SCALE="0",
    ),
    build_case(
        "utilities_micro",
        USE_UTILITIES_CATEGORY_FILTER="0",
        USE_UTILITIES_SMALL_TARGET_ENHANCE="0",
        USE_UTILITIES_BOX_REFINER="0",
        USE_UTILITIES_DIRECTIONAL_PRIOR="0",
        USE_UTILITIES_LOCAL_CFAR="0",
        USE_UTILITIES_POSITION_NEIGHBOR_GATE="0",
        USE_UTILITIES_PROPOSAL_PRIOR="1",
        USE_UTILITIES_NORMAL_TEMPLATE="0",
        USE_UTILITIES_MICRO_SCALE="1",
    ),
    build_case(
        "utilities_all3",
        USE_UTILITIES_CATEGORY_FILTER="0",
        USE_UTILITIES_SMALL_TARGET_ENHANCE="0",
        USE_UTILITIES_BOX_REFINER="0",
        USE_UTILITIES_DIRECTIONAL_PRIOR="0",
        USE_UTILITIES_LOCAL_CFAR="0",
        USE_UTILITIES_POSITION_NEIGHBOR_GATE="0",
        USE_UTILITIES_PROPOSAL_PRIOR="1",
        USE_UTILITIES_NORMAL_TEMPLATE="1",
        USE_UTILITIES_MICRO_SCALE="1",
    ),
]


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v30{bank_suffix}" / f"feature_bank_v30{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v30" / "feature_bank_v30.pth"


def case_output_suffix(case_name: str) -> str:
    return f"_{case_name}"


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
    title = env_str("V30_SWEEP_TITLE", title)
    build_bank_mode = env_str("BUILD_BANK", build_bank_default).lower()
    bank_suffix = common_env["BANK_ROOT_SUFFIX"]
    case_indices_raw = env_str("V30_CASE_INDICES", "")
    cases = ALL_CASES
    if case_indices_raw:
        indices = [int(part.strip()) for part in case_indices_raw.split(",") if part.strip()]
        cases = [ALL_CASES[i] for i in indices]

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"TITLE={title}")
    print(f"BANK_ROOT_SUFFIX={bank_suffix!r}")
    print(f"BANK_PATH={bank_file_path(bank_suffix)}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
    print(f"TOTAL_CASES={len(cases)}")

    build_bank_once = resolve_build_bank(build_bank_mode, bank_suffix)
    if build_bank_once:
        print("\n=== BUILD FEATURE BANK (shared) ===")
        run_cmd(
            "experiments/01_build_feature_bank_v30.py",
            common_env,
            {"BANK_SUFFIX": bank_suffix, "OUTPUT_SUFFIX": ""},
        )
    else:
        print("\n=== SKIP FEATURE BANK BUILD: shared bank exists or BUILD_BANK=0 ===")

    for case in cases:
        case_name = str(case["name"])
        output_suffix = case_output_suffix(case_name)
        case_env = {
            "BANK_SUFFIX": bank_suffix,
            "OUTPUT_SUFFIX": output_suffix,
            **dict(case["env"]),
        }

        print(f"\n=== CASE: {case_name} ===")
        print(f"OUTPUT_SUFFIX={output_suffix}")
        print(f"USE_UTILITIES_PROPOSAL_PRIOR={case_env.get('USE_UTILITIES_PROPOSAL_PRIOR')}")
        print(f"USE_UTILITIES_NORMAL_TEMPLATE={case_env.get('USE_UTILITIES_NORMAL_TEMPLATE')}")
        print(f"USE_UTILITIES_MICRO_SCALE={case_env.get('USE_UTILITIES_MICRO_SCALE')}")

        run_cmd("experiments/02_inference_auto_v30.py", common_env, case_env)
        run_cmd("experiments/03_evaluate_v30.py", common_env, case_env)

    print(f"\n{title} completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        run_sweep(
            build_bank_default="auto",
            title="V30 utilities structural-mechanism sweep",
        )
    )
