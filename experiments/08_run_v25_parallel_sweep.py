"""
V25 parallel quick sweep runner.

用途：
- 自动拆成两路并行运行
- 父进程按需只建一次 bank，再拆成前后两组
- 两路输出 case 不重叠，互不影响
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
    }


ALL_CASES = [
    ("base", 0, 0, 0, 0),
    ("nfuse", 1, 0, 0, 0),
]

SCRIPT_07 = BASE_DIR / "experiments" / "07_run_v25_quick_sweep.py"


def bank_file_path(bank_suffix: str) -> Path:
    if bank_suffix:
        return BASE_DIR / "outputs" / f"feature_banks_v25{bank_suffix}" / f"feature_bank_v25{bank_suffix}.pth"
    return BASE_DIR / "outputs" / "feature_banks_v25" / "feature_bank_v25.pth"


def resolve_build_bank(common_env: dict[str, str], default_mode: str) -> bool:
    mode = env_str("BUILD_BANK", default_mode).lower()
    if mode == "1":
        return True
    if mode == "0":
        return False
    if mode == "auto":
        return not bank_file_path(common_env["BANK_SUFFIX"]).exists()
    raise ValueError(f"Unsupported BUILD_BANK={mode!r}, expected auto/0/1")


def spawn_runner(case_indices: list[int], build_bank: str, title: str, common_env: dict[str, str]) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(common_env)
    env["BUILD_BANK"] = build_bank
    env["V25_CASE_INDICES"] = ",".join(str(i) for i in case_indices)
    env["V25_SWEEP_TITLE"] = title
    cmd = [PYTHON, str(SCRIPT_07)]
    print(">>>", " ".join(cmd), f"[{title}]")
    return subprocess.Popen(cmd, cwd=str(BASE_DIR), env=env)


def main() -> int:
    common_env = build_common_env()
    bank_suffix = common_env["BANK_SUFFIX"]

    first_half = list(range(0, len(ALL_CASES) // 2))
    second_half = list(range(len(ALL_CASES) // 2, len(ALL_CASES)))

    print(f"BASE_DIR={BASE_DIR}")
    print(f"PYTHON={PYTHON}")
    print(f"BANK_SUFFIX={bank_suffix!r}")
    print(f"BANK_PATH={bank_file_path(bank_suffix)}")
    print(f"MAX_IMAGES_PER_CATEGORY={common_env['MAX_IMAGES_PER_CATEGORY']}")
    print(f"TOTAL_CASES={len(ALL_CASES)}")
    print(f"PARALLEL_SPLIT={len(first_half)}+{len(second_half)}")

    build_bank = resolve_build_bank(common_env, "auto")
    print(f"BUILD_BANK={int(build_bank)}")
    print(f"BUILD_BANK_MODE={env_str('BUILD_BANK', 'auto').lower()}")

    if build_bank:
        env = os.environ.copy()
        env.update(common_env)
        cmd = [PYTHON, str(BASE_DIR / "experiments" / "01_build_feature_bank_v25.py")]
        print(">>>", " ".join(cmd), "[V25 parallel build bank]")
        subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)
    else:
        print("\n=== SKIP FEATURE BANK BUILD: existing bank detected or BUILD_BANK=0 ===")

    proc_a = spawn_runner(first_half, "0", "V25 parallel part A", common_env)
    proc_b = spawn_runner(second_half, "0", "V25 parallel part B", common_env)

    rc_a = proc_a.wait()
    rc_b = proc_b.wait()
    if rc_a != 0 or rc_b != 0:
        raise SystemExit(rc_a or rc_b)

    print("\nV25 parallel sweep completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
