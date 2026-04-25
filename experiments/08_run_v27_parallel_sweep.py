"""
V27 parallel sweep wrapper.

用途：
- V27 已固定为单主线 `base`
- 不再维护多 case 并行调度
- 保留 `08` 入口，仅转调 `07_run_v27_quick_sweep.py`
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPT_07 = BASE_DIR / "experiments" / "07_run_v27_quick_sweep.py"


def main() -> int:
    env = os.environ.copy()
    cmd = [PYTHON, str(SCRIPT_07)]
    print("V27 已固定为单主线，08 直接转调 07。")
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
