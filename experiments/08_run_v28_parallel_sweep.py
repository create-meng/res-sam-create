"""
V28 parallel mechanism sweep runner.

用途：
- 将 V28 的 4 个固定机制 case 拆成两路并行
- 不做参数搜索，只做机制组合对照
- 共享同一个 feature bank
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPT_07 = BASE_DIR / "experiments" / "07_run_v28_quick_sweep.py"


def main() -> int:
    common_env = os.environ.copy()

    runners = [
        {"name": "part_a", "case_indices": "0,1"},
        {"name": "part_b", "case_indices": "2,3"},
    ]

    procs: list[subprocess.Popen] = []
    try:
        for i, runner in enumerate(runners):
            env = common_env.copy()
            env["V28_CASE_INDICES"] = runner["case_indices"]
            env["BUILD_BANK"] = "1" if i == 0 else "0"
            cmd = [PYTHON, str(SCRIPT_07)]
            print(f"[{runner['name']}] >>> {' '.join(cmd)}  (V28_CASE_INDICES={runner['case_indices']}, BUILD_BANK={env['BUILD_BANK']})")
            procs.append(subprocess.Popen(cmd, cwd=str(BASE_DIR), env=env))

        rc = 0
        for proc in procs:
            proc.wait()
            if proc.returncode != 0 and rc == 0:
                rc = proc.returncode
        return rc
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
