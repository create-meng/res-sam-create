"""
V29 parallel utilities new-mechanism sweep runner.

用途：
- 将 V29 的 5 个固定机制 case 拆成两路并行
- 共享同一个 feature bank
- 自动等待共享 bank 生成完成后再启动第二路
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
SCRIPT_07 = BASE_DIR / "experiments" / "07_run_v29_quick_sweep.py"
BANK_PATH = BASE_DIR / "outputs" / "feature_banks_v29" / "feature_bank_v29.pth"
WAIT_TIMEOUT_SEC = 60 * 60


def wait_for_shared_bank(timeout_sec: int = WAIT_TIMEOUT_SEC) -> None:
    print(f"[wait] waiting for shared feature bank: {BANK_PATH}")
    start = time.time()
    while True:
        if BANK_PATH.exists() and BANK_PATH.stat().st_size > 0:
            print(f"[wait] shared feature bank ready: {BANK_PATH}")
            return
        if time.time() - start > timeout_sec:
            raise TimeoutError(f"Timed out waiting for shared feature bank: {BANK_PATH}")
        time.sleep(2.0)


def main() -> int:
    common_env = os.environ.copy()
    cmd = [PYTHON, str(SCRIPT_07)]

    env_a = common_env.copy()
    env_a["V29_CASE_INDICES"] = "0,1,2"
    env_a["BUILD_BANK"] = "auto"
    print(f"[part_a] >>> {' '.join(cmd)}  (V29_CASE_INDICES={env_a['V29_CASE_INDICES']}, BUILD_BANK={env_a['BUILD_BANK']})")
    proc_a = subprocess.Popen(cmd, cwd=str(BASE_DIR), env=env_a)

    proc_b = None
    try:
        wait_for_shared_bank()

        env_b = common_env.copy()
        env_b["V29_CASE_INDICES"] = "3,4"
        env_b["BUILD_BANK"] = "0"
        print(f"[part_b] >>> {' '.join(cmd)}  (V29_CASE_INDICES={env_b['V29_CASE_INDICES']}, BUILD_BANK={env_b['BUILD_BANK']})")
        proc_b = subprocess.Popen(cmd, cwd=str(BASE_DIR), env=env_b)

        rc = 0
        proc_a.wait()
        if proc_a.returncode != 0:
            rc = proc_a.returncode
        if proc_b is not None:
            proc_b.wait()
            if proc_b.returncode != 0 and rc == 0:
                rc = proc_b.returncode
        return rc
    finally:
        for proc in [proc_a, proc_b]:
            if proc is not None and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
