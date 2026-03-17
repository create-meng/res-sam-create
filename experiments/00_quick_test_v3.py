from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], cwd: Path, env: dict[str, str], log_fp):
    header = [
        "=" * 80,
        "RUN:",
        f"  CWD: {str(cwd)}",
        f"  CMD: {' '.join(cmd)}",
        f"  MAX_IMAGES_PER_CATEGORY: {env.get('MAX_IMAGES_PER_CATEGORY', '')}",
        "=" * 80,
    ]
    for line in header:
        print(line)
        log_fp.write(line + "\n")
    log_fp.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log_fp.write(line)
    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def _python_cmd() -> list[str]:
    """Return a python command that runs inside conda env `res-sam` when needed."""
    if os.environ.get("CONDA_DEFAULT_ENV", "") == "res-sam":
        return [sys.executable]
    return ["conda", "run", "-n", "res-sam", "python"]


def main():
    parser = argparse.ArgumentParser(
        description="Temporary one-click sanity test for Res-SAM V3 (10 samples per category by default)."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=10,
        help="Max images per category for quick test (default: 10).",
    )
    parser.add_argument(
        "--skip-build-feature-bank",
        action="store_true",
        help="Skip feature bank building even if feature bank file is missing.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    experiments_dir = repo_root / "experiments"

    logs_dir = repo_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"quick_test_v3_{time.strftime('%Y%m%d_%H%M%S')}.log"

    feature_bank = repo_root / "outputs" / "feature_banks_v3" / "feature_bank_v3.pth"

    env = os.environ.copy()
    if args.max_images and args.max_images > 0:
        env["MAX_IMAGES_PER_CATEGORY"] = str(args.max_images)

    py_cmd = _python_cmd()

    print("=" * 80)
    print("Res-SAM V3 quick test")
    print("LOG:")
    print(" ", str(log_path))
    print("=" * 80)

    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write("Res-SAM V3 quick test\n")
        log_fp.write(f"LOG_PATH: {str(log_path)}\n")
        log_fp.write(f"REPO_ROOT: {str(repo_root)}\n")
        log_fp.write(f"MAX_IMAGES_PER_CATEGORY: {env.get('MAX_IMAGES_PER_CATEGORY', '')}\n")
        log_fp.write("\n")
        log_fp.flush()

    # Step 1: Feature bank
        if not args.skip_build_feature_bank:
            if feature_bank.exists():
                msg = f"[SKIP] Feature bank exists: {feature_bank}"
                print(msg)
                log_fp.write(msg + "\n")
                log_fp.flush()
            else:
                _run([*py_cmd, str(experiments_dir / "01_build_feature_bank_v3.py")], cwd=repo_root, env=env, log_fp=log_fp)

        _run([*py_cmd, str(experiments_dir / "02_inference_auto_v3.py")], cwd=repo_root, env=env, log_fp=log_fp)
        _run([*py_cmd, str(experiments_dir / "03_inference_click_v3.py")], cwd=repo_root, env=env, log_fp=log_fp)
        _run([*py_cmd, str(experiments_dir / "04_evaluate_and_visualize_v3.py")], cwd=repo_root, env=env, log_fp=log_fp)

        print("\nDONE: quick test finished.")
        log_fp.write("\nDONE: quick test finished.\n")
        log_fp.flush()


if __name__ == "__main__":
    main()
