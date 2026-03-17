import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, env: dict[str, str]):
    print("=" * 80)
    print("RUN:")
    print("  CWD:", str(cwd))
    print("  CMD:", " ".join(cmd))
    print("  MAX_IMAGES_PER_CATEGORY:", env.get("MAX_IMAGES_PER_CATEGORY", ""))
    print("=" * 80)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


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

    feature_bank = repo_root / "outputs" / "feature_banks_v3" / "feature_bank_v3.pth"

    env = os.environ.copy()
    if args.max_images and args.max_images > 0:
        env["MAX_IMAGES_PER_CATEGORY"] = str(args.max_images)

    py = sys.executable

    # Step 1: Feature bank
    if not args.skip_build_feature_bank:
        if feature_bank.exists():
            print(f"[SKIP] Feature bank exists: {feature_bank}")
        else:
            _run([py, str(experiments_dir / "01_build_feature_bank_v3.py")], cwd=repo_root, env=env)

    # Step 2: Fully automatic inference
    _run([py, str(experiments_dir / "02_inference_auto_v3.py")], cwd=repo_root, env=env)

    # Step 3: Click-guided inference
    _run([py, str(experiments_dir / "03_inference_click_v3.py")], cwd=repo_root, env=env)

    # Step 4: Evaluation (automatic + click)
    _run([py, str(experiments_dir / "04_evaluate_and_visualize_v3.py")], cwd=repo_root, env=env)

    print("\nDONE: quick test finished.")


if __name__ == "__main__":
    main()
