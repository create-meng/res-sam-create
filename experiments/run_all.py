"""
Res-SAM experiment runner for active branches only.

Archived V3 scripts live under:
archive/experiments_v3_snapshot_20260326/

Active runnable branches in experiments/:
- v4
- v5
"""

import argparse
import os
import subprocess
import sys


ACTIVE_VERSIONS = ("v4", "v5")

STEPS_BY_VERSION = {
    "v4": {
        1: {
            "name": "Feature Bank Construction",
            "script": "01_build_feature_bank_v4.py",
            "description": "Build feature bank for v4",
        },
        2: {
            "name": "Fully Automatic Inference",
            "script": "02_inference_auto_v4.py",
            "description": "Run automatic inference for v4",
        },
        3: {
            "name": "Evaluation Metrics",
            "script": "04_evaluate_and_visualize_v4.py",
            "description": "Run evaluation and visualization for v4",
        },
        4: {
            "name": "Click-guided Inference",
            "script": "03_inference_click_v4.py",
            "description": "Run click-guided inference for v4",
        },
        5: {
            "name": "Clustering",
            "script": "05_clustering_v4.py",
            "description": "Run clustering analysis for v4",
        },
        6: {
            "name": "Ablation w/o SAM",
            "script": "06_ablation_wo_sam_v4.py",
            "description": "Run ablation without SAM for v4",
        },
        7: {
            "name": "Cross-environment",
            "script": "07_cross_environment_v4.py",
            "description": "Run cross-environment evaluation for v4",
        },
    },
    "v5": {
        1: {
            "name": "Feature Bank Construction",
            "script": "01_build_feature_bank_v5.py",
            "description": "Build feature bank for v5",
        },
        2: {
            "name": "Fully Automatic Inference",
            "script": "02_inference_auto_v5.py",
            "description": "Run automatic inference for v5",
        },
        3: {
            "name": "Evaluation Metrics",
            "script": "04_evaluate_and_visualize_v5.py",
            "description": "Run evaluation and visualization for v5",
        },
        4: {
            "name": "Click-guided Inference",
            "script": "03_inference_click_v5.py",
            "description": "Run click-guided inference for v5",
        },
        5: {
            "name": "Clustering",
            "script": "05_clustering_v5.py",
            "description": "Run clustering analysis for v5",
        },
        6: {
            "name": "Ablation w/o SAM",
            "script": "06_ablation_wo_sam_v5.py",
            "description": "Run ablation without SAM for v5",
        },
        7: {
            "name": "Cross-environment",
            "script": "07_cross_environment_v5.py",
            "description": "Run cross-environment evaluation for v5",
        },
    },
}


def _python_cmd() -> list[str]:
    return [sys.executable]


def _parse_steps(step_arg: str | None, valid_steps: list[int]) -> list[int]:
    if not step_arg:
        return valid_steps
    if "-" in step_arg:
        start, end = map(int, step_arg.split("-"))
        return [s for s in range(start, end + 1) if s in valid_steps]
    if "," in step_arg:
        return [int(s.strip()) for s in step_arg.split(",") if int(s.strip()) in valid_steps]
    step_num = int(step_arg)
    return [step_num] if step_num in valid_steps else []


def run_step(version: str, step_num: int) -> bool:
    steps = STEPS_BY_VERSION[version]
    if step_num not in steps:
        print(f"Error: invalid step number {step_num} for {version}")
        return False

    step = steps[step_num]
    script_path = os.path.join(os.path.dirname(__file__), step["script"])

    print(f"\n{'=' * 70}")
    print(f"{version.upper()} Step {step_num}: {step['name']}")
    print(f"Description: {step['description']}")
    print(f"Script: {step['script']}")
    print("=" * 70)

    if not os.path.exists(script_path):
        print(f"Error: script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [*_python_cmd(), script_path],
            cwd=os.path.dirname(__file__),
            check=False,
        )
        if result.returncode == 0:
            print(f"\nStep {step_num} completed successfully")
            return True
        print(f"\nStep {step_num} failed with return code {result.returncode}")
        return False
    except Exception as exc:
        print(f"\nStep {step_num} failed with error: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run active Res-SAM experiment branches (v4/v5)."
    )
    parser.add_argument(
        "--version",
        choices=ACTIVE_VERSIONS,
        default="v4",
        help="Which active branch to run.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="Steps to run, e.g. 1 or 2,4 or 1-7",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available steps for the selected version.",
    )
    args = parser.parse_args()

    steps = STEPS_BY_VERSION[args.version]
    valid_steps = list(steps.keys())

    if args.list:
        print(f"\nAvailable steps for {args.version}:")
        print("-" * 70)
        for num, step in steps.items():
            print(f"  {num}. {step['name']}: {step['description']}")
        print("\nArchived V3 scripts are not run from experiments/ anymore.")
        return

    steps_to_run = _parse_steps(args.step, valid_steps)
    if not steps_to_run:
        raise SystemExit("No valid steps selected.")

    print("=" * 70)
    print("Res-SAM Active Experiment Runner")
    print("=" * 70)
    print(f"Version: {args.version}")
    print(f"Steps to run: {steps_to_run}")
    print("Archived V3 snapshot: archive/experiments_v3_snapshot_20260326/")

    results = {}
    for step_num in steps_to_run:
        results[step_num] = run_step(args.version, step_num)
        if not results[step_num] and step_num < max(steps_to_run):
            print(f"\nWarning: step {step_num} failed. Continuing...")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for step_num in steps_to_run:
        status = "Success" if results[step_num] else "Failed"
        print(f"  Step {step_num} ({steps[step_num]['name']}): {status}")

    total = len(steps_to_run)
    success = sum(results.values())
    print(f"\nTotal: {success}/{total} steps completed successfully")


if __name__ == "__main__":
    main()
