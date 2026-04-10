"""
Res-SAM 当前主线实验运行器。

已归档的 V3 脚本位于：
已归档/experiments_v3_snapshot_20260326/

当前可运行主线：
- v6（论文优先主线）
"""

import argparse
import os
import subprocess
import sys

_RUN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RUN_ROOT not in sys.path:
    sys.path.insert(0, _RUN_ROOT)

from experiments.paper_constants import preflight_faiss_or_raise


ACTIVE_VERSIONS = ("v6",)

STEPS_BY_VERSION = {
    "v6": {
        1: {
            "name": "Feature Bank 构建",
            "script": "01_build_feature_bank_v6.py",
            "description": "为 v6 构建 Feature Bank",
        },
        2: {
            "name": "全自动推理",
            "script": "02_inference_auto_v6.py",
            "description": "运行 v6 全自动推理",
        },
        3: {
            "name": "评估与可视化",
            "script": "03_evaluate_and_visualize_v6.py",
            "description": "运行 v6 评估与可视化",
        },
        4: {
            "name": "异常聚类",
            "script": "04_clustering_v6.py",
            "description": "运行 v6 异常聚类",
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
        print(f"错误：{version} 不存在步骤 {step_num}")
        return False

    step = steps[step_num]
    script_path = os.path.join(os.path.dirname(__file__), step["script"])

    print(f"\n{'=' * 70}")
    print(f"{version.upper()} 第 {step_num} 步：{step['name']}")
    print(f"说明：{step['description']}")
    print(f"脚本：{step['script']}")
    print("=" * 70)

    if not os.path.exists(script_path):
        print(f"错误：未找到脚本 {script_path}")
        return False

    try:
        result = subprocess.run(
            [*_python_cmd(), script_path],
            cwd=os.path.dirname(__file__),
            check=False,
        )
        if result.returncode == 0:
            print(f"\n步骤 {step_num} 执行成功")
            return True
        print(f"\n步骤 {step_num} 执行失败，返回码 {result.returncode}")
        return False
    except Exception as exc:
        print(f"\n步骤 {step_num} 执行失败：{exc}")
        return False


def _run_faiss_preflight() -> None:
    """与 01-07 入口脚本一致的 faiss 预检查。"""
    print("[预检查] faiss（如需跳过可设置 RES_SAM_SKIP_FAISS_PREFLIGHT=1）...")
    preflight_faiss_or_raise()


def main():
    parser = argparse.ArgumentParser(
        description="运行当前 Res-SAM 论文优先主线（v6）。"
    )
    parser.add_argument(
        "--version",
        choices=ACTIVE_VERSIONS,
        default="v6",
        help="选择要运行的当前主线版本。",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        help="要运行的步骤，例如 1、2,4 或 1-4",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所选版本可用步骤。",
    )
    args = parser.parse_args()

    steps = STEPS_BY_VERSION[args.version]
    valid_steps = list(steps.keys())

    if args.list:
        print(f"\n{args.version} 可用步骤：")
        print("-" * 70)
        for num, step in steps.items():
            print(f"  {num}. {step['name']}: {step['description']}")
        print("\n已归档的 V3 脚本不再通过 experiments/ 直接运行。")
        return

    steps_to_run = _parse_steps(args.step, valid_steps)
    if not steps_to_run:
        raise SystemExit("未选择有效步骤。")

    _run_faiss_preflight()

    print("=" * 70)
    print("Res-SAM 当前主线实验运行器")
    print("=" * 70)
    print(f"版本：{args.version}")
    print(f"将运行步骤：{steps_to_run}")
    print("数据集：固定为当前唯一保留的增强标注数据集")
    print("V3 归档快照：已归档/experiments_v3_snapshot_20260326/")

    results = {}
    for step_num in steps_to_run:
        results[step_num] = run_step(args.version, step_num)
        if not results[step_num] and step_num < max(steps_to_run):
            print(f"\n警告：步骤 {step_num} 失败，继续执行后续步骤。")

    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    for step_num in steps_to_run:
        status = "成功" if results[step_num] else "失败"
        print(f"  步骤 {step_num}（{steps[step_num]['name']}）：{status}")

    total = len(steps_to_run)
    success = sum(results.values())
    print(f"\n总计：{success}/{total} 个步骤执行成功")


if __name__ == "__main__":
    main()
