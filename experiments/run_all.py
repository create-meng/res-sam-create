"""
Res-SAM 复现实验 - 主运行脚本

按顺序执行所有复现步骤：
1. 构建 Feature Bank
2. Fully Automatic 推理
3. 评估指标计算
4. Click-guided 推理
5. 异常聚类

使用方法：
    python run_all.py              # 运行全部步骤
    python run_all.py --step 1     # 只运行步骤 1
    python run_all.py --step 2,3   # 运行步骤 2 和 3
"""

import sys
import os
import argparse
import subprocess

# 步骤定义
STEPS = {
    1: {
        'name': 'Feature Bank Construction',
        'script': '01_build_feature_bank_v3.py',
        'description': '从 normal 数据构建特征银行',
    },
    2: {
        'name': 'Fully Automatic Inference',
        'script': '02_inference_auto_v3.py',
        'description': '自动模式批量推理',
    },
    3: {
        'name': 'Evaluation Metrics',
        'script': '04_evaluate_and_visualize_v3.py',
        'description': '计算评估指标 (Table 2)',
    },
    4: {
        'name': 'Click-guided Inference',
        'script': '03_inference_click_v3.py',
        'description': '点击引导模式推理 (Table 1)',
    },
    5: {
        'name': 'Anomaly Clustering',
        'script': '05_clustering_v3.py',
        'description': '异常聚类分析 (Table 3)',
    },
}


def run_step(step_num):
    """运行单个步骤"""
    if step_num not in STEPS:
        print(f"Error: Invalid step number {step_num}")
        return False
    
    step = STEPS[step_num]
    script_path = os.path.join(os.path.dirname(__file__), step['script'])
    
    print(f"\n{'='*70}")
    print(f"Step {step_num}: {step['name']}")
    print(f"Description: {step['description']}")
    print(f"Script: {step['script']}")
    print("=" * 70)
    
    if not os.path.exists(script_path):
        print(f"Error: Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            check=False,
        )
        
        if result.returncode == 0:
            print(f"\n✓ Step {step_num} completed successfully")
            return True
        else:
            print(f"\n✗ Step {step_num} failed with return code {result.returncode}")
            return False
    except Exception as e:
        print(f"\n✗ Step {step_num} failed with error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Res-SAM 复现实验运行脚本')
    parser.add_argument('--step', type=str, default=None,
                       help='要运行的步骤，例如: 1 或 2,3 或 1-3')
    parser.add_argument('--list', action='store_true',
                       help='列出所有步骤')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Steps:")
        print("-" * 70)
        for num, step in STEPS.items():
            print(f"  {num}. {step['name']}: {step['description']}")
        print()
        return
    
    # 确定要运行的步骤
    steps_to_run = []
    
    if args.step:
        # 解析步骤参数
        if '-' in args.step:
            # 范围: 1-3
            start, end = map(int, args.step.split('-'))
            steps_to_run = list(range(start, end + 1))
        elif ',' in args.step:
            # 列表: 1,2,3
            steps_to_run = [int(s.strip()) for s in args.step.split(',')]
        else:
            # 单个步骤
            steps_to_run = [int(args.step)]
    else:
        # 运行全部
        steps_to_run = list(STEPS.keys())
    
    print("=" * 70)
    print("Res-SAM Paper Reproduction Experiments")
    print("=" * 70)
    print(f"Steps to run: {steps_to_run}")
    
    # 运行步骤
    results = {}
    for step_num in steps_to_run:
        success = run_step(step_num)
        results[step_num] = success
        
        if not success and step_num < max(steps_to_run):
            print(f"\nWarning: Step {step_num} failed. Continuing with next step...")
    
    # 总结
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for step_num in steps_to_run:
        status = "✓ Success" if results[step_num] else "✗ Failed"
        print(f"  Step {step_num} ({STEPS[step_num]['name']}): {status}")
    
    total = len(steps_to_run)
    success = sum(results.values())
    print(f"\nTotal: {success}/{total} steps completed successfully")
    
    if success == total:
        print("\nAll experiments completed!")
        print("\nMain outputs:")
        print("  - outputs/feature_banks_v3/feature_bank_v3.pth")
        print("  - outputs/predictions_v3/auto_predictions_v3.json")
        print("  - outputs/predictions_v3/click_predictions_v3.json")
        print("  - experiments/04_evaluate_and_visualize_v3_report.md")
        print("  - outputs/metrics_v3/clustering_v3.json")


if __name__ == "__main__":
    main()
