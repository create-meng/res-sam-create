# Res-SAM 论文复现

> 论文: Reservoir-enhanced Segment Anything Model for GPR Anomaly Detection
> Nature Communications (2025)

本项目复现 Res-SAM 论文实验，结合 SAM (Segment Anything Model) 和 2D-ESN (Echo State Network) 进行 GPR 地下异常检测。

## 复现状态

### V1 (初步复现)
- [x] 特征银行构建
- [x] 自动模式推理 (支持断点续传)
- [x] 点击引导模式推理 (支持断点续传)
- [x] 评估指标计算 (Table 2)
- [x] 异常聚类分析 (Table 3) - FCM准确率: 93.65%

### V2 (完整实现) ⭐
- [x] SAM 集成模块 (`PatchRes/sam_integration.py`)
- [x] Res-SAM 核心模块 (`PatchRes/ResSAM.py`)
- [x] Feature Bank 构建 V2 (`04_build_feature_bank_v2.py`)
- [x] Fully Automatic 推理 V2 (`05_inference_auto_v2.py`)
- [x] 评估和可视化 (`06_evaluate_and_visualize.py`)

## 核心问题诊断

V1 版本复现效果不佳的原因：

| 问题 | 原因 |
|------|------|
| Precision 极低 (6.6%) | **缺失 SAM 模块** - 直接对整图提取 patch |
| AUC = 0.5 | Feature Bank 与测试数据环境不匹配 |
| utilities 完全失败 | SAM 未正确识别候选区域 |

V2 版本修正：完整实现论文流程 `SAM生成候选区域 → 2D-ESN分析 → 异常判定`

## 快速开始

```bash
# 1. 安装环境
conda env create -f environment-cpu.yml
conda activate res-sam

# 2. 安装 segment-anything
pip install segment-anything

# 3. 下载 SAM 权重 (推荐 vit_b)
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# 放置在 ./models/ 目录

# 4. 运行 V2 实验
cd experiments
python 04_build_feature_bank_v2.py  # 构建 Feature Bank
python 05_inference_auto_v2.py       # 运行推理
python 06_evaluate_and_visualize.py  # 评估结果
```

## 实验脚本

### V2 版本 (推荐)

| 脚本 | 功能 |
|------|------|
| `04_build_feature_bank_v2.py` | 构建 Feature Bank (支持多源数据) |
| `05_inference_auto_v2.py` | Fully Automatic 模式推理 |
| `06_evaluate_and_visualize.py` | 评估指标 + 可视化 |

### V1 版本 (参考)

| 脚本 | 功能 |
|------|------|
| `01_build_feature_bank.py` | 构建 Feature Bank |
| `02_inference_auto.py` | 自动模式推理 |
| `03_inference_click.py` | 点击引导模式推理 |
| `04_evaluate.py` | 计算评估指标 |
| `05_clustering.py` | 异常聚类 |

## 关键参数

```python
# 论文参数
window_size = 50      # 2D-ESN patch 大小
stride = 5            # 滑动窗口步长 (论文原始值)
hidden_size = 30      # ESN 隐藏层大小
anomaly_threshold = 0.5  # 异常判定阈值

# SAM 参数
sam_model_type = "vit_b"  # SAM 模型类型
```

## 项目结构

```
Res-SAM/
├── PatchRes/
│   ├── ESN_2D_nobatch.py   # 2D-ESN 实现
│   ├── PatchRes.py         # V1 核心模块
│   ├── ResSAM.py           # V2 核心模块 ⭐
│   ├── sam_integration.py  # SAM 集成 ⭐
│   └── functions.py        # 工具函数
├── experiments/
│   ├── 04_build_feature_bank_v2.py  # V2 Feature Bank ⭐
│   ├── 05_inference_auto_v2.py      # V2 推理 ⭐
│   ├── 06_evaluate_and_visualize.py # V2 评估 ⭐
│   └── ANALYSIS.md          # 问题诊断报告
├── data/GPR_data/
│   ├── intact/              # 正常样本
│   ├── augmented_cavities/  # 空洞异常
│   └── augmented_utilities/ # 管道异常
└── outputs/
    ├── feature_banks_v2/    # V2 Feature Bank
    └── predictions_v2/      # V2 预测结果
```

## 论文关键发现

1. **环境一致性** (Fig.6): Feature Bank 必须与测试数据来自同一环境，否则 AUC 下降 0.112-0.161
2. **SAM 作用** (Fig.5): SAM 能有效减少 2D-ESN 拟合次数，提升效率
3. **FCM 聚类** (Table 3): 模糊聚类更适合边界模糊的 GPR 异常

---

详细分析见 `experiments/ANALYSIS.md`
