# Res-SAM 论文复现

> 论文: Reservoir-enhanced Segment Anything Model for GPR Anomaly Detection
> Nature Communications (2025)

本项目复现 Res-SAM 论文实验，结合 SAM (Segment Anything Model) 和 2D-ESN (Echo State Network) 进行 GPR 地下异常检测。

## 复现状态

### V3 (严格对齐论文口径) ⭐
- [x] SAM 集成模块 (`PatchRes/sam_integration.py`)
- [x] Res-SAM 核心模块 (`PatchRes/ResSAM.py`)
- [x] Feature Bank 构建 V3 (`experiments/01_build_feature_bank_v3.py`)
- [x] Fully Automatic 推理 V3 (`experiments/02_inference_auto_v3.py`)
- [x] Click-guided 推理 V3 (`experiments/03_inference_click_v3.py`)
- [x] 评估和可视化 V3 (`experiments/04_evaluate_and_visualize_v3.py`)
- [x] 异常聚类 V3 (`experiments/05_clustering_v3.py`)

> 旧版本脚本（V1/V2）已归档：`archive/legacy_experiments/`
> 旧输出目录已归档：`archive/legacy_outputs/`

## 核心问题诊断

V1 版本复现效果不佳的原因：

| 问题 | 原因 |
|------|------|
| Precision 极低 (6.6%) | **缺失 SAM 模块** - 直接对整图提取 patch |
| AUC = 0.5 | Feature Bank 与测试数据环境不匹配 |
| utilities 完全失败 | SAM 未正确识别候选区域 |

V2 版本修正：完整实现论文流程 `SAM生成候选区域 → 2D-ESN分析 → 异常判定`

## 快速开始

### 1) 安装环境（方式 1：Pip Requirements，推荐）

创建环境并安装依赖：

```bash
conda create -n res-sam python=3.9 -y
conda activate res-sam

# CPU
pip install -r requirements-cpu.txt

# GPU (CUDA 11.8)
pip install -r requirements-gpu.txt
```

如果你遇到 pip 的 `hash mismatch`（例如 `nvidia-cudnn-cu11` 的 sha256 不一致），通常是第三方镜像/缓存导致：

- 建议使用官方源（本仓库 `requirements-gpu.txt` 已默认使用 PyTorch 官方 cu118 index + 官方 PyPI）。
- 可尝试清理缓存/禁用缓存后重装：

```bash
pip cache purge
pip install --no-cache-dir -r requirements-gpu.txt
```

说明（版本口径锁定）：

- **CPU**：`requirements-cpu.txt` 固定 `torch==2.3.1 / torchvision==0.18.1 / torchaudio==2.3.1`
- **GPU (CUDA 11.8)**：`requirements-gpu.txt` 固定 `torch==2.3.1+cu118 / torchvision==0.18.1+cu118 / torchaudio==2.3.1+cu118`

这样做是为了：

- 避免 `torch/torchvision/torchaudio` 版本随时间漂移导致结果不可复现。
- 避免在部分 Python/平台组合下出现 `torch==2.1.2+cu118` 无可用 wheel 的安装失败问题（会表现为 pip 只能看到 `2.2.0+cu118` 及更高版本）。

验证 CUDA 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 1') 安装环境（方式 2：Conda YML，备选）

```bash
# CPU
conda env create -f environment-cpu.yml

# GPU (CUDA 11.8)
conda env create -f environment-gpu.yml

conda activate res-sam
```

### 2) 获取 SAM 权重（Git LFS）

```bash
git lfs install
git lfs pull
```

校验权重文件大小应为几百 MB（不是几 KB 的 LFS 指针文件）：

```bash
ls -lh sam/sam_vit_b_01ec64.pth
```

### 3) 运行 V3 实验

```bash
cd experiments
python 01_build_feature_bank_v3.py
python 02_inference_auto_v3.py
python 04_evaluate_and_visualize_v3.py
```

Windows（或不手动 activate 环境）可用：

```bash
conda run -n res-sam python experiments/01_build_feature_bank_v3.py
conda run -n res-sam python experiments/02_inference_auto_v3.py
conda run -n res-sam python experiments/04_evaluate_and_visualize_v3.py
```

10 样本快速验证（每类最多 10 张；用于检查流程/输出字段，不作为最终出数）：

```bash
set MAX_IMAGES_PER_CATEGORY=10
conda run -n res-sam python experiments/02_inference_auto_v3.py
conda run -n res-sam python experiments/03_inference_click_v3.py
conda run -n res-sam python experiments/04_evaluate_and_visualize_v3.py
```

## 实验脚本

### V3 版本 (推荐)

| 脚本 | 功能 |
|------|------|
| `01_build_feature_bank_v3.py` | 构建 Feature Bank (严格按论文对齐) |
| `02_inference_auto_v3.py` | Fully Automatic 模式推理 (V3) |
| `03_inference_click_v3.py` | Click-guided 模式推理 (Table 1) |
| `04_evaluate_and_visualize_v3.py` | 评估指标 + 可视化 (Table 2 + Table 1 汇总) |
| `05_clustering_v3.py` | 异常聚类 (Table 3 / 区域级再拟合) |

### V1/V2 版本 (已归档)

已归档至：`archive/legacy_experiments/`

## 关键参数

```python
# 论文参数
window_size = 50      # 2D-ESN patch 大小 (论文默认值)
stride = 5            # 滑动窗口步长 (论文原始值)
hidden_size = 30      # ESN 隐藏层大小
anomaly_threshold = 0.5  # 异常判定阈值

# SAM 参数
sam_model_type = "vit_b"  # SAM 模型类型 (内存占用低)
image_size = (369, 369)   # 论文原始尺寸

# 断点参数
checkpoint_interval = 20   # 每20张图片保存断点

# AUC 口径
normal_auc_dir = "data/GPR_data/intact"  # auto/click 推理脚本默认将该目录作为 normal_auc 加入，仅用于 ROC/AUC（不参与检测指标）
```

## V3 输出文件格式（口径固化）

为避免不同实验之间的预处理口径（灰度/resize/normalize）发生漂移，V3 的关键输出 json 文件已统一支持以下结构：

```json
{
  "meta": {
    "version": "V3",
    "timestamp": "...",
    "feature_bank_path": "...",
    "feature_bank_metadata_path": "...",
    "preprocess_signature": {
      "color_mode": "L",
      "resize": {"enabled": true, "size_hw": [369, 369]},
      "normalize": {"enabled": true, "method": "(x-mean)/(std+1e-8)", "per_image": true}
    },
    "image_size_hw": [369, 369],
    "window_size": 50,
    "stride": 5,
    "hidden_size": 30,
    "beta_threshold": 0.5
  },
  "results": {"...": "..."}
}
```

- **Fully automatic**：`outputs/predictions_v3/auto_predictions_v3.json`
- **Click-guided**：`outputs/predictions_v3/click_predictions_v3.json`

说明：
- `meta.preprocess_signature` 来自 `outputs/feature_banks_v3/metadata.json`（由 `01_build_feature_bank_v3.py` 写入）。
- 评估/聚类脚本读取上述输出时已兼容旧格式（直接 dict）与新格式（`{meta, results}`）。

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
│   ├── 01_build_feature_bank_v3.py  # V3 Feature Bank ⭐
│   ├── 02_inference_auto_v3.py      # V3 自动推理 ⭐
│   ├── 03_inference_click_v3.py     # V3 点击引导推理 ⭐
│   ├── 04_evaluate_and_visualize_v3.py # V3 评估 ⭐
│   ├── 05_clustering_v3.py          # V3 聚类 ⭐
│   └── run_all.py                   # 主运行脚本
├── sam/
│   └── sam_vit_b_01ec64.pth # SAM 权重
├── data/GPR_data/
│   ├── intact/              # 正常样本
│   ├── augmented_cavities/  # 空洞异常
│   └── augmented_utilities/ # 管道异常
└── outputs/                 # (运行后生成) V3 输出目录
```

## 论文关键发现

1. **环境一致性** (Fig.6): Feature Bank 必须与测试数据来自同一环境，否则 AUC 下降 0.112-0.161
2. **SAM 作用** (Fig.5): SAM 能有效减少 2D-ESN 拟合次数，提升效率
3. **FCM 聚类** (Table 3): 模糊聚类更适合边界模糊的 GPR 异常

## V2 版本改进记录

| 日期 | 改进内容 |
|------|----------|
| 2026-03-12 | 重写 detect_automatic 严格按照论文流程 |
| 2026-03-12 | 修复内存问题：切换到 SAM vit_b，预构建 NN 搜索器 |
| 2026-03-12 | 添加进度条 (tqdm) 和日志系统 |
| 2026-03-12 | 降低 window_size 到 30 匹配 SAM 候选区域 |
| 2026-03-13 | 优化断点功能，添加每张图片处理日志 |

---

详细分析见 `experiments/ANALYSIS.md`
