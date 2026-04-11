# Res-SAM 论文复现

> 论文: Reservoir-enhanced Segment Anything Model for GPR Anomaly Detection
> Nature Communications (2025)

本项目复现 Res-SAM 论文实验，结合 SAM (Segment Anything Model) 和 2D-ESN (Echo State Network) 进行 GPR 地下异常检测。

## 推荐正流程

如果你是第一次跑项目，建议按下面顺序执行：

1. 安装环境
2. 获取 SAM 权重
3. 先做快速自检（可选，但推荐）
4. 运行 **V7** 主流程（逐个运行 `*_v7.py`）
5. 查看输出结果
6. 再根据需要运行聚类 / 消融 / 跨环境实验

对应命令的最短路径如下（当前主线为 **V7**；V8 为并行对照分支；V6 已归档至 `已归档/v6_snapshot_20260411_clean/`；V3 已归档至 `已归档/experiments_v3_snapshot_20260326/`）：

```bash
# 若尚未克隆（OpenI 示例；浅克隆 + 跳过 LFS 自动检出）
export GIT_LFS_SKIP_SMUDGE=1
git clone --depth 1 https://openi.pcl.ac.cn/create-meng/res-sam.git && cd res-sam

conda create -n res-sam python=3.9 -y
conda activate res-sam
pip install -r requirements-gpu.txt   # 或 requirements-cpu.txt

git lfs install
git lfs pull
# 若 OpenI 等无法拉取 sam/sam_vit_l_0b3195.pth，见下文「SAM 权重」用手动下载。

python experiments/01_build_feature_bank_v7.py
python experiments/02_inference_auto_v7.py
python experiments/03_evaluate_and_visualize_v7.py
python experiments/04_clustering_v7.py
```

---

## 复现状态

### V4 / V5（历史实验分支）
- [x] SAM 集成模块 (`PatchRes/sam_integration.py`)
- [x] Res-SAM 核心模块 (`PatchRes/ResSAM.py`)
- [x] Feature Bank / 自动推理 / 评估 / click / 聚类 / 消融 / 跨环境：对应历史脚本位于 `experiments/*_v4.py`、`experiments/*_v5.py`，仅供追溯参考
- [x] Eq.(9) β 默认值与作者仓库 `PatchRes/main.py` 对齐：`experiments/paper_constants.py` 中 `DEFAULT_BETA_THRESHOLD`（当前 **0.1**）

### V7（当前推荐主线：论文优先 + 独立输出目录）✅
- **定位**：V7 是当前唯一维护主线，遵循“论文优先；论文未写明时再采用官方公开代码默认值”的口径，并使用独立输出目录 `outputs/*_v7/`。
- **运行入口**：依次运行 `python experiments/01_build_feature_bank_v7.py` → `02_inference_auto_v7.py` → `03_evaluate_and_visualize_v7.py` → `04_clustering_v7.py`
- **实现对齐点（作者开源代码口径）**
  - 点击 SAM：`PatchRes/sam_integration.py` 使用 `multimask_output=True` 并取 `scores.argmax()`
  - 2D-ESN：`ESN_2D` 构造器默认 `alpha=0.9`，但当前 V6 主流程在论文未给定该值时沿用作者公开实现，显式传入 `alpha=5`；特征定义按论文公式真实求解 `f=[W_out,b]`，特征维 `2*hidden_size+1`（默认 61）
  - 近邻默认严格 Faiss：未显式放宽时禁止 sklearn 回退；如需应急回退，设置 `RES_SAM_ALLOW_SKLEARN_KNN=1`
- **重要**：由于 V7 当前按论文使用 `f=[W_out,b]`，特征维为 61，因此请使用 V7 的 Step 1 重新构建 `outputs/feature_banks_v7/feature_bank_v7.pth`，不要混用旧产物。
- **唯一保留的数据映射**
  - feature bank normal 源：`data/GPR_data/augmented_intact`（随机抽 20 张建库）
  - 评估集：`data/GPR_data/augmented_cavities`、`augmented_utilities`、`augmented_intact`
  - 不再保留其他数据模式和未标注原始目录

### V8（并行对照分支）
- **定位**：V8 用于和 V7 并行做小样本/机制对照，不作为默认推荐入口。
- **运行入口**：依次运行 `python experiments/01_build_feature_bank_v8.py` → `02_inference_auto_v8.py` → `03_evaluate_and_visualize_v8.py` → `04_clustering_v8.py`

### V6（已归档快照，勿再当活动修改基线）
- 归档位置：`已归档/v6_snapshot_20260411_clean/`
- 包含内容：`experiments/*_v6.py`、`outputs/*_v6/`、V6 日志，以及一份共享运行上下文快照

### V3（已归档快照，勿再当修改基线）
- 见 `已归档/experiments_v3_snapshot_20260326/`

> 旧版本脚本（V1/V2）已归档：`已归档/legacy_experiments/`
> 旧输出目录已归档：`已归档/legacy_outputs/`

## 从安装到运行的一条线

### 第 0 步：克隆仓库（推荐浅克隆）

为减少 Git 传输与本地对象数量、加快首次拉取，建议 **`git clone --depth 1`**（只取默认分支最新提交；复现主线通常不需要完整历史）。若日后需要完整提交历史，可执行 `git fetch --unshallow`，或删除目录后重新全量克隆。

检出时建议 **`GIT_LFS_SKIP_SMUDGE=1`**：工作区里先落 LFS 指针文本，再按需 `git lfs pull`，避免大文件在 clone/checkout 阶段失败。

OpenI：

```bash
export GIT_LFS_SKIP_SMUDGE=1
git clone --depth 1 https://openi.pcl.ac.cn/create-meng/res-sam.git
cd res-sam
```

GitHub：

```bash
export GIT_LFS_SKIP_SMUDGE=1
git clone --depth 1 https://github.com/create-meng/res-sam-create.git
cd res-sam-create
```

一键脚本（脚本内已浅克隆 + skip smudge）：`bash scripts/setup_from_scratch_openi.sh`、`bash scripts/setup_from_scratch_github.sh`。

### 第 1 步：安装环境

#### 方式 1：Pip Requirements（推荐）

创建环境并安装依赖：

```bash
conda create -n res-sam python=3.9 -y
conda activate res-sam

# CPU
pip install -r requirements-cpu.txt

# GPU (CUDA 11.8)
pip install -r requirements-gpu.txt
```

国内镜像（可选，用于加速下载；不改变版本口径）：

```bash
# 临时指定镜像（示例：清华 PyPI）
pip install -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若你在云平台/国内环境下载 PyTorch CUDA wheels 很慢（`torch-2.3.1+cu118` 等从 `download.pytorch.org` 下载缓慢），可改为使用“PyTorch wheels 镜像”作为 `--index-url`（示例：南京大学镜像；安装的包版本与本仓库固定口径一致）：

```bash
pip install "torch==2.3.1+cu118" "torchvision==0.18.1+cu118" "torchaudio==2.3.1+cu118" \
  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple

# 然后安装本仓库其余依赖（避免重复解析 torch 相关依赖）
pip install -r requirements-gpu.txt --no-deps
```

说明：部分镜像站仅提供 wheels 列表页（`-f/--find-links`）且可能不包含带本地版本号的 `+cu118` 包名，出现 `No matching distribution found for torch==...+cu118` 时建议优先使用上述 `--index-url .../cu118` 形式。

说明：

- `requirements-gpu.txt` 已固定使用 PyTorch 官方 CUDA wheels 索引（`--index-url https://download.pytorch.org/whl/cu118`），并保留官方 PyPI 作为兜底。
- 若你需要使用国内 PyPI 镜像加速，本仓库也已将其写入 `--extra-index-url`，保证 `torch/torchvision/torchaudio` 仍从 PyTorch 官方索引获取。

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

#### 方式 2：Conda YML（备选）

```bash
# CPU
conda env create -f environment-cpu.yml

# GPU (CUDA 11.8)
conda env create -f environment-gpu.yml

conda activate res-sam
```

### 第 2 步：获取 SAM 权重

```bash
git lfs install
git lfs pull
```

如果提示 `git: 'lfs' is not a git command`、`git lfs: command not found`，说明当前环境没有安装 Git LFS。常见处理方案如下。

#### 方案 1：通过 Conda 安装 Git LFS

如果你已经在使用 Conda 环境，这是最方便的方案之一：

```bash
conda install -c conda-forge git-lfs
git lfs install
git lfs pull
```

#### 方案 2：通过系统包管理器安装 Git LFS

Linux（Debian / Ubuntu）：

```bash
sudo apt-get update
sudo apt-get install git-lfs
git lfs install
git lfs pull
```

macOS（Homebrew）：

```bash
brew install git-lfs
git lfs install
git lfs pull
```

Windows（常见方式）：

```powershell
winget install GitHub.GitLFS
git lfs install
git lfs pull
```

如果你使用 `choco`，也可以：

```powershell
choco install git-lfs
git lfs install
git lfs pull
```

#### 方案 3：不依赖 Git LFS，直接手动下载权重

如果你暂时不想安装 Git LFS，也可以只手动下载本项目运行所需的 SAM 权重文件到 `sam/` 目录。

**本仓库主线 SAM 为 ViT-L**：`sam/sam_vit_l_0b3195.pth`（约 1.2GB），与论文作者公开仓库 `zhouxr6066/Res-SAM` 的 `sam/sam.py`、`experiments` 与 `scripts/setup_after_clone_*.sh` 默认一致。**GitHub** 远程可用 Git LFS 托管该完整权重文件。**OpenI** 等平台往往无法上传或拉取如此大的 LFS 对象：请仅 `clone` 后按下方官方地址手动下载到 `sam/`（文件名需一致）。

```bash
# Linux / WSL（ViT-L，Meta 官方；aria2 多线程；无 root 时在 apt-get 前加 sudo）
# 请在仓库根目录执行（与 scripts/、sam/ 同级；权重写入克隆自带的 sam/，与 sam/sam.py 同级）
apt-get update && apt-get install -y aria2
aria2c -x 16 -s 16 -k 1M \
  -o sam_vit_l_0b3195.pth \
  -d sam \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
```

```powershell
# Windows PowerShell（ViT-L）
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -OutFile sam\sam_vit_l_0b3195.pth
```

#### 方案 4：通过 SSH / SCP 直接拷贝已有权重文件

如果你本地电脑、实验室服务器或云服务器上已经有 `sam_vit_l_0b3195.pth`，也可以直接通过 `scp` 拷贝到当前项目的 `sam/` 目录，而不必重新下载。

例如：从远程服务器拷贝到当前机器

```bash
scp -P 42034 root@xj-member.bitahub.com:/root/res-sam-create/sam_vit_l_0b3195.pth sam/
```

例如：从本地电脑拷贝到远程服务器

```bash
scp -P 42034 sam/sam_vit_l_0b3195.pth root@xj-member.bitahub.com:/root/res-sam-create/
```

如果远程目标目录还没有 `sam/` 文件夹，也可以显式指定完整目标文件名：

```bash
scp -P 42034 root@xj-member.bitahub.com:/root/res-sam-create/sam_vit_l_0b3195.pth sam/sam_vit_l_0b3195.pth
```

这种方式适合“只想先把主流程跑起来”的情况；如果仓库里后续还有别的 LFS 文件需要拉取，再补装 Git LFS 即可。

#### 方案 5：只跳过 SAM 权重，其它 LFS 文件照常拉取

如果你的问题不是“完全不用 Git LFS”，而是“仓库里其它 LFS 文件都想拉，只是 `sam/sam_vit_l_0b3195.pth` 这个权重不想走 LFS（或远程未托管）”，可以用下面这组命令：

```bash
# 先跳过 LFS smudge，避免 checkout 阶段失败；浅克隆减轻对象与流量
export GIT_LFS_SKIP_SMUDGE=1

git clone --depth 1 https://openi.pcl.ac.cn/create-meng/res-sam.git
cd res-sam

# 拉取除该权重以外的其它 LFS 对象（数据集等）
git lfs pull --exclude="sam/sam_vit_l_0b3195.pth"
```

然后再单独下载权重到 `sam/` 目录（官方地址）：

```bash
# Linux / WSL（ViT-L；在仓库根目录执行；无 root 时在 apt-get 前加 sudo）
apt-get update && apt-get install -y aria2
aria2c -x 16 -s 16 -k 1M \
  -o sam_vit_l_0b3195.pth \
  -d sam \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
```

```powershell
# Windows PowerShell（ViT-L）
Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -OutFile sam\sam_vit_l_0b3195.pth
```

校验权重文件大小应为约 1.2GB（ViT-L），不是几 KB 的 LFS 指针文件：

```bash
ls -lh sam/sam_vit_l_0b3195.pth
```

### 第 3 步：快速自检（可选但推荐）

`00_quick_test_v3.py` 已随 V3 归档至 `已归档/experiments_v3_snapshot_20260326/`。快速验证建议用 **少量图片** 跑当前 `v7` 主线：

```bash
set MAX_IMAGES_PER_CATEGORY=10
conda run -n res-sam python experiments/02_inference_auto_v7.py
conda run -n res-sam python experiments/03_evaluate_and_visualize_v7.py
conda run -n res-sam python experiments/04_clustering_v7.py
```

### 第 4 步：运行 V7 主流程

当前主线流程建议按下面顺序执行（`v7` 为默认维护主线，`v8` 为并行对照分支）：

```bash
python experiments/01_build_feature_bank_v7.py
python experiments/02_inference_auto_v7.py
python experiments/03_evaluate_and_visualize_v7.py
python experiments/04_clustering_v7.py
```

也可以进入 `experiments` 目录运行；脚本统一按仓库根目录解析路径。

Windows（或不手动 activate 环境）可用：

```bash
conda run -n res-sam python experiments/01_build_feature_bank_v7.py
conda run -n res-sam python experiments/02_inference_auto_v7.py
conda run -n res-sam python experiments/03_evaluate_and_visualize_v7.py
conda run -n res-sam python experiments/04_clustering_v7.py
```

### 第 5 步：按需运行扩展实验

#### 异常聚类

```bash
python experiments/04_clustering_v7.py
```

#### 消融实验：w/o SAM

```bash
# 如需追溯历史实验，请查看已归档或历史脚本说明
```

#### Cross-environment 实验

```bash
# 当前维护重点为 v7/v8 的建库、全自动与评估
```

### 第 6 步：查看输出

V7 关键输出主要在：

- `outputs/feature_banks_v7/`
- `outputs/predictions_v7/`
- `outputs/checkpoints_v7/`

其中最常用的是：

- **Fully automatic**：`outputs/predictions_v7/auto_predictions_v7.json`
### 第 7 步：如需上传或下载结果文件

在本地终端运行，例如：

```bash
scp -P 42034 outputs.zip root@xj-member.bitahub.com:/root/res-sam-create/
```

如需从远程服务器下载结果文件到本地，也可以执行：

```bash
scp -P 42034 root@xj-member.bitahub.com:/root/res-sam-create/outputs.zip .
```

---

## 实验脚本

### V7 版本（当前推荐）

| 脚本 | 功能 |
|------|------|
| `01_build_feature_bank_v7.py` | 构建 Feature Bank |
| `02_inference_auto_v7.py` | Fully Automatic 模式推理 |
| `03_evaluate_and_visualize_v7.py` | 评估指标 + 可视化 |
| `04_clustering_v7.py` | 异常聚类 |

### V8 版本（并行对照）

| 脚本 | 功能 |
|------|------|
| `01_build_feature_bank_v8.py` | 构建 Feature Bank |
| `02_inference_auto_v8.py` | Fully Automatic 模式推理 |
| `03_evaluate_and_visualize_v8.py` | 评估指标 + 可视化 |
| `04_clustering_v8.py` | 异常聚类 |


### V3 版本（已归档快照）

见 `已归档/experiments_v3_snapshot_20260326/`（含 `00_quick_test_v3.py` 等）。

### V1/V2 版本（已归档）

已归档至：`已归档/legacy_experiments/`

## 关键参数

```python
# 论文参数
window_size = 50      # 2D-ESN patch 大小 (论文默认值)
stride = 5            # 滑动窗口步长 (论文原始值)
hidden_size = 30      # ESN 隐藏层大小
anomaly_threshold = beta  # 异常判定阈值（论文 Eq.(9) 的 β）

# SAM 参数（本仓库主线 vit_l，与作者公开仓库一致）
sam_model_type = "vit_l"
image_size = (369, 369)   # 论文原始尺寸

# 断点参数
checkpoint_interval = 20   # 每20张图片保存断点

# AUC 口径
normal_auc_dir = "data/GPR_data/augmented_intact"  # 当前主线将该目录作为 normal_auc 加入，仅用于 ROC/AUC（不参与检测指标）
```

说明（β / beta_threshold 口径锁定）：

- 论文 Eq.(7)-(9) 定义 anomaly likelihood 为特征到 Feature Bank 的最近邻 L2 距离，并使用单一阈值 β 判别异常；当前 V7/V8 中该距离基于真实求解的 `f=[W_out,b]` 计算；论文未给出 β 的具体数值。
- 当前主线（V7/V8）中 β 为预先设定的固定阈值（predefined threshold），由于论文未给出具体数值，默认值沿用作者公开仓库 `PatchRes/main.py`，集中在 `experiments/paper_constants.py` 的 **`DEFAULT_BETA_THRESHOLD`（当前 0.1）**；`CONFIG["beta_threshold"]` 可覆盖。
- 评测脚本里 **预测框与 GT 的 IoU 是否算匹配** 使用 **`EVAL_DETECTION_IOU_THRESHOLD`（0.5，正文约 p.4）**，与 β 不是同一参数。
  - `experiments/02_inference_auto_v7.py` / `02_inference_auto_v8.py` 等：`CONFIG["beta_threshold"]` 即为 β。
  - 推理输出 `meta.beta_threshold` 记录本次运行实际使用的 β。

设备说明：

- `CONFIG["device"]` 默认使用 `auto`：若检测到 CUDA 则使用 GPU，否则回退到 CPU；也可显式指定 `cpu`/`cuda` 以适配云平台限制。

## KNN 后端说明

KNN 后端（用于 Feature Bank 1-NN 检索）默认策略：

- 默认策略：涉及 Feature Bank 1-NN 检索的主线脚本先执行 strict-faiss 预检；只有显式设置 `RES_SAM_ALLOW_SKLEARN_KNN=1` 时才允许 `sklearn` 回退。
- 未显式指定 `RES_SAM_KNN_BACKEND` 时，运行期默认优先尝试 `faiss_gpu`（CUDA 可用）或 `faiss_cpu`；strict-faiss 默认下不会自动回退到 `sklearn`。
- 可用环境变量强制指定：`RES_SAM_KNN_BACKEND=sklearn|faiss_cpu|faiss_gpu`；若指定 `sklearn`，建议同时设置 `RES_SAM_ALLOW_SKLEARN_KNN=1`。
- `experiments/03_evaluate_and_visualize_v7.py` / `03_evaluate_and_visualize_v8.py` 属于纯评估/可视化脚本：若输入是已生成的预测结果 JSON，则不执行 faiss 预检，也不需要 Feature Bank 1-NN 检索。

## V7/V8 输出文件格式（当前主线）

为避免不同实验之间的预处理口径（灰度/resize/normalize）发生漂移，主线输出的 json 已统一支持以下结构（示例为 V7）：

```json
{
  "meta": {
    "version": "V7",
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
    "beta_threshold": 0.1
  },
  "results": {"...": "..."}
}
```

- **Fully automatic**：`outputs/predictions_v7/auto_predictions_v7.json`
说明：

- `meta.preprocess_signature` 来自 `outputs/feature_banks_v7/metadata.json`（由 `01_build_feature_bank_v7.py` 写入）；V8 同理写入 `outputs/feature_banks_v8/metadata.json`。
- 评估/聚类脚本读取上述输出时已兼容旧格式（直接 dict）与新格式（`{meta, results}`）。

## 常见失败原因（必须先排查）

| 现象 | 根因 | 处理 |
|------|------|------|
| `cavities` 结果为空 / checkpoint 直接跳过 | `segment-anything` 未安装导致 SAM 加载失败，旧断点可能将“全失败”误写为 completed | 安装 `segment-anything` 后删除对应 checkpoint 或重跑；主线脚本已在日志/断点 summary 中记录 `missing_segment_anything` 并避免全失败写 completed=true |

## 核心问题诊断

V1 版本复现效果不佳的原因：

| 问题 | 原因 |
|------|------|
| Precision 极低 (6.6%) | **缺失 SAM 模块** - 直接对整图提取 patch |
| AUC = 0.5 | Feature Bank 与测试数据环境不匹配 |
| utilities 完全失败 | SAM 未正确识别候选区域 |

V2 版本修正：完整实现论文流程 `SAM生成候选区域 → 2D-ESN分析 → 异常判定`

## 项目结构（当前主线视角）

```text
Res-SAM/
├── PatchRes/
│   ├── ESN_2D_nobatch.py    # 2D-ESN 实现
│   ├── PatchRes.py          # V1 核心模块
│   ├── ResSAM.py            # V2 核心模块 ⭐
│   ├── sam_integration.py   # SAM 集成 ⭐
│   └── functions.py         # 工具函数
├── experiments/
│   ├── 01_build_feature_bank_v7.py           # Feature Bank ⭐
│   ├── 02_inference_auto_v7.py              # 自动推理 ⭐
│   ├── 03_evaluate_and_visualize_v7.py      # 评估与可视化 ⭐
│   ├── 04_clustering_v7.py                  # 异常聚类 ⭐
│   ├── 01_build_feature_bank_v8.py
│   ├── 02_inference_auto_v8.py
│   ├── 03_evaluate_and_visualize_v8.py
│   ├── 04_clustering_v8.py
│   ├── paper_constants.py                    # DEFAULT_BETA_THRESHOLD 等
│   └── dataset_layout.py / resize_policy.py
├── sam/
│   └── sam_vit_l_0b3195.pth            # SAM ViT-L（GitHub 可用 LFS 托管完整文件；OpenI 常需手动下载）
├── data/GPR_data/
│   ├── augmented_intact/               # 正常样本
│   ├── augmented_cavities/             # 空洞异常
│   └── augmented_utilities/            # 管道异常
└── outputs/                            # (运行后生成) 如 predictions_v6、feature_banks_v6
```

## 论文关键发现

1. **环境一致性** (Fig.6): Feature Bank 必须与测试数据来自同一环境，否则 AUC 下降 0.112-0.161
2. **SAM 作用** (Fig.5): SAM 能有效减少 2D-ESN 拟合次数，提升效率
3. **FCM 聚类** (Table 3): 模糊聚类更适合边界模糊的 GPR 异常

## 2026-03 历史口径整理说明

- `03_inference_click_v4.py` / `03_inference_click_v5.py` 的 click 语义已统一为覆盖全部 GT bbox 采样，不再只取第一个框。
- `06_ablation_wo_sam_v4.py` / `06_ablation_wo_sam_v5.py` 已与主线统一使用单一 `beta_threshold`（默认见 `experiments/paper_constants.py`，与作者脚本 0.1 对齐），且 click 消融同样覆盖全部 GT bbox。
- `05_clustering_v4.py` / `05_clustering_v5.py` 已按论文 Methods / Table 3 调整为对最终 anomaly region 直接再次进行 2D-ESN 拟合，不再使用 patch 特征均值聚合作为 region feature。
- `04_evaluate_and_visualize_v4.py` / `04_evaluate_and_visualize_v5.py` 的论文数值现仅作为参考对照，不再表述为与当前唯一评测数据集的严格同口径复现。

## V2 版本改进记录

| 日期 | 改进内容 |
|------|----------|
| 2026-03-12 | 重写 detect_automatic 严格按照论文流程 |
| 2026-03-12 | 修复内存问题：曾试用 SAM vit_b；主线已统一为 vit_l（GitHub 托管大文件） |
| 2026-03-12 | 添加进度条 (tqdm) 和日志系统 |
| 2026-03-12 | 降低 window_size 到 30 匹配 SAM 候选区域 |
| 2026-03-13 | 优化断点功能，添加每张图片处理日志 |

---

详细分析见 `experiments/ANALYSIS.md`
## 2026-04-08 主线说明

- 当前可运行实验分支仅保留 `v6`。
- 实现口径采用“论文优先；若论文未明确给出参数或工程细节，再回退到官方公开代码选择”。
- `v6` 当前使用论文公式特征定义 `f=[W_out,b]`。
- `v7` 已删除，不再作为可运行分支保留。
- 当前唯一保留的数据映射使用带标注的数据集：
  - `data/GPR_data/augmented_intact`
  - `data/GPR_data/augmented_cavities`
  - `data/GPR_data/augmented_utilities`
- `beta_threshold` 默认值保留为 `0.1`，原因是论文 Eq.(9) 只给出了符号 `β`，未给出固定数值，因此此处沿用官方公开代码默认值。
- 当前 V7/V8 的候选区细化遵循论文描述：先取 SAM 候选区域的最小外接矩形，再在该矩形内逐点提取 centered patch；自动模式粗筛先按 coarse region 的 region feature 与 Feature Bank 比较，再进入逐点细化。

### 推荐运行方式

```bash
python experiments/01_build_feature_bank_v7.py
python experiments/02_inference_auto_v7.py
python experiments/03_evaluate_and_visualize_v7.py
python experiments/04_clustering_v7.py
```




