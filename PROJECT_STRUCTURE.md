# Res-SAM 复现项目目录结构

## 项目总览

```
第五次搜集材料/
├── Res-SAM/                         # 主工作目录
│   ├── PatchRes/                    # 核心模块：2D-ESN + 异常检测
│   │   ├── PatchRes.py              # PatchRes 主类
│   │   ├── ESN_2D_nobatch.py        # 双向回声状态网络
│   │   ├── common.py                # 最近邻检索（faiss/sklearn）
│   │   ├── functions.py             # 数据加载、mask 生成
│   │   ├── main.py                  # 独立演示脚本
│   │   └── features.pth             # [预置] 特征银行（需重建）
│   ├── sam/                         # SAM 模型
│   │   └── sam_vit_l_0b3195.pth     # SAM ViT-L（GitHub LFS；OpenI 常需手动下载）
│   ├── ui/                          # GUI 界面
│   │   ├── ui_funcs.py              # 主窗口逻辑
│   │   └── utils.py                 # 工作线程
│   ├── data/                        # Mendeley 数据集
│   │   └── GPR_data/
│   │       ├── intact/              # Normal 样本（75 张）
│   │       ├── cavities/            # 空洞（80 张，无标注）
│   │       ├── Utilities/           # 管道（139 张，无标注）
│   │       ├── augmented_cavities/  # 增强空洞 + 标注（553 张）
│   │       │   ├── *.jpg
│   │       │   └── annotations/
│   │       │       ├── VOC_XML_format/  # VOC 格式 bbox
│   │       │       └── Yolo_format/     # YOLO 格式 bbox
│   │       └── augmented_utilities/ # 增强管道 + 标注（786 张）
│   │           ├── *.jpg
│   │           └── annotations/
│   ├── experiments/                 # 复现实验脚本
│   ├── outputs/                     # 输出结果
│   │   ├── feature_banks/           # 特征银行
│   │   ├── predictions/             # 预测结果
│   │   └── metrics/                 # 评估指标
│   ├── main.py                      # GUI 入口
│   ├── PLAN.md                      # 复现计划书
│   └── PROJECT_STRUCTURE.md         # 本文件
│
├── DOWNLOAD_URLS.txt                # 数据下载地址
├── Res-SAM原论文.pdf                # 论文原文
├── Res-SAM技术研究报告.docx         # 技术报告
└── 任务.txt                         # 任务说明
```

## 数据集说明

### Mendeley Open-source 数据集
- **来源**: https://data.mendeley.com/datasets/ww7fd9t325/1
- **论文对应**: Table 1/2 中的 "Open-source" 数据集（285 B-scans）

### 类别划分
| 类别 | 目录 | 数量 | 有标注 | 用途 |
|------|------|------|--------|------|
| Normal (intact) | `intact/` | 75 | 否 | Feature Bank 构建 |
| Cavity | `augmented_cavities/` | 553 | 是 (VOC/YOLO) | 测试集 |
| Utility/Pipe | `augmented_utilities/` | 786 | 是 (VOC/YOLO) | 测试集 |

### 标注格式
**VOC XML**:
```xml
<object>
    <name>cavities</name>
    <bndbox>
        <xmin>33</xmin>
        <ymin>53</ymin>
        <xmax>123</xmax>
        <ymax>222</ymax>
    </bndbox>
</object>
```

**YOLO txt**: `class cx cy w h`（归一化坐标）

## 复现实验数据划分

### Feature Bank 初始化
- 从 `intact/` 随机抽取 **20 张** normal 样本
- 用 2D-ESN 拟合生成特征银行

### 测试集
- `augmented_cavities/` 全部图片（有 GT bbox）
- `augmented_utilities/` 全部图片（有 GT bbox）
- 每张图都有对应的 VOC XML 标注

## 关键参数（论文 + 本仓库主线）

| 参数 | 值 | 说明 |
|------|-----|------|
| window_size | 50×50 | patch 尺寸（论文默认） |
| stride | 5 | 滑动步长（论文默认；非 5~10 泛称） |
| hidden_size | 30 | reservoir 神经元数 |
| beta_threshold (Eq.9 β) | 默认 **0.1** | 与作者 `PatchRes/main.py` 对齐；见 `experiments/paper_constants.py` 的 `DEFAULT_BETA_THRESHOLD`，可在各脚本 `CONFIG` 覆盖 |
| init_normal_samples | 20 | Feature Bank 初始样本数 |
| 评测 IoU | 0.5 | 预测与 GT 是否匹配（与 β 不同）；见 `EVAL_DETECTION_IOU_THRESHOLD` |

## 实验脚本目录（现状）

主线为 `experiments/*_v4.py`、`*_v5.py` 与 `run_all.py`（`--version v4` \| `v5`）。下文「待创建目录」为早期草稿，**以仓库内实际 `experiments/` 为准**。

```
experiments/
├── paper_constants.py
├── 01_build_feature_bank_v4.py / v5.py
├── 02_inference_auto_v4.py / v5.py
├── … 
└── run_all.py

# 运行生成（仓库根目录下，非 experiments 子目录）
outputs/feature_banks_v4/、outputs/predictions_v4/ 等
```
