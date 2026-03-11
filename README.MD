# Res-SAM 论文复现

> 原始项目: [Res-SAM](https://github.com/xxx/res-sam) (请参考原仓库获取完整文档)

本项目旨在复现 Res-SAM 论文实验，并探讨可能的创新改进方向。

## 复现状态

- [x] 特征银行构建
- [x] 自动模式推理 (支持断点续传)
- [x] 点击引导模式推理 (支持断点续传)
- [x] 评估指标计算 (Table 2)
- [x] 异常聚类分析 (Table 3) - FCM准确率: 93.65%

## 快速开始

```bash
# 安装环境
conda env create -f environment-cpu.yml
conda activate res-sam

# 下载SAM权重到 sam/ 目录
# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

# 运行实验
cd experiments
python run_all.py
```

## 实验脚本

| 脚本 | 功能 |
|------|------|
| `01_build_feature_bank.py` | 构建特征银行 |
| `02_inference_auto.py` | 自动模式批量推理 |
| `03_inference_click.py` | 点击引导模式推理 |
| `04_evaluate.py` | 计算评估指标 |
| `05_clustering.py` | 异常聚类 |
| `run_all.py` | 一键运行全部 |

## 关键参数

```python
window_size = 50      # 2D-ESN patch大小
stride = 10           # 滑动窗口步长
hidden_size = 30      # ESN隐藏层大小
image_size = 256      # 图像尺寸
```

## 创新探讨

(待补充)

---

详细使用说明请参考原仓库文档。
