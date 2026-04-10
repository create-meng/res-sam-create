# TileRes 异常检测说明

## 概览

本目录提供基于 TileRes / PatchRes 的 GPR 图像异常检测最小示例。

## 仓库说明

- 当前持续维护的完整 Res-SAM 复现主线位于 `../experiments/*_v6.py`。
- 实现口径采用“论文优先；若论文未明确给出参数或工程细节，再回退到官方公开代码选择”。
- 本目录中的示例仍是最小化的 PatchRes / TileRes 演示，不是完整的端到端 Res-SAM 主流程。

## 数据结构

项目数据按如下方式组织：

- **测试图像**：`./data/images/`
  - 将测试集对应图像放在该目录。

- **训练正常图像**：`./data/normal/`
  - 将用于训练异常检测模型的正常图像放在该目录。

## 运行演示

建议使用 **python>=3.8**，推荐 **python=3.9**。

如果你要运行 Res-SAM 主线实验，请查看 `../experiments/` 下的 `*_v6.py` 以及 `../experiments/run_all.py`。V3 归档快照位于 `../已归档/experiments_v3_snapshot_20260326/`。

请先确保已经安装依赖，可通过 `requirements.txt` 安装：

```bash
pip install -r requirements.txt
```

运行 TileRes 异常检测：

```bash
python main.py
```

## 输出结果

异常检测结果将保存在以下目录：

- **异常框图像**：`./output/frames/`
  - 该目录保存绘制了异常框的图像。

- **异常掩码**：`./output/masks/`
  - 该目录保存生成的异常掩码。
