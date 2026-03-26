# Res-SAM V3 评估报告

## 数据集对应关系
- 论文参考结果: Open-source (paper Table 1/2)
- 本地评估数据: intact + augmented_cavities + augmented_utilities
- 类别映射: cavities -> cavities, utilities -> pipes/utilities, normal_auc -> intact
- 说明: Local evaluation uses intact + augmented_cavities + augmented_utilities as the nearest runnable proxy to the paper open-source setting.

## 全自动模式直接对比

| 指标 | 论文 | 本地 | 差值 |
|---|---:|---:|---:|
| Precision | 0.842 | 0.159 | -0.683 |
| Recall | 0.877 | 0.152 | -0.725 |
| F1 | 0.859 | 0.155 | -0.704 |
| AUC | 0.832 | 0.700 | -0.132 |

## 全自动模式统计

- TP: 241
- FP: 1273
- FN: 1346
- Region 级粗筛丢弃数: 1022

## 分类别结果

| 类别 | TP | FP | FN | Precision | Recall | F1 | 粗筛丢弃数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cavities | 46 | 246 | 554 | 0.158 | 0.077 | 0.103 | 185 |
| utilities | 195 | 1027 | 792 | 0.160 | 0.198 | 0.177 | 837 |
| normal_auc | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 | 0 |

*生成时间: 2026-03-26 20:43:11*
*版本: V3（论文对齐评估脚本）*

## Click-guided 模式直接对比

| 点击配置 | 论文 AUC | 本地 AUC | 差值 | 论文 F1 | 本地 F1 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| 5/5 | 0.823 | 0.452 | -0.371 | 0.852 | 0.079 | -0.773 |
| 5/3 | 0.827 | 0.428 | -0.399 | 0.863 | 0.086 | -0.777 |
| 3/1 | 0.832 | 0.473 | -0.359 | 0.850 | 0.086 | -0.764 |
