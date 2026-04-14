# V14 实验计划

## V14 = V13 + 方案1（自适应beta）+ 方案2（top-1框）

Res-SAM 核心框架不变：SAM 候选区域 → ESN 特征 → Feature Bank 比较

---

## 方案1：自适应 beta 校准（01_build_feature_bank_v14.py）

**问题**：V13 粗筛丢弃率 = 0%，beta=0.1 在背景去除后变成全通，FP=181

**根因**：背景去除后特征距离整体缩小，固定 beta=0.1 不再有效

**改动**：
- 建库完成后，对 Feature Bank 所有 patch 计算 1-NN L2 距离分布
- 取 p95 作为自适应 beta，写入 `metadata.json`
- 推理脚本从 metadata 读取 `adaptive_beta`，不再使用固定值
- 这样 beta 会随 Feature Bank 的实际分布自动调整

**预期**：粗筛丢弃率从 0% 恢复到合理水平，FP 大幅减少，Precision 提升

---

## 方案2：top-1 pred 框（02_inference_auto_v14.py）

**问题**：每张图平均 6 个 pred 框，GT 只有 1 个，多余的框全是 FP

**改动**：
- 推理后对每张图只保留 anomaly_score 最高的 1 个框
- `top_k_preds = 1`，可配置

**预期**：Precision 大幅提升（FP 接近 0），Recall 略降（最高分框不一定是 TP），F1 整体上升

---

## 继承 V13
- 小样本分组采样（20张，按原始图ID分组）
- PatchCore coreset（patch内部贪心去重，ratio=0.5）
- GPR row_mean 背景去除（建库+推理一致）
- merge_all_anomaly_patches=True

---

## 运行命令

```bash
# 建库（同时计算自适应beta，约需1~2分钟）
conda run -n res-sam python Res-SAM/experiments/01_build_feature_bank_v14.py

# 推理（先用3张快速验证）
set MAX_IMAGES_PER_CATEGORY=3
conda run -n res-sam python Res-SAM/experiments/02_inference_auto_v14.py

# 评估
conda run -n res-sam python Res-SAM/experiments/03_evaluate_and_visualize_v14.py
```

---

## 版本对比

| 版本 | F1(IoU>0.5) | Precision | Recall | AUC | 核心改动 |
|---|---|---|---|---|---|
| V7 基线 | 0.061 | 0.041 | 0.118 | 0.746 | 论文语义 |
| V13（30张/类） | 0.080 | 0.052 | 0.167 | 0.793 | 分组采样+coreset+背景去除+merge_all |
| V14（预期） | >0.15 | >0.2 | ~0.15 | >0.79 | +自适应beta+top-1框 |

---

## 下一步（V15 候选）
- GPR 行列双向背景去除（row+column mean）
- pred 框向外扩展（基于统计的固定比例）
- ESN hidden_size 30→100
