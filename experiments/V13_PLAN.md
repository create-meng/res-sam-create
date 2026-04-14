# V13 实验计划

## V13 = V12 + 三项增强（保留 Res-SAM 骨架）

Res-SAM 核心框架不变：SAM 候选区域 → ESN 特征 → Feature Bank 比较

### 增强1：小样本分组采样 + PatchCore Coreset（01_build_feature_bank_v13.py）

**问题**：V12 随机采样 20 张，可能集中在少数原始图的增强版本，(2) 系列纹理未覆盖，
导致正常图被误判为高分异常（正常图 mean_score=0.551 > 异常图 0.318~0.403）

**改动**：
- **保留小样本特性**：仍然只用 20 张正常图建库，符合 Res-SAM 论文设计
- **分组采样**：按原始图 ID 分组，每组选 1 张，保证覆盖所有原始图纹理
- **PatchCore coreset**：从 20 张图的 patch 特征中贪心选出最具代表性的子集（coreset_ratio=0.5）
  - 去除重复 patch，Feature Bank 更紧凑多样
  - 不是增加样本数量，而是改进"选哪 20 张"和"patch 如何入库"

**预期**：AUC 提升，正常图误判减少

### 增强2：GPR 背景去除预处理（01 建库 + 02 推理）

**问题**：GPR B-scan 的水平条带是背景噪声，干扰 ESN 特征提取

**改动**：
- 对每张图做 row_mean 背景去除（减去每行均值，再重新归一化）
- 建库和推理使用相同的预处理，保证特征空间一致

**预期**：异常双曲线特征更突出，正常区域更接近零，判别能力提升

### 增强3：merge_all_anomaly_patches=True（02 推理）

**来源**：V12 模拟验证有效
- 模拟结果：F1 0.060→0.137，Precision 0.041→0.148，FP 284→75
- V10/V11 同时调高 beta 抵消了收益，V13 beta 保持 0.1 不变

**改动**：把一个 region 内所有异常 patch 合并为一个最小外接矩形

---

## 运行命令

```bash
# 建库（20张分组采样 → coreset，约需1~2分钟）
conda run -n res-sam python Res-SAM/experiments/01_build_feature_bank_v13.py

# 推理（先用3张快速验证）
set MAX_IMAGES_PER_CATEGORY=3
conda run -n res-sam python Res-SAM/experiments/02_inference_auto_v13.py

# 评估
conda run -n res-sam python Res-SAM/experiments/03_evaluate_and_visualize_v13.py
```

## 版本对比

| 版本 | F1(IoU>0.5) | AUC | 核心改动 |
|---|---|---|---|
| V7 基线 | 0.061 | 0.746 | 论文语义 |
| V12（模拟） | ~0.137 | ~0.747 | merge_all |
| V13（预期） | >0.15 | >0.75 | 分组采样+coreset + 背景去除 + merge_all |
