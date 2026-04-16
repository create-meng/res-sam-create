# V15 实验计划

## V15 = V14 + 方案5（hidden_size=100）+ 方案3（行列双向背景去除）

---

## 方案5：ESN hidden_size 30 → 100（01_build_feature_bank_v15.py）

**问题**：V14 诊断——TP 框 score 系统性低于 FP 框（10/10 TP 框 score < FP 均值），
top-1 框是 TP 的比例 = 0/51（0%）。ESN 评分方向有问题，特征判别能力不足。

**改动**：
- hidden_size: 30 → 100
- ESN 特征维度: 61 → 201（含 bias）
- 更高维度的 reservoir 有更强的特征表达能力，有望分离 TP/FP 分布

**预期**：TP/FP score 分布分离，F1 提升

---

## 方案3：GPR 行列双向背景去除（01 建库 + 02 推理）

**问题**：V13/V14 只做行均值去除（水平条带），GPR B-scan 还有垂直方向的直达波

**改动**：
- background_removal_method: "row_mean" → "both"（先减行均值，再减列均值）
- 建库和推理使用相同预处理

**预期**：特征更纯净，AUC 进一步提升

---

## 放弃（V14 诊断证明无效）
- top-1 框过滤：TP score 低于 FP，top-1 永远不是 TP
- score 阈值过滤：TP/FP 分布完全混叠，任何阈值都会同时过滤 TP

---

## 继承
- 小样本分组采样（20张，按原始图ID分组）
- PatchCore coreset（ratio=0.5）
- 自适应 beta（p95 of 1-NN dists，从 metadata 读取）
- merge_all_anomaly_patches=True

---

## 运行命令

```bash
# 建库（hidden_size=100，约需2~3分钟）
conda run -n res-sam python Res-SAM/experiments/01_build_feature_bank_v15.py

# 推理（先用3张快速验证）
set MAX_IMAGES_PER_CATEGORY=3
conda run -n res-sam python Res-SAM/experiments/02_inference_auto_v15.py

# 评估
conda run -n res-sam python Res-SAM/experiments/03_evaluate_and_visualize_v15.py
```

---

## 版本对比

| 版本 | F1(IoU>0.5) | Precision | Recall | AUC | hidden_size | 背景去除 |
|---|---|---|---|---|---|---|
| V7 基线 | 0.061 | 0.041 | 0.118 | 0.746 | 30 | 无 |
| V13 | 0.080 | 0.052 | 0.167 | 0.793 | 30 | row_mean |
| V14 | 0.018 | 0.020 | 0.017 | 0.881 | 30 | row_mean |
| V15（预期） | >0.10 | >0.10 | >0.10 | >0.88 | **100** | **both** |
