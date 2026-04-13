# V12 详细探索计划

## 基线与现状

**V12 基线 = V7 语义（最接近论文）+ 图像级 AUC 评估**

| 指标 | V12 基线 | 论文目标 | 差距 |
|---|---:|---:|---:|
| 图像级 AUC | 0.746 | 0.832 | −0.086 |
| F1 (IoU>0.5) | 0.061 | 0.859 | −0.798 |
| Precision | — | 0.842 | — |
| Recall | — | 0.877 | — |

**数据集**：augmented_intact（正常，建库+负类）/ augmented_cavities / augmented_utilities（异常，正类）

---

## 根因分析（V7~V11 诊断结论）

### 问题 A：图像级 AUC = 0.746（目标 0.832）

根因：Feature Bank 判别能力不足，正常图与异常图的 max_score 分布重叠。

- 正常图（normal_auc）：max_score 均值 ≈ 0.215，其中约 50% 的图 max_score = 0（无 pred 框）
- 异常图：max_score 均值 ≈ 0.428
- 分离度存在但不够大，部分正常图被误判为高分

**子根因**：
1. Feature Bank 用 augmented_intact（旋转/翻转增强后的图）建库，增强操作可能引入分布偏移
2. Feature Bank 覆盖的正常特征空间不够完整（随机选 20 张，stride=5）
3. 图像级分数 = max(anomaly_scores)，对 FP 框非常敏感

### 问题 B：F1 极低（0.061）

根因：pred 框太小（~74px），GT 框太大（~224px），IoU 结构性 < 0.5。

- augmented 数据集图像 224×224，GT 框覆盖大部分图像
- 论文数据集图像 369×369，GT 框相对更小
- 这是数据集结构性差异，不是实现 bug

---

## 探索方向与实验设计

### 方向 A：提升图像级 AUC（核心目标）

#### A1. 用 real_world 数据集跑图像级 AUC（最高优先级）

**动机**：论文 Table 2 的 AUC=0.832 是在 real_world 数据集（369×369）上测的，不是 augmented 数据集。
augmented 数据集是 224×224 的增强版，论文并未在此数据集上报告 AUC。

**实验设计**：
- 数据来源：`Res-SAM/data/GPR_data/` 下是否有 real_world 子目录？
  - 若无，需从 Mendeley 数据集下载 `real_world/` 目录
  - 若有，直接使用
- Feature Bank：用 `real_world/normal`（100 张，369×369）建库
- 推理：对 `real_world/` 下所有类别（normal + 5 个异常类）跑推理
- 评估：图像级 AUC，正类=5 个异常类，负类=normal
- 预期：这才是论文的真实评估口径，AUC 应接近 0.832

**实现步骤**：
1. 检查 `data/GPR_data/` 是否有 real_world 数据
2. 若无，在 `dataset_layout.py` 中新增 `DATASET_REAL_WORLD` 模式
3. 新建 `01_build_feature_bank_v12a.py`（real_world/normal 建库）
4. 新建 `02_inference_auto_v12a.py`（real_world 全量推理）
5. 复用 `03_evaluate_and_visualize_v12.py` 评估

**论文对齐**：论文 Table 2 fully automatic 设置，Methods 中 real_world 数据集描述。

---

#### A2. Feature Bank L2 归一化（工程优化，语义不变）

**动机**：当前 `_score_features_against_bank` 用原始 L2 距离，特征向量长度不均匀时距离度量偏差大。
L2 归一化后特征在单位球面上，余弦距离等价于 L2 距离，判别边界更清晰。

**实现**：在 `ResSAM._score_features_against_bank` 中，查询向量和 Feature Bank 向量都做 L2 归一化后再计算距离。
或者在 `build_feature_bank` 时对存储的特征做归一化，推理时对查询特征同样归一化。

**注意**：
- 归一化后 beta_threshold 的物理含义变化（原来是 L2 距离，归一化后是 [0, √2] 范围的距离）
- 需要重新校准 beta 或用 p99 自动校准（参考 V10 的 `DEFAULT_BETA_THRESHOLD_V10 = 0.183`）
- 这是**工程优化但语义不变**（论文 Eq.(7)-(9) 只说 1-NN L2，未规定是否归一化）

**实验设计**：
- 在 `01_build_feature_bank_v12.py` 中加 `normalize_features: bool` 开关
- 建库时对 feature_bank 做 L2 归一化并存储
- 推理时对查询特征同样归一化
- beta 用 p99 自动校准（`_calibrate_beta_threshold`）
- 对比：归一化前后的图像级 AUC

**论文对齐**：工程优化，不改变 Eq.(7)-(9) 的 1-NN L2 语义。

---

#### A3. 增大 Feature Bank 覆盖（工程优化）

**动机**：当前随机选 20 张图建库，stride=5，每张 369×369 图约 (369-50)/5 × (369-50)/5 ≈ 4096 个 patch。
20 张图共约 81920 个 patch，但正常特征空间可能未被充分覆盖。

**实验设计**：
- 方案 A3a：增加建库图像数量（20 → 50 张）
- 方案 A3b：减小 stride（5 → 2），更密集采样
- 方案 A3c：两者结合

**注意**：stride=2 时每张图 patch 数约 (369-50)/2 × (369-50)/2 ≈ 25600，50 张图共约 128 万 patch，
内存和 faiss 索引构建时间会显著增加，需评估可行性。

**论文对齐**：论文未明确规定建库图像数量和 stride，属于工程默认。

---

#### A4. 图像级分数聚合方式（工程优化）

**动机**：当前图像级分数 = `max(anomaly_scores)`，对单个 FP 框非常敏感。
正常图如果有一个高分 FP 框，整张图就被判为高分异常。

**实验设计**：
- 方案 A4a：`mean(top-k anomaly_scores)`，k=3，减少单点噪声
- 方案 A4b：`sum(anomaly_scores)`，累积分数
- 方案 A4c：`max(anomaly_scores) * num_pred_boxes`，惩罚多框

**注意**：这改变了图像级分数的定义，不是论文显式公式，属于**工程优化但语义不变**。
论文 Table 2 AUC 的图像级分数聚合方式论文未明确说明。

---

### 方向 B：改善 Pre/Rec/F1（次要目标，数据集结构性问题）

#### B1. 降低 IoU 阈值评估（评估口径调整）

**动机**：augmented 数据集 GT 框大（224px），pred 框小（74px），IoU 结构性 < 0.5。
降低 IoU 阈值能更公平地反映定位能力。

**实验设计**：
- 在 `03_evaluate_and_visualize_v12.py` 中增加多阈值评估：IoU ∈ {0.1, 0.2, 0.3, 0.5}
- 不改变主指标（IoU>0.5），额外报告低阈值结果

**论文对齐**：主指标保持论文口径（IoU>0.5），低阈值结果作为补充分析。

---

#### B2. 中心点命中率评估（补充指标）

**动机**：V11 诊断显示 59% 的 pred 框中心在 GT 框内，说明定位是对的，只是框太小。
"中心点命中"能反映定位能力，与 IoU 互补。

**实验设计**：
- 在 `03_evaluate_and_visualize_v12.py` 中增加 `center_hit_rate` 指标
- 判据：pred 框中心 (cx, cy) 是否在任意 GT 框内
- 报告：center_hit_rate = TP_center / (TP_center + FP_center)

**论文对齐**：补充指标，不替代论文口径。

---

#### B3. 使用 VOC 标注尺寸做坐标映射（修复潜在 bug）

**动机**：当前推理时将图像 resize 到 369×369，但 GT 框是原始尺寸（224×224）的坐标。
坐标映射逻辑在 `02_inference_auto_v12.py` 中：`scale_x = target_w / proc_w`。
需要验证这个映射是否正确，特别是 augmented 数据集（原始 224×224 → resize 到 369×369）。

**实验设计**：
- 打印几张图的 pred_bboxes 和 gt_bboxes，目视检查坐标是否对齐
- 如果发现映射错误，修复后重新评估

**论文对齐**：这是实现正确性检查，不改变论文语义。

---

### 方向 C：诊断与分析

#### C1. Feature Bank 分布可视化

**动机**：了解 Feature Bank 的特征分布，找到正常图和异常图分数重叠的原因。

**实验设计**：
- 对 Feature Bank 中的特征做 PCA/t-SNE 可视化
- 对比正常图 patch 特征和异常图 patch 特征的分布
- 分析 max_score 分布：正常图 vs 异常图的直方图

**输出**：可视化图表，帮助理解 AUC 差距的根因。

---

#### C2. SAM 候选区域质量分析

**动机**：SAM 生成的候选区域质量直接影响后续 ESN 评分。
如果 SAM 在正常图上生成了大量高分候选区域，会导致 FP 增多，AUC 下降。

**实验设计**：
- 统计正常图和异常图的候选区域数量分布
- 统计粗筛（coarse filtering）的剔除率
- 分析哪类图像的 FP 最多

---

## 推荐执行顺序

### 第一阶段：验证论文真实口径（1~2 天）

**A1**（real_world 数据集 AUC）是最高优先级。
这能直接回答"我们的实现在论文数据集上能达到多少 AUC"。

执行前提：
1. 检查 `data/GPR_data/` 是否有 real_world 数据
2. 若无，从 Mendeley 下载（`https://data.mendeley.com/datasets/ww7fd9t325/1/`）

---

### 第二阶段：改善 augmented 数据集上的 AUC（2~3 天）

按改动量从小到大：

1. **A2**（Feature Bank L2 归一化）：改动最小，只改评分函数，不需要重新建库
2. **A4**（图像级分数聚合）：只改评估脚本，不需要重新推理
3. **A3a**（增加建库图像数量 20→50）：需要重新建库和推理

---

### 第三阶段：补充分析（1 天）

1. **B1**（多 IoU 阈值评估）：只改评估脚本
2. **B2**（中心点命中率）：只改评估脚本
3. **B3**（坐标映射验证）：诊断性检查
4. **C1**（Feature Bank 分布可视化）：理解根因

---

## 实验命名规范

| 实验 ID | 描述 | 脚本后缀 |
|---|---|---|
| v12 | 基线（augmented，当前实现） | `_v12` |
| v12a | real_world 数据集 AUC | `_v12a` |
| v12b | Feature Bank L2 归一化 | `_v12b` |
| v12c | 增加建库图像数量（50 张） | `_v12c` |
| v12d | 图像级分数聚合方式对比 | 在 `03_evaluate_and_visualize_v12.py` 中加参数 |

---

## 论文对齐声明

| 改动 | 分类 |
|---|---|
| A1 real_world 数据集 | 论文一致（Table 2 真实口径） |
| A2 Feature Bank L2 归一化 | 工程优化但语义不变 |
| A3 增加建库图像数量/stride | 工程优化但语义不变（论文未规定） |
| A4 图像级分数聚合 | 工程优化但语义不变（论文未规定聚合方式） |
| B1 多 IoU 阈值 | 工程优化但语义不变（主指标保持 IoU>0.5） |
| B2 中心点命中率 | 补充指标，不替代论文口径 |
| B3 坐标映射验证 | 实现正确性检查 |

---

## 诊断结论（2026-04-13，基于 V7 推理结果）

**方案5（bbox膨胀）验证：无效。**
- 膨胀后 TP 不升反降，最大 F1=0.066（x4.0），与原始 0.061 基本持平
- 根本原因：FP 太多（290 个 pred 框中只有 12 个 TP），膨胀让 FP 框更大，反而更容易误匹配

**真正的根因：粗筛几乎不过滤**
- SAM 平均每张图生成 3.6 个候选区域，粗筛只丢弃 0.1 个（beta=0.1 太宽松）
- 每张图只有 1 个 GT，所以平均 2.6 个 FP/图
- 76% 的图有 pred 框但没有 TP——pred 框位置本身就不对

**实施方案（已写入 02_inference_auto_v12.py）**：
- 方案4：`coarse_beta_threshold=0.25`（严格粗筛，减少 FP region）
- 方案1：`merge_all_anomaly_patches=True`（所有异常 patch 合并为一个框，扩大 TP 框覆盖）

## 当前 V12 状态

- 基线：V7 语义（num_normal_samples=20, beta=0.1, region feature 粗筛）
- 新增：图像级 AUC 评估（`03_evaluate_and_visualize_v12.py`）
- **已修改**：`02_inference_auto_v12.py` 加入方案1+4（coarse_beta=0.25, merge_all=True）
- outputs/ 目录：仅有 logs/，尚未跑过推理
- 待执行：先跑 01 建库，再跑 02 推理，再跑 03 评估
