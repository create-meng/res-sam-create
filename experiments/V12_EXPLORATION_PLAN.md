# V12 探索计划

## 基线
V12 = V7（最接近论文语义的版本）+ 图像级AUC评估

V7基线指标（50张/类）：
- F1 = 0.061
- 图像级AUC = 0.746（论文口径）
- 论文目标：AUC = 0.832

## 核心问题

经过V7~V11的诊断，确认两个独立问题：

### 问题A：图像级AUC=0.746，距论文0.832差0.086
根因：Feature Bank判别能力不足
- 正常图(normal_auc) max_score均值=0.215，但有一半图max_score=0（无pred框）
- 异常图 max_score均值=0.428
- 分离度存在但不够大，部分正常图被误判为高分

### 问题B：Pre/Rec/F1极低（F1=0.06）
根因：pred框太小（74px），GT框太大（224px），IoU结构性<0.5
- 这是augmented数据集（224×224）与论文数据（369×369）的结构性差异
- 不是实现问题，是数据集问题

---

## V12探索方向

### 方向1：改善Feature Bank的判别能力（针对问题A）

**1a. 用原始intact图像建库（非augmented）**
- 当前用augmented_intact（旋转/翻转增强后），ESN特征可能因增强操作产生分布偏移
- 尝试：从download/official_gpr_data.zip解压real_world/normal（100张，369×369）建库
- 预期：real_world数据与augmented数据不同环境，可能更差（论文Fig.6跨环境实验）
- 但值得验证

**1b. 改变Feature Bank的构建方式：用滑窗全覆盖而非随机采样**
- 当前：随机选20张图，每张图滑窗提取patch
- 尝试：选20张图但用更小的stride（stride=1）提取更密集的patch
- 预期：Feature Bank覆盖更全面，正常特征空间更完整

**1c. 对Feature Bank做归一化/标准化**
- 当前：原始L2距离空间，正常图和异常图的分数分布重叠
- 尝试：对Feature Bank的特征做L2归一化，使特征在单位球面上，距离更有意义
- 这是工程优化，不改变论文语义

### 方向2：改善Pre/Rec/F1（针对问题B）

**2a. 改变评估口径：用"中心点命中"代替IoU>0.5**
- 59%的pred框中心在GT框内，说明定位是对的
- 用"pred框中心是否在GT框内"作为TP判据
- 这不是论文口径，但能反映定位能力

**2b. 改变评估口径：降低IoU阈值到0.1或0.2**
- IoU>0.1时TP=44，F1=0.221（vs IoU>0.5时TP=12，F1=0.06）
- 对于augmented数据集（GT框大、pred框小），0.1更合理

**2c. 改变bbox生成：用SAM mask的外接矩形直接作为pred框**
- 不做patch级细化，直接用SAM找到的候选区域作为输出
- 这样pred框可以覆盖整个候选区域，IoU能达到0.5
- 但这偏离了论文的patch级精化流程

### 方向3：验证论文的真实数据集效果

**3a. 用real_world数据（369×369）跑推理，算图像级AUC**
- real_world有6类各100张，normal作为负类，其余5类作为正类
- 不需要bbox标注，只需要图像级标签
- 这才是论文Table 2的真实数据集

---

## 推荐执行顺序

1. **先跑方向3a**（real_world图像级AUC）
   - 最快验证：用real_world/normal建Feature Bank，对real_world全量跑推理
   - 能直接和论文0.832对比
   - 需要修改dataset_layout.py支持real_world数据

2. **再跑方向1c**（Feature Bank特征归一化）
   - 改动最小，只改_score_features_against_bank里的距离计算
   - 不需要重新建库，直接重新评分

3. **最后跑方向2b**（降低IoU阈值）
   - 不改代码，只改评估口径
   - 能更公平地反映augmented数据集上的定位能力

---

## 当前V12状态
- 基线：V7语义（num_normal_samples=20, beta=0.1, region feature粗筛）
- 新增：图像级AUC评估（03_evaluate_and_visualize_v12.py）
- 待实现：上述探索方向
