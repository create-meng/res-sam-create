# Res-SAM 复现结果对比分析

## 一、复现结果概览

### Table 2: Fully Automatic 模式评估指标

| 类别 | TP | FP | FN | Precision | Recall | F1 | AUC |
|------|----|----|----|-----------|--------|-----|-----|
| cavities | 89 | 464 | 0 | 0.161 | 1.0 | 0.277 | 0.5 |
| utilities | 0 | 786 | 0 | 0.0 | 0 | 0 | 0.5 |
| **overall** | 89 | 1250 | 0 | **0.066** | **1.0** | **0.125** | **0.5** |

### Table 3: 异常聚类结果

| 方法 | Accuracy | ARI | NMI |
|------|----------|-----|-----|
| K-Means | 54.8% | -0.005 | 0.001 |
| Agglomerative (AC) | 54.8% | -0.005 | 0.001 |
| **FCM** | **93.65%** | **0.761** | **0.696** |

---

## 二、问题诊断

### 问题1: Precision 极低 (0.066)

**现象**: 
- Recall = 1.0 (所有异常都被检测到)
- Precision = 0.066 (大量误检)
- utilities 类别完全失败 (Precision=0)

**可能原因**:

1. **SAM 分割问题**
   - SAM 在 GPR 图像上的分割效果不佳
   - SAM 原本针对自然图像训练，对 B-scan 的双曲线特征不敏感
   - 自动模式下 SAM 可能生成过多候选区域

2. **阈值设置问题**
   - `anomaly_threshold = 0.1` 可能过低
   - 导致大量正常 patch 被判定为异常

3. **Feature Bank 构建问题**
   - Feature Bank 可能不够代表性
   - 正常样本数量不足或多样性不够

### 问题2: AUC = 0.5 (无区分能力)

**现象**: AUC = 0.5 等同于随机猜测

**可能原因**:

1. **异常分数计算问题**
   - 所有样本的异常分数分布相近
   - 特征距离计算可能存在问题

2. **Feature Bank 与测试数据不匹配**
   - Feature Bank 来自 `intact/` 目录
   - 测试数据来自 `augmented_cavities/` 和 `augmented_utilities/`
   - 两者可能来自不同环境/设备

### 问题3: utilities 类别完全失败

**现象**: TP=0, 所有预测都是 FP

**可能原因**:

1. **数据标注问题**
   - utilities (管道/井盖) 的 GT bbox 可能与 cavities 不同
   - 模型可能检测到了 utilities 但 bbox 与 GT 不匹配

2. **特征差异**
   - utilities 的 GPR 特征与 cavities 差异较大
   - Feature Bank 可能无法覆盖 utilities 的正常特征

---

## 三、聚类结果分析

### FCM 表现优异 (93.65%)

**原因分析**:
- FCM (模糊C均值) 允许软聚类，更适合边界模糊的异常
- GPR 异常类别 (cavity vs utility) 在特征空间有一定区分度

### K-Means 和 AC 表现差

**原因**:
- 硬聚类不适合边界模糊的异常
- ARI ≈ 0 说明聚类结果与真实标签几乎无关

---

## 四、与论文预期差距

| 指标 | 预期 | 实际 | 差距原因 |
|------|------|------|----------|
| Precision | >0.5 | 0.066 | SAM分割效果差、阈值过低 |
| Recall | >0.8 | 1.0 | 阈值过低导致过度检测 |
| F1 | >0.6 | 0.125 | Precision拖累 |
| AUC | >0.7 | 0.5 | 特征提取或距离计算问题 |
| FCM Acc | >0.9 | 0.937 | ✅ 接近预期 |

---

## 五、改进建议

### 短期改进

1. **调整阈值**
   ```python
   # 尝试更高阈值
   anomaly_threshold = 0.3  # 原为 0.1
   ```

2. **优化 SAM 分割**
   - 使用 click-guided 模式替代 fully automatic
   - 或使用更适合 GPR 的分割方法

3. **增加 Feature Bank 多样性**
   - 使用更多 normal 样本
   - 确保与测试数据来自同一环境

### 长期创新方向

1. **替换 SAM**
   - 使用专门针对 GPR 训练的分割模型
   - 或使用传统边缘检测 + 双曲线拟合

2. **改进特征提取**
   - 尝试其他 Reservoir Computing 变体
   - 或使用轻量级 CNN 提取特征

3. **自适应阈值**
   - 根据特征距离分布自动确定阈值
   - 使用 Otsu 或类似方法

---

## 六、下一步行动

- [ ] 尝试提高 anomaly_threshold 到 0.3-0.5
- [ ] 检查 Feature Bank 与测试数据的来源一致性
- [ ] 使用 click-guided 模式重新评估
- [ ] 分析单张图像的 SAM 分割效果
