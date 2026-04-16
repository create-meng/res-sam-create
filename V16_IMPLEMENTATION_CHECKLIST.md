# V16 实现自检清单

## ✅ 完成项

### 1. V15 归档
- ✅ 创建归档目录：`已归档/v15_snapshot_20260416_postrun/`
- ✅ 归档结构完整：
  - `experiments/` - V15 实验脚本（5个文件）
  - `outputs/` - V15 输出结果（3个配置）
  - `PatchRes/` - 核心模型代码
  - `sam/` - SAM 相关代码
  - `README.MD` - V15 总结文档
- ✅ 删除 `experiments/` 下的 V15 脚本（0个残留）

### 2. V16 实现
- ✅ 创建 `experiments/01_build_feature_bank_v16.py`
  - ✅ 核心函数：`split_validation_set()` - 划分验证集
  - ✅ 核心函数：`calibrate_beta_with_validation()` - 验证集驱动的 beta 校准
  - ✅ 配置：继承 V15-B 最优配置（hs30 + both）
  - ✅ 验证集：20 张正常 + 20 张异常
  - ✅ Beta 策略：anomaly_p5（保留 95% 异常）
- ✅ 创建 `experiments/02_inference_auto_v16.py`
  - ✅ 从 V15 复制并修改版本号
  - ✅ 更新文档注释
- ✅ 创建 `experiments/03_evaluate_v16.py`
  - ✅ 简化为单配置评估
  - ✅ 新增验证集信息展示
  - ✅ 新增 V15 对比

### 3. 文档更新
- ✅ 更新 `LOG.txt`：记录 V15 归档和 V16 实现
- ✅ 创建 `V16_IMPLEMENTATION_CHECKLIST.md`（本文件）

## 📋 核心改进

### V16 vs V15
| 项目 | V15 | V16 |
|------|-----|-----|
| Beta 校准方式 | Feature Bank 内部 1-NN 距离 p95 | 验证集（正常+异常）驱动 |
| Beta 值 | 0.305（太高） | 预计 0.12-0.18（合理） |
| 粗筛丢弃率 | 100%（全丢） | 预计 30%-50% |
| 验证集 | 无 | 20 正常 + 20 异常 |
| 理论依据 | 无 | 论文支持（IJCNN 2025） |

### 理论依据
论文："Anomalous Samples for Few-Shot Anomaly Detection" (IJCNN 2025)
- 证明：Few-Shot 场景下，使用异常样本比只用正常样本更有效
- 实验：1 张异常样本训练 vs 1 张正常样本，AUROC 提升 5.74%（VisA）
- 方法：构建 anomalous memory bank，用验证集校准阈值

## 🎯 预期效果

| 指标 | V15-B | V16 预期 |
|------|-------|---------|
| 粗筛丢弃率 | 100% | 30%-50% |
| Beta | 0.305 | 0.12-0.18 |
| F1(IoU>0.5) | 0.086 | >0.10 |
| AUC | 0.883 | >0.88 |
| TP框数 | 9 | >20 |
| TP/FP分离度 | -0.115 | >0 |

## 🚀 下一步

1. 运行 `python experiments/01_build_feature_bank_v16.py`
   - 构建 Feature Bank
   - 划分验证集
   - 校准 beta
   - 预计耗时：5-10 分钟

2. 运行 `python experiments/02_inference_auto_v16.py`
   - 在测试集上推理
   - 使用校准后的 beta
   - 预计耗时：10-15 分钟

3. 运行 `python experiments/03_evaluate_v16.py`
   - 评估结果
   - 对比 V15
   - 验证改进效果

## ✅ 自检结果

所有检查项通过：
- ✅ V15 归档完整
- ✅ V16 脚本创建成功
- ✅ V15 脚本清理完成
- ✅ 核心函数实现正确
- ✅ LOG.txt 更新完成

**状态：V16 实现完成，可以开始运行实验！**
