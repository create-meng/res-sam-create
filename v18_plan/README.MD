# V18 方案设计

Created: 2026-04-19

---

## 背景：为什么V18要回到V16

### V7-V17-3 的核心教训

经过12个版本的实验，得出以下结论：

1. **V16是唯一有效的基础**：TP=10，F1=0.083，patch合并方法覆盖全图不会漏检
2. **V17系列heatmap方向是结构性失败**：
   - SAM候选太少 → TP消失
   - SAM候选太多 → 速度暴增33倍（145s/张）
   - 形态学膨胀受候选区域边界限制，无法解决框太小问题
3. **V16唯一的问题**：FP=170太多（Precision=0.056，FP/TP=17:1）

### V18的核心目标

**在保持V16 TP=10的基础上，将FP从170减少到50以下，F1从0.083提升到0.15+**

---

## V18 方案：全图patch-level异常分数图 + 多级FP抑制

### 核心思路

V16的patch合并方法（merge_all）存在一个根本问题：
- 它把所有异常patch合并成一个大框，导致框太大或太小
- 没有利用patch-level的空间分布信息

**V18的新思路**：
1. 保留V16的ESN特征提取和Feature Bank比较
2. 不再用merge_all合并patch，而是**生成全图patch-level异常分数图**
3. 对分数图做**连通分量分析**，每个连通区域生成一个bbox
4. 对bbox做**多级FP抑制**

这个思路来自PatchCore（CVPR 2022）的anomaly map生成方式，但结合了Res-SAM的小样本特性。

---

## V18 具体实现

### 改动1：全图patch-level异常分数图（核心）

**当前V16的做法**：
```
滑窗提取patch → ESN特征 → Feature Bank比较 → 异常patch合并(merge_all) → bbox
```

**V18的新做法**：
```
滑窗提取patch → ESN特征 → Feature Bank比较 → 生成全图分数图 → 阈值化 → 连通分量 → bbox
```

**具体实现**：
- 对每个patch位置，记录其异常分数，填入对应的全图分数图（H×W）
- 分数图用高斯平滑（sigma=2）消除噪声
- 用beta阈值化得到二值图
- 连通分量分析，每个连通区域的外接矩形作为bbox
- 过滤掉面积太小的连通区域（min_area = GT平均面积的10% ≈ 3000px²）

**预期效果**：
- bbox直接来自patch分数的空间分布，尺寸更接近真实异常区域
- 不再受SAM候选区域边界限制
- 速度与V16相当（不需要SAM生成候选）

### 改动2：基于尺寸的FP抑制

**依据**：
- GT框平均面积：32240px²（224×166px）
- V16 pred框平均面积：约5000px²（太小）
- 真实异常区域面积应该在GT的20%-200%范围内

**实现**：
- 过滤掉面积 < 3000px²（GT的10%）的框 → 去除噪声小框
- 过滤掉面积 > 80000px²（GT的250%）的框 → 去除过大的背景框
- 这两个阈值基于GT统计，不是拍脑袋

**预期效果**：
- 去除大量小噪声框（V16的FP中很多是小框）
- 不影响真实异常框（面积在合理范围内）

### 改动3：基于分数的per-image自适应阈值

**当前V16的做法**：全局beta阈值（0.1399），所有图片用同一个阈值

**V18的新做法**：per-image自适应阈值
- 对每张图，计算patch分数的分布
- 如果图中最高分 < beta（说明这张图可能是正常图），不输出任何框
- 如果图中最高分 >= beta，用 `max_score * 0.7` 作为该图的阈值
  - 这样每张图只保留分数在最高分70%以上的区域
  - 相当于per-image的top-30%过滤

**预期效果**：
- 正常图：最高分通常 < beta，直接过滤，FP大幅减少
- 异常图：最高分 >= beta，只保留高分区域，精准定位

### 改动4：NMS（继承V17的后处理）

- 对同一张图的多个bbox做NMS（IoU>0.5的重叠框只保留最高分）
- 每张图最多保留top-3框（不是top-1，避免漏检）

---

## V18 参数设计

```python
CONFIG = {
    # 继承V16
    "hidden_size": 30,
    "background_removal_method": "both",
    "merge_all_anomaly_patches": False,  # ← 关键改动：不再merge_all
    "beta_threshold": 0.1399,  # 从metadata读取（V16的p5值）
    "use_adaptive_beta": True,
    
    # V18新增：全图分数图
    "use_score_map": True,           # 使用全图patch分数图
    "score_map_smooth_sigma": 2.0,   # 高斯平滑sigma
    
    # V18新增：连通分量过滤
    "min_bbox_area": 3000,           # 最小bbox面积（GT的10%）
    "max_bbox_area": 80000,          # 最大bbox面积（GT的250%）
    
    # V18新增：per-image自适应阈值
    "use_per_image_threshold": True,
    "per_image_threshold_ratio": 0.7,  # max_score * 0.7
    
    # V18新增：NMS
    "nms_iou_threshold": 0.5,
    "top_k_per_image": 3,
    
    # 不再使用SAM（速度问题）
    "use_pixel_heatmap": False,
}
```

---

## V18 预期效果

### 理论分析

**FP减少的来源**：
1. 尺寸过滤：去除小噪声框（预计减少FP 30-40%）
2. per-image自适应阈值：正常图不输出框（预计减少FP 20-30%）
3. NMS：去除重叠框（预计减少FP 10-20%）
4. 全图分数图：更精准的bbox，减少误判（预计减少FP 10-20%）

**TP保持的原因**：
1. 全图分数图覆盖全图，不会漏检（与V16的patch合并一样）
2. 尺寸过滤的阈值基于GT统计，真实异常框不会被过滤
3. per-image阈值只在最高分<beta时才过滤，异常图不受影响

**预期指标**：
| 指标 | V16 | V18预期 |
|------|-----|---------|
| TP (IoU>0.5) | 10 | 8-12 |
| FP | 170 | 40-80 |
| Precision | 0.056 | 0.10-0.20 |
| Recall | 0.167 | 0.133-0.200 |
| F1 (IoU>0.5) | 0.083 | 0.12-0.18 |
| AUC | 0.821 | 0.83-0.87 |

---

## V18 与已验证失败方向的区别

| 失败方向 | 为什么失败 | V18的区别 |
|---------|-----------|----------|
| bbox膨胀后处理（V12） | 在已有框上膨胀，FP框也变大 | V18不膨胀框，而是改变bbox生成方式 |
| pixel-level heatmap（V17） | 依赖SAM候选区域，候选太少TP消失 | V18不依赖SAM，直接用全图分数图 |
| 降低SAM阈值（V17-3） | 候选暴增33倍，速度不可接受 | V18完全不用SAM |
| 单纯调beta（V7-V11） | 全局阈值无法区分正常图和异常图 | V18用per-image自适应阈值 |

---

## V18 实施计划

### 步骤1：创建V18推理脚本
- 基于V16的推理脚本（不是V17）
- 实现全图patch分数图生成
- 实现连通分量bbox生成
- 实现per-image自适应阈值
- 实现NMS后处理

### 步骤2：创建V18评估脚本
- 基于V17的评估脚本（已有多IoU阈值评估）
- 更新对比数据（V16, V17, V18）

### 步骤3：运行并分析
- 先用max_images_per_category=30测试
- 分析FP减少情况
- 如果FP减少但TP也减少，调整尺寸过滤阈值
- 如果效果好，运行全量数据

---

## 注意事项

1. **不改变Feature Bank**：继续使用V17的feature_bank_v17.pth（20张正常图）
2. **不改变ESN**：hidden_size=30，window_size=50，stride=5
3. **不改变beta校准**：继续从metadata读取adaptive_beta
4. **速度预期**：与V16相当（约5-6秒/张），因为不用SAM

---

## 参考文献

1. PatchCore (Roth et al., CVPR 2022)：patch-level anomaly map生成方式
2. Res-SAM (Zhou et al., Nature Communications 2025)：原始框架
3. WinCLIP (Jeong et al., CVPR 2023)：per-image自适应阈值思路
