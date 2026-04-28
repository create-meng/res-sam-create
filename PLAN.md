# Res-SAM 论文复现计划书

> 说明：本文件主要保留早期规划痕迹，已经不是当前主线执行说明。
> 当前实际主线以 `experiments/01_build_feature_bank_v6.py`、
> `02_inference_auto_v6.py`、`03_evaluate_and_visualize_v6.py`、
> `04_clustering_v6.py` 以及 `experiments/run_all.py --step 1-4` 为准。
> 文中涉及 click-guided、多数据模式、旧编号 Step 5 等内容，均视为历史存档，不再代表当前实现状态。

## 一、论文概述

**标题**: Res-SAM: Reservoir-enhanced Segment Anything Model for Underground Anomaly Detection in GPR B-scans

**核心创新**: 将 Segment Anything Model (SAM) 与双向上回声状态网络 (2D-ESN) 结合，实现 GPR B-scan 地下异常检测，**无需深度学习训练**。

**论文 DOI**: https://doi.org/10.1038/s41467-025-67382-4

---

## 二、方法论详解

### 2.1 整体框架（两阶段）

```
┌─────────────────────────────────────────────────────────────┐
│                    Phase 1: Feature Collection              │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Normal   │ -> │ Sliding  │ -> │ 2D-ESN   │ -> Feature   │
│  │ B-scans  │    │ Window   │    │ Fitting  │    Bank M    │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Phase 2: Anomaly Detection                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐│
│  │ Test     │ -> │ SAM      │ -> │ Patch    │ -> │ Feature││
│  │ B-scan   │    │ Regions  │    │ Fitting  │    | Compare││
│  └──────────┘    └──────────┘    └──────────┘    └────────┘│
│                                         ↓                    │
│                              ┌──────────────────┐           │
│                              │ Anomaly Heatmap  │           │
│                              │ + Bounding Box   │           │
│                              └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 2D-ESN 数学原理

**状态更新方程**:
```
h_{x,y} = tanh(W_x · h_{x-1,y} + W_y · h_{x,y-1} + W_in · u_{x,y})
```

**读出方程**:
```
v_{x,y} = W_out · [h_{x-1,y}; h_{x,y-1}] + b
```

**特点**:
- 双向传播：水平方向 + 垂直方向
- 捕捉 B-scan 的时空动态特征
- 仅需 ridge regression 求解 W_out，无需梯度训练

### 2.3 异常评分

**特征距离**:
```
s(f*) = min_{f ∈ M} ||f* - f||_2
```

**二分类器**:
```
H(f*) = 0  if s(f*) > β  (异常)
H(f*) = 1  if s(f*) ≤ β  (正常)
```

其中 β 是异常阈值。

### 2.4 两种运行模式

| 模式 | 输入 | 流程 |
|------|------|------|
| **Click-guided** | 用户点击（正/负） | 点击 → SAM 分割 → 候选区域 → 特征比对 |
| **Fully Automatic** | 无需交互 | SAM 全图分割 → 粗筛 → 细化 → 输出框 |

---

## 三、数据集准备

### 3.1 使用的数据集

**Mendeley Open-source Dataset** (本地已有):
- 路径: `../Intelligent recognition.../GPR_data/`
- 论文对应: Table 1/2 中的 "Open-source" (285 B-scans)

### 3.2 数据划分策略

```
┌────────────────────────────────────────────────────┐
│                    数据划分                         │
├────────────────────────────────────────────────────┤
│  intact/ (75张)                                    │
│    ├── init_20张 → Feature Bank 构建               │
│    └── 剩余55张 → 测试集 Normal 样本               │
│                                                    │
│  augmented_cavities/ (553张) → 测试集 Anomaly      │
│  augmented_utilities/ (786张) → 测试集 Anomaly    │
└────────────────────────────────────────────────────┘
```

### 3.3 标注解析

**VOC XML → bbox**:
```python
import xml.etree.ElementTree as ET

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find('object')
    bbox = obj.find('bndbox')
    return {
        'xmin': int(bbox.find('xmin').text),
        'ymin': int(bbox.find('ymin').text),
        'xmax': int(bbox.find('xmax').text),
        'ymax': int(bbox.find('ymax').text)
    }
```

---

## 四、复现实验步骤

### Step 1: Feature Bank 构建

**脚本**: `experiments/_feature_bank.py`

**输入**: `intact/` 中随机 20 张图片

**流程**:
1. 读取 20 张 normal B-scan
2. 每张图做滑窗提取 patch (50×50, stride=5)
3. 每个 patch 用 2D-ESN 拟合，得到动态特征 f
4. 所有特征存入 `outputs/feature_banks/features.pth`

**代码框架**:
```python
from PatchRes import PatchRes
from PatchRes.functions import random_select_images_in_one_folder

# 参数
window_size = 50
stride = 5
hidden_size = 30

# 初始化
TR = PatchRes(
    hidden_size=hidden_size,
    stride=stride,
    window_size=[window_size, window_size]
)

# 加载 normal 数据
normal_data = random_select_images_in_one_folder(
    data_folder='../Intelligent.../GPR_data/intact/',
    num=20,
    rand_select=True
)

# 构建 feature bank
features_list = []
for i in range(normal_data.shape[0]):
    features_list.append(TR.fit(normal_data[i].unsqueeze(0)))

# 保存
features = torch.cat(features_list)
torch.save(features, 'outputs/feature_banks/features.pth')
```

---

### Step 2: Fully Automatic 推理

**脚本**: `experiments/_inference_auto.py`

**输入**: 测试集图片 + Feature Bank

**输出**: 每张图的预测 bbox + 异常分数

**流程**:
1. 加载 Feature Bank
2. 对每张测试图:
   - SAM 自动分割得到候选区域
   - 每个候选区域内 patch 特征比对
   - 异常 patch 合并 → 最终 bbox
   - 记录最大异常分数作为该图置信度

**代码框架**:
```python
from PatchRes import PatchRes
from sam import SamPredictor

# 加载 feature bank
TR = PatchRes(features='outputs/feature_banks/features.pth')
TR.fit(0)  # 加载预置特征

# 加载 SAM
sam = SamPredictor()

# 遍历测试集
results = []
for img_path in test_images:
    img = load_image(img_path)
    
    # SAM 自动分割
    masks = sam.generate(img)
    
    # 对每个 mask 区域做特征比对
    for mask in masks:
        # 提取 mask 区域内的 patch
        # 2D-ESN 拟合
        # 与 feature bank 比对
        # 判断是否异常
        pass
    
    # 合并异常区域，输出 bbox
    pred_bbox = merge_anomaly_patches(...)
    anomaly_score = compute_anomaly_score(...)
    
    results.append({
        'image': img_path,
        'pred_bbox': pred_bbox,
        'score': anomaly_score
    })
```

---

### Step 3: 评估指标计算

**脚本**: `experiments/04_evaluate.py`

**指标**:

| 指标 | 公式 | 说明 |
|------|------|------|
| IoU | \|\|A∩B\|\| / \|\|A∪B\|\| | 预测框与真值框重叠度 |
| Precision | TP / (TP+FP) | 检测正确率 |
| Recall | TP / (TP+FN) | 检测召回率 |
| F1 | 2×P×R / (P+R) | 综合指标 |
| AUC | ROC 曲线下面积 | 异常分数区分能力 |

**判定规则**:
- IoU > 0.5 → TP (正确检测)
- IoU ≤ 0.5 → FP (误检)
- 有 GT 但无预测 → FN (漏检)

**代码框架**:
```python
def compute_iou(box1, box2):
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])
    
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (box1['xmax']-box1['xmin'])*(box1['ymax']-box1['ymin']) + \
            (box2['xmax']-box2['xmin'])*(box2['ymax']-box2['ymin']) - inter
    
    return inter / union if union > 0 else 0

def evaluate(predictions, ground_truths):
    tp, fp, fn = 0, 0, 0
    
    for img_id in ground_truths:
        gt = ground_truths[img_id]
        pred = predictions.get(img_id, None)
        
        if pred is None:
            fn += 1
        else:
            iou = compute_iou(gt, pred)
            if iou > 0.5:
                tp += 1
            else:
                fp += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

---

### Step 4: Click-guided 模式

**脚本**: `experiments/03_inference_click.py`

**关键**: 用 GT bbox 模拟用户点击

**点击生成策略**:
```python
def generate_clicks(gt_bbox, pos_count, neg_count, img_size):
    # 正点击：在 GT bbox 内均匀采样
    pos_points = sample_points_in_bbox(gt_bbox, pos_count)
    
    # 负点击：在 bbox 外随机采样
    neg_points = sample_points_outside_bbox(gt_bbox, neg_count, img_size)
    
    return pos_points, neg_points
```

**实验配置** (论文 Table 1):
| 配置 | 正点击 | 负点击 |
|------|--------|--------|
| 5/5 | 5 | 5 |
| 5/3 | 5 | 3 |
| 3/1 | 3 | 1 |

---

### Step 5: 异常聚类 (Table 3)

**脚本**: `experiments/05_clustering.py`

**流程**:
1. 对检测到的异常区域重新用 2D-ESN 拟合
2. 提取类别特征
3. 用 K-Means / Agglomerative / FCM 聚类
4. 计算 Acc / ARI / NMI

---

## 五、复现目标

### Table 2: Fully Automatic 模式

| 方法 | Precision | Recall | F1 | AUC |
|------|-----------|--------|-----|-----|
| Res-SAM (论文) | - | - | - | - |
| 我们的复现 | ? | ? | ? | ? |

### Table 1: Click-guided 模式

| Prompt | AUC | F1 |
|--------|-----|-----|
| 5/5 | - | - |
| 5/3 | - | - |
| 3/1 | - | - |

### Table 3: 异常聚类

| 方法 | Accuracy | ARI | NMI |
|------|----------|-----|-----|
| K-Means | - | - | - |
| AC | - | - | - |
| FCM | - | - | - |

---

## 六、执行顺序

```
1. 创建目录结构
   └── experiments/, outputs/

2. Step 1: Feature Bank 构建
   └── _feature_bank.py

3. Step 2: Fully Automatic 推理
   └── _inference_auto.py

4. Step 3: 评估指标
   └── 04_evaluate.py

5. Step 4: Click-guided 推理
   └── 03_inference_click.py

6. Step 5: 异常聚类
   └── 05_clustering.py

7. 生成最终报告
   └── 汇总 Table 1/2/3 结果
```

---

## 七、注意事项

1. **环境一致性**: Feature Bank 必须用与测试集同环境的 normal 数据构建
2. **参数对齐**: 使用论文参数 (window=50, stride=5, hidden=30, β=0.1)
3. **随机种子**: 固定随机种子保证可复现
4. **数据预处理**: 标准化 (mean=0, std=1) 是关键步骤
5. **CPU 环境**: 当前环境为 CPU-only，注意内存管理

---

## 八、预期产出

1. **可复现的实验脚本**: 5 个 Python 脚本
2. **Feature Bank**: 基于本地数据构建的特征库
3. **预测结果**: 每张测试图的 bbox 和异常分数
4. **评估报告**: Table 1/2/3 的复现结果
5. **对比分析**: 与论文结果的差异分析
