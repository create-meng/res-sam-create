"""
验证：如果推理时保持 224×224（不 resize）+ merge_all，
理论上 IoU 能达到多少？

用 V9 的 pred_bboxes_resized（369 坐标系）缩回 224，
再模拟 merge_all（把同一张图的所有 pred 框合并成一个最小外接矩形），
计算 IoU。
"""
import json, os, sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_PATH = os.path.join(BASE_DIR, "已归档", "v9_snapshot_20260412_postrun",
                         "outputs", "predictions_v9", "auto_predictions_v9.json")

with open(PRED_PATH, "r", encoding="utf-8") as f:
    obj = json.load(f)
results = obj.get("results", obj) if isinstance(obj, dict) else obj

def iou(a, b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    if x2<=x1 or y2<=y1: return 0.0
    inter=(x2-x1)*(y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter) if (ua+ub-inter)>0 else 0.0

def match(preds, gts, thresh=0.5):
    matched=[False]*len(gts); tp=fp=0
    for p in preds:
        best,bi=0.0,-1
        for gi,g in enumerate(gts):
            if matched[gi]: continue
            v=iou(p,g)
            if v>best: best,bi=v,gi
        if bi>=0 and best>thresh: matched[bi]=True; tp+=1
        else: fp+=1
    fn=sum(1 for m in matched if not m)
    return tp,fp,fn

def scale(box, sx, sy):
    return [int(box[0]*sx), int(box[1]*sy), int(box[2]*sx), int(box[3]*sy)]

def merge_all(boxes):
    if not boxes: return None
    x1=min(b[0] for b in boxes); y1=min(b[1] for b in boxes)
    x2=max(b[2] for b in boxes); y2=max(b[3] for b in boxes)
    return [x1,y1,x2,y2]

# 收集有GT的记录
records = []
for cat, cat_results in results.items():
    if cat == "normal_auc": continue
    for r in cat_results:
        if r.get("exclude_from_det_metrics", False): continue
        gt = r.get("gt_bboxes", [])
        pred_resized = r.get("pred_bboxes_resized", [])  # 369坐标
        gt_w = r.get("gt_width", 224); gt_h = r.get("gt_height", 224)
        if not gt: continue
        records.append({"pred_369": pred_resized, "gt_224": gt,
                        "gt_w": gt_w, "gt_h": gt_h, "cat": cat})

print(f"总记录数: {len(records)}")
print("="*65)
print(f"{'场景':<35} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6}")
print("="*65)

def report(label, all_preds_gts, thresh=0.5):
    ttp=tfp=tfn=0
    for preds,gts in all_preds_gts:
        tp,fp,fn=match(preds,gts,thresh)
        ttp+=tp; tfp+=fp; tfn+=fn
    p=ttp/(ttp+tfp) if (ttp+tfp) else 0
    r=ttp/(ttp+tfn) if (ttp+tfn) else 0
    f=2*p*r/(p+r) if (p+r) else 0
    print(f"{label:<35} {ttp:>4} {tfp:>4} {tfn:>4} {p:>6.3f} {r:>6.3f} {f:>6.3f}")

# 场景1：当前做法（369→224缩放，不merge）
s1 = []
for r in records:
    sx=r["gt_w"]/369.0; sy=r["gt_h"]/369.0
    preds=[scale(p,sx,sy) for p in r["pred_369"]]
    s1.append((preds, r["gt_224"]))
report("1. 当前(369→224, 不merge)", s1)

# 场景2：369→224缩放 + merge_all（把同图所有pred合并）
s2 = []
for r in records:
    sx=r["gt_w"]/369.0; sy=r["gt_h"]/369.0
    preds=[scale(p,sx,sy) for p in r["pred_369"]]
    merged=merge_all(preds)
    s2.append(([merged] if merged else [], r["gt_224"]))
report("2. 当前(369→224) + merge_all", s2)

# 场景3：保持224（pred_resized直接缩回224），不merge
# pred_resized 是 369 坐标，缩回 224 = 场景1，所以场景3 = 场景1
# 但如果推理时直接在 224 上跑，window_size=50 在 224 坐标系下
# pred 框会更大（相对GT），这里用 pred_resized 乘以 (224/369) 模拟
s3 = []
for r in records:
    sx=r["gt_w"]/369.0; sy=r["gt_h"]/369.0
    preds=[scale(p,sx,sy) for p in r["pred_369"]]
    s3.append((preds, r["gt_224"]))
# 场景3 = 场景1（因为 pred_resized 已经是 369 坐标，缩回 224 就是当前做法）

# 场景4：模拟 224 推理 + merge_all
# 关键：在 224 坐标系下，window_size=50 的 patch 中心范围更大
# 假设 pred 框中心位置不变，但 window_size 在 224 坐标系下是 50px（不是 50*224/369≈30px）
# 即：把 pred_resized 的中心保留，但把框大小改为 50×50（224坐标系）
s4 = []
WIN = 50
for r in records:
    sx=r["gt_w"]/369.0; sy=r["gt_h"]/369.0
    preds_224_native = []
    for p in r["pred_369"]:
        # 中心点缩回 224
        cx = int(((p[0]+p[2])/2) * sx)
        cy = int(((p[1]+p[3])/2) * sy)
        # 在 224 坐标系下用原始 window_size=50
        x1=max(0,cx-WIN//2); y1=max(0,cy-WIN//2)
        x2=min(r["gt_w"],cx+WIN//2); y2=min(r["gt_h"],cy+WIN//2)
        preds_224_native.append([x1,y1,x2,y2])
    merged=merge_all(preds_224_native)
    s4.append(([merged] if merged else [], r["gt_224"]))
report("4. 224原生window + merge_all", s4)

# 场景5：场景4 但不 merge（每个 patch 独立）
s5 = []
for r in records:
    sx=r["gt_w"]/369.0; sy=r["gt_h"]/369.0
    preds_224_native = []
    for p in r["pred_369"]:
        cx=int(((p[0]+p[2])/2)*sx); cy=int(((p[1]+p[3])/2)*sy)
        x1=max(0,cx-WIN//2); y1=max(0,cy-WIN//2)
        x2=min(r["gt_w"],cx+WIN//2); y2=min(r["gt_h"],cy+WIN//2)
        preds_224_native.append([x1,y1,x2,y2])
    s5.append((preds_224_native, r["gt_224"]))
report("5. 224原生window, 不merge", s5)

# 场景6：场景4 用 IoU>0.1 阈值
print()
print("--- IoU>0.1 阈值 ---")
report("4. 224原生window + merge_all (IoU>0.1)", s4, thresh=0.1)
report("2. 当前 + merge_all (IoU>0.1)", s2, thresh=0.1)
report("1. 当前, 不merge (IoU>0.1)", s1, thresh=0.1)

# 额外：分析场景4中 merge 后的框尺寸
print("\n--- 场景4 merge后框尺寸分析 ---")
merged_sizes = []
for r in records:
    sx=r["gt_w"]/369.0; sy=r["gt_h"]/369.0
    preds_224_native = []
    for p in r["pred_369"]:
        cx=int(((p[0]+p[2])/2)*sx); cy=int(((p[1]+p[3])/2)*sy)
        x1=max(0,cx-WIN//2); y1=max(0,cy-WIN//2)
        x2=min(r["gt_w"],cx+WIN//2); y2=min(r["gt_h"],cy+WIN//2)
        preds_224_native.append([x1,y1,x2,y2])
    m=merge_all(preds_224_native)
    if m: merged_sizes.append((m[2]-m[0], m[3]-m[1]))

if merged_sizes:
    ws=[s[0] for s in merged_sizes]; hs=[s[1] for s in merged_sizes]
    print(f"merge后框宽: mean={np.mean(ws):.1f} median={np.median(ws):.1f}")
    print(f"merge后框高: mean={np.mean(hs):.1f} median={np.median(hs):.1f}")
    print(f"GT框宽: mean={np.mean([g[2]-g[0] for r in records for g in r['gt_224']]):.1f}")
    print(f"GT框高: mean={np.mean([g[3]-g[1] for r in records for g in r['gt_224']]):.1f}")
