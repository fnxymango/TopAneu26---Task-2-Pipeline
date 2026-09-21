#!/usr/bin/env python3
"""'M1 이 얼마나 짧은가' 가 early bifurcation 과 M1-M2 junction 을 가르는지 — 현재 피처에 없는 신호 검증."""
import json, os, numpy as np, nibabel as nib, collections
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; A=f"{R}/code/sblee/nnunet/analysis"
vlab=json.load(open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"]
vid={k:int(v) for k,v in vlab.items()}
rows=json.load(open(f"{A}/c10_feat_train_NEW.json"))
items=[(r["case"], r["gt_loc"][:2], "early" if "early bifurcation" in r["gt_loc"] else "junction")
       for r in rows if "M1 early bifurcation" in r["gt_loc"] or "M1-M2 junction" in r["gt_loc"]]
res=collections.defaultdict(list)
for case, side, kind in items:
    p=f"{R}/dataset/TopAneu/vessel_masks/{case}.nii.gz"
    if not os.path.exists(p): continue
    im=nib.load(p); v=np.asanyarray(im.dataobj); sp=np.array(im.header.get_zooms()[:3],float)
    m=(v==vid.get(side+"M1",-1))
    if m.sum()==0: res[kind].append((0.0,0.0)); continue
    idx=np.argwhere(m)*sp
    vol=float(m.sum()*np.prod(sp))
    ext=float(np.linalg.norm(idx.max(0)-idx.min(0)))       # M1 성분의 bbox 대각 = 대략적 길이
    res[kind].append((vol,ext))
def auc(a,b):
    from itertools import product
    a=np.asarray(a); b=np.asarray(b)
    return sum((x<y)+0.5*(x==y) for x,y in product(a,b))/(len(a)*len(b))
print("train GT 혈관마스크에서 잰 같은쪽 M1 크기 (병변 단위)\n")
print(f"{'집단':<12s} {'n':>3s} {'M1 부피 mm3 중앙':>18s} {'M1 길이 mm 중앙':>16s}")
for k in ("early","junction"):
    V=[x[0] for x in res[k]]; E=[x[1] for x in res[k]]
    print(f"{k:<12s} {len(V):3d} {np.median(V):18.0f} {np.median(E):16.1f}")
ve=[x[0] for x in res["early"]]; vj=[x[0] for x in res["junction"]]
ee=[x[1] for x in res["early"]]; ej=[x[1] for x in res["junction"]]
print(f"\n분리력 AUC   M1 부피 {auc(ve,vj):.3f}   M1 길이 {auc(ee,ej):.3f}   (0.5 = 구분 못함)")
print("참고 · 기존 최강 피처(혈관중첩 M2) AUC 0.943")
