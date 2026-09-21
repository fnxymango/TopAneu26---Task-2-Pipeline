#!/usr/bin/env python3
"""M1 계열 병변이 M1/M2 경계에 '걸쳐' 있는지 정량화 — 주석 불일치인가 knife-edge 인가."""
import json, os, numpy as np, nibabel as nib, collections
from scipy import ndimage
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; A=f"{R}/code/sblee/nnunet/analysis"
sp=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
lc=sp["location_classes"]; vlab=json.load(open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"]
vid={k:int(v) for k,v in vlab.items()}
diag=json.load(open(f"{R}/experiments/D1_newdata/adj_diag_test.json"))
m1cases=sorted({r["case"] for r in diag if "5.2" in r["name"] or "5.3" in r["name"]})
print(f"M1/M2 계열 병변을 가진 test 케이스 {len(m1cases)}개\n")
print(f"{'케이스':<26s} {'GT 라벨':<28s} {'M1복셀':>7s} {'M2복셀':>7s} {'M2비율':>7s}  판정")
edge=0; clear_ok=0; clear_bad=0
for case in m1cases:
    loc=np.asanyarray(nib.load(f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz").dataobj)
    gv=np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{case}.nii.gz").dataobj)
    for c in [int(x) for x in np.unique(loc) if x]:
        nm=lc[str(c)]
        if "5.2" not in nm and "5.3" not in nm: continue
        side=nm[:2]
        lab,n=ndimage.label(loc==c)
        for i in range(1,n+1):
            m=ndimage.binary_dilation(lab==i, iterations=3)
            a=int((m & (gv==vid.get(side+"M1",-1))).sum()); b=int((m & (gv==vid.get(side+"M2",-1))).sum())
            tot=a+b; frac=b/tot if tot else float("nan")
            exp_m2 = "5.3" in nm          # 관례: junction→M2, early→M1
            if tot==0: verdict="혈관 없음"
            elif 0.2<=frac<=0.8: verdict="경계에 걸침 (knife-edge)"; edge+=1
            elif (frac>0.8)==exp_m2: verdict="관례 일치"; clear_ok+=1
            else: verdict="★관례 반대 (주석 불일치)"; clear_bad+=1
            print(f"{case:<26s} {nm:<28s} {a:7d} {b:7d} {frac:7.2f}  {verdict}")
tot=edge+clear_ok+clear_bad
print(f"\n합계 {tot}건 · 관례 일치 {clear_ok} ({clear_ok/tot:.0%}) · 경계에 걸침 {edge} ({edge/tot:.0%}) · 관례 반대 {clear_bad} ({clear_bad/tot:.0%})")
