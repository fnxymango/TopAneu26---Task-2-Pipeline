#!/usr/bin/env python
"""검출 강도 사다리의 손익 — 회수 vs 환각 (2026-08-26).

오라클 환산: 미검출 회수 +0.0092/병변 · 환각 제거 +0.0022/병변
→ 회수 1개는 환각 4.2개까지 감당한다. 단, 회수한 병변을 **틀리게 라벨하면** FP+FN 2중이라
   회수 자체가 손해가 될 수 있다. 여기서는 먼저 blob 수지만 센다.
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L

P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.TOPANEU_ROOT / "dataset" / "TopAneu" / "location_masks"
_, _, test_ids = L.case_ids_by_split()
LADDER = ["aneu_test_probavgf", "aneu_test_vote4f", "aneu_test_vote3f", "aneu_test_vote2f",
          "aneu_test_probavg", "aneu_test_vote4", "aneu_test_vote3", "aneu_test_vote2",
          "aneu_test_unionu80", "aneu_test_unionu40", "aneu_test_unionu20"]
res = {}
for d in LADDER:
    if not (P / d).exists():
        continue
    hit = fp = nblob = 0
    gtn = 0
    for cid in test_ids:
        f = P / d / f"{cid}.nii.gz"
        if not f.exists():
            continue
        pa = np.asarray(nib.load(str(f)).dataobj) > 0
        ga = np.asarray(nib.load(str(GT / f"{cid}.nii.gz")).dataobj)
        gl, gk = ndi.label(ga > 0); gtn += gk
        pl, pk = ndi.label(pa); nblob += pk
        # 병변 기준 적중: GT 성분마다 겹치는 pred 성분이 있으면 적중
        for i in range(1, gk + 1):
            if pa[gl == i].any(): hit += 1
        # 환각: 어떤 GT 성분과도 안 겹치는 pred 성분
        for j in range(1, pk + 1):
            if not (ga[pl == j] > 0).any(): fp += 1
    res[d] = (hit, fp, nblob, gtn)

base = res["aneu_test_probavgf"]
print(f"GT 병변 {base[3]} / test {len(test_ids)}케이스\n")
print(f"  {'검출본':<20}{'blob':>6}{'적중':>6}{'환각':>6}{'Δ적중':>7}{'Δ환각':>7}{'추정Δ':>9}{'판정':>8}")
for d, (h, f_, n, _g) in res.items():
    dh, df = h - base[0], f_ - base[1]
    est = 0.0092 * dh - 0.0022 * df
    v = "" if d == "aneu_test_probavgf" else ("이득" if est > 0 else "손해")
    print(f"  {d.replace('aneu_test_',''):<20}{n:>6}{h:>6}{f_:>6}{dh:>+7}{df:>+7}{est:>+9.4f}{v:>8}")
print("\n  추정Δ = 0.0092·Δ적중 − 0.0022·Δ환각  (covered_gt MCC, 오라클 기울기)")
print("  ※ 회수한 병변을 오답 라벨하면 FP+FN 2중이라 실제는 이보다 나쁘다 — e2e 로 확인 필요")
