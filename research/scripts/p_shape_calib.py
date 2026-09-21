#!/usr/bin/env python
"""S4 복셀 형태 보정 스크리닝 (2026-08-27) — Dice/VolSim/HD95 축.

공식 랭킹은 6지표 평균. MCC 는 intersection>0 만 보므로 blob 팽창/수축은 MCC 중립이고
(라벨 안 바뀜·클래스 존재 안 바뀜), TP 케이스의 Dice·VolSim·HD95 만 움직인다.
X5(P55ff) blob 과 GT 병변의 부피비·Dice 를 재고, 팽창/수축 반경별로 TP 병변 평균 Dice/VolSim 을 본다.
반경은 val 로 고르고 test 로 확인.
"""
import os, sys, numpy as np, nibabel as nib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage as ndi
import d9xx_lib as L
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.TOPANEU_ROOT / "dataset" / "TopAneu" / "location_masks"
_, val_ids, test_ids = L.case_ids_by_split()
RAD = [-1, 0, 1, 2]   # 음수 = 수축(복셀), 양수 = 팽창(복셀)

def dice(a, b):
    s = a.sum() + b.sum(); return 2.0 * (a & b).sum() / s if s else 0.0
def volsim(a, b):
    s = a.sum() + b.sum(); return 1 - abs(int(a.sum()) - int(b.sum())) / s if s else 0.0

for sp, ids in (("val", val_ids), ("test", test_ids)):
    ratios = []; res = {r: [] for r in RAD}
    for cid in ids:
        fb = P / f"aneu_{sp}_P55ff" / f"{cid}.nii.gz"
        a = np.asarray(nib.load(str(fb)).dataobj) > 0
        if not a.any(): continue
        g = np.asarray(nib.load(str(GT / f"{cid}.nii.gz")).dataobj) > 0
        pl, pk = ndi.label(a); gl, gk = ndi.label(g)
        for j in range(1, pk + 1):
            m = pl == j; hit = np.unique(gl[m]); hit = hit[hit > 0]
            if len(hit) == 0: continue                      # 환각 — 형태보정과 무관
            gm = gl == hit[0]
            ratios.append(m.sum() / gm.sum())
            # 국소 bbox 에서만 팽창/수축
            idx = np.argwhere(m | gm); lo = np.maximum(idx.min(0) - 4, 0); hi = idx.max(0) + 5
            sl = tuple(slice(l_, h_) for l_, h_ in zip(lo, hi))
            ms, gs = m[sl], gm[sl]
            for r in RAD:
                mm = ndi.binary_dilation(ms, iterations=r) if r > 0 else \
                     (ndi.binary_erosion(ms, iterations=-r) if r < 0 else ms)
                if not mm.any(): mm = ms                     # 수축으로 사라지면 원본 유지 (TP 보존)
                res[r].append((dice(mm, gs), volsim(mm, gs)))
    ratios = np.array(ratios)
    print(f"\n[{sp}] TP blob {len(ratios)}개 · 부피비 pred/GT 중앙 {np.median(ratios):.2f}  "
          f"(25% {np.percentile(ratios,25):.2f} / 75% {np.percentile(ratios,75):.2f})")
    print(f"  {'반경':>5}{'Dice':>8}{'VolSim':>8}")
    for r in RAD:
        v = np.array(res[r]); print(f"  {r:>+5d}{v[:,0].mean():>8.3f}{v[:,1].mean():>8.3f}")
print("\n완료")
