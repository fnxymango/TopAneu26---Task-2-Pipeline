#!/usr/bin/env python
"""수술적 회수 — 놓친 병변의 조건에 맞는 blob 만 붙인다 (2026-08-26).

미검출 24개 특성화: 지름 <3mm 가 14개, ICA(3.x) 계열이 13개. 전역 완화(vote2f)는
8개 회수에 환각 44개라 죽는다. **작고 ICA 근처인 blob 만** 골라 붙이면
회수는 대부분 살리고 환각은 버릴 수 있는지 본다.

현행 probavgf 는 그대로 두고, 느슨한 검출본의 '추가 blob' 만 조건부로 채택한다.
→ 잘 되는 병변은 건드리지 않는다.
"""
import os, sys, json, itertools, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L

P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.TOPANEU_ROOT / "dataset" / "TopAneu" / "location_masks"
_, _, test_ids = L.case_ids_by_split()
VN = L.vessel_dense_names()
ICA = {k for k, v in VN.items() if "ICA" in v}           # 4,6,35,36
print("ICA dense id:", sorted(ICA), flush=True)

SRC = ["aneu_test_vote3f", "aneu_test_vote2f", "aneu_test_unionu20"]
DIA = [3.0, 4.0, 5.0, 99.0]          # 추가 blob 최대 지름 (mm)
NEAR = [0.0, 5.0, 10.0, 1e9]         # ICA 까지 거리 상한 (mm); 1e9 = 부위 무제한

# 케이스별로 한 번만 읽고 모든 조합을 동시에 센다
tab = {(s, d, r): [0, 0] for s in SRC for d in DIA for r in NEAR}   # [Δ적중, Δ환각]
base_hit = base_fp = gtn = 0

for n, cid in enumerate(test_ids, 1):
    gi = nib.load(str(GT / f"{cid}.nii.gz"))
    ga = np.asarray(gi.dataobj); sp = gi.header.get_zooms()[:3]
    vmm = float(np.prod(sp))
    gl, gk = ndi.label(ga > 0); gtn += gk
    cur = np.asarray(nib.load(str(P / "aneu_test_probavgf" / f"{cid}.nii.gz")).dataobj) > 0
    cl, ck = ndi.label(cur)
    for i in range(1, gk + 1):
        if cur[gl == i].any(): base_hit += 1
    for j in range(1, ck + 1):
        if not (ga[cl == j] > 0).any(): base_fp += 1
    # ICA 거리장
    vf = P / "vespp_test" / f"{cid}.nii.gz"
    ves = np.asarray(nib.load(str(vf)).dataobj).astype(np.int16) if vf.exists() else None
    icam = np.isin(ves, list(ICA)) if ves is not None else None
    dist = ndi.distance_transform_edt(~icam, sampling=sp) if (icam is not None and icam.any()) \
        else np.full(ga.shape, 1e9, np.float32)
    for s in SRC:
        f = P / s / f"{cid}.nii.gz"
        if not f.exists(): continue
        pa = np.asarray(nib.load(str(f)).dataobj) > 0
        pl, pk = ndi.label(pa)
        for j in range(1, pk + 1):
            m = pl == j
            if cur[m].any(): continue                    # 이미 현행에 있는 blob
            nv = int(m.sum())
            dia = 2.0 * (3.0 * nv * vmm / (4 * np.pi)) ** (1 / 3)
            dmin = float(dist[m].min())
            gids = [int(v) for v in np.unique(gl[m]) if v > 0]
            # 이 blob 이 현행에서 못 잡은 GT 를 새로 덮는가
            newgt = [g for g in gids if not cur[gl == g].any()]
            for d, r in itertools.product(DIA, NEAR):
                if dia <= d and dmin <= r:
                    if newgt: tab[(s, d, r)][0] += len(set(newgt))
                    else:     tab[(s, d, r)][1] += 1
    if n % 20 == 0: print(f"  {n}/{len(test_ids)}", flush=True)

print(f"\n현행 probavgf: GT {gtn} · 적중 {base_hit} · 환각 {base_fp}\n")
print(f"  {'출처':<10}{'지름≤':>7}{'ICA거리≤':>9}{'Δ적중':>7}{'Δ환각':>7}{'추정Δ':>9}")
rows = []
for (s, d, r), (dh, df) in tab.items():
    est = 0.0092 * dh - 0.0022 * df
    rows.append((est, s, d, r, dh, df))
for est, s, d, r, dh, df in sorted(rows, reverse=True):
    ss = s.replace("aneu_test_", "")
    ds = "∞" if d > 90 else f"{d:.0f}mm"
    rs = "무제한" if r > 1e8 else f"{r:.0f}mm"
    print(f"  {ss:<10}{ds:>7}{rs:>9}{dh:>+7}{df:>+7}{est:>+9.4f}")
json.dump({f"{s}|{d}|{r}": v for (s, d, r), v in tab.items()},
          open(L.TOPANEU_ROOT / "experiments" / "surgical_recover.json", "w"), indent=1)
print("\n  추정Δ = 0.0092·Δ적중 − 0.0022·Δ환각 (상한; 오답 라벨이면 실제는 더 나쁨)")
