#!/usr/bin/env python
"""조각 배치 진단 (2026-08-26).

K=2 방출이 val 에서 2등 적중 예측 15% 대비 실측 4.4% 밖에 안 나왔다.
가설: 예측 blob 과 GT 병변이 부분적으로만 겹쳐서, blob **중심** 3복셀이 GT 밖으로 나간다.
→ 2등이 정답이어도 intersection=0 이라 TP 가 아니라 FP 가 된다.

중심 뭉치 배치 vs blob 전체에 흩뿌린 배치의 GT 적중률을 비교한다.
"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L

P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.TOPANEU_ROOT / "dataset" / "TopAneu" / "location_masks"
_, val_ids, test_ids = L.case_ids_by_split()

for split, ids, adir in (("val", val_ids, "aneu_val_probavgf"),
                         ("test", test_ids, "aneu_test_probavgf")):
    tot = ov = cen3 = cen10 = sc3 = sc10 = 0
    frac = []
    for cid in ids:
        f = P / adir / f"{cid}.nii.gz"
        if not f.exists(): continue
        pa = np.asarray(nib.load(str(f)).dataobj) > 0
        if not pa.any(): continue
        ga = np.asarray(nib.load(str(GT / f"{cid}.nii.gz")).dataobj) > 0
        pl, pk = ndi.label(pa)
        for j in range(1, pk + 1):
            idx = np.argwhere(pl == j); tot += 1
            g = ga[idx[:, 0], idx[:, 1], idx[:, 2]]
            if not g.any(): continue          # 환각 blob — 어차피 FP
            ov += 1; frac.append(g.mean())
            c = idx.mean(axis=0)
            order = np.argsort(((idx - c) ** 2).sum(axis=1))
            for n, acc in ((3, "c3"), (10, "c10")):
                hit = g[order[:min(n, len(idx))]].any()
                if n == 3 and hit: cen3 += 1
                if n == 10 and hit: cen10 += 1
            # 흩뿌리기: 전체를 균등 간격으로 n개 표집
            for n in (3, 10):
                sel = np.linspace(0, len(idx) - 1, min(n, len(idx))).astype(int)
                hit = g[sel].any()
                if n == 3 and hit: sc3 += 1
                if n == 10 and hit: sc10 += 1
    fr = np.array(frac)
    print(f"\n[{split}]  예측 blob {tot}  그중 GT 와 겹치는 것 {ov}")
    print(f"  blob 복셀 중 GT 안에 있는 비율:  중앙값 {np.median(fr):.3f}  평균 {fr.mean():.3f}  "
          f"25%분위 {np.percentile(fr,25):.3f}")
    print(f"  {'배치':<16}{'GT 적중':>9}{'적중률':>9}")
    for lab, v in (("중심 3복셀", cen3), ("중심 10복셀", cen10),
                   ("흩뿌림 3복셀", sc3), ("흩뿌림 10복셀", sc10)):
        print(f"  {lab:<16}{v:>9}{v/max(ov,1):>9.3f}")
