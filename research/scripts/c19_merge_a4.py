"""C19 — A4(lesionscale crop) 마스크로 검출 경계 재작성.

검출기가 잡은 **위치는 그대로 두고**, 각 성분의 경계만 A4 예측으로 교체한다.
A4는 병변 중심 고해상도 crop으로 학습돼 경계가 더 정확할 것으로 기대.
목표 지표는 DICE/VolSim/HD95(랭킹 3/6). MCC는 클래스 판정이 안 바뀌므로 중립이어야 하며,
**MCC를 깎지 않는지 확인**하는 것이 이 실험의 안전 조건이다.

병합 규칙: 검출 성분마다 A4 마스크와 겹치는 부분이 있으면 그 성분을 A4 형태로 교체,
없으면 원래 성분을 유지(위치 정보를 잃지 않기 위해).
"""
import argparse
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L

ST = np.ones((3,3,3), dtype=bool)

ap = argparse.ArgumentParser()
ap.add_argument("--det-dir", required=True); ap.add_argument("--a4-dir", required=True)
ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()

ids = L.case_ids_by_split()[{"train":0,"val":1,"test":2}[a.split]]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
n_rep = n_keep = 0
for i, cid in enumerate(ids, 1):
    dp, a4p = Path(a.det_dir)/f"{cid}.nii.gz", Path(a.a4_dir)/f"{cid}.nii.gz"
    if not dp.exists(): continue
    di = nib.load(dp); det = np.asanyarray(di.dataobj) > 0
    res = np.zeros(det.shape, dtype=np.int16)
    if a4p.exists():
        a4 = np.asanyarray(nib.load(a4p).dataobj) > 0
        lab, n = ndi.label(det, structure=ST)
        a4lab, _ = ndi.label(a4, structure=ST)
        for l in range(1, n+1):
            sel = lab == l
            hit = set(int(x) for x in np.unique(a4lab[sel]) if x > 0)
            if hit:                                  # A4 형태로 교체
                for h in hit: res[a4lab == h] = 1
                n_rep += 1
            else:                                    # 위치 유지
                res[sel] = 1; n_keep += 1
    else:
        res[det] = 1
    nib.save(nib.Nifti1Image(res, di.affine, di.header), out/f"{cid}.nii.gz")
    if i % 20 == 0 or i == len(ids): print(f"  {i}/{len(ids)}", flush=True)
print(f"[c19] 경계 교체 {n_rep}성분 · 원형 유지 {n_keep}성분 -> {out}")
