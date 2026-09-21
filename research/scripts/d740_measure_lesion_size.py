#!/usr/bin/env python3
"""D740(병변스케일 고정crop) 준비 — 훈련셋 병변 bbox 크기 분포를 재서
고정 window size W를 정한다.

방법: train split(417케이스 중 학습에 쓰는 부분)의 location_mask를 connected-component로
분리, 각 병변의 mm 단위 bbox 변(extent)을 축별로 측정. 이 분포의 최댓값 기준(잘리는 병변이
없어야 하므로 percentile이 아니라 max)에 여유마진을 더하고, nnU-Net 5단계 다운샘플링(2^5=32)의
배수로 올림해서 W를 확정한다.
"""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from d9xx_lib import DATA, case_ids_by_split

LOCDIR = DATA / "location_masks"
MARGIN_MM = 20.0        # 병변 주변 문맥(모혈관 형태 파악용) 여유마진, D730의 25mm과 같은 급
DOWNSAMPLE_MULT = 32     # nnU-Net 기본 5단계 풀링 = 2^5

train_ids, val_ids, test_ids = case_ids_by_split()
print(f"[d740] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

extents_mm = []   # (dz, dy, dx) per lesion, mm
extents_vox = []
n_lesions = 0
n_cases_with_lesion = 0
missing = []

for cid in train_ids:
    p = LOCDIR / f"{cid}.nii.gz"
    if not p.exists():
        missing.append(cid)
        continue
    nii = nib.load(str(p))
    arr = np.asanyarray(nii.dataobj)
    spacing = np.array(nii.header.get_zooms()[:3], dtype=float)
    lesions, n = ndimage.label(arr > 0, structure=np.ones((3, 3, 3)))
    if n == 0:
        continue
    n_cases_with_lesion += 1
    for lid in range(1, n + 1):
        idx = np.argwhere(lesions == lid)
        if idx.shape[0] == 0:
            continue
        lo = idx.min(0); hi = idx.max(0) + 1
        ext_vox = hi - lo
        ext_mm = ext_vox * spacing
        extents_vox.append(ext_vox)
        extents_mm.append(ext_mm)
        n_lesions += 1

if missing:
    print(f"[d740] location_mask 없는 케이스 {len(missing)}건 (스킵): {missing[:5]}...")

extents_mm = np.array(extents_mm)          # (n_lesions, 3)
extents_vox = np.array(extents_vox)
max_dim_mm = extents_mm.max(axis=1)        # 병변별 최장축(mm) — 등방aeous window 기준

print(f"\n[d740] train split 병변 개수: {n_lesions} (병변 있는 케이스 {n_cases_with_lesion}/{len(train_ids)})")
print(f"[d740] 축별 extent(mm) 통계 (z,y,x):")
for ax, name in enumerate(["z", "y", "x"]):
    v = extents_mm[:, ax]
    print(f"    {name}: mean={v.mean():.2f} median={np.median(v):.2f} p95={np.percentile(v,95):.2f} p99={np.percentile(v,99):.2f} max={v.max():.2f}")

print(f"\n[d740] 병변별 최장축(mm): mean={max_dim_mm.mean():.2f} median={np.median(max_dim_mm):.2f} "
      f"p95={np.percentile(max_dim_mm,95):.2f} p99={np.percentile(max_dim_mm,99):.2f} max={max_dim_mm.max():.2f}")

# max 기준은 outlier(거대동맥류 1건, topaneu_center5_mr_478, 55mm/70만voxel)에 너무 민감하고
# 등방 큐브로 강제하면 slice축(spacing 0.5mm, 원본이 149~248vox로 얕은 케이스 존재)이 원본
# 볼륨보다 커지는 경우까지 생겨 "작은 window" 취지가 무너짐. 그래서:
#   1) p99(33.08mm) 기준으로 잡음 — 거대동맥류(p99 초과, ~1%)는 애초에 "너무 작아서 안 잡히는"
#      문제의 대상이 아니라서(이미 크고 눈에 잘 띔) 이 방법의 표적이 아님 -> W 밖이면
#      D740 학습에서 제외하고 기존 방식(D720/D730)으로만 커버.
#   2) spacing이 이방성(in-plane 0.38mm, slice 0.5mm)이라 등방 큐브 대신 축별 window로.
p99_mm = np.percentile(max_dim_mm, 99)
n_excluded = int((max_dim_mm > p99_mm).sum())
raw_W_mm = p99_mm + 2 * MARGIN_MM
print(f"\n[d740] raw W(mm) = p99_lesion_extent({p99_mm:.2f}) + 2*margin({MARGIN_MM}) = {raw_W_mm:.2f} mm")
print(f"[d740] p99 초과(거대병변, W에서 제외 대상): {n_excluded}/{n_lesions}건")

all_spacings = []
for cid in train_ids:
    p = LOCDIR / f"{cid}.nii.gz"
    if p.exists():
        all_spacings.append(nib.load(str(p)).header.get_zooms()[:3])
med_spacing = np.median(np.array(all_spacings), axis=0)
print(f"[d740] train 대표(median) spacing: {med_spacing} mm/vox")

raw_W_vox_axis = raw_W_mm / med_spacing              # 축별(이방성) voxel 크기
W_vox_axis = np.ceil(raw_W_vox_axis / DOWNSAMPLE_MULT).astype(int) * DOWNSAMPLE_MULT
print(f"[d740] raw W(vox, 축별) = {raw_W_vox_axis}")
print(f"[d740] W 확정(32배수 올림, 축별) = {tuple(int(x) for x in W_vox_axis)} voxel")
W_vox = int(W_vox_axis.max())  # 참고용 스칼라(등방 비교용), 실제 사용은 축별 값

out = {
    "n_lesions_train": int(n_lesions),
    "n_cases_with_lesion_train": int(n_cases_with_lesion),
    "extent_mm_stats": {
        ax: {"mean": float(extents_mm[:, i].mean()), "median": float(np.median(extents_mm[:, i])),
             "p95": float(np.percentile(extents_mm[:, i], 95)), "p99": float(np.percentile(extents_mm[:, i], 99)),
             "max": float(extents_mm[:, i].max())}
        for i, ax in enumerate(["z", "y", "x"])
    },
    "max_dim_mm_stats": {
        "mean": float(max_dim_mm.mean()), "median": float(np.median(max_dim_mm)),
        "p95": float(np.percentile(max_dim_mm, 95)), "p99": float(np.percentile(max_dim_mm, 99)),
        "max": float(max_dim_mm.max()),
    },
    "margin_mm": MARGIN_MM,
    "p99_dim_mm": float(p99_mm),
    "n_lesions_excluded_gt_p99": n_excluded,
    "raw_W_mm": float(raw_W_mm),
    "median_spacing_mm": [float(x) for x in med_spacing],
    "raw_W_vox_axis": [float(x) for x in raw_W_vox_axis],
    "downsample_mult": DOWNSAMPLE_MULT,
    "W_vox_axis_final": [int(x) for x in W_vox_axis],
    "outlier_case_excluded": "topaneu_center5_mr_478 (55mm/70만voxel 거대동맥류, W 기준 산정에서 제외)",
}
outp = Path(__file__).resolve().parents[1] / "analysis" / "d740_window_size.json"
outp.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(outp, "w"), indent=2, ensure_ascii=False)
print(f"\n[d740] 저장: {outp}")
