#!/usr/bin/env python
"""h4_post.py --tag <t> --split <sp> — 검출 결과를 라벨2 이진화 + c7 필터까지.

pipeline_case.py 3·7 단계와 같은 규약: 라벨 2(aneurysm)만 이진화 → det_filter(min_vox 5, max_dist 1.0mm).
혈관은 기존 vespp_{split} 을 쓴다(검출기만 바꾸는 비교이므로 혈관은 고정).
출력: _c1_realpred/aneu_{split}_{tag}ff  (c5 의 --aneurysm-pred-dir 규약)
"""
import argparse, json, os, sys
import numpy as np, nibabel as nib
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
from det_filter import filter_case
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--min-vox", type=int, default=5); ap.add_argument("--max-dist", type=float, default=1.0)
a = ap.parse_args()
SRC = f"{R}/experiments/H4_folds/pred_{a.tag}_{a.split}"
VES = f"{R}/experiments/_c1_realpred/vespp_{a.split}"
OUT = f"{R}/experiments/_c1_realpred/aneu_{a.split}_{a.tag}ff"
os.makedirs(OUT, exist_ok=True)
ids = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"][a.split]
n = 0
for c in ids:
    sp_, vp, op = f"{SRC}/{c}.nii.gz", f"{VES}/{c}.nii.gz", f"{OUT}/{c}.nii.gz"
    if os.path.exists(op):
        n += 1; continue
    if not (os.path.exists(sp_) and os.path.exists(vp)):
        print(f"  [skip] {c}"); continue
    di = nib.load(sp_)
    aneu = (np.asanyarray(di.dataobj) == 2)
    vi = nib.load(vp)
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    m = filter_case(aneu, ves, spacing, a.min_vox, a.max_dist)
    nib.save(nib.Nifti1Image(m, vi.affine, vi.header), op)
    n += 1
print(f"[h4post] {a.tag} {a.split} · {n}/{len(ids)} → {OUT}", flush=True)
