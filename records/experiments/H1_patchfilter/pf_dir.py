#!/usr/bin/env python
"""통합 컨테이너의 패치 CNN 환각필터를 예측 디렉터리 전체에 건다.

컨테이너의 run_patch_filter.py 를 케이스 하나 -> 디렉터리 전체로 바꾼 것뿐이다.
필터 자체(src/patch_filter.py · src/features.py)는 컨테이너에서 **바이트 그대로** 복사했고,
호출 순서·정규화·임계도 run_patch_filter.py 와 동일하다:

    ch0 = 뇌마스크 intensity z-score (혈관 EDT 정규화가 아니다)
    혈관 = stage-1 V5 후처리판(vespp) — 컨테이너가 TOPANEU_EXPORT_VESSEL 로 내보내는 것과 같다
    임계 = patchclf/meta.json 의 값 (추론 시점에 다시 고르지 않는다)

**SimpleITK 로 읽는다.** 컨테이너와 축 순서를 맞추기 위해서다 — sitk 는 (z,y,x), nibabel 은
(x,y,z) 로 주는데 3D CNN 은 축 치환에 불변이 아니라 nibabel 로 읽으면 학습 때와 다른 패치가
들어간다. spacing 도 sitk 규약대로 GetSpacing()[::-1] 로 뒤집어 같은 순서로 맞춘다.

모달리티는 dataset_split.json 의 modality 필드에서 읽는다(파일명 추측 금지).

사용: pf_dir.py --pred <52클래스 dir> --vessel <vespp dir> --image <원본 dir> --split test|val --out <dir>
"""
import argparse, json, os, sys, time
from pathlib import Path

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from src import patch_filter as PF, features as FE

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--vessel", required=True)
ap.add_argument("--image", required=True)
ap.add_argument("--split", required=True, choices=["train", "val", "test"])
ap.add_argument("--patchclf", default=str(APP / "patchclf"))
ap.add_argument("--out", required=True)
ap.add_argument("--threshold", default="")
ap.add_argument("--device", default="auto")
ap.add_argument("--report", default="")
ap.add_argument("--quiet", action="store_true")
a = ap.parse_args()

net, thr = PF.load(Path(a.patchclf))
if a.threshold:
    thr = float(a.threshold)

dev = a.device
if dev == "auto":
    dev = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            dev = "cuda"
    except ImportError:
        pass

MODAL = {c["case_id"]: c["modality"]
         for c in json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["cases"][a.split]}

os.makedirs(a.out, exist_ok=True)
cases = sorted(p.name[:-7] for p in Path(a.pred).glob("*.nii.gz"))
rows, t0 = [], time.time()
for i, cid in enumerate(cases, 1):
    ti = time.time()
    si = sitk.ReadImage(f"{a.pred}/{cid}.nii.gz")
    seg = sitk.GetArrayFromImage(si).astype(np.uint8)
    out_path = f"{a.out}/{cid}.nii.gz"
    if not seg.any():
        sitk.WriteImage(si, out_path, True)
        rows.append({"case": cid, "n_blob": 0, "n_drop": 0, "sec": round(time.time() - ti, 1)})
        continue

    ves = sitk.GetArrayFromImage(sitk.ReadImage(f"{a.vessel}/{cid}.nii.gz")).astype(np.uint8)
    ip = f"{a.image}/{cid}_0000.nii.gz"
    if not os.path.exists(ip):
        ip = f"{a.image}/{cid}.nii.gz"
    img = sitk.GetArrayFromImage(sitk.ReadImage(ip)).astype(np.float32)
    sp = list(si.GetSpacing())[::-1]
    is_ct = MODAL[cid].upper().startswith("CT")

    bm, _ = FE.brain_mask_intensity(img, sp, is_ct)
    zm = np.array(img.shape) / np.array(bm.shape)
    bmask = ndi.zoom(bm.astype(np.uint8), zm, order=0).astype(bool)
    if bmask.shape != img.shape:                      # zoom can land a voxel short
        b2 = np.zeros(img.shape, bool)
        sl = tuple(slice(0, min(x, y)) for x, y in zip(bmask.shape, img.shape))
        b2[sl] = bmask[sl]
        bmask = b2
    img_norm = FE.zscore_in_mask(img, bmask)

    seg2, nb, nd, sc = PF.filter_blobs(seg, net, thr, img_norm, ves, sp, (0, 0, 0), device=dev)
    o = sitk.GetImageFromArray(seg2.astype(np.uint8))
    o.CopyInformation(si)
    sitk.WriteImage(o, out_path, True)
    rows.append({"case": cid, "modality": MODAL[cid], "n_blob": int(nb), "n_drop": int(nd),
                 "scores": [round(float(x), 6) for x in sc], "sec": round(time.time() - ti, 1)})
    if not a.quiet:
        print(f"  {i}/{len(cases)} {cid} · blob {nb} · drop {nd} · {time.time()-ti:.1f}s", flush=True)

tb = sum(r["n_blob"] for r in rows); td = sum(r["n_drop"] for r in rows)
summary = {"pred": a.pred, "out": a.out, "threshold": thr, "device": dev, "split": a.split,
           "n_cases": len(cases), "n_blob": tb, "n_drop": td,
           "sec_total": round(time.time() - t0, 1), "cases": rows}
if a.report:
    json.dump(summary, open(a.report, "w"), indent=1)
print(f"[pf] {len(cases)}케이스 · blob {tb} 중 {td} 제거 · {time.time()-t0:.1f}s · {dev}", flush=True)
