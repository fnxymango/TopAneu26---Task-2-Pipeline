#!/usr/bin/env python
"""크롭 데이터셋 v2 — patch-CNN 용 (2026-08-26).

C24/C25 와 다른 점:
  (a) **병변 마스크를 채널로 넣는다.** C24 는 크롭 중심만 맞추고 어느 성분이 대상인지
      알려주지 않았다. 근처에 다른 병변/후보가 있으면 모델이 대상을 특정할 수 없다.
  (b) **혈관 라벨을 ves/36 로 넣지 않는다.** 라벨 id 는 명목형인데 36으로 나누면
      8(R-Pcom)과 9(L-Pcom)가 거의 같은 값이 되고 모델이 엉뚱한 순서를 배운다.
      대신 의미 있는 이진 채널로 쪼갠다: 전체혈관 / 큰줄기 / 작은곁가지.
  (c) **창을 32mm/64^3 = 0.5mm 등방** 으로 잡는다. C24 는 73mm/96^3 = 0.76mm 였는데,
      가르려는 게 지름 1mm 안팎의 Pcom·AChA 라 그 해상도로는 1복셀도 안 된다.
      분기 기원은 병변에서 10mm 안에 있으므로 32mm 면 충분히 덮는다.

채널
  0  영상 (TopAneuAdaptiveNorm)
  1  이 후보 병변 마스크
  2  혈관 전체 (이진)
  3  큰 줄기   (ICA/M1/M2/A1A2/BA/VA/P1P2 등 학습중앙값 >1000복셀)
  4  작은 곁가지 (Pcom/AChA/OA/PICA/SCA/AICA/Acom/A3/M3/P3P4)
"""
import argparse, json, os
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L

ST = np.ones((3, 3, 3), bool)
WIN_MM = 32.0
OUT = 64
RAW = L.TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset800_TopAneuVessel417" / "imagesTr"
VMAP = json.load(open(L.TOPANEU_ROOT / "dataset/TopAneu/vessel_mapping.json"))["labels"]
SMALL = {"R-Pcom","L-Pcom","Acom","R-SCA","L-SCA","R-AICA","L-AICA","R-PICA","L-PICA",
         "R-AChA","L-AChA","R-OA","L-OA","R-A3","L-A3","R-M3","L-M3","R-P3P4","L-P3P4",
         "3rd-A2","3rd-A3"}
SMALL_ID = {v for k, v in VMAP.items() if k in SMALL}
BIG_ID = {v for k, v in VMAP.items() if k not in SMALL and v > 0}


def adaptive_norm(img):
    img = img.astype(np.float32, copy=True); eps = 1e-8
    if float((img < -300).mean()) > 0.01:
        np.clip(img, -200.0, 800.0, out=img)
        fg = img[img > -100.0]
        m = float(fg.mean()) if fg.size else float(img.mean())
        s = float(fg.std()) if fg.size else float(img.std())
        return (img - m) / max(s, eps)
    out = np.zeros_like(img); fg = img > 0
    if fg.sum():
        v = img[fg]; lo, hi = np.percentile(v, (0.5, 99.9)); v = np.clip(v, lo, hi)
        out[fg] = (v - float(v.mean())) / max(float(v.std()), eps)
    return out


def crop_at(vol, cen, half, fill=0.0):
    lo = np.round(cen - half).astype(int); hi = lo + np.round(half * 2).astype(int)
    out = np.full(tuple(hi - lo), fill, dtype=np.float32)
    a = np.maximum(lo, 0); b = np.minimum(hi, np.array(vol.shape))
    if np.any(b <= a): return out
    da = a - lo; db = da + (b - a)
    out[da[0]:db[0], da[1]:db[1], da[2]:db[2]] = vol[a[0]:b[0], a[1]:b[1], a[2]:b[2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--aneu-dir", default=None, help="주면 검출마스크 기준(e2e), 없으면 GT")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    id2name, _ = L.official_location_names()
    ids = L.case_ids_by_split()[{"train": 0, "val": 1, "test": 2}[a.split]]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vd = Path(a.vessel_dir); meta = []
    for i, cid in enumerate(ids, 1):
        ip = RAW / f"{cid}_0000.nii.gz"; vp = vd / f"{cid}.nii.gz"
        sp_ = (Path(a.aneu_dir) / f"{cid}.nii.gz") if a.aneu_dir else (L.DATA / "location_masks" / f"{cid}.nii.gz")
        gp = L.DATA / "location_masks" / f"{cid}.nii.gz"
        if not (ip.exists() and vp.exists() and sp_.exists()): continue
        ii = nib.load(ip); img = np.asanyarray(ii.dataobj).astype(np.float32)
        ves = np.asanyarray(nib.load(vp).dataobj).astype(np.int16)
        src = np.asanyarray(nib.load(sp_).dataobj)
        gtv = np.asanyarray(nib.load(gp).dataobj) if gp.exists() else None
        if img.shape != ves.shape or img.shape != src.shape: continue
        spacing = np.array(ii.header.get_zooms()[:3], dtype=float)
        img = adaptive_norm(img)
        vall = (ves > 0).astype(np.float32)
        vbig = np.isin(ves, list(BIG_ID)).astype(np.float32)
        vsml = np.isin(ves, list(SMALL_ID)).astype(np.float32)
        lab, n = ndi.label(src > 0, structure=ST)
        half = (WIN_MM / 2.0) / spacing
        for l in range(1, n + 1):
            m = lab == l
            idx = np.argwhere(m); cen = idx.mean(axis=0)
            gt = None
            if gtv is not None:
                vals, cnts = np.unique(gtv[m], return_counts=True)
                vals, cnts = vals[vals > 0], cnts[vals > 0]
                if len(vals): gt = id2name.get(int(vals[np.argmax(cnts)]))
            ch = np.stack([crop_at(img, cen, half), crop_at(m.astype(np.float32), cen, half),
                           crop_at(vall, cen, half), crop_at(vbig, cen, half),
                           crop_at(vsml, cen, half)])
            zoom = [1.0] + [OUT / s for s in ch.shape[1:]]
            ch = ndi.zoom(ch, zoom, order=1).astype(np.float16)
            key = f"{cid}__{l}"
            np.savez_compressed(out / f"{key}.npz", x=ch)
            meta.append(dict(key=key, case=cid, lesion_mask_idx=l, gt_loc=gt,
                             n_vox=int(m.sum())))
        if i % 40 == 0: print(f"  {a.split} {i}/{len(ids)} · 누적크롭 {len(meta)}", flush=True)
    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False)
    lab_n = sum(1 for x in meta if x["gt_loc"])
    print(f"[{a.split}] 크롭 {len(meta)}건 (라벨 있는 것 {lab_n}) -> {out}")


if __name__ == "__main__":
    main()
