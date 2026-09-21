"""C24-a — 병변 중심 크롭 데이터셋 생성.

현재 분류 피처 112차원은 전부 마스크에서 계산한 기하량이고 **원본 화소값이 하나도 없다.**
AChA 분기점 검출률이 43.2% 라 케이스의 57% 에서는 Pcom 과 AChA 의 입력이 동일해
어떤 알고리즘도 구분할 수 없다. 영상에는 찍혀 있으나 분할 임계를 못 넘은 정보를 회수한다.

크롭 규격은 D740 측정치를 따른다(analysis/d740_window_size.json):
  병변 최대변 중앙값 5.0mm / p95 19.8mm / p99 33.1mm, margin 20mm -> 창 73mm

채널:
  0  TopAneuAdaptiveNorm 적용 영상  — CTA는 clip[-200,800]+전경 z-score,
                                      MRA는 퍼센타일[0.5,99.9] clip+전경 z-score (배경 0 유지)
  1  36클래스 혈관 라벨 / 36 (0~1 정규화)
  2  혈관까지 거리맵 (mm), 10mm 에서 포화

C14 가 2회 실패한 원인 중 하나가 **패치 좌표 매핑 미검증**이라, 여기서는 크롭 중심에
병변이 실제로 들어갔는지 `lesion_frac`(크롭 안 병변 voxel 비율)을 같이 저장해 게이트에서 확인한다.

사용:
  python c24_build_crops.py --split train --out <dir>
  python c24_build_crops.py --split val --aneu-dir <검출마스크> --out <dir>   # e2e용
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

import d9xx_lib as L

ST = np.ones((3, 3, 3), dtype=bool)
WIN_MM = 73.0
OUT_VOX = 96
RAW = L.TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset800_TopAneuVessel417" / "imagesTr"


def adaptive_norm(img):
    """TopAneuAdaptiveNorm 과 동일 로직 (nnU-Net import 회피용 인라인)."""
    img = img.astype(np.float32, copy=True)
    eps = 1e-8
    if float((img < -300).mean()) > 0.01:                 # ── CTA ──
        np.clip(img, -200.0, 800.0, out=img)
        fg = img[img > -100.0]
        m = float(fg.mean()) if fg.size else float(img.mean())
        s = float(fg.std()) if fg.size else float(img.std())
        return (img - m) / max(s, eps)
    out = np.zeros_like(img)                              # ── MRA ── 배경 0 유지
    fg = img > 0
    if fg.sum():
        v = img[fg]
        lo, hi = np.percentile(v, (0.5, 99.9))
        v = np.clip(v, lo, hi)
        out[fg] = (v - float(v.mean())) / max(float(v.std()), eps)
    return out


def crop_at(vol, center_vox, half_vox, fill=0.0):
    """center 중심 half_vox 반경 크롭. 경계 밖은 fill 로 패딩."""
    lo = np.round(center_vox - half_vox).astype(int)
    hi = lo + np.round(half_vox * 2).astype(int)
    out = np.full(tuple(hi - lo), fill, dtype=np.float32)
    src_lo = np.maximum(lo, 0)
    src_hi = np.minimum(hi, np.array(vol.shape))
    if np.any(src_hi <= src_lo):
        return out
    dst_lo = src_lo - lo
    dst_hi = dst_lo + (src_hi - src_lo)
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1], dst_lo[2]:dst_hi[2]] = \
        vol[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1], src_lo[2]:src_hi[2]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--aneu-dir", default=None,
                    help="주면 검출마스크 기준(e2e용), 없으면 GT location_masks 기준")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    id2name, _ = L.official_location_names()
    ids = L.case_ids_by_split()[{"train": 0, "val": 1, "test": 2}[args.split]]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ves_dir = Path(args.vessel_dir)
    meta, n = [], 0

    for i, cid in enumerate(ids, 1):
        img_p = RAW / f"{cid}_0000.nii.gz"
        ves_p = ves_dir / f"{cid}.nii.gz"
        src_p = (Path(args.aneu_dir) / f"{cid}.nii.gz") if args.aneu_dir \
            else (L.DATA / "location_masks" / f"{cid}.nii.gz")
        if not (img_p.exists() and ves_p.exists() and src_p.exists()):
            continue
        ii = nib.load(img_p)
        img = np.asanyarray(ii.dataobj).astype(np.float32)
        ves = np.asanyarray(nib.load(ves_p).dataobj).astype(np.int16)
        src = np.asanyarray(nib.load(src_p).dataobj)
        if img.shape != ves.shape or img.shape != src.shape:
            print(f"  스킵 {cid}: shape 불일치"); continue
        spacing = np.array(ii.header.get_zooms()[:3], dtype=float)

        img = adaptive_norm(img)
        dist = (ndi.distance_transform_edt(~(ves > 0), sampling=spacing)
                if (ves > 0).any() else np.full(img.shape, 10.0))
        np.clip(dist, 0, 10.0, out=dist)

        lab, nl = ndi.label(src > 0, structure=ST)
        half_vox = (WIN_MM / 2.0) / spacing
        for l in range(1, nl + 1):
            m = lab == l
            idx = np.argwhere(m)
            cen = idx.mean(axis=0)
            gt = None
            if args.aneu_dir is None:                      # GT 라벨에서 클래스 추출
                vals, cnts = np.unique(src[m], return_counts=True)
                gt = id2name.get(int(vals[np.argmax(cnts)]))
            ch = np.stack([crop_at(img, cen, half_vox),
                           crop_at(ves.astype(np.float32), cen, half_vox) / 36.0,
                           crop_at(dist, cen, half_vox, fill=10.0) / 10.0])
            les = crop_at(m.astype(np.float32), cen, half_vox)
            # 등방 96^3 으로 리샘플
            zoom = [1.0] + [OUT_VOX / s for s in ch.shape[1:]]
            ch = ndi.zoom(ch, zoom, order=1).astype(np.float16)
            les = ndi.zoom(les, zoom[1:], order=1)
            frac = float(les.sum()) / max(float(m.sum()), 1.0) * np.prod(
                [s / OUT_VOX for s in (2 * half_vox)])
            key = f"{cid}__{l}"
            np.savez_compressed(out / f"{key}.npz", x=ch)
            meta.append({"key": key, "case": cid, "lesion_idx": int(l),
                         "gt_loc": gt, "n_vox": int(m.sum()),
                         "lesion_in_crop": float(les.sum()),
                         "center_vox": [float(v) for v in cen]})
            n += 1
        if i % 25 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}  누적 크롭 {n}", flush=True)

    json.dump(meta, open(out / "meta.json", "w"), ensure_ascii=False)
    lab_n = sum(1 for m in meta if m["gt_loc"])
    empty = sum(1 for m in meta if m["lesion_in_crop"] < 1)
    print(f"[c24-build] {args.split}: 크롭 {n} (라벨 {lab_n}) · "
          f"병변이 크롭에 안 잡힌 것 {empty} -> {out}")


if __name__ == "__main__":
    main()
