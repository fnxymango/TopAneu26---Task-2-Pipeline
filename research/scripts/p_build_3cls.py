#!/usr/bin/env python
"""P1 — pjh Stage1 재현용 3-class 데이터셋 빌드 (2026-08-25).

REIMPLEMENT_jslee_pjh.md §3.2~3.3 을 따른다.

라벨:  0 background · 1 vessel · 2 aneurysm
  **동맥류를 나중에 그려 혈관 위에 덮는다.** 겹치는 복셀을 혈관으로 두면 검출 대상이
  사라진다 — 문서가 명시한 함정이다.

정규화: nnU-Net 기본 ZScore 가 아니라 **모달리티별 robust z** 를 빌드 시점에 끝낸다.
  기본 ZScore 는 케이스 전체 평균/표준편차라 CTA 에서 뼈에 끌려 모달리티 대역비가 272배로
  실패했다(문서 §3.2). foreground 정의가 모달리티마다 다르다:
      CTA   arr > -300 HU
      MRA   arr > max(1.0, 0.02 * p99.5)
      -> (x - median_fg) / IQR_fg          케이스마다 따로
  따라서 dataset.json 의 channel_names 는 **noNorm** 으로 선언한다.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d9xx_lib as L

DS = "Dataset722_TopAneuPjh3cls417"
OUT = L.RAW / DS


def modality(cid):
    """케이스 id 에 _ct_ / _mr_ 가 들어 있다."""
    if "_ct_" in cid:
        return "CT"
    if "_mr_" in cid:
        return "MR"
    raise ValueError(cid)


def robust_z(arr, mod):
    a = arr.astype(np.float32)
    if mod == "CT":
        fg = a > -300.0
    else:
        p995 = np.percentile(a, 99.5)
        fg = a > max(1.0, 0.02 * float(p995))
    if fg.sum() < 100:                       # 전경이 거의 없으면 전체로 후퇴
        fg = np.ones_like(a, dtype=bool)
    v = a[fg]
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    if iqr < 1e-6:
        iqr = float(v.std()) or 1.0
    return (a - med) / iqr, med, iqr, float(fg.mean())


def main():
    (OUT / "imagesTr").mkdir(parents=True, exist_ok=True)
    (OUT / "labelsTr").mkdir(parents=True, exist_ok=True)
    # images/ 는 {cid}_0000.nii.gz, 마스크는 {cid}.nii.gz 로 접미사가 다르다.
    ids = sorted(p.name[:-len("_0000.nii.gz")] for p in (L.DATA / "images").glob("*_0000.nii.gz"))
    print(f"[p1] 케이스 {len(ids)}", flush=True)
    stats = []
    for i, cid in enumerate(ids):
        ip = OUT / "imagesTr" / f"{cid}_0000.nii.gz"
        lp = OUT / "labelsTr" / f"{cid}.nii.gz"
        if ip.exists() and lp.exists():
            continue
        img = nib.load(L.DATA / "images" / f"{cid}_0000.nii.gz")
        arr = np.asanyarray(img.dataobj)
        mod = modality(cid)
        norm, med, iqr, fgr = robust_z(arr, mod)
        nib.save(nib.Nifti1Image(norm.astype(np.float32), img.affine, img.header), ip)

        ves = np.asanyarray(nib.load(L.DATA / "vessel_masks" / f"{cid}.nii.gz").dataobj)
        loc = np.asanyarray(nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz").dataobj)
        seg = np.zeros(ves.shape, dtype=np.uint8)
        seg[ves > 0] = 1
        seg[loc > 0] = 2                      # ★ 동맥류를 나중에 — 혈관 위에 덮는다
        li = nib.Nifti1Image(seg, img.affine, img.header)
        li.set_data_dtype(np.uint8)
        nib.save(li, lp)
        stats.append((cid, mod, med, iqr, fgr, int((seg == 1).sum()), int((seg == 2).sum())))
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(ids)}", flush=True)

    json.dump({
        "channel_names": {"0": "noNorm"},
        "labels": {"background": 0, "vessel": 1, "aneurysm": 2},
        "numTraining": len(ids),
        "file_ending": ".nii.gz",
        "name": DS,
        "description": "pjh Stage1 재현: 3-class(bg/vessel/aneurysm) + 모달리티별 robust z (빌드시 정규화 완료)",
    }, open(OUT / "dataset.json", "w"), indent=2)

    if stats:
        a = np.array([[s[2], s[3], s[4]] for s in stats], dtype=float)
        ct = [s for s in stats if s[1] == "CT"]; mr = [s for s in stats if s[1] == "MR"]
        print(f"\n[p1] 정규화 통계  CT {len(ct)}건 · MR {len(mr)}건")
        for lab, sub in (("CT", ct), ("MR", mr)):
            if not sub: continue
            m = np.array([[s[2], s[3], s[4]] for s in sub], dtype=float)
            print(f"  {lab}  median_fg 중앙 {np.median(m[:,0]):>9.2f} · IQR 중앙 {np.median(m[:,1]):>8.2f} "
                  f"· 전경비율 중앙 {np.median(m[:,2]):.3f}")
        v = np.array([s[5] for s in stats]); an = np.array([s[6] for s in stats])
        print(f"  라벨 복셀  vessel 중앙 {np.median(v):,.0f} · aneurysm 중앙 {np.median(an):,.0f} "
              f"· aneurysm 0 인 케이스 {(an==0).sum()}/{len(stats)}")
    print(f"[p1] 완료 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
