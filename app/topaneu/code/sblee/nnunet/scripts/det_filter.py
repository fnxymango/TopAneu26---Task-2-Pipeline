#!/usr/bin/env python
"""검출 후처리 필터 — GT 불요판 (2026-08-28, 도커/번들용).

c7_detect_postproc.py 는 스윕·평가용이라 GT(labelsTr)와 split 목록을 읽는다. 배포 환경엔
정답이 없으므로, **확정된 설정(min_vox=5, max_dist=1.0mm)만 그대로 적용**하는 부분을 떼어냈다.
로직은 c7 의 component_stats + save_best 분기와 동일하다(성분 라벨링 구조도 3x3x3).

사용: python det_filter.py --aneu <bin.nii.gz> --vessel <36cls.nii.gz> --out <f.nii.gz>
      [--min-vox 5] [--max-dist 1.0]
입력 aneu 는 5폴드 확률평균 결과에서 **라벨 2(aneurysm)만 이진화**한 마스크다.
"""
import argparse
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

ST = np.ones((3, 3, 3), dtype=bool)


def filter_case(aneu, ves, spacing, min_vox=5, max_dist=1.0):
    lab, n = ndi.label(aneu > 0, structure=ST)
    if n == 0:
        return np.zeros_like(lab, dtype=np.int16)
    dist = ndi.distance_transform_edt(~(ves > 0), sampling=spacing) if (ves > 0).any() \
        else np.full(aneu.shape, np.inf)
    keep = [l for l in range(1, n + 1)
            if (m := (lab == l)).sum() >= min_vox and float(dist[m].min()) <= max_dist]
    return np.isin(lab, keep).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aneu", required=True); ap.add_argument("--vessel", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-vox", type=int, default=5)
    ap.add_argument("--max-dist", type=float, default=1.0)
    a = ap.parse_args()
    ai = nib.load(a.aneu); an = np.asanyarray(ai.dataobj)
    ves = np.asanyarray(nib.load(a.vessel).dataobj)
    sp = np.array(ai.header.get_zooms()[:3], dtype=float)
    m = filter_case(an, ves, sp, a.min_vox, a.max_dist)
    nib.save(nib.Nifti1Image(m, ai.affine, ai.header), a.out)
    print(f"[det_filter] 성분 {int(m.max()) if m.any() else 0} · 유지복셀 {int((m>0).sum())} → {a.out}")


if __name__ == "__main__":
    main()
