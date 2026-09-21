#!/usr/bin/env python
"""GrandChallenge .mha ↔ 우리 파이프라인(.nii.gz/nibabel) 변환 (2026-08-28, 도커용).

우리 코드는 전부 nibabel 로 읽고 `header.get_zooms()[:3]` 을 spacing 으로 쓴다. GC 템플릿은
SimpleITK 로 .mha 를 읽어 넘긴다. 두 라이브러리는 축 순서(sitk z,y,x / nibabel x,y,z)와
좌표 규약(LPS / RAS)이 달라, 배열만 전치해 넘기면 affine 이 어긋난다.

가장 안전한 경로는 **SimpleITK 로 .nii.gz 를 써서 왕복**하는 것이다 — sitk 가 LPS→RAS 를
정확히 처리하므로 nibabel 이 읽을 때 affine 이 복원된다. 아래 to_nifti/from_nifti 가 그것이고,
검증(verify)은 원본 .nii.gz → .mha → .nii.gz 왕복이 복셀단위로 같은지 본다.
"""
import argparse, os, tempfile
import numpy as np


def to_nifti(sitk_image, path):
    """sitk.Image → 디스크의 .nii.gz (nibabel 이 그대로 읽을 수 있는 형태)."""
    import SimpleITK as sitk
    sitk.WriteImage(sitk_image, path, useCompression=True)
    return path


def from_nifti(path):
    """우리 파이프라인이 만든 .nii.gz → sitk.Image (GC 출력용 .mha 로 쓸 수 있음)."""
    import SimpleITK as sitk
    return sitk.ReadImage(path)


def verify(src_nii):
    """원본 .nii.gz → .mha → .nii.gz 왕복이 배열·spacing 을 보존하는지."""
    import SimpleITK as sitk, nibabel as nib
    a = nib.load(src_nii); aa = np.asanyarray(a.dataobj)
    with tempfile.TemporaryDirectory() as d:
        m = os.path.join(d, "x.mha"); n2 = os.path.join(d, "x.nii.gz")
        sitk.WriteImage(sitk.ReadImage(src_nii), m)
        sitk.WriteImage(sitk.ReadImage(m), n2, useCompression=True)
        b = nib.load(n2); bb = np.asanyarray(b.dataobj)
    same_shape = aa.shape == bb.shape
    maxdiff = float(np.abs(aa.astype(np.float64) - bb.astype(np.float64)).max()) if same_shape else float("nan")
    zoom_a = tuple(round(float(z), 6) for z in a.header.get_zooms()[:3])
    zoom_b = tuple(round(float(z), 6) for z in b.header.get_zooms()[:3])
    aff = float(np.abs(a.affine - b.affine).max())
    print(f"[verify] shape {aa.shape} == {bb.shape} : {same_shape}")
    print(f"[verify] 최대 복셀차 {maxdiff:.3e} · zooms {zoom_a} vs {zoom_b} · affine 최대차 {aff:.3e}")
    ok = same_shape and maxdiff < 1e-3 and zoom_a == zoom_b and aff < 1e-4
    print(f"[verify] {'통과' if ok else '★실패'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", required=True)
    verify(ap.parse_args().verify)
