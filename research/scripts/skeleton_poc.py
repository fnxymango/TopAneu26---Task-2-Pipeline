#!/usr/bin/env python3
"""
Skeleton PoC — GT 36-class 뇌혈관 마스크에서 centerline 추출 + 시각화.

챌린지 공식 metric과 동일하게: 라벨 병합 → binary → skimage skeletonize(3D, Lee).
추가로 skeleton voxel에 원 혈관 라벨을 되붙여 '혈관별 색' skeleton을 만든다
(= 동맥류 위치 할당의 근거가 될 해부학적 centerline).

출력(outputs/):
  skeleton_poc_<case>.png       3면(axial/coronal/sagittal) MIP: 혈관 + centerline(혈관별 색)
  skeleton_poc_<case>_skel.nii.gz   혈관 라벨이 붙은 skeleton (3D Slicer 확인용)

Usage: skeleton_poc.py [case_id]   (기본: topaneu_center2_mr_002)
"""
import sys, os
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
    def skel3d(b): return skeletonize(b, method="lee").astype(bool)   # Lee = 챌린지 skeletonize_3d
except Exception:
    from skimage.morphology import skeletonize_3d
    def skel3d(b): return skeletonize_3d(b).astype(bool)

case = sys.argv[1] if len(sys.argv) > 1 else "topaneu_center2_mr_002"
VMASK = f"/home/user/dataset/TopAneu/vessel_masks/{case}.nii.gz"
IMG   = f"/home/user/TopAneu/seg/sblee/nnunet/nnUNet_raw/Dataset520_TopAneuBinary/imagesTr/{case}_0000.nii.gz"
OUTDIR = "/home/user/TopAneu/seg/sblee/outputs"
os.makedirs(OUTDIR, exist_ok=True)

vimg = sitk.ReadImage(VMASK)
mask = sitk.GetArrayFromImage(vimg).astype(np.int16)      # [z,y,x], 0..36
binary = mask > 0
labels_present = sorted(int(x) for x in np.unique(mask) if x != 0)
print(f"[case] {case}  shape={mask.shape}  vessel labels={len(labels_present)}: {labels_present}")

# ---- skeleton (challenge-canonical: merged binary → Lee thinning) ----
skel = skel3d(binary)
# skeleton voxel에 원 혈관 라벨 되붙이기 (혈관별 색 centerline)
skel_lab = np.where(skel, mask, 0).astype(np.int16)
nves, nskel = int(binary.sum()), int(skel.sum())
print(f"[skeleton] vessel voxels={nves:,}  skeleton voxels={nskel:,}  (지름 압축비 {nves/max(nskel,1):.1f}x)")

# skeleton NIfTI 저장 (라벨 보존, 원본 geometry 복사)
so = sitk.GetImageFromArray(skel_lab); so.CopyInformation(vimg)
skel_path = f"{OUTDIR}/skeleton_poc_{case}_skel.nii.gz"
sitk.WriteImage(so, skel_path)
print(f"[save] {skel_path}")

# ---- 시각화 ----
# angio 배경 (있으면), 없으면 binary
try:
    ai = sitk.GetArrayFromImage(sitk.ReadImage(IMG)).astype(np.float32)
    if ai.shape != mask.shape: ai = None
except Exception:
    ai = None

LUT = cm.nipy_spectral(np.linspace(0, 1, 37))  # 0..36
def colorize(lab2d):
    rgb = LUT[np.clip(lab2d, 0, 36)][..., :3]
    rgb[lab2d == 0] = 0
    return rgb

# skeleton은 얇아서 dilation으로 두껍게 (라벨 보존: grey dilation)
skel_lab_fat = ndi.grey_dilation(skel_lab, size=(3, 3, 3))
skel_lab_fat = np.where(skel_lab > 0, skel_lab, np.where(skel, skel_lab, skel_lab_fat)) if False else skel_lab_fat

axes_names = ["axial (z-MIP)", "coronal (y-MIP)", "sagittal (x-MIP)"]
fig, axg = plt.subplots(2, 3, figsize=(15, 9), facecolor="black")
for col, ax_i in enumerate([0, 1, 2]):
    m_mip = np.rot90(mask.max(axis=ax_i))
    s_mip = np.rot90(skel_lab_fat.max(axis=ax_i))
    b_mip = np.rot90(binary.max(axis=ax_i).astype(float))
    ang = np.rot90(ai.max(axis=ax_i)) if ai is not None else b_mip

    # row0: 36-class 혈관 마스크
    a0 = axg[0, col]; a0.imshow(colorize(m_mip)); a0.set_title(axes_names[col], color="w", fontsize=11)
    a0.axis("off")
    # row1: angio(회색) + centerline(혈관별 색)
    a1 = axg[1, col]
    vmax = np.percentile(ang[ang > 0], 99) if (ang > 0).any() else 1
    a1.imshow(ang, cmap="gray", vmin=0, vmax=vmax)
    srgb = colorize(s_mip); mask_ov = (s_mip > 0)
    overlay = np.zeros((*s_mip.shape, 4)); overlay[..., :3] = srgb; overlay[..., 3] = mask_ov * 1.0
    a1.imshow(overlay); a1.axis("off")

axg[0, 0].text(-0.05, 0.5, "vessels (36-class)", color="w", rotation=90,
               va="center", ha="right", transform=axg[0, 0].transAxes, fontsize=12)
axg[1, 0].text(-0.05, 0.5, "centerline over angio", color="w", rotation=90,
               va="center", ha="right", transform=axg[1, 0].transAxes, fontsize=12)
fig.suptitle(f"{case}  —  vessel labels={len(labels_present)}, skeleton {nskel:,} vox "
             f"({nves/max(nskel,1):.0f}x thinner)", color="w", fontsize=13)
fig.tight_layout(rect=[0.02, 0, 1, 0.97])
png = f"{OUTDIR}/skeleton_poc_{case}.png"
fig.savefig(png, dpi=110, facecolor="black"); print(f"[save] {png}")
