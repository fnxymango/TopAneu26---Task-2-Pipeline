#!/usr/bin/env python3
"""
예측 마스크(val_predict_out) → skeleton(centerline)만 추출해서 이미지화.

챌린지 공식 metric과 동일: 라벨 병합 → binary → skimage skeletonize(Lee thinning).
skeleton voxel에 원 혈관 라벨을 되붙여 혈관별 색 centerline을 만든다.

출력(<exp>/val_skeletons/):
  <case>_skel.nii.gz    혈관 라벨이 붙은 skeleton (3D Slicer 확인용)
  <case>_skel.png       skeleton만 3면(axial/coronal/sagittal) MIP
  _overview.png         15케이스 axial MIP 한 장 + 혈관 색 범례

Usage: skeleton_from_predictions.py [pred_dir] [out_dir]
"""
import sys, os, glob, json
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
    def skel3d(b): return skeletonize(b, method="lee").astype(bool)
except Exception:
    from skimage.morphology import skeletonize_3d
    def skel3d(b): return skeletonize_3d(b).astype(bool)

_ROOT = os.environ.get("TOPANEU_ROOT", os.path.expanduser("~/topaneu_sblee"))
EXP = f"{_ROOT}/experiments/D600_vessel_skelrec_resencm_250ep"
PRED = sys.argv[1] if len(sys.argv) > 1 else f"{EXP}/val_predict_out"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{EXP}/val_skeletons"
DS_JSON = os.environ.get("SKEL_DS_JSON", f"{_ROOT}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json")
os.makedirs(OUT, exist_ok=True)

NAME = {v: k for k, v in json.load(open(DS_JSON))["labels"].items()}
# 36클래스는 연속 컬러맵(nipy_spectral)으로는 초록 계열이 뭉쳐 구분이 안 됨
# → tab20+tab20b 범주형 팔레트로 인접 라벨이 최대한 다른 색이 되게 배치
_PAL = np.vstack([cm.tab20(np.arange(20)), cm.tab20b(np.arange(20))])
LUT = np.zeros((37, 4)); LUT[1:] = _PAL[:36]
AXES = ["axial (z-MIP)", "coronal (y-MIP)", "sagittal (x-MIP)"]
MARGIN = 6


def colorize(lab2d):
    rgb = LUT[np.clip(lab2d, 0, 36)][..., :3]
    rgb[lab2d == 0] = 0
    return rgb


def crop(lab2d):
    """skeleton 바운딩박스로 잘라 검은 여백 제거."""
    ys, xs = np.nonzero(lab2d)
    if len(ys) == 0:
        return lab2d
    y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN + 1, lab2d.shape[0])
    x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN + 1, lab2d.shape[1])
    return lab2d[y0:y1, x0:x1]


cases = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{PRED}/*.nii.gz"))
print(f"[in] {PRED}  ({len(cases)} cases)")
stats, thumbs = [], []

for case in cases:
    img = sitk.ReadImage(f"{PRED}/{case}.nii.gz")
    mask = sitk.GetArrayFromImage(img).astype(np.int16)          # [z,y,x], 0..36
    binary = mask > 0
    skel = skel3d(binary)
    skel_lab = np.where(skel, mask, 0).astype(np.int16)          # 혈관별 색 centerline

    nves, nskel = int(binary.sum()), int(skel.sum())
    present = sorted(int(x) for x in np.unique(skel_lab) if x != 0)
    stats.append({"case": case, "vessel_voxels": nves, "skeleton_voxels": nskel,
                  "compression": round(nves / max(nskel, 1), 1),
                  "n_labels_on_skeleton": len(present),
                  "labels": [NAME[i] for i in present]})

    so = sitk.GetImageFromArray(skel_lab); so.CopyInformation(img)
    sitk.WriteImage(so, f"{OUT}/{case}_skel.nii.gz")

    # skeleton은 1-voxel 두께라 MIP에서 거의 안 보임 → 라벨 보존 dilation
    fat = ndi.grey_dilation(skel_lab, size=(3, 3, 3))

    fig, axg = plt.subplots(1, 3, figsize=(15, 5.6), facecolor="black")
    for col in range(3):
        axg[col].imshow(colorize(crop(np.rot90(fat.max(axis=col)))))
        axg[col].set_title(AXES[col], color="w", fontsize=11)
        axg[col].axis("off")
    fig.legend(handles=[Patch(facecolor=LUT[i][:3], label=NAME[i]) for i in present],
               loc="lower center", ncol=10, facecolor="black", labelcolor="w",
               edgecolor="0.4", fontsize=8, framealpha=1)
    fig.suptitle(f"{case}  —  skeleton only  |  {nskel:,} vox ({nves/max(nskel,1):.0f}x thinner)"
                 f"  |  {len(present)}/36 vessels", color="w", fontsize=13)
    fig.tight_layout(rect=[0, 0.14, 1, 0.94])
    fig.savefig(f"{OUT}/{case}_skel.png", dpi=110, facecolor="black")
    plt.close(fig)

    thumbs.append((case, colorize(crop(np.rot90(fat.max(axis=0)))), len(present)))
    print(f"  {case}: vessel {nves:>7,} → skel {nskel:>6,} ({nves/max(nskel,1):>4.0f}x), "
          f"{len(present)}/36 labels")

# ---- 전체 개관 (axial MIP 15장 + 범례) ----
ncol, nrow = 5, int(np.ceil(len(thumbs) / 5))
fig, axg = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.4 * nrow + 1.6), facecolor="black")
for ax in np.ravel(axg): ax.axis("off")
for ax, (case, rgb, n) in zip(np.ravel(axg), thumbs):
    ax.imshow(rgb)
    ax.set_title(f"{case.replace('topaneu_', '')}  ({n}/36)", color="w", fontsize=9)

seen = sorted({l for s in stats for l in s["labels"]},
              key=lambda n: [k for k, v in NAME.items() if v == n][0])
handles = [Patch(facecolor=LUT[[k for k, v in NAME.items() if v == n][0]][:3], label=n) for n in seen]
fig.legend(handles=handles, loc="lower center", ncol=8, facecolor="black",
           labelcolor="w", edgecolor="0.4", fontsize=8, framealpha=1)
fig.suptitle("D600 val 15 — predicted vessel skeletons (axial MIP, colored by vessel label)",
             color="w", fontsize=14)
fig.tight_layout(rect=[0, 0.13, 1, 0.96])
fig.savefig(f"{OUT}/_overview.png", dpi=120, facecolor="black")
plt.close(fig)

json.dump(stats, open(f"{OUT}/skeleton_stats.json", "w"), indent=2)
print(f"\n[out] {OUT}  (PNG {len(cases)+1}, NIfTI {len(cases)}, skeleton_stats.json)")
