#!/usr/bin/env python3
"""분류기 약점 지도의 4개 영역이 실제 뇌혈관 어디인지 — CTA 실제 케이스 MIP 위에 표시."""
import json
import numpy as np, nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
CID = "topaneu_center2_ct_161"
plt.rcParams["font.family"] = "DejaVu Sans"

VID = {k: int(v) for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}

TERR = {
    "ICA": ["R-ICA-C1-C5","L-ICA-C1-C5","R-ICA-C6-C7","L-ICA-C6-C7",
            "R-Pcom","L-Pcom","R-AChA","L-AChA","R-OA","L-OA"],
    "Posterior (VA/BA/PCA)": ["BA","R-VA","L-VA","R-P1P2","L-P1P2","R-P3P4","L-P3P4",
                              "R-SCA","L-SCA","R-AICA","L-AICA","R-PICA","L-PICA"],
    "MCA": ["R-M1","L-M1","R-M2","L-M2","R-M3","L-M3"],
    "ACA / Acom": ["Acom","R-A1A2","L-A1A2","R-A3","L-A3","3rd-A2","3rd-A3"],
}
COL = {"ICA": (0.98, 0.33, 0.28), "Posterior (VA/BA/PCA)": (0.33, 0.66, 1.00),
       "MCA": (1.00, 0.78, 0.16), "ACA / Acom": (0.40, 0.92, 0.52)}
STAT = {"ICA": (64.3, "45%", 42), "Posterior (VA/BA/PCA)": (47.8, "28%", 18),
        "MCA": (81.5, "15%", 27), "ACA / Acom": (81.0, "11%", 20)}
ORDER = list(TERR); TID = {n: i + 1 for i, n in enumerate(ORDER)}
lut = np.zeros(max(VID.values()) + 1, np.uint8)
for n, vs in TERR.items():
    for v in vs: lut[VID[v]] = TID[n]

img = nib.as_closest_canonical(nib.load(f"{R}/experiments/_c1_realpred/in_test/{CID}_0000.nii.gz"))
ves = nib.as_closest_canonical(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{CID}.nii.gz"))
I = np.asanyarray(img.dataobj).astype(np.float32)
T = lut[np.clip(np.asanyarray(ves.dataobj).astype(np.int16), 0, lut.size - 1)]
nz = np.argwhere(T > 0); lo = np.maximum(nz.min(0) - 12, 0)
hi = np.minimum(nz.max(0) + 13, np.array(T.shape))
sl = tuple(slice(a, b) for a, b in zip(lo, hi)); I = I[sl]; T = T[sl]
z = img.header.get_zooms()[:3]
p1, p99 = np.percentile(I, [45, 99.7]); I = np.clip((I - p1) / max(p99 - p1, 1e-6), 0, 1)

def draw(ax, axis, flip_v, flip_h, title, left, right):
    g = I.max(axis=axis)
    m = T > 0
    lab = np.where(m.any(axis=axis),
                   np.take_along_axis(T, np.expand_dims(np.argmax(m, axis=axis), axis),
                                      axis=axis).squeeze(axis), 0)
    g, lab = g.T, lab.T                      # 세로축을 이미지 행으로
    if flip_v: g, lab = g[::-1], lab[::-1]
    if flip_h: g, lab = g[:, ::-1], lab[:, ::-1]
    asp = [z[2] / z[1], z[2] / z[0], z[1] / z[0]][axis]
    ax.imshow(g, cmap="gray", vmin=0, vmax=1, aspect=asp, interpolation="bilinear")
    rgba = np.zeros(lab.shape + (4,))
    for n in ORDER:
        s = lab == TID[n]; rgba[s, :3] = COL[n]; rgba[s, 3] = 0.92
    ax.imshow(rgba, aspect=asp, interpolation="nearest")
    ax.set_title(title, color="w", fontsize=12.5, pad=7)
    ax.text(0.015, 0.5, left, transform=ax.transAxes, color="#dfe6ee", fontsize=11,
            va="center", fontweight="bold")
    ax.text(0.985, 0.5, right, transform=ax.transAxes, color="#dfe6ee", fontsize=11,
            va="center", ha="right", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#2b323b")

fig = plt.figure(figsize=(16, 7.6), facecolor="#0b0d10")
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.85, 1.25], wspace=0.05,
                      left=0.015, right=0.99, top=0.865, bottom=0.045)
# RAS 축: 0 = L->R, 1 = P->A, 2 = I->S
for i, (axis, fv, fh, t, l, r) in enumerate([
        (2, True,  False, "Axial  (from above)",  "R", "L"),
        (1, True,  False, "Coronal  (from front)", "R", "L"),
        (0, True,  False, "Sagittal  (from side)", "P", "A")]):
    a = fig.add_subplot(gs[i]); a.set_facecolor("#0b0d10")
    draw(a, axis, fv, fh, t, l, r)

ax = fig.add_subplot(gs[3]); ax.set_facecolor("#0b0d10"); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
y = 0.99
ax.text(0, y, "Classification accuracy by territory", color="w", fontsize=14,
        va="top", fontweight="bold"); y -= 0.062
ax.text(0, y, "submission config  ·  test+val  ·  5 seeds  ·  107 lesions",
        color="#8e99a8", fontsize=10, va="top"); y -= 0.055
for n in ORDER:
    acc, share, k = STAT[n]
    ax.add_patch(plt.Rectangle((0, y - 0.040), 0.048, 0.034, color=COL[n], clip_on=False))
    ax.text(0.068, y - 0.004, n, color="w", fontsize=12.5, va="top", fontweight="bold")
    ax.text(1.0, y - 0.004, f"{acc:.1f}%", color=COL[n], fontsize=13.5, va="top",
            ha="right", fontweight="bold")
    y -= 0.052
    ax.add_patch(plt.Rectangle((0.068, y - 0.028), 0.932, 0.019, color="#1d232b", clip_on=False))
    ax.add_patch(plt.Rectangle((0.068, y - 0.028), 0.932 * acc / 100, 0.019,
                               color=COL[n], clip_on=False))
    y -= 0.043
    ax.text(0.068, y, f"{share} of all errors   ·   {k} lesions",
            color="#aab4c2", fontsize=10, va="top")
    y -= 0.058

y -= 0.012
ax.plot([0, 1], [y, y], color="#2b323b", lw=1); y -= 0.045
for t, c in (("Posterior is worst (47.8%) - only 1-2 training", "w"),
             ("  samples per class, so no feature can build", "#aab4c2"),
             ("  a decision boundary there.", "#aab4c2"),
             ("ICA is the biggest prize (45% of all errors) -", "w"),
             ("  42 lesions over 7 sub-segments, but the", "#aab4c2"),
             ("  vessel label merges C6 and C7 into one.", "#aab4c2")):
    ax.text(0, y, t, color=c, fontsize=10.2, va="top",
            fontweight="bold" if c == "w" else "normal")
    y -= 0.036

fig.suptitle(f"Four vascular territories on a real CTA case  ·  {CID}",
             color="w", fontsize=16.5, y=0.955)
fig.text(0.015, 0.008,
         "GT vessel mask (36 classes) grouped into 4 territories, overlaid on a maximum-intensity projection.  "
         "Pcom / AChA / OA are counted with ICA because they define its sub-segments.",
         color="#5f6875", fontsize=8.6)
out = f"{R}/experiments/V1_vessel_axis/territories.png"
fig.savefig(out, dpi=130, facecolor="#0b0d10")
print("saved", out)
