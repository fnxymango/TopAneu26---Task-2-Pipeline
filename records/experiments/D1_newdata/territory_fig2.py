#!/usr/bin/env python3
"""세분화판 — 혈관 라벨이 허용하는 최대 세분도로 칠하고, 병변 세부구간별 정확도를 붙인다."""
import json
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
CID = "topaneu_center2_ct_161"
plt.rcParams["font.family"] = "DejaVu Sans"
VID = {k: int(v) for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}

# 혈관 라벨이 허용하는 만큼만 쪼갠다 — C6 와 C7 은 라벨이 하나라 못 쪼갠다
GRP = [
    ("ICA  C1-C5",            ["R-ICA-C1-C5","L-ICA-C1-C5"],              (0.86,0.16,0.16)),
    ("ICA  C6+C7  (merged!)", ["R-ICA-C6-C7","L-ICA-C6-C7"],              (1.00,0.45,0.38)),
    ("ICA  branches  OA/Pcom/AChA", ["R-OA","L-OA","R-Pcom","L-Pcom","R-AChA","L-AChA"], (1.00,0.62,0.72)),
    ("MCA  M1",  ["R-M1","L-M1"],                                          (1.00,0.72,0.05)),
    ("MCA  M2",  ["R-M2","L-M2"],                                          (0.98,0.85,0.35)),
    ("MCA  M3",  ["R-M3","L-M3"],                                          (0.96,0.94,0.68)),
    ("ACA  Acom",["Acom","3rd-A2","3rd-A3"],                               (0.13,0.82,0.42)),
    ("ACA  A1-A2",["R-A1A2","L-A1A2"],                                     (0.47,0.93,0.60)),
    ("ACA  A3",  ["R-A3","L-A3"],                                          (0.73,0.97,0.78)),
    ("Post.  VA", ["R-VA","L-VA"],                                         (0.16,0.80,0.90)),
    ("Post.  BA", ["BA"],                                                  (0.20,0.48,0.98)),
    ("Post.  PCA  P1-P4", ["R-P1P2","L-P1P2","R-P3P4","L-P3P4"],           (0.60,0.55,0.98)),
    ("Post.  cerebellar  PICA/AICA/SCA", ["R-PICA","L-PICA","R-AICA","L-AICA","R-SCA","L-SCA"], (0.36,0.72,0.75)),
]
lut = np.zeros(max(VID.values()) + 1, np.uint8)
for i, (_, vs, _) in enumerate(GRP, 1):
    for v in vs: lut[VID[v]] = i
assert sorted(v for _, vs, _ in GRP for v in vs) == sorted(k for k in VID if k != "background")

img = nib.as_closest_canonical(nib.load(f"{R}/experiments/_c1_realpred/in_test/{CID}_0000.nii.gz"))
ves = nib.as_closest_canonical(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{CID}.nii.gz"))
I = np.asanyarray(img.dataobj).astype(np.float32)
T = lut[np.clip(np.asanyarray(ves.dataobj).astype(np.int16), 0, lut.size - 1)]
nz = np.argwhere(T > 0)
sl = tuple(slice(a, b) for a, b in zip(np.maximum(nz.min(0)-12, 0),
                                       np.minimum(nz.max(0)+13, np.array(T.shape))))
I, T = I[sl], T[sl]
z = img.header.get_zooms()[:3]
p1, p99 = np.percentile(I, [45, 99.7]); I = np.clip((I-p1)/max(p99-p1, 1e-6), 0, 1)

def draw(ax, axis, title, left, right):
    g = I.max(axis=axis); m = T > 0
    lab = np.where(m.any(axis=axis),
                   np.take_along_axis(T, np.expand_dims(np.argmax(m, axis=axis), axis),
                                      axis=axis).squeeze(axis), 0)
    g, lab = g.T[::-1], lab.T[::-1]
    asp = [z[2]/z[1], z[2]/z[0], z[1]/z[0]][axis]
    ax.imshow(g, cmap="gray", vmin=0, vmax=1, aspect=asp, interpolation="bilinear")
    rgba = np.zeros(lab.shape + (4,))
    for i, (_, _, c) in enumerate(GRP, 1):
        s = lab == i; rgba[s, :3] = c; rgba[s, 3] = 0.93
    ax.imshow(rgba, aspect=asp, interpolation="nearest")
    ax.set_title(title, color="w", fontsize=12.5, pad=6)
    ax.text(0.015, 0.5, left, transform=ax.transAxes, color="#dfe6ee", fontsize=11,
            va="center", fontweight="bold")
    ax.text(0.985, 0.5, right, transform=ax.transAxes, color="#dfe6ee", fontsize=11,
            va="center", ha="right", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#2b323b")

# 병변 세부구간 정확도 (weakmap2.py 와 같은 집계)
BARS = [  # (영역, 라벨, 정확도%, 병변수, 오답비중, 색)
    ("Post.", "BA trunk & branches", 40.0, 7, "13%", (0.20,0.48,0.98)),
    ("Post.", "VA / PICA",           47.5, 8, "13%", (0.16,0.80,0.90)),
    ("ICA",   "C7  (Pcom/AChA/nonBranch)", 50.7, 15, "22%", (1.00,0.45,0.38)),
    ("ACA",   "A1",                  60.0, 3, "4%",  (0.47,0.93,0.60)),
    ("ACA",   "A2 and distal",       60.0, 5, "6%",  (0.73,0.97,0.78)),
    ("ICA",   "C6  (OA / nonOA)",    65.7, 14, "14%", (1.00,0.62,0.72)),
    ("Post.", "BA tip",              66.7, 3, "3%",  (0.60,0.55,0.98)),
    ("ICA",   "C7 terminus",         66.7, 3, "3%",  (0.86,0.16,0.16)),
    ("MCA",   "M1 trunk",            66.7, 3, "3%",  (1.00,0.72,0.05)),
    ("ICA",   "C1-C5",               82.0, 10, "5%", (0.86,0.16,0.16)),
    ("MCA",   "M1 bifurcation (5.2/5.3)", 83.3, 24, "12%", (0.98,0.85,0.35)),
    ("ACA",   "Acom complex",        95.0, 12, "2%", (0.13,0.82,0.42)),
]

fig = plt.figure(figsize=(17.5, 11.2), facecolor="#0b0d10")
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], width_ratios=[1, 1, 0.85, 1.05],
                      wspace=0.05, hspace=0.16, left=0.015, right=0.985, top=0.915, bottom=0.055)
for i, (axis, t, l, r) in enumerate([(2, "Axial  (from above)", "R", "L"),
                                     (1, "Coronal  (from front)", "R", "L"),
                                     (0, "Sagittal  (from side)", "P", "A")]):
    a = fig.add_subplot(gs[0, i]); a.set_facecolor("#0b0d10"); draw(a, axis, t, l, r)

lg = fig.add_subplot(gs[0, 3]); lg.set_facecolor("#0b0d10"); lg.axis("off")
lg.set_xlim(0, 1); lg.set_ylim(0, 1)
y = 1.0
lg.text(0, y, "Vessel groups  (13)", color="w", fontsize=13.5, va="top", fontweight="bold")
y -= 0.055
lg.text(0, y, "the finest split the vessel label allows", color="#8e99a8", fontsize=9.5, va="top")
y -= 0.058
for name, vs, c in GRP:
    lg.add_patch(plt.Rectangle((0, y-0.036), 0.045, 0.030, color=c, clip_on=False))
    bold = "(merged" in name
    lg.text(0.065, y-0.004, name, color="#ff8a7a" if bold else "#dde4ec",
            fontsize=10.6, va="top", fontweight="bold" if bold else "normal")
    y -= 0.062
y -= 0.02
lg.text(0, y, "The 36 vessel labels cannot separate", color="#ff8a7a", fontsize=10,
        va="top", fontweight="bold"); y -= 0.045
lg.text(0, y, "ICA C6 from C7 - they share one label.", color="#ff8a7a", fontsize=10, va="top")
y -= 0.045
lg.text(0, y, "But 5 lesion classes live inside it.", color="#aab4c2", fontsize=10, va="top")

ax = fig.add_subplot(gs[1, :]); ax.set_facecolor("#0b0d10")
ys = np.arange(len(BARS))[::-1]
for (terr, name, acc, n, share, c), yy in zip(BARS, ys):
    ax.barh(yy, 100, color="#171c23", height=0.66)
    ax.barh(yy, acc, color=c, height=0.66)
    ax.text(-1.2, yy, f"{terr}   {name}", color="#dde4ec", fontsize=11.3, va="center", ha="right")
    ax.text(acc+1.2, yy, f"{acc:.1f}%", color=c, fontsize=11.6, va="center", fontweight="bold")
    ax.text(106.5, yy, f"{n:>2d} lesions", color="#8e99a8", fontsize=10, va="center")
    ax.text(118.0, yy, f"{share:>4s} of errors", color="#8e99a8", fontsize=10, va="center")
ax.set_xlim(-34, 131); ax.set_ylim(-0.9, len(BARS)-0.1)
ax.axvline(69.0, color="#4a5361", lw=1.1, ls="--")
ax.text(69.3, len(BARS)-0.55, "overall 69.0%", color="#7d8794", fontsize=9.6, va="center")
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("Accuracy by lesion sub-segment  ·  submission config  ·  test+val  ·  5 seeds  ·  107 lesions  ·  535 seed-judgements",
             color="w", fontsize=13, pad=12, loc="left")

fig.suptitle(f"Cerebral vessels, finely split  ·  real CTA case {CID}",
             color="w", fontsize=17, y=0.972)
fig.text(0.015, 0.012,
         "Top: GT vessel mask (36 labels) grouped into 13 vessel groups, over a maximum-intensity projection.   "
         "Bottom: classification accuracy per lesion sub-segment (12 groups that actually carry detected lesions; PCA and M2/M3 carry none).",
         color="#5f6875", fontsize=8.8)
out = f"{R}/experiments/V1_vessel_axis/territories_fine.png"
fig.savefig(out, dpi=120, facecolor="#0b0d10")
print("saved", out)
