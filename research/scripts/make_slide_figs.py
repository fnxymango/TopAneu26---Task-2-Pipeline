"""발표자료용 데이터 그림 생성 (2026-08-16).

한글 폰트가 없는 환경이라 **그림 안 라벨은 전부 라틴 문자**로 둔다
(혈관명 R-ICA-C6-C7, 지표명 MCC/Dice, 피처블록 dist/ov/bp 등 원래 라틴이라 자연스럽다).
한글 설명은 슬라이드 텍스트 상자에 둔다.

팔레트는 차분한 계열 — 웜 뉴트럴 지면 + 슬레이트 잉크 + 저채도 액센트.
"""
import json, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = Path(os.environ["TOPANEU_ROOT"])
A = R / "code" / "sblee" / "nnunet" / "analysis"
OUT = R / "code" / "sblee" / "outputs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

PAPER = "#F5F4F1"; INK = "#2A2E33"; MUTE = "#7A8189"; GRID = "#DEDCD7"
ACC = "#4F6B72"; OK = "#6A8A6E"; NG = "#A87A72"; HOLD = "#B0995F"; SOFT = "#C9CFD1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlecolor": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(" ", name)


# 1 ─ 손실 분해 -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 2.1))
segs = [("Achieved", .265, ACC), ("Detection gap", .069, HOLD), ("Classification gap", .666, NG)]
left = 0
for lab, v, c in segs:
    ax.barh(0, v, left=left, height=.5, color=c, edgecolor=PAPER, linewidth=2)
    ax.text(left + v / 2, 0, f"{v:.3f}", ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
    ax.text(left + v / 2, -.42, lab, ha="center", va="top", color=MUTE, fontsize=10.5)
    left += v
ax.set_xlim(0, 1); ax.set_ylim(-.75, .45); ax.set_yticks([]); ax.grid(False)
ax.set_xlabel("adjusted MCC  (val)", color=MUTE, labelpad=8)
ax.spines["left"].set_visible(False); ax.spines["bottom"].set_color(GRID)
save(fig, "fig_loss_decomp.png")

# 2 ─ 검출기 동작점 ---------------------------------------------------------
pts = [("A5-2 raw", 55, .721, SOFT), ("A6-2 raw", 152, .814, SOFT),
       ("2/2 consensus", 29, .698, HOLD), ("A5-2 + gate", 12, .721, OK),
       ("A6-2 + gate", 53, .814, OK)]
fig, ax = plt.subplots(figsize=(8, 4.6))
for lab, fp, se, c in pts:
    ax.scatter(fp, se, s=210, color=c, edgecolor=INK, linewidth=1.1, zorder=3)
    dy = .012 if lab != "2/2 consensus" else -.026
    ax.annotate(lab, (fp, se), xytext=(6, 10 if dy > 0 else -16),
                textcoords="offset points", fontsize=10, color=INK)
ax.annotate("", xy=(14, .712), xytext=(53, .712),
            arrowprops=dict(arrowstyle="->", color=ACC, lw=1.6))
ax.text(33, .705, "free post-processing", ha="center", va="top",
        fontsize=9.5, color=ACC, style="italic")
ax.set_xlabel("False positives (total, val 42 cases)"); ax.set_ylabel("Lesion sensitivity")
ax.set_xlim(0, 168); ax.set_ylim(.66, .855)
save(fig, "fig_detector_tradeoff.png")

# 3 ─ 피처 블록 ablation ----------------------------------------------------
rows = [("dist  (baseline)", .361, SOFT), ("+ ov", .379, OK), ("+ bp", .349, OK),
        ("+ pos  (C10)", .455, OK), ("+ rel  (C9)", .366, NG),
        ("+ geo  (C15)", .368, NG), ("+ enc  (C14)", .261, NG)]
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=.62, edgecolor="none")
for yi, (lab, v, c) in zip(y, rows):
    ax.text(v + .006, yi, f"{v:.3f}", va="center", fontsize=10.5,
            color=INK, fontweight="bold" if v == .455 else "normal")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
ax.axvline(.361, color=MUTE, lw=1, ls=(0, (4, 3)))
ax.set_xlim(0, .52); ax.set_xlabel("macro-recall  (patient-grouped 5-fold CV, 268 lesions)")
ax.grid(axis="y", visible=False)
save(fig, "fig_feature_ablation.png")

# 4 ─ 누적 개선 -------------------------------------------------------------
stages = ["C2 kNN\n(8/11)", "C5\nbranch-point", "+ β + C10\ncoords", "+ 5-fold\nvote2"]
vals = [.1587, .1739, .2038, .2207]
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(vals))
ax.plot(x, vals, color=ACC, lw=2.2, marker="o", ms=9, mfc=PAPER, mec=ACC, mew=2.2, zorder=3)
ax.fill_between(x, .14, vals, color=ACC, alpha=.09)
for xi, v in zip(x, vals):
    ax.annotate(f"{v:.4f}", (xi, v), xytext=(0, 13), textcoords="offset points",
                ha="center", fontsize=11, color=INK, fontweight="bold")
ax.axhline(.3259, color=NG, lw=1.3, ls=(0, (5, 3)))
ax.text(3.05, .3259, "  classification ceiling  0.3259", va="center",
        fontsize=9.5, color=NG, ha="left")
ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=10)
ax.set_ylim(.14, .35); ax.set_ylabel("official MCC  (test 83, held-out)")
ax.grid(axis="x", visible=False)
save(fig, "fig_progress.png")

# 5 ─ 6지표 레이더 ----------------------------------------------------------
labs = ["MCC", "Precision", "Recall", "Dice", "VolSim", "HD95\n(inverted)"]
series = [("single + filter", [.2038, .1720, .2540, .1331, .1499, 1 - .5887], SOFT),
          ("vote2", [.2207, .1955, .2962, .1286, .1456, 1 - .6490], HOLD),
          ("prob. average", [.2175, .1900, .2880, .1304, .1470, 1 - .5827], ACC)]
ang = np.linspace(0, 2 * np.pi, len(labs), endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(5.6, 5.2), subplot_kw=dict(polar=True))
for lab, v, c in series:
    vv = v + v[:1]
    ax.plot(ang, vv, color=c, lw=2, label=lab)
    ax.fill(ang, vv, color=c, alpha=.10)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labs, fontsize=10, color=INK)
ax.set_ylim(0, .48); ax.set_yticks([.1, .2, .3, .4])
ax.set_yticklabels([".1", ".2", ".3", ".4"], fontsize=8.5, color=MUTE)
ax.grid(color=GRID); ax.spines["polar"].set_color(GRID)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=3,
          frameon=False, fontsize=9.5, labelcolor=INK)
save(fig, "fig_radar.png")

# 6 ─ 혼동 흐름 (Pcom 흡인) --------------------------------------------------
srcs = [("L-3.5 AChA", 3), ("R-3.5 AChA", 3), ("R-3.6 nonBranch", 3),
        ("L-3.3 C6-nonOA", 2), ("L-3.6 nonBranch", 2), ("R-3.2 C6-OA", 2)]
fig, ax = plt.subplots(figsize=(8.6, 4.3))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(srcs) + .6)
for i, (lab, n) in enumerate(srcs):
    yy = len(srcs) - i - .3
    ax.text(3.15, yy, lab, ha="right", va="center", fontsize=11, color=INK)
    ax.add_patch(FancyArrowPatch((3.35, yy), (6.05, len(srcs) / 2),
                 arrowstyle="-|>", mutation_scale=13, color=NG,
                 alpha=.25 + .16 * n, lw=.9 + .55 * n, shrinkA=0, shrinkB=6))
    ax.text(3.5, yy + .16, f"{n}", fontsize=8.5, color=MUTE)
box = FancyBboxPatch((6.15, len(srcs) / 2 - .45), 3.3, .9,
                     boxstyle="round,pad=0.06", fc="#EDE7E4", ec=NG, lw=1.4)
ax.add_patch(box)
ax.text(7.8, len(srcs) / 2 + .1, "R/L-3.4  ICA C7-Pcom", ha="center", va="center",
        fontsize=12, color=INK, fontweight="bold")
ax.text(7.8, len(srcs) / 2 - .22, "n = 29  (most frequent ICA class)",
        ha="center", va="center", fontsize=9, color=MUTE)
ax.text(1.7, len(srcs) + .35, "rare ICA classes  (n = 2 ~ 5)", fontsize=10,
        color=MUTE, ha="center", style="italic")
save(fig, "fig_confusion_flow.png")

# 7 ─ c7 스윕 히트맵 (실데이터) ----------------------------------------------
d = json.load(open(A / "c7_detect_sweep_aneu_val.json"))
mv = sorted({r["min_vox"] for r in d["grid"]})
md = sorted({r["max_dist_mm"] for r in d["grid"]})
Z = np.zeros((len(mv), len(md))); S = np.zeros_like(Z)
for r in d["grid"]:
    Z[mv.index(r["min_vox"]), md.index(r["max_dist_mm"])] = r["fp_total"]
    S[mv.index(r["min_vox"]), md.index(r["max_dist_mm"])] = r["lesion_sensitivity"]
fig, ax = plt.subplots(figsize=(7.4, 3.8))
im = ax.imshow(Z, cmap="BuPu", aspect="auto", origin="lower")
for i in range(len(mv)):
    for j in range(len(md)):
        ax.text(j, i, f"{int(Z[i,j])}\n{S[i,j]:.2f}", ha="center", va="center",
                fontsize=9, color="white" if Z[i, j] > Z.max() * .55 else INK)
ax.set_xticks(range(len(md)))
ax.set_xticklabels(["1", "2", "3", "5", "∞"], color=MUTE)
ax.set_yticks(range(len(mv))); ax.set_yticklabels([str(v) for v in mv], color=MUTE)
ax.set_xlabel("max distance to vessel (mm)"); ax.set_ylabel("min component size (voxels)")
ax.set_title("A5-2  ·  FP total (upper) / lesion sensitivity (lower)",
             fontsize=10.5, color=MUTE, pad=10)
ax.grid(False)
ax.add_patch(plt.Rectangle((-.5 + 3, -.5 + 3), 1, 1, fill=False, ec=OK, lw=2.4))
save(fig, "fig_c7_heatmap.png")

# 8 ─ 52 vs 36 클래스 붕괴 ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8.6, 3.4))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4)
loc = ["3.2 C6-OA-junction", "3.3 C6-nonOA", "3.4 C7-Pcom-junction",
       "3.5 C7-AChA-junction", "3.6 C7-nonBranch", "3.7 C7-terminus"]
for i, l in enumerate(loc):
    yy = 3.62 - i * .58
    ax.text(3.55, yy, l, ha="right", va="center", fontsize=10.5, color=INK)
    ax.add_patch(FancyArrowPatch((3.7, yy), (6.0, 1.9), arrowstyle="-|>",
                 mutation_scale=11, color=MUTE, alpha=.55, lw=1.0, shrinkA=0, shrinkB=6))
b = FancyBboxPatch((6.1, 1.45), 3.2, .9, boxstyle="round,pad=0.06",
                   fc="#E4E8E9", ec=ACC, lw=1.5)
ax.add_patch(b)
ax.text(7.7, 2.05, "R-ICA-C6-C7", ha="center", fontsize=12.5, color=INK, fontweight="bold")
ax.text(7.7, 1.72, "single vessel class", ha="center", fontsize=9.5, color=MUTE)
ax.text(1.9, 3.98, "6 location classes  ×  R/L  =  12", fontsize=10.5, color=ACC,
        ha="center", fontweight="bold")
ax.text(1.9, .18, "117 / 397 lesions  (29%)", fontsize=10, color=MUTE, ha="center", style="italic")
save(fig, "fig_class_collapse.png")

print(f"\n[완료] {OUT}")
