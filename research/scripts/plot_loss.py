#!/usr/bin/env python3
"""
Plot train & val loss (+ EMA pseudo-Dice) for one experiment.

Usage: plot_loss.py <EXP_DIR>
Primary source : nnU-Net training_log_*.txt  (available from epoch 0)
Fallback       : checkpoint_*.pth['logging']  (if log parse is empty)
Outputs        : <EXP_DIR>/loss_curve.png , <EXP_DIR>/metrics.json
Safe to re-run anytime (live monitoring).
"""
import sys, re, json, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

exp = Path(sys.argv[1])
# config.json의 fold를 우선 사용. 5-fold를 한 폴더에 합치면 glob 마지막이 fold_4가 되어
# 엉뚱한 fold의 곡선을 그리게 된다.
_f = None
try: _f = json.load(open(exp / "config.json")).get("fold")
except Exception: pass
fold_dirs = (sorted(exp.glob(f"results/*/*/fold_{_f}")) if _f is not None else []) \
            or sorted(exp.glob("results/*/*/fold_*"))
if not fold_dirs:
    print(f"[plot_loss] no fold dir under {exp}/results yet; skip"); sys.exit(0)
fold = fold_dirs[-1]

# ---- primary: parse training log ----
ep, tr, va, dice = [], [], [], []
logs = sorted(fold.glob("training_log_*.txt"),
              key=lambda f: open(f, errors="ignore").read().count("train_loss"))
if logs:
    cur = None
    per = {}
    for line in open(logs[-1], errors="ignore"):
        m = re.search(r":\s*Epoch (\d+)\s*$", line)
        if m: cur = int(m.group(1)); per.setdefault(cur, {}); continue
        if cur is None: continue
        m = re.search(r":\s*train_loss\s+(-?[\d.]+)", line)
        if m: per[cur]["tr"] = float(m.group(1)); continue
        m = re.search(r":\s*val_loss\s+(-?[\d.]+)", line)
        if m: per[cur]["va"] = float(m.group(1)); continue
        m = re.search(r"Pseudo dice\s+\[([^\]]*)\]", line)
        if m:
            raw = re.findall(r"np\.float\d+\(([^)]+)\)", m.group(1)) or re.findall(r"-?\d+\.?\d*", m.group(1))
            vals = [float(x) for x in raw]
            vals = [v for v in vals if v == v]   # drop nan
            if vals: per[cur]["dice"] = sum(vals) / len(vals)
    for e in sorted(per):
        d = per[e]
        if "tr" in d and "va" in d:
            ep.append(e); tr.append(d["tr"]); va.append(d["va"]); dice.append(d.get("dice"))

# ---- fallback: checkpoint logging dict ----
if not ep:
    for name in ("checkpoint_final.pth", "checkpoint_best.pth", "checkpoint_latest.pth"):
        ck = fold / name
        if ck.exists():
            import torch
            L = torch.load(ck, map_location="cpu", weights_only=False).get("logging", {})
            tr = list(L.get("train_losses", [])); va = list(L.get("val_losses", []))
            dice = list(L.get("ema_fg_dice", [])) or [None] * len(tr)
            ep = list(range(len(tr)))
            break

if not ep:
    print(f"[plot_loss] no metrics parsed for {exp.name} yet; skip"); sys.exit(0)

# ---- plot ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, tr, color="#1f77b4", lw=1.8, label="train loss")
ax.plot(ep, va, color="#d62728", lw=1.8, label="val loss")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.grid(alpha=.3)
best_i = min(range(len(va)), key=lambda i: va[i])
ax.scatter([ep[best_i]], [va[best_i]], color="#d62728", zorder=5)
ax.annotate(f"best val {va[best_i]:.4f}@{ep[best_i]}",
            (ep[best_i], va[best_i]), textcoords="offset points", xytext=(5, 8), fontsize=8)
if any(d is not None for d in dice):
    ax2 = ax.twinx()
    xs = [e for e, d in zip(ep, dice) if d is not None]
    ys = [d for d in dice if d is not None]
    ax2.plot(xs, ys, color="#2ca02c", lw=1.2, ls="--", alpha=.8, label="EMA pseudo-Dice")
    ax2.set_ylabel("pseudo-Dice", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
lines, labs = ax.get_legend_handles_labels()
ax.legend(lines, labs, loc="upper right", fontsize=9)
ax.set_title(f"{exp.name}  (epochs: {len(ep)})")
fig.tight_layout()
out = exp / "loss_curve.png"
fig.savefig(out, dpi=130); plt.close(fig)

json.dump({"epochs": ep, "train_loss": tr, "val_loss": va,
           "ema_pseudo_dice": dice,
           "best_val_loss": va[best_i], "best_val_epoch": ep[best_i],
           "last_epoch": ep[-1]},
          open(exp / "metrics.json", "w"), indent=2)
print(f"[plot_loss] {out}  (epochs {ep[0]}..{ep[-1]}, best val {va[best_i]:.4f}@{ep[best_i]})")
