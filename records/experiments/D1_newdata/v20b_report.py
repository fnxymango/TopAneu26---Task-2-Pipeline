#!/usr/bin/env python3
"""V2-0b 보고서 — 판정규칙은 v20b_icasplit.py 머리말에 결과 보기 전에 고정."""
import json, os
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
rec = json.load(open(f"{D}/v20b_icasplit.json"))
JS = (0.40, 0.75)


def dice(p, g, k):
    a, b = p == k, g == k
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def score(t, g, c1, c2):
    p = np.where(t < c1, 0, np.where(t < c2, 1, 2))
    return [dice(p, g, k) for k in (0, 1, 2)]


print("# V2-0b — ICA C6/C7/terminus 분할 기준 (축 버그 수정 · 분지 재정의)\n")
print(f"fine GT side {len(rec)} (train·val 만 · test 10케이스는 열지 않음)\n")

# ── 0. 축 건전성 ───────────────────────────────────────────────────────────
inv = [r for r in rec if r.get("gt_c6_end") and r.get("gt_c7_end")
       and r["gt_c6_end"] > r["gt_c7_end"]]
print(f"## 0. 축 건전성\n\n뒤집힌 side **{len(inv)}/{len(rec)}** (V2-0 에서는 4/42)\n")
for r in inv:
    print(f"  ★{r['case']} {r['side']}  C6끝 {r['gt_c6_end']:.3f} > C7끝 {r['gt_c7_end']:.3f}")
if inv:
    print("\n→ **축 버그가 남아 있다. 판정하지 않는다.**")

good = [r for r in rec if r not in inv]
c6 = np.array([r["gt_c6_end"] for r in good if r.get("gt_c6_end") is not None])
c7 = np.array([r["gt_c7_end"] for r in good if r.get("gt_c7_end") is not None])
print(f"\n수작업 경계  C6끝 중앙 {np.median(c6):.3f} ({c6.min():.3f}~{c6.max():.3f})"
      f"  ·  C7끝 중앙 {np.median(c7):.3f} ({c7.min():.3f}~{c7.max():.3f})")

# ── 1. 분지 위치 ───────────────────────────────────────────────────────────
print("\n## 1. 분지 위치 (이 21케이스)\n")
print("| 분지 | 잡힌 side | t 중앙 | 수작업 경계와 차이 |")
print("|---|---|---|---|")
for key, nm, ref in (("t_OA", "OA", None), ("t_Pcom", "Pcom", float(np.median(c6))),
                     ("t_AChA", "AChA", None), ("t_A1", "A1 기시", float(np.median(c7))),
                     ("t_M1", "M1 기시", None)):
    v = np.array([r[key] for r in good if r.get(key) is not None])
    if not len(v):
        print(f"| {nm} | 0 | - | - |"); continue
    d = "" if ref is None else f"{np.median(v)-ref:+.3f}"
    print(f"| {nm} | {len(v)}/{len(good)} = {len(v)/len(good):.0%} | {np.median(v):.3f} | {d} |")

# ── 2. 세 분할 ─────────────────────────────────────────────────────────────
tr = [r for r in good if r["split"] == "train"]
c1_tr = float(np.median([r["gt_c6_end"] for r in tr if r.get("gt_c6_end") is not None]))
c2_tr = float(np.median([r["gt_c7_end"] for r in tr if r.get("gt_c7_end") is not None]))
print(f"\n## 2. 세 분할 비교\n")
print(f"(C) train 재추정 절단점 = **{c1_tr:.3f} / {c2_tr:.3f}**"
      f"  (jskim 0.400/0.750 · 차이 {c1_tr-JS[0]:+.3f} / {c2_tr-JS[1]:+.3f})\n")

res = {"A jskim 0.40/0.75": [], "B 분지기준(Pcom·A1)": [], "C train 재추정": []}
nfull = 0
for r in good:
    f = f"{D}/v20b_cache/{r['case']}_{r['side']}"
    if not os.path.exists(f + "_t.npy"):
        continue
    t = np.load(f + "_t.npy"); g = np.load(f + "_g.npy")
    m = g >= 0
    t, g = t[m], g[m]
    res["A jskim 0.40/0.75"].append(score(t, g, *JS))
    res["C train 재추정"].append(score(t, g, c1_tr, c2_tr))
    # B — 절단별 폴백: 그 분지가 없을 때만 그 절단만 백분위로 대체
    p, a = r.get("t_Pcom"), r.get("t_A1")
    b1 = p if p is not None else c1_tr
    b2 = a if a is not None else c2_tr
    if p is not None and a is not None and b1 < b2:
        nfull += 1
    if b1 >= b2:
        b1, b2 = c1_tr, c2_tr
    res["B 분지기준(Pcom·A1)"].append(score(t, g, b1, b2))

print("| 분할 | side | C6 Dice | C7 Dice | terminus Dice | 평균 |")
print("|---|---|---|---|---|---|")
avg = {}
for k, v in res.items():
    if not v:
        continue
    a = np.nanmean(np.array(v), axis=0)
    avg[k] = float(np.nanmean(a))
    print(f"| {k} | {len(v)} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} | **{np.nanmean(a):.3f}** |")
cov = nfull / max(len(good), 1)
print(f"\n두 절단 모두 분지로 결정된 side {nfull}/{len(good)} = {cov:.0%}")

# ── 3. 판정 ───────────────────────────────────────────────────────────────
print("\n## 3. 판정\n")
if inv:
    print("축 뒤집힘이 남아 측정 실패. 판정 보류."); raise SystemExit(0)
A = avg["A jskim 0.40/0.75"]; B = avg["B 분지기준(Pcom·A1)"]; C = avg["C train 재추정"]
print(f"1. B({B:.3f}) ≥ A({A:.3f}) ? → {'예' if B >= A else '아니오'}")
print(f"2. B 가 A 보다 0.05 이상 낮은가? → {'예' if (A-B) >= 0.05 else '아니오'}")
print(f"3. 분지 커버리지 {cov:.0%} ≥ 40% ? → {'예' if cov >= 0.40 else '아니오'}")
leak = abs(c1_tr-JS[0]) > 0.05 or abs(c2_tr-JS[1]) > 0.05
print(f"4. C 절단점이 0.40/0.75 와 ±0.05 초과 차이? → {'예 · 누수 영향 실재, A 사용 금지' if leak else '아니오'}")
pick = "B 분지기준(Pcom·A1)" if (B >= A and cov >= 0.40) else "C train 재추정"
print(f"\n**→ V2-A 에서 쓸 분할 규칙: {pick}**")
if leak and pick == "A jskim 0.40/0.75":
    print("⚠ A 는 누수로 금지. 차선을 쓴다.")
