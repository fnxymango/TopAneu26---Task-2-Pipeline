#!/usr/bin/env python3
"""V2-0 보고서 — 세 가지 ICA 분할 기준을 수작업 fine GT 와 대조한다.

판정규칙은 v20_icasplit.py 머리말에 결과 보기 전에 고정.
**test 10건은 fine GT 가 있어도 보지 않는다** — 보는 순간 우리도 오염된다.
"""
import json, os, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
rec = json.load(open(f"{D}/v20_icasplit.json"))
JS = (0.40, 0.75)


def dice(p, g, k):
    a = p == k; b = g == k
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan


def score(t, g, c1, c2):
    p = np.where(t < c1, 0, np.where(t < c2, 1, 2))
    return [dice(p, g, k) for k in (0, 1, 2)]


fine = [r for r in rec if r.get("has_fine")]
allow = [r for r in fine if r["split"] in ("train", "val")]
print(f"# V2-0 — ICA C6/C7/terminus 분할 기준\n")
print(f"fine GT side {len(fine)} · 그중 **볼 수 있는 것(train·val) {len(allow)}** · "
      f"test {len(fine)-len(allow)} 는 열지 않음\n")

# ── 0. 이 파일들이 수작업인가 규칙 산출물인가 ──────────────────────────────
print("## 0. fine GT 가 수작업인지 규칙 산출물인지\n")
print("각 side 에서 수작업 경계를 t 로 환산한 값. 전부 0.40/0.75 면 규칙으로 찍어낸 것이다.\n")
print("| split | 케이스 | side | C6 끝 t | C7 끝 t |")
print("|---|---|---|---|---|")
for r in sorted(allow, key=lambda x: (x["case"], x["side"])):
    a, b = r.get("gt_c6_end"), r.get("gt_c7_end")
    print(f"| {r['split']} | {r['case']} | {r['side']} | "
          f"{a:.3f} | {b:.3f} |" if a is not None and b is not None else
          f"| {r['split']} | {r['case']} | {r['side']} | - | - |")
ce = np.array([r["gt_c6_end"] for r in allow if r.get("gt_c6_end") is not None])
c7 = np.array([r["gt_c7_end"] for r in allow if r.get("gt_c7_end") is not None])
if len(ce):
    print(f"\nC6 끝 t   중앙 {np.median(ce):.3f}  범위 {ce.min():.3f}~{ce.max():.3f}  (jskim 고정 0.400)")
    print(f"C7 끝 t   중앙 {np.median(c7):.3f}  범위 {c7.min():.3f}~{c7.max():.3f}  (jskim 고정 0.750)")
    if ce.std() < 1e-6 and c7.std() < 1e-6:
        print("\n→ **산포 0. 수작업이 아니라 0.40/0.75 규칙을 찍어낸 파일이다.**")
    else:
        print("\n→ 산포가 있다. 수작업 경계다.")

# ── 1. 분지 위치 전수 분포 (라벨 불필요) ───────────────────────────────────
print("\n## 1. 분지가 실제로 t 어디에 있나 — train 전수 (라벨 안 봄)\n")
tr = [r for r in rec if r["split"] == "train"]
print("| 분지 | 잡힌 side | t 중앙 | 25~75% | jskim 절단점과 비교 |")
print("|---|---|---|---|---|")
for key, nm, ref in (("t_OA", "OA (안동맥)", None), ("t_Pcom", "Pcom", 0.40),
                     ("t_AChA", "AChA", 0.75), ("t_M1", "M1 (종말)", 1.0),
                     ("t_A1", "A1 (종말)", 1.0)):
    v = np.array([r[key] for r in tr if r.get(key) is not None])
    if not len(v):
        print(f"| {nm} | 0 | - | - | - |"); continue
    cmp = "" if ref is None else f"{np.median(v)-ref:+.3f}"
    print(f"| {nm} | {len(v)}/{len(tr)} = {len(v)/len(tr):.0%} | {np.median(v):.3f} | "
          f"{np.percentile(v,25):.3f}~{np.percentile(v,75):.3f} | {cmp} |")

# ── 2. 세 분할의 Dice ──────────────────────────────────────────────────────
print("\n## 2. 세 분할 비교 (train·val 10케이스에서만)\n")
tr_fine = [r for r in allow if r["split"] == "train"]
c1_tr = float(np.median([r["gt_c6_end"] for r in tr_fine if r.get("gt_c6_end") is not None])) if tr_fine else JS[0]
c2_tr = float(np.median([r["gt_c7_end"] for r in tr_fine if r.get("gt_c7_end") is not None])) if tr_fine else JS[1]
print(f"(C) train 재추정 절단점 = **{c1_tr:.3f} / {c2_tr:.3f}**  (jskim 0.400/0.750 · "
      f"차이 {c1_tr-JS[0]:+.3f} / {c2_tr-JS[1]:+.3f})\n")

res = collections.defaultdict(list)
nbranch = 0
for r in allow:
    f = f"{D}/v20_cache/{r['case']}_{r['side']}"
    if not os.path.exists(f + "_t.npy"):
        continue
    t = np.load(f + "_t.npy"); g = np.load(f + "_g.npy")
    m = g >= 0
    t, g = t[m], g[m]
    res["A jskim 0.40/0.75"].append(score(t, g, *JS))
    res["C train 재추정"].append(score(t, g, c1_tr, c2_tr))
    p, a = r.get("t_Pcom"), r.get("t_AChA")
    if p is not None and a is not None and p < a:
        nbranch += 1
        res["B 분지기준"].append(score(t, g, p, a))
    else:
        res["B 분지기준(폴백)"].append(score(t, g, c1_tr, c2_tr))

print("| 분할 | side | C6 Dice | C7 Dice | terminus Dice | 평균 |")
print("|---|---|---|---|---|---|")
avg = {}
for k in ("A jskim 0.40/0.75", "C train 재추정", "B 분지기준", "B 분지기준(폴백)"):
    v = res.get(k)
    if not v:
        continue
    a = np.nanmean(np.array(v), axis=0)
    avg[k] = float(np.nanmean(a))
    print(f"| {k} | {len(v)} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} | **{np.nanmean(a):.3f}** |")

# B 전체(분지 있으면 분지, 없으면 폴백)
allB = res.get("B 분지기준", []) + res.get("B 분지기준(폴백)", [])
if allB:
    a = np.nanmean(np.array(allB), axis=0)
    avg["B 전체"] = float(np.nanmean(a))
    print(f"| **B 전체(분지+폴백)** | {len(allB)} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} | **{np.nanmean(a):.3f}** |")

print(f"\n분지(Pcom·AChA 둘 다)가 잡힌 side {nbranch}/{len(allow)} = {nbranch/max(len(allow),1):.0%}")

# ── 3. 판정 ───────────────────────────────────────────────────────────────
print("\n## 3. 판정 (v20_icasplit.py 머리말 규칙)\n")
A = avg.get("A jskim 0.40/0.75", np.nan); B = avg.get("B 전체", np.nan); C = avg.get("C train 재추정", np.nan)
cov = nbranch / max(len(allow), 1)
print(f"1. B({B:.3f}) ≥ A({A:.3f}) ? → {'예 · 분지기준 채택' if B >= A else '아니오'}")
print(f"2. B 가 A 보다 0.05 이상 낮은가? → {'예 · C 로 간다' if (A-B) >= 0.05 else '아니오'}")
print(f"3. 분지 커버리지 {cov:.0%} ≥ 40% ? → {'예' if cov >= 0.40 else '아니오 · C 로 간다'}")
print(f"4. C 절단점이 0.40/0.75 와 ±0.05 초과 차이? → "
      f"{'예 · 누수 영향 실재, 앞으로 A 사용 금지' if (abs(c1_tr-JS[0])>0.05 or abs(c2_tr-JS[1])>0.05) else '아니오'}")
pick = "B 분지기준" if (B >= A and cov >= 0.40) else "C train 재추정"
print(f"\n**→ V2-A 에서 쓸 분할 규칙: {pick}**")
