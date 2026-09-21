#!/usr/bin/env python3
"""M1 길이 전수조사 판정 — 게이트는 m1census.py 머리말에 결과 보기 전에 고정.

  1. AUC ≥ 0.90            단일 축으로 갈린다
  2. 겹침 구간 표본 < 20%   회색지대가 좁다
  3. 좌우 각각 성립         R·L 각각 AUC ≥ 0.90
  셋 다 만족 → V1-A 착수 · 하나라도 미달 → 이 축을 닫는다
"""
import json, re, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
cen = json.load(open(f"{D}/m1census.json"))
rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]

L = np.array([v["len_mm"] for v in cen.values()])
print("# M1 길이 전수조사 — 5.2 early bifurcation ↔ 5.3 M1-M2 junction\n")
print(f"## 1. 모집단 분포 (train {len(cen)} side · 라벨 안 봄)\n")
print(f"| n | 중앙 | 평균±표준편차 | 10% | 25% | 75% | 90% | 범위 |")
print(f"|---|---|---|---|---|---|---|---|")
print(f"| {len(L)} | {np.median(L):.1f}mm | {L.mean():.1f}±{L.std():.1f} | "
      f"{np.percentile(L,10):.1f} | {np.percentile(L,25):.1f} | {np.percentile(L,75):.1f} | "
      f"{np.percentile(L,90):.1f} | {L.min():.1f}~{L.max():.1f} |")

# 병변 ↔ 같은쪽 M1 길이
def side_of(n):
    return n[0] if n and n[0] in "RL" else None
grp = collections.defaultdict(list)
miss = 0
for r in rows:
    n = r["gt_loc"]
    if "M1 early bifurcation" in n: k = "5.2 early"
    elif "M1-M2 junction" in n:     k = "5.3 M1-M2"
    else: continue
    s = side_of(n)
    v = cen.get(f"{r['case']}|{s}-")
    if v is None: miss += 1; continue
    grp[k].append((v["len_mm"], s, r["case"]))

print(f"\n## 2. 병변별 같은쪽 M1 길이 (측정 실패 {miss}건 제외)\n")
print("| 집단 | n | 중앙 | 평균±표준편차 | 범위 | 모집단 백분위 중앙 |")
print("|---|---|---|---|---|---|")
for k in ("5.2 early", "5.3 M1-M2"):
    v = np.array([x[0] for x in grp[k]])
    if not len(v): continue
    pct = np.array([(L < x).mean() for x in v])
    print(f"| {k} | {len(v)} | {np.median(v):.1f}mm | {v.mean():.1f}±{v.std():.1f} | "
          f"{v.min():.1f}~{v.max():.1f} | {np.median(pct):.0%} |")

def auc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b): return float("nan")
    return float(sum((x < y) + 0.5 * (x == y) for x in a for y in b) / (len(a) * len(b)))

E = [x[0] for x in grp["5.2 early"]]; J = [x[0] for x in grp["5.3 M1-M2"]]
au = auc(E, J); au = max(au, 1 - au)
# 겹침 구간
lo, hi = max(min(E), min(J)), min(max(E), max(J))
ov = sum(lo <= x <= hi for x in E + J) / max(len(E) + len(J), 1)
print(f"\n## 3. 게이트\n")
print("| 게이트 | 기준 | 실측 | 판정 |")
print("|---|---|---|---|")
print(f"| 1. 분리력 | AUC ≥ 0.90 | **{au:.3f}** | {'통과' if au>=0.90 else '**미달**'} |")
print(f"| 2. 회색지대 | 겹침 표본 < 20% | **{ov:.0%}** ({lo:.1f}~{hi:.1f}mm) | {'통과' if ov<0.20 else '**미달**'} |")
side_ok = True
for s in ("R", "L"):
    e = [x[0] for x in grp["5.2 early"] if x[1] == s]
    j = [x[0] for x in grp["5.3 M1-M2"] if x[1] == s]
    a = auc(e, j); a = max(a, 1 - a) if len(e) and len(j) else float("nan")
    ok = (a >= 0.90) if a == a else False
    side_ok &= ok
    print(f"| 3. {s} 측 | AUC ≥ 0.90 | **{a:.3f}** (n={len(e)}/{len(j)}) | {'통과' if ok else '**미달**'} |")
allok = au >= 0.90 and ov < 0.20 and side_ok
print(f"\n**판정 → {'세 게이트 전부 통과 · V1-A 착수' if allok else '미달 · **이 축을 닫는다**'}**")
print("\n참고 · 기존 피처 중 최강(혈관중첩 M2) AUC 0.943, "
      "기존 '종말부까지 정규화거리' 는 5.2 표본의 57%가 5.3 범위 안에 있었다.")
