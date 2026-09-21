#!/usr/bin/env python3
"""공용 — 두 태그의 공식 개정 eval 7지표 비교표 (판정 보조용).

환경변수: BASE(기준 태그) · TAG(후보 태그) · SEEDS(콤마) · [LABEL_BASE] [LABEL_TAG]
출력: 시드평균 Δ · 시드별 개선지표 수 · 주요 지표의 평균±표준편차 · TP/FN/FP/TN 합계 변화
"""
import json
import os

import numpy as np

H = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/H1_patchfilter/scores"
M = ["PRECISION", "RECALL", "F1", "MCC", "DICE", "VOLSIM", "HD95"]
BASE = os.environ["BASE"]
TAG = os.environ["TAG"]
SEEDS = [int(x) for x in os.environ["SEEDS"].split(",")]
LB = os.environ.get("LABEL_BASE", BASE)
LT = os.environ.get("LABEL_TAG", TAG)


def g(t, sp, s):
    return json.load(open(f"{H}/{t}_{sp}_s{s}.json"))["new"]


def conf(t, sp, s):
    c = json.load(open(f"{H}/{t}_{sp}_s{s}.json"))["counts"]
    k = lambda p: sum(v for kk, v in c.items() if kk.startswith(p + "_"))
    return k("TP"), k("FN"), k("FP"), k("TN")


def better(m, d):
    return d < 0 if m == "HD95" else d > 0


def main():
    print(f"## 7지표 — `{LT}` − `{LB}` · 시드 {SEEDS} 평균 (HD95 는 낮을수록 개선)\n")
    print("| split | " + " | ".join(M) + " | 개선 |\n|" + "---|" * (len(M) + 2))
    for sp in ("test", "val"):
        d = {m: float(np.mean([g(TAG, sp, s)[m] - g(BASE, sp, s)[m] for s in SEEDS])) for m in M}
        print(f"| {sp} | " + " | ".join(f"{d[m]:+.4f}" for m in M) +
              f" | {sum(better(m, d[m]) for m in M)}/7 |")

    print("\n### 시드별 개선 지표 수 (7 중)\n")
    print("| split | " + " | ".join(f"s{s}" for s in SEEDS) + " |\n|" + "---|" * (len(SEEDS) + 1))
    for sp in ("test", "val"):
        row = []
        for s in SEEDS:
            row.append(str(sum(better(m, g(TAG, sp, s)[m] - g(BASE, sp, s)[m]) for m in M)))
        print(f"| {sp} | " + " | ".join(row) + " |")

    print("\n### 주요 지표 평균±표준편차 · 후보가 나은 시드 수\n")
    print("| split | 지표 | 기준 | 후보 | Δ | 후보 승 |\n|---|---|---|---|---|---|")
    for sp in ("test", "val"):
        for m in ("MCC", "F1", "RECALL", "PRECISION"):
            a = np.array([g(BASE, sp, s)[m] for s in SEEDS])
            b = np.array([g(TAG, sp, s)[m] for s in SEEDS])
            sd = lambda v: v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"| {sp} | {m} | {a.mean():.4f}±{sd(a):.4f} | {b.mean():.4f}±{sd(b):.4f} | "
                  f"{b.mean()-a.mean():+.4f} | {(b>a).sum()}/{len(SEEDS)} |")

    print("\n### 존재 기반 합계 (시드평균)\n")
    print("| split | TP | FN | FP | TN |\n|---|---|---|---|---|")
    for sp in ("test", "val"):
        a = np.array([conf(BASE, sp, s) for s in SEEDS], float).mean(0)
        b = np.array([conf(TAG, sp, s) for s in SEEDS], float).mean(0)
        print(f"| {sp} | {a[0]:.1f} → **{b[0]:.1f}** ({b[0]-a[0]:+.1f}) | {a[1]:.1f} → {b[1]:.1f} ({b[1]-a[1]:+.1f}) "
              f"| {a[2]:.1f} → {b[2]:.1f} ({b[2]-a[2]:+.1f}) | {a[3]:.1f} → {b[3]:.1f} ({b[3]-a[3]:+.1f}) |")


if __name__ == "__main__":
    main()
