#!/usr/bin/env python3
"""FRAC 보고 — gC 2등 조각 복셀 지분 스윕 결과표 (판정은 FRAC.sh 머리말 고정 규칙).

기준 b1on_pf(= FRAC 0) 대비 각 지분값의 7지표 평균 Δ 와, 존재기반 4지표 불변 여부를 찍는다.
불변 검사는 코드 가정 검증이다 — TP 성립에 1복셀이면 충분하므로 지분을 바꿔도
PRECISION·RECALL·F1·MCC 는 정의상 변하지 않아야 한다. 깨지면 가정이 틀린 것이다.
"""
import json
import os
import sys

import numpy as np

H = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/H1_patchfilter/scores"
M = ["PRECISION", "RECALL", "F1", "MCC", "DICE", "VOLSIM", "HD95"]
EXIST = ["PRECISION", "RECALL", "F1", "MCC"]   # 정의상 불변이어야 하는 것
SHAPE = ["DICE", "VOLSIM"]                      # 지분에 비례해 개선되어야 하는 것
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
TAGS = [t for t in os.environ["TAGS"].split(",") if t]     # 예: b1frac035,b1frac050,b1frac065
LABELS = os.environ["LABELS"].split(",")                    # 예: 0.35,0.50,0.65
BASE = os.environ.get("BASE", "b1on_pf")


def g(tag, sp, s):
    return json.load(open(f"{H}/{tag}_{sp}_s{s}.json"))["new"]


def counts(tag, sp, s):
    c = json.load(open(f"{H}/{tag}_{sp}_s{s}.json"))["counts"]
    k = lambda p: sum(v for kk, v in c.items() if kk.startswith(p + "_"))
    return k("TP"), k("FN"), k("FP"), k("TN")


def main():
    print("# FRAC — gC 2등 조각 복셀 지분 스윕 (판정 규칙은 FRAC.sh 머리말 고정)\n")
    print(f"기준 `{BASE}` (= 지분 3복셀 고정 · FRAC 0) · 시드 {SEEDS} · 공식 개정 eval 7지표\n")

    print("## 1) 7지표 평균 Δ (기준 대비 · HD95 는 낮을수록 개선)\n")
    print("| 지분 | split | " + " | ".join(M) + " | 개선 |")
    print("|---|---|" + "---|" * (len(M) + 1))
    rows = {}
    for tag, lab in zip(TAGS, LABELS):
        for sp in ("test", "val"):
            try:
                d = {m: float(np.mean([g(tag, sp, s)[m] - g(BASE, sp, s)[m] for s in SEEDS])) for m in M}
            except FileNotFoundError as e:
                print(f"| {lab} | {sp} | 채점본 없음: {os.path.basename(str(e).split(chr(39))[-2])} |")
                continue
            imp = sum((d[m] > 0) if m != "HD95" else (d[m] < 0) for m in M)
            rows[(lab, sp)] = d
            print(f"| {lab} | {sp} | " + " | ".join(f"{d[m]:+.4f}" for m in M) + f" | {imp}/7 |")

    print("\n## 2) 존재기반 4지표 불변 검사 (시드별 최대 절대 편차)\n")
    print("자유도 없음 가정의 검증이다. 0 이 아니면 가정이 틀렸다는 뜻이므로 판정을 멈춘다.\n")
    print("| 지분 | split | " + " | ".join(EXIST) + " | TP/FP 변화 | 판정 |")
    print("|---|---|" + "---|" * (len(EXIST) + 2))
    ok_inv = {}
    for tag, lab in zip(TAGS, LABELS):
        for sp in ("test", "val"):
            try:
                mx = {m: max(abs(g(tag, sp, s)[m] - g(BASE, sp, s)[m]) for s in SEEDS) for m in EXIST}
                dtp = float(np.mean([counts(tag, sp, s)[0] - counts(BASE, sp, s)[0] for s in SEEDS]))
                dfp = float(np.mean([counts(tag, sp, s)[2] - counts(BASE, sp, s)[2] for s in SEEDS]))
            except FileNotFoundError:
                continue
            good = all(v < 1e-9 for v in mx.values())
            ok_inv[(lab, sp)] = good
            print(f"| {lab} | {sp} | " + " | ".join(f"{mx[m]:.2e}" for m in EXIST) +
                  f" | ΔTP {dtp:+.1f} · ΔFP {dfp:+.1f} | {'불변' if good else '★깨짐'} |")

    print("\n## 3) 판정\n")
    print("| 지분 | DICE·VOLSIM 양쪽 개선 | 4지표 불변 | HD95 악화 ≤ +3 | 종합 |")
    print("|---|---|---|---|---|")
    passed = []
    for lab in LABELS:
        if (lab, "test") not in rows or (lab, "val") not in rows:
            continue
        shape_ok = all(rows[(lab, sp)][m] > 0 for sp in ("test", "val") for m in SHAPE)
        inv_ok = ok_inv.get((lab, "test"), False) and ok_inv.get((lab, "val"), False)
        hd_ok = all(rows[(lab, sp)]["HD95"] <= 3.0 for sp in ("test", "val"))
        both = shape_ok and inv_ok and hd_ok
        if both:
            passed.append(lab)
        print(f"| {lab} | {'○' if shape_ok else '✗'} | {'○' if inv_ok else '✗'} | "
              f"{'○' if hd_ok else '✗'} | {'통과' if both else '미달'} |")

    print()
    if not passed:
        print("**판정 → 통과한 지분값 없음 · FRAC 축 닫음**")
    else:
        # 값 선택은 val 로만 한다 (test 로 고르지 않는다 · 사전 고정)
        best = max(passed, key=lambda L: rows[(L, "val")]["DICE"] + rows[(L, "val")]["VOLSIM"])
        print(f"**판정 → 통과 {passed} · val 기준 선택값 = {best}**")
        print(f"\n선택은 val 의 DICE+VOLSIM 합으로만 했다(test 미사용). "
              f"val Δ: DICE {rows[(best,'val')]['DICE']:+.4f} · VOLSIM {rows[(best,'val')]['VOLSIM']:+.4f} "
              f"· HD95 {rows[(best,'val')]['HD95']:+.2f}")
        print(f"\n참고(선택 미사용) test Δ: DICE {rows[(best,'test')]['DICE']:+.4f} · "
              f"VOLSIM {rows[(best,'test')]['VOLSIM']:+.4f} · HD95 {rows[(best,'test')]['HD95']:+.2f}")
        print(f"\n다음: 시드 5~9 복제에서 같은 조건을 만족해야 채택 권고(반영은 사용자 결정).")


if __name__ == "__main__":
    sys.exit(main())
