#!/usr/bin/env python3
"""h1_abs.py <scores_dir> — 팔별 5시드 평균 절대값. 신 eval 6지표 + 구 eval ÷52 + covered_gt.

covered_gt 는 구 eval(÷52)을 그 split 의 GT 등장 클래스 수로 환산한 것이다
(test 36 · val 33). P·R·MCC·DICE·VOLSIM 은 정확히 비례하므로 환산이 정당하고,
HD95 는 비례하지 않으므로 환산하지 않고 ÷52 값만 적는다.
"""
import sys, os, json
SD = sys.argv[1]
M = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM", "HD95"]
PRESENT = {"test": 36, "val": 33}
TAGS = [("b1on","제출본 (구검출기·구피처·gC ON)"),("b1on_pf","  + 패치필터 = 실제 제출본"),("b1Non","① 개정피처 (gC ON)"),("b1Non_pf","  + 패치필터"),("b1Noff","② 개정피처 + gC OFF"),("b1Noff_pf","  + 패치필터"),("e9off","③ E9 ResEncL + 개정피처 + gC OFF"),("e9off_pf","  + 패치필터")]

def load(p):
    if not os.path.exists(p):
        return None
    s = open(p).read(); i = len(s)
    while True:
        i = s.rfind('{', 0, i)
        if i < 0:
            return None
        try:
            d = json.loads(s[i:])
            if isinstance(d, dict) and "label" in d:
                return d
        except Exception:
            pass

for sp in ("test", "val"):
    print(f"### {sp} (GT 등장 클래스 {PRESENT[sp]})\n")
    print("| 구성 | " + " | ".join(M) + " | ÷52 MCC | cov MCC |")
    print("|---|" + "---|" * (len(M) + 2))
    for tag, name in TAGS:
        ds = [load(f"{SD}/{tag}_{sp}_s{s}.json") for s in range(5)]
        ds = [d for d in ds if d]
        if not ds:
            continue
        new = [f"{sum(float(d['new'][m]) for d in ds)/len(ds):.4f}" for m in M]
        o = [d.get("old") or {} for d in ds]
        om = [float(x["MCC"]) for x in o if "MCC" in x]
        o52 = sum(om) / len(om) if om else float("nan")
        cov = o52 * 52 / PRESENT[sp] if om else float("nan")
        print(f"| {name} | " + " | ".join(new) + f" | {o52:.4f} | **{cov:.4f}** |")
    print()
