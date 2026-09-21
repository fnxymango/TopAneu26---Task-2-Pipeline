#!/usr/bin/env python3
"""h3_abs.py <scores_dir> — 환경변수 H3_TAGS 의 태그들을 5시드 평균 절대값으로.
covered_gt 는 구 eval(÷52)을 GT 등장 클래스 수로 환산(test 36 · val 33). HD95 는 비례하지 않아 환산 안 함."""
import sys, os, json
SD = sys.argv[1]
M = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM", "HD95"]
PRESENT = {"test": 36, "val": 33}
TAGS = os.environ.get("H3_TAGS", "").split()
def load(p):
    if not os.path.exists(p): return None
    s = open(p).read(); i = len(s)
    while True:
        i = s.rfind('{', 0, i)
        if i < 0: return None
        try:
            d = json.loads(s[i:])
            if isinstance(d, dict) and "label" in d: return d
        except Exception: pass
for sp in ("test", "val"):
    print(f"### {sp} (GT 등장 클래스 {PRESENT[sp]})\n")
    print("| 구성 | " + " | ".join(M) + " | ÷52 MCC | cov MCC |")
    print("|---|" + "---|" * (len(M) + 2))
    for tag in TAGS:
        ds = [load(f"{SD}/{tag}_{sp}_s{s}.json") for s in range(5)]
        ds = [d for d in ds if d]
        if not ds: continue
        new = [f"{sum(float(d['new'][m]) for d in ds)/len(ds):.4f}" for m in M]
        om = [float((d.get('old') or {})['MCC']) for d in ds if (d.get('old') or {}).get('MCC') is not None]
        o52 = sum(om)/len(om) if om else float('nan')
        cov = o52 * 52 / PRESENT[sp] if om else float('nan')
        print(f"| {tag} | " + " | ".join(new) + f" | {o52:.4f} | **{cov:.4f}** |")
    print()
