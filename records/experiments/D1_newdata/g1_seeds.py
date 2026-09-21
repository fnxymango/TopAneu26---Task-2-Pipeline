#!/usr/bin/env python3
"""gC 5시드 최종 판정 (규칙 11:05 고정): split 별 seed 0~4 (on−off) 지표별 평균 → 개선 ≥4/6 AND 평균ΔMCC ≥ 0, 두 split 모두."""
import sys, math
sys.path.insert(0, "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata")
import json, os
G = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/G1_gc_neweval"
M = ["PRECISION","RECALL","MCC","DICE","VOLSIM","HD95"]
def L(t):
    p = f"{G}/scores/{t}.json"
    if not os.path.exists(p): p = f"{G}/scores_par/{t}.json"
    if not os.path.exists(p): return None
    s = open(p).read(); i = len(s)
    while True:
        i = s.rfind('{', 0, i)
        if i < 0: return None
        try:
            d = json.loads(s[i:])
            if isinstance(d, dict) and "label" in d: return d
        except Exception: pass
HIGHER = {"PRECISION":1,"RECALL":1,"MCC":1,"DICE":1,"VOLSIM":1,"HD95":-1}
def val(d,m):
    v = d["new"].get(m); return float("nan") if v is None else float(v)
splits = sys.argv[1:] or ["test","val"]
verdict = {}
for sp in splits:
    rows = {m: [] for m in M}; seeds = []
    for sd in (0,1,2,3,4):
        suf = "" if sd == 3 else f"_s{sd}"
        on, off = L(f"b1gcON_{sp}{suf}"), L(f"b1gcOFF_{sp}{suf}")
        if not on or not off: continue
        seeds.append(sd)
        for m in M: rows[m].append((val(off,m), val(on,m)))
    if not seeds: print(f"[{sp}] 채점 결과 없음"); continue
    print(f"\n[B단계 번들 검출기 · {sp} · 시드 {seeds}]")
    print(f"{'지표':10s} " + " ".join(f"s{sd:<8d}" for sd in seeds) + "   평균Δ    개선")
    nimp = 0; dmcc = None
    for m in M:
        ds = [o2-o1 for o1,o2 in rows[m]]
        mean = sum(ds)/len(ds); imp = (mean*HIGHER[m]) > 0
        if m == "MCC": dmcc = mean
        nimp += imp
        print(f"{m:10s} " + " ".join(f"{d:+9.4f}" for d in ds) + f" {mean:+9.4f}  {'○' if imp else '×'}")
    ok = nimp >= 4 and dmcc >= 0
    verdict[sp] = ("유지" if ok else "미충족", nimp, dmcc, len(seeds))
    print(f"→ 개선 {nimp}/6 · 평균ΔMCC {dmcc:+.4f} · {'충족' if ok else '미충족'} (시드 {len(seeds)}개)")
print("\n판정:", {k: v[0] for k,v in verdict.items()})
if all(k in verdict for k in ("test","val")):
    fin = "gC 유지" if all(v[0]=="유지" for v in verdict.values()) else "gC OFF 권고 (번들 topk n=2 → 1)"
    print("최종:", fin)
