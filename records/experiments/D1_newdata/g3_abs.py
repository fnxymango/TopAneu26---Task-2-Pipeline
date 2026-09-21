#!/usr/bin/env python3
import json, os
M=["PRECISION","RECALL","MCC","DICE","VOLSIM","HD95"]
G="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/G3_gcoff_candidates/scores"
def L(p):
    if not os.path.exists(p): return None
    s=open(p).read(); i=len(s)
    while True:
        i=s.rfind('{',0,i)
        if i<0: return None
        try:
            d=json.loads(s[i:])
            if isinstance(d,dict) and "label" in d: return d
        except Exception: pass
for sp in ("test","val"):
    print(f"### {sp}\n\n| 후보 | n | P | R | MCC | Dice | VS | HD95(mm) | 구MCC÷52 |\n|---|---|---|---|---|---|---|---|---|")
    for name,tag in (("B1 구번들","b1off"),("B2 재학습","b2off"),("E9 10폴드","e9off")):
        ds=[d for d in (L(f"{G}/{tag}_{sp}_s{sd}.json") for sd in range(5)) if d]
        if not ds: continue
        m=lambda k: sum(float(d["new"][k]) for d in ds)/len(ds)
        old=[float(d["old"]["MCC"]) for d in ds if d.get("old") and d["old"].get("MCC") is not None]
        print(f"| {name} | {len(ds)} | {m('PRECISION'):.4f} | {m('RECALL'):.4f} | **{m('MCC'):.4f}** | {m('DICE'):.4f} | {m('VOLSIM'):.4f} | {m('HD95'):.1f} | {sum(old)/len(old):.4f} |")
    print()
