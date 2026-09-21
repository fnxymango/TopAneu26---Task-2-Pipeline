#!/usr/bin/env python3
"""탑1 규칙(결과 보기 전 고정): 새 eval · gC OFF · 5시드 평균 · 지표별 순위(1~3)를 test 6 + val 6 = 12개 평균, 최소가 탑1. 동률 시 test MCC."""
import json, os
M=["PRECISION","RECALL","MCC","DICE","VOLSIM","HD95"]; HI={"HD95":-1}
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
C={"B1 구번들":"b1off","B2 재학습":"b2off","E9 10폴드":"e9off"}
mean={}
for name,tag in C.items():
    for sp in ("test","val"):
        ds=[d for d in (L(f"{G}/{tag}_{sp}_s{sd}.json") for sd in range(5)) if d]
        for m in M: mean[(name,sp,m)]=sum(float(d["new"][m]) for d in ds)/len(ds)
ranks={n:[] for n in C}
print("| 후보 | " + " | ".join(f"{sp}·{m}" for sp in ("test","val") for m in M) + " | 평균순위 |"); print("|---|"+"---|"*13)
for sp in ("test","val"):
    for m in M:
        order=sorted(C, key=lambda n: -mean[(n,sp,m)]*HI.get(m,1))
        for r,n in enumerate(order,1): ranks[n].append(r)
for n in C:
    print(f"| {n} | " + " | ".join(str(r) for r in ranks[n]) + f" | **{sum(ranks[n])/12:.2f}** |")
best=sorted(C, key=lambda n:(sum(ranks[n])/12, -mean[(n,"test","MCC")]))
print(f"\n**탑1 (새 eval · gC OFF · 순위평균): {best[0]}**  (2위 {best[1]}, 3위 {best[2]})")
print("test만 순위평균:", {n: round(sum(ranks[n][:6])/6,2) for n in C}, "· val만:", {n: round(sum(ranks[n][6:])/6,2) for n in C})
