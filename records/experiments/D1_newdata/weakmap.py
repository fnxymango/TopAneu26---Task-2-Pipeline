#!/usr/bin/env python3
"""제출본(b1on_pf) 기준 분류 약점 지도 — intweak_b1on_pf.json 재집계."""
import json, collections, re
D="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
L=json.load(open(f"{D}/intweak_b1on_pf.json"))["lesions"]
det=[x for x in L if x["detected"]]
tot=sum(x["n_seed"] for x in det); ok=sum(x["n_ok"] for x in det)
print(f"검출된 GT 병변 {len(det)}/{len(L)} · 시드판정 {tot}회 · 정답 {ok} · **오답 {tot-ok} ({(tot-ok)/tot:.1%})**\n")

def terr(n):
    m=re.match(r"(?:[RL]-)?(\d+)\.", n)
    g=m.group(1) if m else "?"
    return {"1":"후순환(VA/BA/PCA)","2":"후순환(VA/BA/PCA)","3":"ICA","4":"ACA/Acom","5":"MCA"}.get(g,"기타")

print("## 영역별\n")
print("| 영역 | 병변 | 시드판정 | 정답률 | 오답 수 | 전체 오답 중 |")
print("|---|---|---|---|---|---|")
agg=collections.defaultdict(lambda:[0,0,0])
for x in det:
    a=agg[terr(x["name"])]; a[0]+=1; a[1]+=x["n_seed"]; a[2]+=x["n_ok"]
for k,(n,t,o) in sorted(agg.items(), key=lambda kv:-(kv[1][1]-kv[1][2])):
    print(f"| {k} | {n} | {t} | **{o/t:.1%}** | {t-o} | {(t-o)/(tot-ok):.0%} |")

print("\n## 크기별 (최대직경 mm)\n")
def band(d):
    return "<3mm" if d<3 else "3~5mm" if d<5 else "5~10mm" if d<10 else "≥10mm"
print("| 크기 | 병변 | 시드판정 | 정답률 | 오답 수 |")
print("|---|---|---|---|---|")
agg2=collections.defaultdict(lambda:[0,0,0])
for x in det:
    a=agg2[band(x["dia"])]; a[0]+=1; a[1]+=x["n_seed"]; a[2]+=x["n_ok"]
for k in ("<3mm","3~5mm","5~10mm","≥10mm"):
    if k in agg2:
        n,t,o=agg2[k]; print(f"| {k} | {n} | {t} | **{o/t:.1%}** | {t-o} |")

allw=[x for x in det if x["n_ok"]==0]; part=[x for x in det if 0<x["n_ok"]<x["n_seed"]]
print(f"\n## 재현성\n\n- 전시드 오답(편향) **{len(allw)}병변** — 오답 {sum(x['n_seed'] for x in allw)}회 = 전체 오답의 {sum(x['n_seed'] for x in allw)/(tot-ok):.0%}")
print(f"- 시드마다 갈림(분산) {len(part)}병변 — 오답 {sum(x['n_seed']-x['n_ok'] for x in part)}회")
print(f"- 전시드 정답 {sum(1 for x in det if x['n_ok']==x['n_seed'])}병변")

print("\n## 전시드 오답 병변 (고칠 대상)\n")
print("| GT 클래스 | 직경mm | 5시드 예측 |")
print("|---|---|---|")
NAME=json.load(open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/dataset/TopAneu/dataset_split.json"))["location_classes"]
NAME={int(k):v for k,v in NAME.items()}
for x in sorted(allw, key=lambda z:z["name"]):
    pr=collections.Counter(NAME.get(p,"미할당") if p else "미할당" for p in x["preds"])
    print(f"| {x['name']} | {x['dia']:.1f} | {', '.join(f'{k}×{v}' for k,v in pr.most_common())} |")
