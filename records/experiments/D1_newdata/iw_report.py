#!/usr/bin/env python3
"""iw_report.py <tag> — intweak_<tag>.json 을 분류층 관점으로 읽는다."""
import json, sys, collections
D="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
TAG=sys.argv[1]
d=json.load(open(f"{D}/intweak_{TAG}.json"))
L=d["lesions"]; FP=d["fps"]; PC=d["per_case"]
det=[x for x in L if x["detected"]]
print(f"# {TAG}\n")
print(f"GT 병변 {len(L)} · 검출됨 {len(det)} ({100*len(det)/len(L):.1f}%) · 미검출 {len(L)-len(det)}")
print()
# --- 분류 정확도 (검출된 병변만) ---
print("## 1. 검출된 병변의 분류 정확도 (시드 5판)")
print()
print("| split | 검출병변 | 5판 모두정답 | 부분정답 | 5판 모두오답 | 평균 정답률 |")
print("|---|---|---|---|---|---|")
for sp in ("test","val","전체"):
    rs=[x for x in det if sp=="전체" or x["split"]==sp]
    if not rs: continue
    allok=sum(1 for x in rs if x["n_ok"]==5); none=sum(1 for x in rs if x["n_ok"]==0)
    part=len(rs)-allok-none
    acc=sum(x["n_ok"] for x in rs)/(5*len(rs))
    print(f"| {sp} | {len(rs)} | {allok} ({100*allok/len(rs):.0f}%) | {part} | {none} ({100*none/len(rs):.0f}%) | **{100*acc:.1f}%** |")
print()
# --- 오분류 유형 ---
print("## 2. 오분류 유형 (검출된 병변 × 5시드 = 시행 단위)")
print()
k=collections.Counter()
for x in det:
    for t in x["kinds"]: k[t]+=1
tot=sum(k.values())
order=["정답","인접분절","인접분절+좌우반전","좌우반전","같은혈관군 원거리","다른혈관군","미할당"]
print("| 유형 | 시행 | 비율 | 오답 중 비율 |")
print("|---|---|---|---|")
wrong=tot-k["정답"]
for t in order:
    if not k[t]: continue
    w=f"{100*k[t]/wrong:.1f}%" if t!="정답" else "—"
    print(f"| {t} | {k[t]} | {100*k[t]/tot:.1f}% | {w} |")
print(f"\n검출된 병변 시행 {tot}회 중 오답 {wrong}회 ({100*wrong/tot:.1f}%)")
print()
# --- 공식지표 FP 분해 ---
print("## 3. 공식 지표 FP 의 출처")
print()
onreal=sum(1 for f in FP if f["on_gt"]>0); hal=len(FP)-onreal
print(f"| 출처 | (케이스×클래스) FP 수 | 비율 |")
print("|---|---|---|")
print(f"| 진짜 병변을 틀린 클래스에 붙임 | {onreal} | {100*onreal/max(len(FP),1):.1f}% |")
print(f"| 환각 blob 에 클래스가 붙음 | {hal} | {100*hal/max(len(FP),1):.1f}% |")
print(f"\n5시드 합계 · 시드당 평균 FP {len(FP)/5:.1f}개")
tp=sum(p["n_tp"] for p in PC)/5; fp=sum(p["n_fp"] for p in PC)/5; fn=sum(p["n_fn"] for p in PC)/5
print(f"\n시드평균 (케이스×클래스) — TP {tp:.1f} · FP {fp:.1f} · FN {fn:.1f}")
print()
# --- 클래스별 ---
print("## 4. 클래스별 (검출된 병변 ≥1개인 클래스)")
print()
by=collections.defaultdict(list)
for x in det: by[x["name"]].append(x)
rows=[]
for nm,rs in by.items():
    acc=sum(x["n_ok"] for x in rs)/(5*len(rs))
    wc=collections.Counter()
    for x in rs:
        for p,t in zip(x["preds"],x["kinds"]):
            if t!="정답": wc[(p,t)]+=1
    top=wc.most_common(1)
    rows.append((acc,len(rs),nm,top[0] if top else None))
rows.sort()
NAME={int(k):v for k,v in json.load(open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/dataset/TopAneu/dataset_split.json"))["location_classes"].items()}
print("| 클래스 | 검출병변 | 정답률 | 최빈 오답 → | 유형 |")
print("|---|---|---|---|---|")
for acc,n,nm,top in rows:
    if top:
        pn = NAME.get(top[0][0],"(미할당)") if top[0][0] else "(미할당)"
        print(f"| {nm} | {n} | {100*acc:.0f}% | {pn} ×{top[1]} | {top[0][1]} |")
    else:
        print(f"| {nm} | {n} | {100*acc:.0f}% | — | — |")
