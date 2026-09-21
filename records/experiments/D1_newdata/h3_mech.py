#!/usr/bin/env python3
"""h3_mech.py <기준태그> <태그...> — 기전 확인.

bpfalse.json 이 표시한 '가짜 곁가지가 5mm 안에 있던 병변' 에서 실제로 정답률이 올랐는지 본다.
vesconf 가 노린 것이 바로 그 병변들이므로, 전체 지표가 올라도 여기서 안 오르면 다른 이유로 오른 것이다.
각 태그의 intweak_<태그>.json 이 있어야 한다 (없으면 그 태그는 건너뛴다).
"""
import json, sys, os
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
bp = {(x["split"], x["case"], x["cls"], round(x["dia"], 4)): x for x in json.load(open(f"{D}/bpfalse.json"))}
spur = {k for k, v in bp.items() if v["spurious_branch"]}
dense = {k for k, v in bp.items() if v["n_bp_near"] >= 8}
print(f"대상: 가짜 곁가지 있던 병변 {len(spur)}개 · 분기점 밀집(8개+) {len(dense)}개 · 전체 {len(bp)}개\n")
print("| 구성 | 전체 | 가짜곁가지 있던 병변 | 밀집 병변 | 나머지 |")
print("|---|---|---|---|---|")
for tag in sys.argv[1:]:
    p = f"{D}/intweak_{tag}.json"
    if not os.path.exists(p):
        print(f"| {tag} | (intweak 없음) | | | |"); continue
    L = {(x["split"], x["case"], x["cls"], round(x["dia"], 4)): x
         for x in json.load(open(p))["lesions"] if x["detected"]}
    def acc(keys):
        rs = [L[k] for k in keys if k in L]
        return f"{100*sum(x['n_ok'] for x in rs)/(5*len(rs)):.1f}% ({len(rs)})" if rs else "—"
    allk = set(L)
    print(f"| {tag} | {acc(allk)} | {acc(spur & allk)} | {acc(dense & allk)} | {acc(allk - spur - dense)} |")
