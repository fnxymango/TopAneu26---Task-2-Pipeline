#!/usr/bin/env python3
"""terrdeep.py <tag> — 현 기준선 영역별 오답 분해 (intweak_<tag>.json). 병변×시드 단위."""
import json, sys, re, collections
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
tag = sys.argv[1]
d = json.load(open(f"{D}/intweak_{tag}.json"))
S = json.load(open("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/dataset/TopAneu/dataset_split.json"))
NAME = {int(k): v for k, v in S["location_classes"].items()}
def terr(nm):
    c = re.sub(r"^[RL]-", "", nm).split()[0]
    g = c.split(".")[0]
    return {"1": "후순환", "2": "후순환", "3": "ICA", "4": "ACA", "5": "MCA"}[g]
L = d["lesions"]
print(f"# {tag} · GT 병변 {len(L)} · 검출 {sum(x['detected']>0 for x in L)}\n")
for T in ("MCA", "ACA", "ICA", "후순환"):
    xs = [x for x in L if terr(x["name"]) == T]
    nd = [x for x in xs if not x["detected"]]
    det = [x for x in xs if x["detected"]]
    tot = sum(x["n_seed"] for x in det); ok = sum(x["n_ok"] for x in det)
    kc = collections.Counter(k for x in det for k in x["kinds"])
    print(f"## {T} — 병변 {len(xs)} · 미검출 {len(nd)} · 검출 {len(det)} · 시드정답 {ok}/{tot} ({ok/max(tot,1):.0%})")
    print("오답유형: " + " · ".join(f"{k} {v}" for k, v in kc.most_common() if k != "정답") + "\n")
    print("| split | 케이스 | GT | 지름 | 정답/5 | 예측(시드별) |\n|---|---|---|---|---|---|")
    for x in sorted(det, key=lambda x: (x["name"], x["n_ok"])):
        if x["n_ok"] == 5: continue
        pr = collections.Counter(NAME.get(p, "미할당") if p else "미할당" for p in x["preds"])
        print(f"| {x['split']} | {x['case'][8:]} | {x['name']} | {x['dia']:.1f} | {x['n_ok']} | {', '.join(f'{k}×{v}' for k,v in pr.most_common())} |")
    print()
# 환각
fc = collections.Counter(terr(f["name"]) for f in d["fps"])
print("## 환각(GT에 없는 케이스×클래스, 시드합) 영역별: " + " · ".join(f"{k} {v}" for k, v in fc.most_common()))
