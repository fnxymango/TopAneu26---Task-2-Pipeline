#!/usr/bin/env python3
"""terr_oof.py — train 5겹×5시드 OOF(deep_ica_oof.json)로 영역별 혼동·top-k 여유. test 안 봄."""
import json, re, collections
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
A = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
oof = json.load(open(f"{D}/deep_ica_oof.json"))
def code(nm):
    b = re.sub(r"^[RL]-", "", nm); c = b.split()[0]
    return ("5.3j" if "M1-M2" in b else "5.3d") if c == "5.3" else c
def terr(nm):
    return {"1": "후순환", "2": "후순환", "3": "ICA", "4": "ACA", "5": "MCA"}[code(nm).split(".")[0]]
for T in ("MCA", "ACA", "후순환", "ICA"):
    idx = [i for i, r in enumerate(rows) if terr(r["gt_loc"]) == T]
    n = t1 = t2 = t3 = 0
    per = collections.defaultdict(lambda: [0, 0, 0, 0])     # n, top1, top2, top3
    conf = collections.Counter(); predc = collections.Counter(); side = 0
    for i in idx:
        g = rows[i]["gt_loc"]
        for s in oof[str(i)]:
            nm = s["names"]; n += 1; p = per[code(g)]; p[0] += 1
            h1 = nm[:1] == [g]; h2 = g in nm[:2]; h3 = g in nm[:3]
            t1 += h1; t2 += h2; t3 += h3; p[1] += h1; p[2] += h2; p[3] += h3
            if nm: predc[code(nm[0])] += 1
            if not h1 and nm:
                if code(nm[0]) == code(g): side += 1
                else: conf[(code(g), code(nm[0]))] += 1
    print(f"## {T} — train 병변 {len(idx)} · 판정 {n} · top1 {t1/n:.0%} · top2 {t2/n:.0%} · top3 {t3/n:.0%} · 좌우만 틀림 {side}\n")
    print("| 클래스 | 판정 | top1 | top2 | top3 | RF가 1등으로 고른 횟수 |\n|---|---|---|---|---|---|")
    for c, (a, b, e, f) in sorted(per.items()):
        print(f"| {c} | {a} | {b/a:.0%} | {e/a:.0%} | {f/a:.0%} | {predc[c]} |")
    print("\n주요 혼동(GT→예측, 판정수): " + " · ".join(f"{g}→{p} {v}" for (g, p), v in conf.most_common(8)) + "\n")
