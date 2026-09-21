#!/usr/bin/env python3
"""prior_screen.py — 로짓 보정 p/prior^α 를 train OOF 에서 스크리닝 (α 격자 0·0.25·0.5·0.75·1 사전 고정).
공식 지표는 클래스 평균이라 희귀 클래스 1개 수정 = 흔한 클래스 수십 개와 같은 무게 → macro-recall 을 주지표로 본다.
prior 는 학습 폴드 대신 train 전체 병변 수(좌우 병합 전 52클래스, 미러 반영해 좌우 합)로 근사."""
import json, re, collections
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
A = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
oof = json.load(open(f"{D}/deep_ica_oof.json"))
def code(nm):
    b = re.sub(r"^[RL]-", "", nm); c = b.split()[0]
    return ("5.3j" if "M1-M2" in b else "5.3d") if c == "5.3" else c
def terr(nm): return {"1": "후순환", "2": "후순환", "3": "ICA", "4": "ACA", "5": "MCA"}[code(nm).split(".")[0]]
cnt = collections.Counter(code(r["gt_loc"]) for r in rows)
for a in (0, 0.25, 0.5, 0.75, 1.0):
    hit = collections.defaultdict(lambda: [0, 0]); fixed = broke = 0
    for i, r in enumerate(rows):
        g = r["gt_loc"]
        for s in oof[str(i)]:
            if not s["names"]: continue
            sc = [p / (cnt.get(code(n), 1) ** a) for n, p in zip(s["names"], s["p"])]
            new = s["names"][max(range(len(sc)), key=sc.__getitem__)]
            h = hit[code(g)]; h[0] += 1; h[1] += new == g
            o = s["names"][0] == g
            fixed += (new == g) and not o; broke += o and (new != g)
    tot = sum(h[0] for h in hit.values()); ok = sum(h[1] for h in hit.values())
    mr = sum(h[1] / h[0] for h in hit.values()) / len(hit)
    terr_r = collections.defaultdict(list)
    for c, h in hit.items(): terr_r[terr(c + " x")].append(h[1] / h[0])
    tt = " · ".join(f"{t} {sum(v)/len(v):.2f}" for t, v in terr_r.items())
    print(f"α {a:<4} top1 {ok/tot:.1%} · macro-recall {mr:.3f} · 고침 {fixed} 망침 {broke} · 영역 macro {tt}")
print("\n클래스별 train 병변 수:", dict(sorted(cnt.items())))
