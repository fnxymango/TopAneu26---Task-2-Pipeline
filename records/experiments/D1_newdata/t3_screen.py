#!/usr/bin/env python3
"""T3 스크리닝 — 비낭형 표본 가중(c5_t3 · TOPANEU_T3=1) vs 기준, train OOF (케이스 5겹 × 시드5 · test·val 안 봄).
유형 가중은 각 폴드 학습 행 안에서만 계산된다(N낭형/N비낭형). 평가 행의 유형은 쓰지 않는다(추론과 같다).
── 관문 (결과 보기 전 고정 · 2026-09-15 · V5·K2·K6 와 같은 틀) ──
 통과 = 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro-recall>0 시드 ≥ 4/5 → T3.sh e2e(K0 장치 · α 0.05)
 참고: 비낭형 병변 top1 · 후순환 top1
"""
import json, os, sys
import numpy as np
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D); os.environ.setdefault("TOPANEU_ROOT", R)


def run_arm(arm):
    os.environ["TOPANEU_T3"] = "1" if arm == "t3" else "0"
    import c5_t3 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in rows}); out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
        pred = {}
        for k in range(5):
            m = C5.fit_model([r for r in rows if fold[r["case"]] != k], ax, kind="rf", mirror=True)
            for i, r in enumerate(rows):
                if fold[r["case"]] == k:
                    pred[i] = C5.predict_one(m, r, 0.5)
        out[sd] = [pred[i] for i in range(len(rows))]
    return arm, out


def main():
    import multiprocessing as mp
    from k2_sym import metrics, code
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    F = json.load(open(f"{D}/k6_type_feat.json"))
    ns = {f"{r['case']}|{r['lesion_mask_idx']}": F[f"{r['case']}|{r['lesion_mask_idx']}"]["type"] in (2, 3) for r in rows}
    truth = [r["gt_loc"] for r in rows]
    with mp.get_context("spawn").Pool(2) as p:
        res = dict(p.map(run_arm, ["base", "t3"]))
    b = {s: metrics(truth, res["base"][s]) for s in range(5)}; t = {s: metrics(truth, res["t3"][s]) for s in range(5)}
    d1 = np.array([t[s][0] - b[s][0] for s in range(5)]); d2 = np.array([t[s][1] - b[s][1] for s in range(5)])
    nsi = [i for i, r in enumerate(rows) if ns[f"{r['case']}|{r['lesion_mask_idx']}"]]
    acc_ns = lambda arm: np.mean([np.mean([res[arm][s][i] == truth[i] for i in nsi]) for s in range(5)])
    post = lambda g: code(g).split(".")[0] in ("1", "2")
    pb = np.mean([metrics(truth, res["base"][s], post)[0] for s in range(5)]); pt = np.mean([metrics(truth, res["t3"][s], post)[0] for s in range(5)])
    ok = d2.mean() >= 0.02 and d1.mean() >= -0.005 and (d2 > 0).sum() >= 4
    B = np.array(list(b.values())).mean(0); T = np.array(list(t.values())).mean(0)
    print("# T3 비낭형 표본 가중 — train OOF 스크리닝\n")
    print("| 팔 | top1 | macro-recall | 비낭형 top1 | 후순환 top1 |\n|---|---|---|---|---|")
    print(f"| base | {B[0]:.3f} | {B[1]:.3f} | {acc_ns('base'):.3f} | {pb:.3f} |\n| T3 | {T[0]:.3f} | {T[1]:.3f} | {acc_ns('t3'):.3f} | {pt:.3f} |")
    print(f"\nΔtop1 {d1.mean():+.3f} · Δmacro {d2.mean():+.3f} · 시드별 {' '.join(f'{x:+.3f}' for x in d2)} ({(d2 > 0).sum()}/5)")
    print(f"\n**관문 → {'통과 · e2e' if ok else '미달 · e2e 안 건다'}**")
    json.dump(dict(ok=bool(ok)), open(f"{D}/t3_gate.json", "w"))


if __name__ == "__main__":
    main()
