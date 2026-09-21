#!/usr/bin/env python3
"""V4-D 2단계 — 검출 blob 학습표 스크리닝 (train OOF 만 · test·val 안 봄).

관문은 v4d_oof.py 머리말에 1단계 결과 보기 전 고정(2026-09-15)한 것을 그대로 쓴다.
 평가 행 = 보류 폴드 케이스의 검출 blob TP 행(v4d_feat_blob_oof.json, gt_loc 있음) — 추론과 같은 입력 조건.
 폴드 = 기준표 케이스를 시드별 순열로 5등분(t2_span · V5 틀과 같음). 학습 행은 보류 폴드 케이스를 모두 뺀다.
 팔 base = 기준 하이브리드표 GT 행 · V4D = 검출 blob TP 행 · V4D+ = 두 표 행 합침.
 설정: C5 rf · mirror · balance(기본) · USE_POS · τ0.5 · β_hi 0 · predict_one(β0.5).
 통과 = 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro>0 시드 ≥ 4/5.
 두 팔 모두 통과하면 Δmacro 큰 쪽 하나만 e2e. 둘 다 미달 → V4-D 닫음.
"""
import json, os, sys, collections
import numpy as np
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D); os.environ.setdefault("TOPANEU_ROOT", R)


def load():
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(os.environ.get("V4D_TABLE", f"{A}/v4d_feat_blob_oof.json"))) if r.get("gt_loc")]
    return base, blob


def job(args):
    arm, sd = args
    os.environ["CLF_SEED"] = str(sd)
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    base, blob = load()
    cases = sorted({r["case"] for r in base})
    perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
    pred = [None] * len(blob)
    for k in range(5):
        tr = {"base": base, "v4d": blob, "v4dp": base + blob}[arm]
        m = C5.fit_model([r for r in tr if fold[r["case"]] != k], ax, kind="rf", mirror=True)
        for i, r in enumerate(blob):
            if fold[r["case"]] == k:
                pred[i] = C5.predict_one(m, r, 0.5)
    return (arm, sd), pred


def main():
    import multiprocessing as mp
    from k2_sym import metrics, code
    base, blob = load()
    assert {r["case"] for r in blob} <= {r["case"] for r in base}
    truth = [r["gt_loc"] for r in blob]
    with mp.Pool(8) as p:
        res = dict(p.map(job, [(a, s) for a in ("base", "v4d", "v4dp") for s in range(5)], chunksize=1))
    M = {k: metrics(truth, v) for k, v in res.items()}
    post = lambda g: code(g).split(".")[0] in ("1", "2")
    print("# V4-D 2단계 — 검출 blob 학습표 스크리닝 (train OOF · 평가 = 보류 폴드 검출 blob TP 행)\n")
    print(f"학습 행: 기준표 {len(base)} · blob TP {len(blob)} · 평가 행 {len(blob)} · 평가 클래스 {len(set(truth))}\n")
    print("| 팔 | top1 | macro-recall | 후순환 top1 |\n|---|---|---|---|")
    for a in ("base", "v4d", "v4dp"):
        t = np.mean([M[(a, s)][0] for s in range(5)]); mr = np.mean([M[(a, s)][1] for s in range(5)])
        pt = np.mean([metrics(truth, res[(a, s)], post)[0] for s in range(5)])
        print(f"| {a} | {t:.3f} | {mr:.3f} | {pt:.3f} |")
    verdict = {}
    for a in ("v4d", "v4dp"):
        d1 = np.array([M[(a, s)][0] - M[("base", s)][0] for s in range(5)])
        d2 = np.array([M[(a, s)][1] - M[("base", s)][1] for s in range(5)])
        ok = bool(d2.mean() >= 0.02 and d1.mean() >= -0.005 and (d2 > 0).sum() >= 4)
        verdict[a] = dict(ok=ok, dmacro=float(d2.mean()), dtop1=float(d1.mean()))
        print(f"\n- {a}: Δtop1 {d1.mean():+.3f} · Δmacro {d2.mean():+.3f} · 시드별 {' '.join(f'{x:+.3f}' for x in d2)} ({(d2 > 0).sum()}/5) → {'통과' if ok else '미달'}")
    passed = [a for a in verdict if verdict[a]["ok"]]
    pick = max(passed, key=lambda a: verdict[a]["dmacro"]) if passed else None
    print(f"\n**관문 → {'통과 · e2e 팔 = ' + pick if pick else '미달 · V4-D 닫음'}**")
    json.dump(dict(pick=pick, **verdict), open(f"{D}/v4d_gate.json", "w"))


if __name__ == "__main__":
    main()
