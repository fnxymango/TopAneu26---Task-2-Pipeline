#!/usr/bin/env python3
"""K1 을 train OOF 로 재본다 (2026-09-16 · 사용자 "test oof로도 해봐" → test 는 학습에 못 넣으므로 train OOF).

K1 은 스크리닝 없이 바로 e2e 로 갔다. 여기서는 같은 장치를 케이스 묶음 5폴드 × 시드 0~4 로 재서
e2e(시드 0~9)에서 본 것 — test 재현율↑·정밀도↓ / val 전부↑ — 이 train 에서도 같은 모양인지 본다.
누수 방지: 보류 폴드 케이스는 **두 표 모두에서** 뺀다.
평가 행 두 가지:
  (a) GT 병변 행(기준 학습표와 같은 조건)
  (b) 검출 blob TP 행(v4d_feat_blob_oof · 추론과 같은 입력 조건)
판정용이 아니라 진단용이다(채택 판정은 K0 e2e 로만 한다).
"""
import json, os, sys, collections
import numpy as np
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D); os.environ.setdefault("TOPANEU_ROOT", R)


def job(args):
    arm, sd = args
    os.environ["CLF_SEED"] = str(sd)
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    pv = [r for r in json.load(open(f"{A}/c10_feat_train_predves_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(f"{A}/v4d_feat_blob_oof.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in base})
    perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
    pg, pb = [None] * len(base), [None] * len(blob)
    for k in range(5):
        tr1 = [r for r in base if fold[r["case"]] != k]
        m = C5.fit_model(tr1, ax, kind="rf", mirror=True)
        if arm == "k1":
            tr2 = [r for r in pv if fold[r["case"]] != k]
            m2 = C5.fit_model(tr2, ax, kind="rf", mirror=True)
            m["clf"] = C5.ProbAvg([m["clf"], m2["clf"]])
        for i, r in enumerate(base):
            if fold[r["case"]] == k:
                pg[i] = C5.predict_one(m, r, 0.5)
        for i, r in enumerate(blob):
            if fold[r["case"]] == k:
                pb[i] = C5.predict_one(m, r, 0.5)
    return (arm, sd), (pg, pb)


def main():
    from multiprocessing import Pool
    from k2_sym import metrics
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(f"{A}/v4d_feat_blob_oof.json")) if r.get("gt_loc")]
    tg, tb = [r["gt_loc"] for r in base], [r["gt_loc"] for r in blob]
    with Pool(10) as p:
        res = dict(p.map(job, [(a, s) for a in ("base", "k1") for s in range(5)], chunksize=1))
    print("# K1 — train OOF 진단 (케이스 묶음 5폴드 × 시드 0~4 · 판정용 아님)\n")
    for lab, truth, idx in (("GT 병변 행 (학습표 조건)", tg, 0), ("검출 blob TP 행 (추론 조건)", tb, 1)):
        print(f"## {lab} · 행 {len(truth)}\n")
        print("| 팔 | top1 | macro-recall |\n|---|---|---|")
        M = {}
        for a in ("base", "k1"):
            v = [metrics(truth, res[(a, s)][idx]) for s in range(5)]
            M[a] = np.array(v)
            print(f"| {a} | {M[a][:,0].mean():.3f} | {M[a][:,1].mean():.3f} |")
        d1 = M["k1"][:, 0] - M["base"][:, 0]; d2 = M["k1"][:, 1] - M["base"][:, 1]
        # 짝지은 고침/망침
        fx = br = 0
        for s in range(5):
            a, b = res[("base", s)][idx], res[("k1", s)][idx]
            for i, t in enumerate(truth):
                fx += (a[i] != t) and (b[i] == t); br += (a[i] == t) and (b[i] != t)
        print(f"\nΔtop1 {d1.mean():+.3f} ({(d1>0).sum()}/5) · Δmacro {d2.mean():+.3f} ({(d2>0).sum()}/5) · "
              f"5시드 합 살아남 {fx} · 새로 틀림 {br} · 순증 {fx-br:+d}\n")


if __name__ == "__main__":
    main()
