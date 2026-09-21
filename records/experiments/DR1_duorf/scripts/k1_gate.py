#!/usr/bin/env python3
"""K1 을 전부 걸까, 애매한 것만 걸까 — train OOF 진단 (2026-09-16 · 판정용 아님).

같은 폴드에서 base 와 K1 의 예측을 나란히 얻고, **base 의 확신**으로 구간을 나눠 고침/망침을 센다.
게이트 모의: base 의 (1등 확률 p1) 또는 (1등−2등 마진) 이 임계 미만인 병변에만 K1 결과를 쓰고
나머지는 base 결과를 그대로 쓴다 → 전역 적용 대비 순증이 커지는 구간이 있는지 본다.
평가 행: 검출 blob TP 행(추론 조건) + GT 병변 행(학습표 조건) 둘 다.
"""
import json, os, sys, collections
import numpy as np
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D); os.environ.setdefault("TOPANEU_ROOT", R)


def job(sd):
    os.environ["CLF_SEED"] = str(sd)
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    pv = [r for r in json.load(open(f"{A}/c10_feat_train_predves_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(f"{A}/v4d_feat_blob_oof.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in base})
    perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
    out = {"gt": [None] * len(base), "blob": [None] * len(blob)}
    for k in range(5):
        m = C5.fit_model([r for r in base if fold[r["case"]] != k], ax, kind="rf", mirror=True)
        m2 = C5.fit_model([r for r in pv if fold[r["case"]] != k], ax, kind="rf", mirror=True)
        mk = dict(m); mk["clf"] = C5.ProbAvg([m["clf"], m2["clf"]])
        for key, rows in (("gt", base), ("blob", blob)):
            for i, r in enumerate(rows):
                if fold[r["case"]] != k:
                    continue
                nm, pb, _ = C5.predict_ranked(m, r, 0.5)
                if nm is None or not len(nm):
                    out[key][i] = None; continue
                p1 = float(pb[0]); p2 = float(pb[1]) if len(pb) > 1 else 0.0
                out[key][i] = (str(nm[0]), p1, p1 - p2, str(C5.predict_one(mk, r, 0.5)))
    return sd, out


def main():
    from multiprocessing import Pool
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(f"{A}/v4d_feat_blob_oof.json")) if r.get("gt_loc")]
    with Pool(5) as p:
        res = dict(p.map(job, range(5)))
    truth = {"gt": [r["gt_loc"] for r in base], "blob": [r["gt_loc"] for r in blob]}
    print("# K1 전역 적용 vs 애매한 것만 적용 — train OOF (시드 0~4 합 · 판정용 아님)\n")
    for key, lab in (("blob", "검출 blob TP 행 (추론 조건)"), ("gt", "GT 병변 행 (학습표 조건)")):
        T = truth[key]
        rec = [(v, T[i]) for sd in range(5) for i, v in enumerate(res[sd][key]) if v]
        print(f"## {lab} · 판정 {len(rec)}\n")
        print("| base 1등 확률 구간 | 판정 수 | 바뀐 수 | 살아남 | 새로 틀림 | 순증 |\n|---|---|---|---|---|---|")
        for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, 1.01)):
            sub = [(v, t) for v, t in rec if lo <= v[1] < hi]
            ch = [(v, t) for v, t in sub if v[3] != v[0]]
            fx = sum(1 for v, t in ch if v[0] != t and v[3] == t); br = sum(1 for v, t in ch if v[0] == t and v[3] != t)
            print(f"| {lo:.1f}~{hi:.1f} | {len(sub)} | {len(ch)} | {fx} | {br} | {fx-br:+d} |")
        print("\n| 게이트(이 조건일 때만 K1) | 적용된 판정 | 살아남 | 새로 틀림 | 순증 | 전체 정답률 |\n|---|---|---|---|---|---|")
        n = len(rec)
        for name, sel in (("전역(항상)", lambda v: True),
                          ("p1 < 0.7", lambda v: v[1] < 0.7), ("p1 < 0.5", lambda v: v[1] < 0.5),
                          ("p1 < 0.3", lambda v: v[1] < 0.3),
                          ("마진 < 0.2", lambda v: v[2] < 0.2), ("마진 < 0.1", lambda v: v[2] < 0.1)):
            app = [(v, t) for v, t in rec if sel(v)]
            ch = [(v, t) for v, t in app if v[3] != v[0]]
            fx = sum(1 for v, t in ch if v[0] != t and v[3] == t); br = sum(1 for v, t in ch if v[0] == t and v[3] != t)
            acc = sum(1 for v, t in rec if (v[3] if sel(v) else v[0]) == t) / n
            print(f"| {name} | {len(app)} | {fx} | {br} | {fx-br:+d} | {acc:.3f} |")
        acc0 = sum(1 for v, t in rec if v[0] == t) / n
        print(f"\nbase 전체 정답률 {acc0:.3f}\n")


if __name__ == "__main__":
    main()
