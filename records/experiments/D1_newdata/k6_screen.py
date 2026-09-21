#!/usr/bin/env python3
"""K6-2 — 예측한 동맥류 유형(비낭형 확률)을 분류기 블록으로 넣는 train OOF 스크리닝 (test·val 안 봄).

K6-1 관문 통과(유형 OOF AUC 0.924). 추론 땐 유형 라벨이 없으므로 **예측 확률**을 피처로 쓴다.
누수 방지(중첩): 바깥 케이스 5겹마다 —
  · 바깥 학습 행의 유형 확률 = 바깥 학습 행 안에서 다시 케이스 5겹 OOF 로 예측
  · 바깥 평가 행의 유형 확률 = 바깥 학습 행 전체로 학습한 유형 RF 로 예측
  그 확률 1차원을 c5 추가 블록 규약(블록 정규화 × 0.5)으로 붙여 c5 fit_model/predict_one 그대로.
미러: 유형은 좌우 무관 → 같은 값.

── 관문 (결과 보기 전 고정 · 2026-09-15 · V5·K2 와 같은 틀) ─────────────────
 통과 = 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro-recall>0 시드 ≥ 4/5 → e2e(추론 쪽 형상 피처 구현 후 K0 장치로)
 표적 구간(후순환 1.x·2.x) top1 은 참고.
"""
import json, os, sys, collections
import numpy as np

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D)
os.environ.setdefault("TOPANEU_ROOT", R)
W = 0.5


def type_rf(X, y, seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(500, class_weight="balanced", random_state=seed, n_jobs=1).fit(X, y)


def run_arm(arm):
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    orig = C5.row_to_vec

    def vec(r, ves_axis, mirror=False):
        v = orig(r, ves_axis, mirror=mirror)
        if arm == "base" or "_ptype" not in r:
            return v
        v = np.concatenate([v, [W * r["_ptype"]]])
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    C5.row_to_vec = vec
    ves_axis = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    F = json.load(open(f"{D}/k6_type_feat.json"))
    keys = [f"{r['case']}|{r['lesion_mask_idx']}" for r in rows]
    X = np.array([F[k]["x"] for k in keys]); Y = np.array([int(F[k]["type"] in (2, 3)) for k in keys])
    cases = sorted({r["case"] for r in rows}); cs = np.array([r["case"] for r in rows])
    out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
        f = np.array([fold[c] for c in cs])
        pred = {}
        for k in range(5):
            tr = np.where(f != k)[0]; te = np.where(f == k)[0]
            if arm != "base":
                p = np.zeros(len(rows))
                # 바깥 평가 행
                p[te] = type_rf(X[tr], Y[tr], sd).predict_proba(X[te])[:, 1]
                # 바깥 학습 행 — 안쪽 케이스 5겹 OOF
                tcs = sorted(set(cs[tr])); iperm = list(np.random.default_rng(100 + sd * 10 + k).permutation(tcs))
                ifold = {c: i % 5 for i, c in enumerate(iperm)}
                fi = np.array([ifold[c] for c in cs[tr]])
                for j in range(5):
                    a, b = tr[fi != j], tr[fi == j]
                    p[b] = type_rf(X[a], Y[a], sd).predict_proba(X[b])[:, 1]
                for i in range(len(rows)):
                    rows[i]["_ptype"] = float(p[i])
            m = C5.fit_model([rows[i] for i in tr], ves_axis, kind="rf", mirror=True)
            for i in te:
                pred[int(i)] = C5.predict_one(m, rows[i], 0.5)
        out[sd] = [pred[i] for i in range(len(rows))]
    return arm, out


def main():
    import multiprocessing as mp
    from k2_sym import metrics, code
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    truth = [r["gt_loc"] for r in rows]
    with mp.Pool(2) as p:
        res = dict(p.map(run_arm, ["base", "type"]))
    base = {s: metrics(truth, res["base"][s]) for s in range(5)}
    tt = {s: metrics(truth, res["type"][s]) for s in range(5)}
    d1 = np.array([tt[s][0] - base[s][0] for s in range(5)]); d2 = np.array([tt[s][1] - base[s][1] for s in range(5)])
    tgt = lambda g: code(g).split(".")[0] in ("1", "2")
    tb = np.mean([metrics(truth, res["base"][s], tgt)[0] for s in range(5)])
    ta = np.mean([metrics(truth, res["type"][s], tgt)[0] for s in range(5)])
    ok = d2.mean() >= 0.02 and d1.mean() >= -0.005 and (d2 > 0).sum() >= 4
    b = np.array(list(base.values())).mean(0); a = np.array(list(tt.values())).mean(0)
    print("# K6-2 유형 확률 블록 — train OOF 스크리닝 (중첩 · 케이스 5겹 × 시드5)\n")
    print("| 팔 | top1 | macro-recall | 표적 후순환 top1 |\n|---|---|---|---|")
    print(f"| base | {b[0]:.3f} | {b[1]:.3f} | {tb:.3f} |\n| +유형확률 | {a[0]:.3f} | {a[1]:.3f} | {ta:.3f} |")
    print(f"\nΔtop1 {d1.mean():+.3f} · Δmacro-recall {d2.mean():+.3f} · 시드별 Δmacro {' '.join(f'{x:+.3f}' for x in d2)} ({(d2 > 0).sum()}/5)")
    print(f"\n**관문 → {'통과 · e2e 진행' if ok else '미달 · e2e 안 건다'}**")
    json.dump(dict(ok=bool(ok), dtop1=float(d1.mean()), dmacro=float(d2.mean())), open(f"{D}/k6_gate2.json", "w"))


if __name__ == "__main__":
    main()
