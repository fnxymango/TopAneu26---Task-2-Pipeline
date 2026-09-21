"""E3 — RF vs ET 를 268병변 환자단위 CV 로 직접 겨룬다 (2026-08-19).

E2 결과: test 에서 두 모델은 **정답 개수가 같았다**(44/63, 3승3패). 그런데 cov.MCC 는
ET 가 +0.0303 높다. 차이가 전부 "어느 클래스를 맞혔나"에서 나온 것이다.
그래서 클래스 구성 운이 안 섞이는 지표(top-1 정확도)를, 표본이 3배 많은 곳
(268병변)에서, 여러 시드로 재본다.

macro-MCC 도 같이 내되 판정은 top-1 로 한다 — macro-MCC 는 43개 클래스로 나누므로
여기서도 희소 클래스 운을 탄다.

⚠️ 이 스크립트의 숫자는 학습코호트(268병변/43클래스) 기준이라 **test 와 비교 불가**이며
   팀 표·노션에 올리지 않는다 (PROJECT_RULES.md 6-1c). 용도는 설정 간 **순위**뿐이다.
"""
import argparse, collections, json
import numpy as np
from sklearn.model_selection import GroupKFold

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8
from c30_mcc_weights import macro_mcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,et")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()
    A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
    C5.USE_POS = True

    rows = [r for r in json.load(open(A / "c10_feat_train.json")) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    y = np.array([r["gt_loc"] for r in rows])
    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    kinds = a.models.split(",")
    print(f"[e3] 병변 {len(rows)} · 환자 {len(set(groups))} · 클래스 {len(set(y))} "
          f"· 모델 {kinds} · β={a.beta} · 시드 {a.seeds} x {a.folds}폴드", flush=True)

    res = {k: {"acc": [], "mcc": []} for k in kinds}
    wins = collections.Counter()
    for s in range(a.seeds):
        idx = np.random.RandomState(s).permutation(len(rows))
        pred = {k: np.empty(len(rows), dtype=object) for k in kinds}
        for tr, te in GroupKFold(n_splits=a.folds).split(idx, y[idx], groups[idx]):
            tri, tei = idx[tr], idx[te]
            sub = [rows[i] for i in tri]
            for k in kinds:
                C5.CONF_TAU = 0.0
                m = C5.fit_model(sub, ves_axis, kind=k, mirror=True, balance=True)
                for i in tei:
                    pred[k][i] = C5.predict_one(m, rows[i], a.beta)
        for k in kinds:
            acc = float(np.mean(pred[k] == y)); mc = macro_mcc(y, pred[k])
            res[k]["acc"].append(acc); res[k]["mcc"].append(mc)
        if len(kinds) == 2:
            k1, k2 = kinds
            d = [(pred[k1][i] == y[i], pred[k2][i] == y[i]) for i in range(len(rows))]
            w1 = sum(1 for p, q in d if p and not q); w2 = sum(1 for p, q in d if q and not p)
            wins[k1] += w1; wins[k2] += w2
            print(f"  시드{s}: {k1} top1 {res[k1]['acc'][-1]:.4f} / {k2} {res[k2]['acc'][-1]:.4f}"
                  f"  · 갈린병변 {w1+w2} ({k1} {w1} : {w2} {k2})", flush=True)

    print(f"\n{'모델':<8}{'top-1 평균':>12}{'표준편차':>10}{'macroMCC 평균':>15}{'표준편차':>10}")
    for k in kinds:
        A_, M_ = np.array(res[k]["acc"]), np.array(res[k]["mcc"])
        print(f"{k:<8}{A_.mean():>12.4f}{A_.std():>10.4f}{M_.mean():>15.4f}{M_.std():>10.4f}")
    if len(kinds) == 2:
        k1, k2 = kinds
        d = np.array(res[k2]["acc"]) - np.array(res[k1]["acc"])
        print(f"\n  top-1 차이 {k2}-{k1}: {d.mean():+.4f} ± {d.std():.4f}  "
              f"(시드별 {' '.join(f'{v:+.4f}' for v in d)})")
        print(f"  누적 승패 {k2} {wins[k2]} : {wins[k1]} {k1}")
        v = "실재" if d.mean() > 2 * (d.std() / np.sqrt(len(d)) + 1e-9) else "노이즈 구간"
        print(f"  [판정] {v}")
    json.dump({"models": kinds, "beta": a.beta, "seeds": a.seeds, "folds": a.folds,
               "results": res, "wins": dict(wins)},
              open(A / "e3_model_cv.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
