#!/usr/bin/env python3
"""K1X — K1 위에 얹을 두 가지 스크리닝 (train OOF 만 · test·val 안 봄 · 2026-09-16 사용자 "이어돌려").

① 확률평균 가중치 w : p = (1−w)·기준표모델 + w·예측혈관표모델. 현재 K1 은 w=0.5. w ∈ {0.3,0.4,0.6,0.7} 를 본다.
② 균형 부트스트랩 RF(brf) : class_weight="balanced_subsample" — 트리마다 부트스트랩을 클래스 균형으로 다시 뽑는다.
   현재는 전체 한 번만 가중하는 "balanced" 라 트리 하나가 희귀 클래스를 아예 못 보는 일이 생긴다. 코드에 구현만 있고 평가 기록 없음.
   두 모델 모두 brf 로 바꾸고 w=0.5 로 둔다.
주 평가행 = 검출 blob TP 행(추론 조건 · v4d_feat_blob_oof) · 참고 = GT 병변 행. 케이스 묶음 5폴드 × 시드 0~4.
── 관문 (결과 보기 전 고정 · 2026-09-16) ────────────────────────────────
 통과 = 추론 조건 행에서 K1(w=0.5·rf) 대비 [Δtop1 ≥ +0.005] ∧ [Δmacro-recall ≥ +0.005] ∧ [Δmacro>0 시드 ≥ 4/5]
 통과 팔이 여럿이면 Δmacro 가 큰 하나만 e2e(K0)로 올린다. 아무 것도 못 넘으면 둘 다 닫는다.
"""
import json, os, sys
import numpy as np
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, D); os.environ.setdefault("TOPANEU_ROOT", R)
ARMS = [("k1", 0.5, "rf"), ("w03", 0.3, "rf"), ("w04", 0.4, "rf"), ("w06", 0.6, "rf"), ("w07", 0.7, "rf"), ("brf", 0.5, "brf")]


class WAvg:
    def __init__(self, ms, w):
        self.models = ms; self.w = w; self.classes_ = ms[0].classes_

    def predict_proba(self, X):
        idx = {c: j for j, c in enumerate(self.classes_)}
        acc = np.zeros((len(X), len(self.classes_)))
        for m, wt in zip(self.models, (1.0 - self.w, self.w)):
            p = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                if c in idx:
                    acc[:, idx[c]] += wt * p[:, j]
        return acc

    def predict(self, X):
        return np.array(self.classes_)[np.argmax(self.predict_proba(X), axis=1)]


def job(args):
    name, w, kind = args
    import c5_location_v2 as C5
    C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
    ax = [C5.L.vessel_dense_names()[i] for i in sorted(C5.L.vessel_dense_names())]
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    pv = [r for r in json.load(open(f"{A}/c10_feat_train_predves_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(f"{A}/v4d_feat_blob_oof.json")) if r.get("gt_loc")]
    cases = sorted({r["case"] for r in base})
    out = {}
    for sd in range(5):
        os.environ["CLF_SEED"] = str(sd)
        perm = list(np.random.default_rng(sd).permutation(cases)); fold = {c: i % 5 for i, c in enumerate(perm)}
        pg, pb = [None] * len(base), [None] * len(blob)
        for k in range(5):
            m1 = C5.fit_model([r for r in base if fold[r["case"]] != k], ax, kind=kind, mirror=True)
            m2 = C5.fit_model([r for r in pv if fold[r["case"]] != k], ax, kind=kind, mirror=True)
            m = dict(m1); m["clf"] = WAvg([m1["clf"], m2["clf"]], w); m["kind"] = "rf"
            for i, r in enumerate(base):
                if fold[r["case"]] == k:
                    pg[i] = C5.predict_one(m, r, 0.5)
            for i, r in enumerate(blob):
                if fold[r["case"]] == k:
                    pb[i] = C5.predict_one(m, r, 0.5)
        out[sd] = (pg, pb)
    return name, out


def main():
    from multiprocessing import Pool
    from k2_sym import metrics
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    blob = [r for r in json.load(open(f"{A}/v4d_feat_blob_oof.json")) if r.get("gt_loc")]
    tg, tb = [r["gt_loc"] for r in base], [r["gt_loc"] for r in blob]
    with Pool(6) as p:
        res = dict(p.map(job, ARMS, chunksize=1))
    print("# K1X — 확률평균 가중치 · 균형 부트스트랩 RF 스크리닝 (train OOF · 시드 0~4)\n")
    print("| 팔 | 추론조건 top1 | 추론조건 macro | Δtop1 | Δmacro | Δmacro>0 시드 | GT행 top1 | 판정 |\n|---|---|---|---|---|---|---|---|")
    M = {n: np.array([metrics(tb, res[n][s][1]) for s in range(5)]) for n, _, _ in ARMS}
    G = {n: np.array([metrics(tg, res[n][s][0]) for s in range(5)]) for n, _, _ in ARMS}
    ok = {}
    for n, w, kind in ARMS:
        d1 = M[n][:, 0] - M["k1"][:, 0]; d2 = M[n][:, 1] - M["k1"][:, 1]
        pas = n != "k1" and d1.mean() >= 0.005 and d2.mean() >= 0.005 and (d2 > 0).sum() >= 4
        ok[n] = bool(pas)
        print(f"| {n} | {M[n][:,0].mean():.3f} | {M[n][:,1].mean():.3f} | {d1.mean():+.3f} | {d2.mean():+.3f} | "
              f"{(d2>0).sum()}/5 | {G[n][:,0].mean():.3f} | {'기준' if n=='k1' else ('통과' if pas else '미달')} |")
    win = max([n for n in ok if ok[n]], key=lambda n: (M[n][:, 1] - M["k1"][:, 1]).mean(), default=None)
    print(f"\n**관문 → {'통과 · ' + win + ' 만 e2e(K0)' if win else '미달 · 두 축 모두 닫음'}**")
    json.dump(dict(win=win, ok=ok), open(f"{D}/k1x_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
