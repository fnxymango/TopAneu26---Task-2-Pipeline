"""C9 — 위치분류기 개선 실험 (c8 진단 결과를 표적으로).

c8(환자단위 5-fold CV, 268병변)이 밝힌 것:
  - macro-recall 0.38 vs top-1 0.68 — 흔한 클래스만 맞히고 희귀 클래스는 놓친다.
    공식지표는 52클래스 **균등평균(macro)** 이므로 이 격차가 최대 손실원.
  - 오류가 ICA(3.x)에 몰리고 전부 3.4 Pcom-junction으로 빨려든다(ICA 최빈 클래스).
    AChA(3.5)는 정확도 0~20%. 학습샘플 AChA 4~5 vs Pcom 29의 불균형이 직접 원인.
  - 좌우 0.902 / 해부그룹 0.951 — 오류는 전부 같은 그룹·같은 쪽 안에서 발생.

여기서 세 가지를 순서대로 검증한다:
  [1] macro 직접 최적화
      - kNN 빈도보정 지수 alpha 스윕: w = sim / prior**alpha  (기존 고정 0.5)
      - RF 확률에 사전확률 역보정: argmax p(c|x) / prior(c)**beta  (beta 스윕)
        공식지표가 macro면 사후확률 argmax는 최적이 아니다 — 균등 사전확률로 옮겨야 한다.
  [2] ICA 분지 상대비교
      절대거리 1/(1+d) 대신 분기점들 사이의 **경쟁적** 인코딩을 추가:
        rel_j = softmax(-d_j / tau)  — "어느 분지가 상대적으로 더 가까운가"
      Pcom/AChA/OA가 C7에서 몇 mm 안에 함께 분지하므로 절대거리로는 분리가 안 된다.
  [3] 기권 제거
      반경 밖이면 피처가 전영이 되어 예측을 포기한다(=확정 오답). --feat 를 큰 반경으로
      재추출한 파일로 바꿔 실행하면 그 효과를 볼 수 있다.

선택 기준은 **macro-recall** (c8은 top-1로 골랐는데 공식지표와 어긋난 실수였다).

사용:
  python c9_classifier_tune.py --feat <c5_feat_train.json> [--folds 5] [--tau 2.0]
"""
import argparse, json, collections
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8


def row_vec(r, ves_axis, blocks, tau, mirror=False):
    """dist(36) + ov(36) + bp(34) + rel(34, 경쟁적 인코딩)."""
    base = C5.row_to_vec(r, ves_axis, mirror=mirror)
    n = len(ves_axis)
    parts = []
    parts.append(base[:n] if "dist" in blocks else np.zeros(n))
    parts.append(base[n:2 * n] if "ov" in blocks else np.zeros(n))
    parts.append(base[2 * n:] if "bp" in blocks else np.zeros(C5.BP_DIM))

    if "rel" in blocks:
        bp = r.get("bp_mm") or [None] * C5.BP_DIM
        d = np.array([np.inf if x is None else float(x) for x in bp])
        if mirror:                                   # 미러 시 쌍 인덱스 재배치
            d2 = np.full(C5.BP_DIM, np.inf)
            for k, p in enumerate(C5.JUNCTION_PAIRS):
                key = frozenset((C5.mirror_name(p[0]), C5.mirror_name(p[1])))
                k2 = next((i for i, q in enumerate(C5.JUNCTION_PAIRS) if frozenset(q) == key), None)
                if k2 is not None:
                    d2[k2] = d[k]
            d = d2
        finite = np.isfinite(d)
        rel = np.zeros(C5.BP_DIM)
        if finite.any():
            z = -(d[finite] - d[finite].min()) / tau   # 최근접 분기점 대비 마진
            e = np.exp(z)
            rel[finite] = e / e.sum()
        parts.append(rel)
    else:
        parts.append(np.zeros(C5.BP_DIM))

    v = np.concatenate(parts)
    nn = np.linalg.norm(v)
    return v / nn if nn > 0 else v


def cv_run(rows, ves_axis, blocks, model, k, alpha, beta, tau, n_folds):
    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y_true = np.array([r["gt_loc"] for r in rows])
    y_pred = np.empty(len(rows), dtype=object)
    n_abstain = 0

    for tr_idx, te_idx in GroupKFold(n_splits=n_folds).split(np.zeros(len(rows)), y_true, groups):
        Xtr, ytr = [], []
        for i in tr_idx:
            Xtr.append(row_vec(rows[i], ves_axis, blocks, tau, False)); ytr.append(y_true[i])
            Xtr.append(row_vec(rows[i], ves_axis, blocks, tau, True))
            ytr.append(C5.mirror_name(y_true[i]))
        Xtr = np.array(Xtr); ytr = np.array(ytr)
        prior = collections.Counter(ytr)

        clf = None
        if model == "rf":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(Xtr, ytr)
            cls = clf.classes_
            pri = np.array([prior[c] for c in cls], dtype=float)

        for i in te_idx:
            v = row_vec(rows[i], ves_axis, blocks, tau, False)
            if np.linalg.norm(v) == 0:
                y_pred[i] = None; n_abstain += 1; continue
            if model == "rf":
                p = clf.predict_proba(v[None, :])[0]
                if beta > 0:
                    p = p / (pri ** beta)            # 사전확률 역보정 -> macro 정렬
                y_pred[i] = str(cls[int(np.argmax(p))])
                continue
            sims = Xtr @ v
            order = np.argsort(-sims)
            acc, used = {}, 0
            for j in order:
                if sims[j] <= 0 or used >= k:
                    break
                w = sims[j] / (prior[ytr[j]] ** alpha if alpha > 0 else 1.0)
                acc[ytr[j]] = acc.get(ytr[j], 0.0) + w
                used += 1
            y_pred[i] = max(acc, key=acc.get) if acc else None
            if y_pred[i] is None:
                n_abstain += 1
    return y_true, y_pred, n_abstain


def score(y_true, y_pred):
    top1 = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    mr, _ = C8.macro_recall(y_true, y_pred)
    ica = [(a, b) for a, b in zip(y_true, y_pred) if C8.group_of(a) == "3"]
    ica_acc = float(np.mean([a == b for a, b in ica])) if ica else None
    return top1, mr, ica_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--tag", default="r10")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.feat)) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    print(f"[c9] 병변 {len(rows)}  {args.folds}-fold 환자단위 CV  tau={args.tau}  feat={Path(args.feat).name}\n")

    trials = []
    def run(label, blocks, model, k=5, alpha=0.5, beta=0.0):
        yt, yp, nab = cv_run(rows, ves_axis, blocks, model, k, alpha, beta, args.tau, args.folds)
        t1, mr, ia = score(yt, yp)
        trials.append({"label": label, "blocks": list(blocks), "model": model, "k": k,
                       "alpha": alpha, "beta": beta, "top1": t1, "macro_recall": mr,
                       "ica_acc": ia, "abstain": nab})
        print(f"{label:<34}{t1:>8.3f}{mr:>10.3f}{(ia if ia is not None else 0):>9.3f}{nab:>8}")

    print(f"{'실험':<34}{'top-1':>8}{'macroRec':>10}{'ICA acc':>9}{'기권':>8}")
    print("--- 기준선 (c8 최적) ---")
    run("baseline dist+ov rf", ("dist", "ov"), "rf")
    run("baseline dist+ov+bp rf", ("dist", "ov", "bp"), "rf")
    run("baseline dist+ov+bp knn5 a=.5", ("dist", "ov", "bp"), "knn", k=5, alpha=0.5)

    print("--- [1] macro 직접 최적화: RF 사전확률 역보정 beta ---")
    for b in (0.25, 0.5, 0.75, 1.0):
        run(f"dist+ov+bp rf beta={b}", ("dist", "ov", "bp"), "rf", beta=b)

    print("--- [1] macro 직접 최적화: kNN 보정지수 alpha ---")
    for a in (0.0, 0.25, 0.75, 1.0):
        run(f"dist+ov+bp knn5 alpha={a}", ("dist", "ov", "bp"), "knn", k=5, alpha=a)

    print("--- [2] ICA 분지 상대비교 (rel 블록 추가) ---")
    run("dist+ov+bp+rel rf", ("dist", "ov", "bp", "rel"), "rf")
    run("dist+ov+rel rf", ("dist", "ov", "rel"), "rf")
    for b in (0.5, 0.75):
        run(f"dist+ov+bp+rel rf beta={b}", ("dist", "ov", "bp", "rel"), "rf", beta=b)
    run("dist+ov+bp+rel knn5 a=.75", ("dist", "ov", "bp", "rel"), "knn", k=5, alpha=0.75)

    best = max(trials, key=lambda t: t["macro_recall"])
    print(f"\n[최적 — macro-recall 기준] {best['label']}")
    print(f"  top-1 {best['top1']:.3f} | macro-recall {best['macro_recall']:.3f} "
          f"| ICA {best['ica_acc']:.3f} | 기권 {best['abstain']}")

    out = Path(args.feat).parent / f"c9_tune_{args.tag}.json"
    json.dump({"feat": args.feat, "tau": args.tau, "folds": args.folds,
               "trials": trials, "best": best}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
