"""C30 — 클래스별 가중치를 macro-MCC 에 직접 맞춘다 (2026-08-17).

진단(오늘): 43클래스 중 21개가 재현율 0 이고, 혼동이 거의 전부 같은 해부학적 가족
안에서 일어난다. 희소 클래스가 같은 가족의 빈발 형제에게 통째로 흡수된다
(R-3.2 n=2 -> R-3.3, R-1.3 n=2 -> R-1.1, AChA 는 좌우 9개 전부 오답).

공식지표의 구조적 비대칭:
  - GT 에 없는 클래스로 예측하면 tp=fn=0 이라 MCC 분모가 0 -> 그 클래스는 **0 그대로**.
    즉 벌점이 없다.
  - 반면 희소 클래스를 하나라도 맞히면 그 클래스 MCC 가 0 에서 크게 뛴다.
  => 희소 클래스로 과감히 예측하는 쪽이 유리한데, 대가는 원래 정답 클래스의 FN 하나뿐이고
     그 클래스가 빈발이면 MCC 손실이 미미하다.

그런데 지금은 전역 지수 하나(β)로 모든 클래스를 똑같이 민다. 그래서 β 를 키우면
희소 클래스는 살아나지만 빈발 클래스 정밀도가 같이 무너져 β=1.0 이 0.5 보다 나빴다.
클래스별로 다른 가중치가 옳은 정식화다.

  argmax_c  w_c * p(c|x)      (w 를 macro-MCC 에 좌표상승으로 맞춤)

43개 자유 파라미터를 268샘플에 맞추므로 **과적합이 진짜 위험**이다. 그래서
  1) w 는 작은 격자에서만 고른다 (연속 최적화 금지)
  2) **중첩 CV** 로 정직한 추정치를 함께 낸다 — 4폴드 OOF 로 w 를 맞추고 남은 폴드에서 평가.
     안쪽 점수와 바깥쪽 점수가 벌어지면 그만큼이 과적합이고, 그 경우 채택하지 않는다.

사용: python c30_mcc_weights.py --feat <c10_feat_train.json>
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8

GRID = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])   # 클래스 가중치 후보
ROUNDS = 3


def macro_mcc(yt, yp, classes=None):
    """GT 에 등장하는 클래스로 평균낸 인스턴스 수준 MCC (covered_gt 분모와 같은 성격)."""
    cs = sorted(set(yt)) if classes is None else classes
    N = len(yt)
    out = []
    for c in cs:
        tp = int(np.sum((yt == c) & (yp == c))); fp = int(np.sum((yt != c) & (yp == c)))
        fn = int(np.sum((yt == c) & (yp != c))); tn = N - tp - fp - fn
        d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out.append(((tp * tn - fp * fn) / d) if d > 0 else 0.0)
    return float(np.mean(out))


def oof_probs(X, Xm, y, groups, folds=5):
    """미러 증강 포함 환자단위 5-fold OOF 확률 (프로덕션 c5 fit_model 과 동일 구성)."""
    classes = sorted(set(y))
    ci = {c: i for i, c in enumerate(classes)}
    P = np.zeros((len(y), len(classes)), np.float32)
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        Xa = np.concatenate([X[tr], Xm[tr]])
        ya = np.concatenate([y[tr], [C5.mirror_name(v) for v in y[tr]]])
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(Xa, ya)
        pr = clf.predict_proba(X[te])
        for j, c in enumerate(clf.classes_):
            if c in ci:
                P[te, ci[c]] = pr[:, j]
    return np.array(classes), P / np.maximum(P.sum(1, keepdims=True), 1e-9)


def fit_weights(classes, P, y, w0=None, rounds=ROUNDS):
    """좌표상승: 클래스 하나씩 격자에서 macro-MCC 가 최대가 되는 가중치를 고른다."""
    w = np.ones(len(classes)) if w0 is None else w0.copy()
    best = macro_mcc(y, classes[np.argmax(P * w, 1)])
    for _ in range(rounds):
        improved = False
        for j in range(len(classes)):
            cur = w[j]
            for v in GRID:
                if v == cur:
                    continue
                w[j] = v
                s = macro_mcc(y, classes[np.argmax(P * w, 1)])
                if s > best + 1e-9:
                    best, cur, improved = s, v, True
            w[j] = cur
        if not improved:
            break
    return w, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
    C5.USE_POS = True
    rows = [r for r in json.load(open(a.feat)) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    X = np.array([C5.row_to_vec(r, ves_axis) for r in rows])
    Xm = np.array([C5.row_to_vec(r, ves_axis, mirror=True) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])
    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])

    classes, P = oof_probs(X, Xm, y, groups, a.folds)
    prior = collections.Counter(y)
    pri = np.array([max(prior.get(c, 1), 1) for c in classes], float)

    def rep(lab, yp):
        z = sum(1 for c in set(y) if np.sum(yp[y == c] == c) == 0)
        print(f"{lab:<30}{np.mean(yp == y):>8.3f}{C8.macro_recall(y, yp)[0]:>10.3f}"
              f"{macro_mcc(y, yp):>10.4f}{z:>6}/{len(set(y))}")
        return macro_mcc(y, yp)

    print(f"{'방식':<30}{'top-1':>8}{'macroRec':>10}{'macroMCC':>10}{'재현0':>12}")
    for b in (0.0, 0.5, 1.0):
        rep(f"전역 β={b}", classes[np.argmax(P / pri ** b, 1)])
    base = macro_mcc(y, classes[np.argmax(P / pri ** 0.5, 1)])

    # ── 클래스별 가중치 (전체 OOF 로 맞춤 = 낙관적 상한) ──
    w, s_in = fit_weights(classes, P, y, w0=1.0 / pri ** 0.5)
    rep("클래스별 w (전체로 맞춤)", classes[np.argmax(P * w, 1)])

    # ── 중첩 CV: 정직한 추정치 ──
    outer = list(GroupKFold(n_splits=a.folds).split(X, y, groups))
    pred_nested = np.empty(len(y), dtype=object)
    for tr, te in outer:
        w_in, _ = fit_weights(classes, P[tr], y[tr], w0=1.0 / pri ** 0.5)
        pred_nested[te] = classes[np.argmax(P[te] * w_in, 1)]
    s_out = rep("클래스별 w (중첩CV, 정직)", pred_nested)

    gap = s_in - s_out
    print(f"\n안쪽(맞춘 데이터) {s_in:.4f} · 바깥쪽(중첩CV) {s_out:.4f} · 과적합 폭 {gap:.4f}")
    print(f"기준선 전역 β=0.5 {base:.4f} 대비 정직한 이득 {s_out - base:+.4f}")
    verdict = "채택 후보" if s_out > base + 0.01 else "기각 — 과적합분을 빼면 이득 없음"
    print(f"[판정] {verdict}")

    json.dump({"classes": list(classes), "weights": [float(v) for v in w],
               "macro_mcc_baseline_b0.5": base, "macro_mcc_insample": s_in,
               "macro_mcc_nested": s_out, "overfit_gap": gap, "verdict": verdict},
              open(A / "c30_mcc_weights.json", "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {A / 'c30_mcc_weights.json'}")


if __name__ == "__main__":
    main()
