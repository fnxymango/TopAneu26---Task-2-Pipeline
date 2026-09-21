"""C21 — 분류기 밀어올리기 3종 (2026-08-17). 목표: test MCC > 0.3.

① 케이스 수준 유일성 제약
   한 케이스에 같은 위치클래스 동맥류가 둘 있는 경우는 드물다(공식 README 명시).
   그런데 지금은 병변마다 독립 분류라 중복 클래스가 나오고, 중복분은 곧바로 FP가 된다.
   확률행렬에 헝가리안 할당을 걸어 케이스 내 클래스를 유일하게 강제한다.

② 그래디언트 부스팅
   지금까지 kNN / RF 만 썼다. 268샘플 x 112차원 소규모 정형 데이터에서는
   HistGradientBoosting 이 RF 를 이기는 경우가 흔하다.

③ 분류기 앙상블
   RF + GB + kNN 의 확률 평균. 서로 다른 실패 모드를 상쇄한다.

모두 환자단위 5-fold CV 로 재고, 선택 기준은 macro-recall(공식지표와 정렬) 과
클래스 균등 MCC 근사치를 함께 본다.

사용: python c21_classifier_push.py --feat <c10_feat_train.json>
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8


def build(rows, ves_axis, idx, mirror=True):
    X, y = [], []
    for i in idx:
        X.append(C5.row_to_vec(rows[i], ves_axis, mirror=False)); y.append(rows[i]["gt_loc"])
        if mirror:
            X.append(C5.row_to_vec(rows[i], ves_axis, mirror=True))
            y.append(C5.mirror_name(rows[i]["gt_loc"]))
    return np.array(X), np.array(y)


def fit_probs(kind, Xtr, ytr, Xte, beta):
    """반환: (클래스배열, 사전확률 보정된 확률행렬)"""
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(Xtr, ytr)
    elif kind == "gb":
        # 소규모/고차원이라 얕고 강하게 규제. class_weight 미지원이라 sample_weight로 균형.
        prior = collections.Counter(ytr)
        sw = np.array([1.0 / prior[v] for v in ytr]); sw *= len(sw) / sw.sum()
        clf = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=.06,
                                             l2_regularization=1.0, min_samples_leaf=3,
                                             random_state=0).fit(Xtr, ytr, sample_weight=sw)
    else:
        raise ValueError(kind)
    P = clf.predict_proba(Xte)
    prior = collections.Counter(ytr)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float)
    if beta > 0:
        P = P / (pri ** beta)
    return clf.classes_, P / P.sum(axis=1, keepdims=True)


def assign_unique(cls, P, case_of):
    """케이스별로 헝가리안 할당 — 같은 케이스 안에서 클래스가 겹치지 않게."""
    out = np.empty(len(P), dtype=object)
    by_case = collections.defaultdict(list)
    for i, c in enumerate(case_of):
        by_case[c].append(i)
    for c, idx in by_case.items():
        if len(idx) == 1:
            out[idx[0]] = cls[int(np.argmax(P[idx[0]]))]; continue
        cost = -np.log(np.clip(P[idx], 1e-9, None))
        r, cc = linear_sum_assignment(cost)
        for ri, ci in zip(r, cc):
            out[idx[ri]] = cls[ci]
    return out


def score(y, pred):
    top1 = float(np.mean(y == pred))
    mr, _ = C8.macro_recall(y, pred)
    ica = [(a, b) for a, b in zip(y, pred) if C8.group_of(a) == "3"]
    return top1, mr, float(np.mean([a == b for a, b in ica]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    C5.USE_POS = True
    rows = [r for r in json.load(open(a.feat)) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])
    case_of = np.array([r["case"] for r in rows])
    dup = sum(v - 1 for v in collections.Counter(
        [(r["case"], r["gt_loc"]) for r in rows]).values() if v > 1)
    print(f"[c21] 병변 {len(rows)} · 같은 케이스에 같은 클래스가 실제로 중복되는 경우 {dup}건\n")

    gkf = list(GroupKFold(n_splits=a.folds).split(np.zeros(len(rows)), y, groups))
    cache = {}

    def run(label, kinds, beta, unique):
        pred = np.empty(len(rows), dtype=object)
        for fi, (tr, te) in enumerate(gkf):
            Xtr, ytr = build(rows, ves_axis, tr)
            Xte = np.array([C5.row_to_vec(rows[i], ves_axis) for i in te])
            Ps, cls = [], None
            for k in kinds:
                key = (fi, k, beta)
                if key not in cache:
                    cache[key] = fit_probs(k, Xtr, ytr, Xte, beta)
                c_, P_ = cache[key]
                cls = c_; Ps.append(P_)
            P = np.mean(Ps, axis=0)
            if unique:
                pred[te] = assign_unique(cls, P, case_of[te])
            else:
                pred[te] = cls[np.argmax(P, axis=1)]
        t1, mr, ic = score(y, pred)
        n_dup = sum(v - 1 for v in collections.Counter(
            list(zip(case_of, pred))).values() if v > 1)
        print(f"{label:<34}{t1:>8.3f}{mr:>10.3f}{ic:>8.3f}{n_dup:>9}")
        return {"label": label, "top1": t1, "macro_recall": mr, "ica": ic, "dup_pred": n_dup}

    print(f"{'실험':<34}{'top-1':>8}{'macroRec':>10}{'ICA':>8}{'중복예측':>9}")
    res = []
    print("--- 기준선 ---")
    res.append(run("rf  β=1.0  (현재 최고)", ["rf"], 1.0, False))
    print("--- ① 케이스 유일성 제약 ---")
    res.append(run("rf  β=1.0  + unique", ["rf"], 1.0, True))
    print("--- ② 그래디언트 부스팅 ---")
    for b in (0.5, 1.0):
        res.append(run(f"gb  β={b}", ["gb"], b, False))
    res.append(run("gb  β=1.0  + unique", ["gb"], 1.0, True))
    print("--- ③ 앙상블 (rf + gb) ---")
    res.append(run("rf+gb  β=1.0", ["rf", "gb"], 1.0, False))
    res.append(run("rf+gb  β=1.0  + unique", ["rf", "gb"], 1.0, True))

    best = max(res, key=lambda r: r["macro_recall"])
    base = res[0]["macro_recall"]
    print(f"\n[최적] {best['label']}  macro-recall {best['macro_recall']:.3f} "
          f"(기준선 {base:.3f}, {best['macro_recall']/base-1:+.1%})")
    out = Path(a.feat).parent / "c21_push_report.json"
    json.dump({"n": len(rows), "gt_duplicates": dup, "trials": res, "best": best},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
