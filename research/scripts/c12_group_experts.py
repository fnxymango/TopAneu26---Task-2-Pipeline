"""C12 — 해부그룹 조건부 전문가.

c8 진단: 해부그룹(1.x 후순환 / 2.x PCA / 3.x ICA / 4.x ACA / 5.x MCA) 정확도가 0.951인데
전체 52-way는 0.62~0.68. 즉 **오류가 전부 그룹 내부에서 발생**한다.
그런데 지금은 한 모델이 52-way를 통째로 푼다.

여기서는 2단으로 나눈다:
  1단 그룹 분류기(5-way, 쉬움)  -> 2단 그룹별 전문가
ICA 전문가는 117병변/14클래스만 다루므로 샘플/클래스 비가 6.2 -> 8.4로 개선되고,
무엇보다 Pcom 흡인(모든 ICA 오류가 3.4로 빨려듦)을 **ICA 안에서만** 균형 맞추면 된다.

앞서 기각한 '26-way x 좌우'와 다르다 — 그건 이미 90% 맞는 축을 다시 맞히는 거였고,
이건 오류가 실제로 집중된 축을 쪼갠다.

두 가지 라우팅을 비교한다:
  hard  : 1단이 고른 그룹의 전문가에게만 보냄 (1단 오류가 그대로 전파)
  soft  : 그룹 확률 x 전문가 확률을 모두 곱해 argmax (1단 오류를 2단이 복구 가능)

사용: python c12_group_experts.py --feat <c10_feat_train.json> [--folds 5]
"""
import argparse, json, collections
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8
import c9_classifier_tune as C9
import c10_landmark_coords as C10

C9.row_vec = C10.row_vec_pos


def rf(X, y, beta):
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(X, y)
    prior = collections.Counter(y)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float)
    return clf, (pri ** beta if beta > 0 else np.ones(len(clf.classes_)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tau", type=float, default=2.0)
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.feat)) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    blocks = ("dist", "ov", "bp", "pos")
    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])
    g = np.array([C8.group_of(v) for v in y])
    print(f"[c12] 병변 {len(rows)}  그룹 분포: {dict(collections.Counter(g))}\n")

    for beta in (0.5, 1.0):
        pred_hard = np.empty(len(rows), dtype=object)
        pred_soft = np.empty(len(rows), dtype=object)
        grp_ok = 0
        for tr, te in GroupKFold(n_splits=args.folds).split(np.zeros(len(rows)), y, groups):
            Xtr, ytr = [], []
            for i in tr:
                Xtr.append(C10.row_vec_pos(rows[i], ves_axis, blocks, args.tau, False)); ytr.append(y[i])
                Xtr.append(C10.row_vec_pos(rows[i], ves_axis, blocks, args.tau, True))
                ytr.append(C5.mirror_name(y[i]))
            Xtr = np.array(Xtr); ytr = np.array(ytr)
            gtr = np.array([C8.group_of(v) for v in ytr])
            Xte = np.array([C10.row_vec_pos(rows[i], ves_axis, blocks, args.tau, False) for i in te])

            gclf, gpri = rf(Xtr, gtr, 0.0)                      # 1단: 그룹 (균형 문제 아님)
            gp = gclf.predict_proba(Xte)
            ghat = gclf.classes_[np.argmax(gp, axis=1)]
            grp_ok += int(np.sum(ghat == g[te]))

            experts = {}
            for gg in np.unique(gtr):
                m = gtr == gg
                if len(np.unique(ytr[m])) < 2:                  # 클래스 1개면 상수 예측
                    experts[gg] = (None, ytr[m][0]); continue
                experts[gg] = (rf(Xtr[m], ytr[m], beta), None)

            for j, i in enumerate(te):
                # hard 라우팅
                e = experts.get(ghat[j])
                if e is None:
                    pred_hard[i] = None
                elif e[1] is not None:
                    pred_hard[i] = e[1]
                else:
                    clf, pri = e[0]
                    p = clf.predict_proba(Xte[j:j + 1])[0] / pri
                    pred_hard[i] = clf.classes_[int(np.argmax(p))]
                # soft 라우팅 — 그룹확률 x 전문가확률
                best, bestp = None, -1.0
                for gi, gg in enumerate(gclf.classes_):
                    e2 = experts.get(gg)
                    if e2 is None:
                        continue
                    if e2[1] is not None:
                        sc = gp[j, gi]
                        if sc > bestp: best, bestp = e2[1], sc
                        continue
                    clf, pri = e2[0]
                    p = clf.predict_proba(Xte[j:j + 1])[0] / pri
                    p = p / p.sum()
                    k = int(np.argmax(p))
                    sc = gp[j, gi] * p[k]
                    if sc > bestp: best, bestp = clf.classes_[k], sc
                pred_soft[i] = best

        for label, pred in (("hard", pred_hard), ("soft", pred_soft)):
            t1 = float(np.mean(y == pred))
            mr, _ = C8.macro_recall(y, pred)
            ica = [(a, b) for a, b in zip(y, pred) if C8.group_of(a) == "3"]
            ia = float(np.mean([a == b for a, b in ica]))
            print(f"beta={beta} {label:>4}: top-1 {t1:.3f} | macroRec {mr:.3f} | ICA {ia:.3f}")
        print(f"          1단 그룹 정확도 {grp_ok/len(rows):.3f}")

    out = Path(args.feat).parent / "c12_cv_report.json"
    json.dump({"note": "그룹 조건부 전문가", "n": len(rows)}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
