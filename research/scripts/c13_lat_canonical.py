"""C13 — 좌우 canonical화 (증강이 아니라 출력공간 축소).

현재 미러는 *증강*이라 출력이 여전히 52-way다. canonical화는 다르다:
  1) 모든 병변을 우측 기준으로 뒤집어 **측면무관 26-way 라벨**을 예측
  2) 좌우는 별도 이진분류기로 붙임
  3) 둘을 합쳐 원래 52-way 라벨 복원
클래스당 실질 샘플이 2배가 된다 (AChA: 좌4+우5 = 9 -> 한 클래스로 9).

c8 진단 기준: 좌우 정확도가 이미 0.902라 이 접근의 상한이 거기 묶인다.
그래서 좌우 전용 이진분류기를 따로 학습해 그 0.902를 얼마나 올릴 수 있는지 먼저 잰다.
(좌우는 "어느 쪽 혈관이 가까운가"라 본래 쉬운 문제 — 52-way 안에 섞여 있어서 손해봤을 수 있다.)

사용: python c13_lat_canonical.py --feat <c10_feat_train.json> [--folds 5]
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

C9.row_vec = C10.row_vec_pos          # pos 블록 포함 (C10에서 최적 확인)


def canon(name):
    """측면무관 라벨. 'R-3.4 ...' / 'L-3.4 ...' -> '3.4 ...'"""
    return name[2:] if name[:2] in ("R-", "L-") else name


def side(name):
    return name[0] if name[:2] in ("R-", "L-") else "M"


def restore(canon_label, s):
    return canon_label if s == "M" else f"{s}-{canon_label}"


def fit_predict(Xtr, ytr, Xte, beta):
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                 random_state=0, n_jobs=-1).fit(Xtr, ytr)
    prior = collections.Counter(ytr)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float)
    P = clf.predict_proba(Xte)
    if beta > 0:
        P = P / (pri ** beta)
    return clf.classes_[np.argmax(P, axis=1)]


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
    print(f"[c13] 병변 {len(rows)}  52-way 클래스 {len(set(y))} -> "
          f"측면무관 {len(set(canon(v) for v in y))}\n")

    for beta in (0.0, 0.5, 1.0):
        y_side = np.empty(len(rows), dtype=object)
        y_canon = np.empty(len(rows), dtype=object)
        for tr, te in GroupKFold(n_splits=args.folds).split(np.zeros(len(rows)), y, groups):
            # 학습셋: 원본 + 미러 (미러는 canonical 라벨이 같고 side만 뒤집힘)
            Xtr, ytr_c, ytr_s = [], [], []
            for i in tr:
                Xtr.append(C10.row_vec_pos(rows[i], ves_axis, blocks, args.tau, False))
                ytr_c.append(canon(y[i])); ytr_s.append(side(y[i]))
                Xtr.append(C10.row_vec_pos(rows[i], ves_axis, blocks, args.tau, True))
                m = C5.mirror_name(y[i])
                ytr_c.append(canon(m)); ytr_s.append(side(m))
            Xtr = np.array(Xtr)
            Xte = np.array([C10.row_vec_pos(rows[i], ves_axis, blocks, args.tau, False) for i in te])
            y_canon[te] = fit_predict(Xtr, np.array(ytr_c), Xte, beta)
            y_side[te] = fit_predict(Xtr, np.array(ytr_s), Xte, 0.0)   # 좌우는 균형 문제 아님

        side_acc = float(np.mean([side(a) == b for a, b in zip(y, y_side)]))
        canon_acc = float(np.mean([canon(a) == b for a, b in zip(y, y_canon)]))
        comb = np.array([restore(c, s) for c, s in zip(y_canon, y_side)])
        t1 = float(np.mean(y == comb))
        mr, _ = C8.macro_recall(y, comb)
        ica = [(a, b) for a, b in zip(y, comb) if C8.group_of(a) == "3"]
        ia = float(np.mean([a == b for a, b in ica]))
        print(f"beta={beta}: 좌우 {side_acc:.3f} | 측면무관26 {canon_acc:.3f} "
              f"| 복원52 top-1 {t1:.3f} | macroRec {mr:.3f} | ICA {ia:.3f}")

    out = Path(args.feat).parent / "c13_cv_report.json"
    json.dump({"note": "좌우 canonical화", "n": len(rows)}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
