"""C8 — 위치분류기 환자단위 교차검증 + 오류 진단.

왜 필요한가(2026-08-14): C5 스윕이 val 43병변으로 모델을 골랐는데, 1~2개 차이가
곧 2~5%p라 RF/kNN 우열이 통계적으로 구분되지 않는다. 여기서는 train 268병변을
**환자단위 5-fold**로 나눠 재선택한다(같은 환자가 train/test 양쪽에 들어가지 않게).

그리고 더 중요한 것 — 어디서 틀리는지 모른 채 피처를 더 넣으면 43개 노이즈에 과적합된다.
그래서 CV 예측으로 다음을 뽑는다:
  1) 피처블록 ablation — 혈관거리 / +sac점유율 / +분기점  각각의 기여분
     (분기점 피처가 실제로 오라클 61.0% 천장을 뚫었는지 확인하는 유일한 방법)
  2) 클래스별 정확도 + 오류 질량 상위 — 어느 클래스를 고쳐야 이득이 큰지
  3) 혼동 쌍 상위 — 무엇과 무엇을 헷갈리는지
  4) 좌우(laterality)와 해부그룹(1.x~5.x) 각각의 정확도
     -> 좌우가 거의 맞는다면 52-way를 26-way + 좌우이진으로 분해할 수 있다

공식지표가 52클래스 균등평균이므로 top-1 정확도와 **macro-recall**을 같이 본다.

사용:
  python c8_classifier_cv.py --feat <c5_feat_train.json> [--folds 5]
"""
import argparse, json, collections, re
from pathlib import Path

import numpy as np

import d9xx_lib as L
import c5_location_v2 as C5


def patient_map():
    rel = json.load(open(L.SPLIT_JSON))
    return {c["case_id"]: c["patient_id"] for part in rel["cases"].values() for c in part}


def laterality(name):
    if name.startswith("R-"):
        return "R"
    if name.startswith("L-"):
        return "L"
    return "M"                      # 중앙 구조 (BA trunk, Acom complex, BA tip ...)


def group_of(name):
    """'R-3.4 ICA C7-Pcom-junction' -> '3' (해부 대분류: 1 후순환/2 PCA/3 ICA/4 ACA/5 MCA)"""
    m = re.search(r"(\d+)\.", name)
    return m.group(1) if m else "?"


def vec_block(r, ves_axis, blocks, mirror=False):
    """피처블록 선택적 구성. blocks 안에 'dist'/'ov'/'bp' 포함 여부로 마스킹."""
    v = C5.row_to_vec(r, ves_axis, mirror=mirror)
    n = len(ves_axis)
    out = v.copy()
    if "dist" not in blocks:
        out[:n] = 0
    if "ov" not in blocks:
        out[n:2 * n] = 0
    if "bp" not in blocks:
        out[2 * n:] = 0
    nn = np.linalg.norm(out)
    return out / nn if nn > 0 else out


def build_xy(rows, ves_axis, blocks, mirror):
    X, y, g = [], [], []
    pm = patient_map()
    for r in rows:
        X.append(vec_block(r, ves_axis, blocks, False)); y.append(r["gt_loc"])
        g.append(pm.get(r["case"], r["case"]))
        if mirror:
            X.append(vec_block(r, ves_axis, blocks, True))
            y.append(C5.mirror_name(r["gt_loc"])); g.append(pm.get(r["case"], r["case"]))
    return np.array(X), np.array(y), np.array(g)


def cv_predict(rows, ves_axis, blocks, model, k, mirror, balance, n_folds):
    """환자단위 fold. 미러 샘플은 학습에만 쓰고 평가는 원본 병변으로만 한다."""
    from sklearn.model_selection import GroupKFold
    pm = patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y_true = np.array([r["gt_loc"] for r in rows])
    y_pred = np.empty(len(rows), dtype=object)

    gkf = GroupKFold(n_splits=n_folds)
    for tr_idx, te_idx in gkf.split(np.zeros(len(rows)), y_true, groups):
        tr_rows = [rows[i] for i in tr_idx]
        Xtr, ytr, _ = build_xy(tr_rows, ves_axis, blocks, mirror)
        prior = collections.Counter(ytr)
        if model == "rf":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(Xtr, ytr)
        for i in te_idx:
            v = vec_block(rows[i], ves_axis, blocks, False)
            if np.linalg.norm(v) == 0:
                y_pred[i] = None; continue
            if model == "rf":
                y_pred[i] = str(clf.predict(v[None, :])[0]); continue
            sims = Xtr @ v
            order = np.argsort(-sims)
            acc, used = {}, 0
            for j in order:
                if sims[j] <= 0 or used >= k:
                    break
                w = sims[j] / (np.sqrt(prior[ytr[j]]) if balance else 1.0)
                acc[ytr[j]] = acc.get(ytr[j], 0.0) + w
                used += 1
            y_pred[i] = max(acc, key=acc.get) if acc else None
    return y_true, y_pred


def macro_recall(y_true, y_pred):
    cls = sorted(set(y_true))
    rec = []
    for c in cls:
        m = y_true == c
        rec.append(float(np.mean(y_pred[m] == c)))
    return float(np.mean(rec)), len(cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.feat)) if r.get("gt_loc")]
    ves_names = L.vessel_dense_names()
    ves_axis, _ = C5.build_feature_axes(ves_names)
    print(f"[c8] 병변 {len(rows)}개, 환자 {len(set(patient_map().get(r['case'], r['case']) for r in rows))}명, "
          f"클래스 {len(set(r['gt_loc'] for r in rows))}종, {args.folds}-fold 환자단위 CV\n")

    # --- 1) 피처블록 ablation x 모델 ---
    configs = []
    for blocks in (("dist",), ("dist", "ov"), ("dist", "bp"), ("dist", "ov", "bp")):
        for model, k, bal in (("knn", 3, False), ("knn", 5, True), ("rf", 5, True)):
            configs.append((blocks, model, k, bal))

    print(f"{'피처블록':<22}{'모델':>6}{'k':>3}{'bal':>5}{'top-1':>8}{'macroRec':>10}")
    results = []
    for blocks, model, k, bal in configs:
        yt, yp = cv_predict(rows, ves_axis, blocks, model, k, True, bal, args.folds)
        acc = float(np.mean([a == b for a, b in zip(yt, yp)]))
        mr, ncls = macro_recall(yt, yp)
        tag = "+".join(blocks)
        print(f"{tag:<22}{model:>6}{k:>3}{str(bal):>5}{acc:>8.3f}{mr:>10.3f}")
        results.append({"blocks": list(blocks), "model": model, "k": k, "balance": bal,
                        "top1": acc, "macro_recall": mr})

    best = max(results, key=lambda r: r["top1"])
    print(f"\n[최적] {'+'.join(best['blocks'])} / {best['model']} k={best['k']} "
          f"bal={best['balance']} -> top-1 {best['top1']:.3f}, macro-recall {best['macro_recall']:.3f}")

    # --- 2) 최적 설정으로 오류 진단 ---
    yt, yp = cv_predict(rows, ves_axis, tuple(best["blocks"]), best["model"],
                        best["k"], True, best["balance"], args.folds)

    lat_ok = float(np.mean([laterality(a) == laterality(b) for a, b in zip(yt, yp) if b]))
    grp_ok = float(np.mean([group_of(a) == group_of(b) for a, b in zip(yt, yp) if b]))
    print(f"\n[분해] 좌우(laterality) 정확도 {lat_ok:.3f} | 해부그룹(1.x~5.x) 정확도 {grp_ok:.3f}")
    print("  -> 좌우가 높다면 52-way를 '26-way x 좌우이진'으로 분해할 여지가 있다")

    print(f"\n=== 클래스별 오류 질량 상위 15 (n x 오답률) ===")
    print(f"{'위치클래스':<34}{'n':>4}{'정확도':>8}{'오류수':>7}")
    per = collections.defaultdict(lambda: [0, 0])
    for a, b in zip(yt, yp):
        per[a][0] += 1
        per[a][1] += int(a == b)
    rank = sorted(per.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))
    for name, (n, ok) in rank[:15]:
        print(f"{name:<34}{n:>4}{ok / n:>8.2f}{n - ok:>7}")

    print(f"\n=== 혼동 쌍 상위 15 (정답 -> 오답) ===")
    conf = collections.Counter((a, b) for a, b in zip(yt, yp) if a != b)
    for (a, b), c in conf.most_common(15):
        print(f"  {c:>2}x  {a[:30]:<30} -> {str(b)[:30]}")

    out = args.out or str(Path(args.feat).parent / "c8_cv_report.json")
    json.dump({"n_lesions": len(rows), "folds": args.folds, "ablation": results,
               "best": best, "laterality_acc": lat_ok, "group_acc": grp_ok,
               "per_class": {k: {"n": v[0], "correct": v[1]} for k, v in per.items()},
               "confusions": [{"true": a, "pred": b, "n": c} for (a, b), c in conf.most_common(40)]},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
