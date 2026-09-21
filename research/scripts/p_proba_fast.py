#!/usr/bin/env python
"""병변별 확률행렬 — 피처행렬 재사용판 (2026-08-26).

앞선 p_export_proba.py 는 (a) 시드마다 케이스를 다시 읽어 추출했고 — row_to_vec 은 시드와
무관하므로 5배 낭비 — (b) OOF 폴드마다 RF 의 classes_ 길이가 달라 배열로 쌓다 터졌다.
여기서는 feat_*.npz 를 재사용하고, **모든 폴드의 확률을 공통 클래스축에 정렬**한다.
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import d9xx_lib as L, c5_location_v2 as C5

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
BETA, TAU = 0.5, 0.5
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
CLASSES = sorted(set(list(yt) + list(ym)))          # 공통 클래스축 (미러 포함)
CI = {c: i for i, c in enumerate(CLASSES)}
print(f"공통 클래스축 {len(CLASSES)}개 · 학습 {len(Xt)}병변 (미러 후 {2*len(Xt)})")


def fit(mask, seed):
    """mask: 학습에 쓸 원본행 불리언. 미러 확장 후 RF."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.concatenate([Xt[mask], Xm[mask]]); y = np.concatenate([yt[mask], ym[mask]])
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1,
                                 class_weight="balanced", random_state=seed, n_jobs=4).fit(X, y)
    pri = np.array([collections.Counter(y)[c] for c in clf.classes_], dtype=float)
    return clf, pri


def proba(clf, pri, X):
    """β/τ 보정 후 공통 클래스축으로 정렬한 (n, C)."""
    P = clf.predict_proba(X)
    hi = P.max(axis=1) >= TAU
    Pc = P.copy()
    Pc[~hi] = Pc[~hi] / (pri ** BETA)
    Pc = Pc / np.maximum(Pc.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    for j, c in enumerate(clf.classes_):
        out[:, CI[c]] = Pc[:, j]
    return out


for tag in ("test", "val", "trainoof"):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True)
    X, truth, case, les = f["X"], f["truth"], f["case"], f["lesion"]
    for sd in range(5):
        o = A / f"proba_{tag}_s{sd}.npz"
        if o.exists(): continue
        if tag in ("test", "val"):
            clf, pri = fit(np.ones(len(Xt), bool), sd)
            P = proba(clf, pri, X)
        else:
            folds = json.load(open(SP720))
            P = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
            done = np.zeros(len(X), bool)
            for k, fd in enumerate(folds):
                va = set(fd["val"])
                sel = np.array([c in va for c in case])
                if not sel.any(): continue
                clf, pri = fit(np.array([c not in va for c in ct]), sd)
                P[sel] = proba(clf, pri, X[sel]); done |= sel
            if not done.all():
                print(f"  경고: OOF 미할당 {int((~done).sum())}행 — 제외")
                keep = done
            else:
                keep = np.ones(len(X), bool)
            P, truth_, case_, les_ = P[keep], truth[keep], case[keep], les[keep]
            np.savez_compressed(o, P=P, classes=np.array(CLASSES, dtype=object),
                                truth=truth_, case=case_, lesion=les_)
            print(f"  [저장] proba_{tag}_s{sd}  n={len(P)}", flush=True); continue
        np.savez_compressed(o, P=P, classes=np.array(CLASSES, dtype=object),
                            truth=truth, case=case, lesion=les)
        print(f"  [저장] proba_{tag}_s{sd}  n={len(P)}", flush=True)
print("완료")
