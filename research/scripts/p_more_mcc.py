#!/usr/bin/env python
"""추가 MCC 인상 후보 3종 병변수준 스크리닝 (2026-08-26).

남은 미측정 축:
  (A) RF 트리 수 500→1500/2500 — 시드 분산 축소. 시드평균이 아니라 **모델 자체**가 좋아지는지.
  (B) 5시드 확률 앙상블 — proba_*.npz 평균 후 argmax. (A) 와 같은 목적, 다른 수단.
  (C) Sinkhorn alpha=0.5 — 292 +0.0104 로 게이트(+0.02) 미달이었지만 val 심판 체계가 생겼으니
      test·val 에서 재판정할 가치가 있다.

전부 병변수준(빠름). 유망하면 공식 e2e 로 넘긴다.
val 용 proba 가 없으므로 여기서 만든다 (feat_val.npz 재사용, 시드당 수초).
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
NONE = "__none__"
BETA, TAU = 0.5, 0.5
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym = ft["X"], ft["Xm"], ft["y"], ft["ym"]
CLASSES = sorted(set(list(yt) + list(ym))); CI = {c: i for i, c in enumerate(CLASSES)}
prior = collections.Counter(list(yt) + list(ym))


def fit(seed, ntree):
    X = np.concatenate([Xt, Xm]); y = np.concatenate([yt, ym])
    clf = RandomForestClassifier(n_estimators=ntree, min_samples_leaf=1,
                                 class_weight="balanced", random_state=seed, n_jobs=6).fit(X, y)
    pri = np.array([prior[c] for c in clf.classes_], dtype=float)
    return clf, pri


def proba(clf, pri, X):
    P = clf.predict_proba(X)
    hi = P.max(axis=1) >= TAU
    Pc = P.copy(); Pc[~hi] = Pc[~hi] / (pri ** BETA)
    Pc = Pc / np.maximum(Pc.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    for j, c in enumerate(clf.classes_): out[:, CI[c]] = Pc[:, j]
    return out


def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))


def sinkhorn_v(P, q, iters=200):
    v = np.ones(P.shape[1])
    for _ in range(iters):
        u = 1.0 / np.maximum((P * v).sum(axis=1), 1e-30)
        v = q / np.maximum((P * u[:, None]).sum(axis=0), 1e-30)
    return v


SETS = {}
for tag in ("test", "val"):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True)
    SETS[tag] = (f["X"], list(f["truth"]))

print("병변수:", {k: len(v[0]) for k, v in SETS.items()}, flush=True)

# ---- (A) 트리 수 ----
print("\n[A] RF 트리 수  (5시드 평균, 병변수준)")
print(f"  {'':6}{'':8}{'top-1':>8}{'macroRec':>10}{'MCC':>9}")
cacheP = {}          # (tag,ntree,seed) -> P  — (B)(C) 에서 재사용
for ntree in (500, 1500, 2500):
    for tag in SETS:
        X, tr = SETS[tag]
        ms = []
        for sd in range(5):
            clf, pri = fit(sd, ntree)
            P = proba(clf, pri, X); cacheP[(tag, ntree, sd)] = P
            ms.append(score(tr, [CLASSES[i] for i in P.argmax(axis=1)]))
        m = np.array(ms).mean(axis=0)
        print(f"  {tag:6}{ntree:>6}  {m[0]:>8.3f}{m[1]:>10.3f}{m[2]:>9.4f}", flush=True)

# ---- (B) 시드 앙상블 ----
print("\n[B] 5시드 확률 앙상블 (트리500)  vs 단일시드 평균")
for tag in SETS:
    X, tr = SETS[tag]
    Pm = np.mean([cacheP[(tag, 500, sd)] for sd in range(5)], axis=0)
    m = score(tr, [CLASSES[i] for i in Pm.argmax(axis=1)])
    print(f"  {tag:6}  top-1 {m[0]:.3f}  macroRec {m[1]:.3f}  MCC {m[2]:.4f}")

# ---- (C) Sinkhorn ----
print("\n[C] Sinkhorn 전역배정  (트리500, 5시드 평균)")
print(f"  {'':6}{'alpha':>6}{'top-1':>8}{'macroRec':>10}{'MCC':>9}")
for tag in SETS:
    X, tr = SETS[tag]
    for a in (0.0, 0.25, 0.5):
        ms = []
        for sd in range(5):
            P = cacheP[(tag, 500, sd)]
            pr = np.array([prior.get(c, 0) for c in CLASSES], dtype=float)
            q = pr / pr.sum() * len(P)
            v = sinkhorn_v(P, q)
            Pa = P * (v ** a)
            ms.append(score(tr, [CLASSES[i] for i in Pa.argmax(axis=1)]))
        m = np.array(ms).mean(axis=0)
        print(f"  {tag:6}{a:>6.2f}{m[0]:>8.3f}{m[1]:>10.3f}{m[2]:>9.4f}", flush=True)
print("\n완료")
