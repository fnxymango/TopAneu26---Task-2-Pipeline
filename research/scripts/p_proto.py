#!/usr/bin/env python
"""Q2 — 판별 메트릭 축소 + 클래스 프로토타입, RF 와 혼합 (2026-08-26).

근거: 오늘 진단에서 정확도가 사실상 학습 샘플 수의 함수였다
      (1-4개 0.19 / 5-9개 0.43 / 10-19개 0.65 / 20+개 0.84).
      클래스당 5샘플 구간에서 RF 500그루는 잎이 사실상 암기가 된다.
      최근접 프로토타입은 그 분산을 줄인다 — few-shot 에서 일관되게 관찰되는 성질이다.

RF 를 대체하지 않고 확률을 혼합한다. w 는 **292 OOF 에서만** 고른다.
OOF 는 폴드마다 프로토타입도 다시 적합한다(누수 방지).
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
NONE = "__none__"
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
CLASSES = sorted(set(list(yt) + list(ym))); CI = {c: i for i, c in enumerate(CLASSES)}


def fit_proto(mask, n_comp=None):
    X = np.concatenate([Xt[mask], Xm[mask]]); y = np.concatenate([yt[mask], ym[mask]])
    keep = np.array([collections.Counter(y)[c] >= 2 for c in y])   # LDA 는 클래스당 2개 이상 필요
    X, y = X[keep], y[keep]
    k = n_comp or min(30, len(set(y)) - 1, X.shape[1])
    lda = LDA(n_components=k, solver="eigen", shrinkage="auto").fit(X, y)
    Z = lda.transform(X)
    mu = {}
    for c in set(y):
        mu[c] = Z[y == c].mean(axis=0)
    return lda, mu


def proto_proba(lda, mu, X, T=1.0):
    Z = lda.transform(X)
    out = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    cs = list(mu); M = np.stack([mu[c] for c in cs])
    d2 = ((Z[:, None, :] - M[None]) ** 2).sum(axis=2)
    p = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / max(T, 1e-6))
    p = p / p.sum(axis=1, keepdims=True)
    for j, c in enumerate(cs): out[:, CI[c]] = p[:, j]
    return out


def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))


WS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
for tag in ("trainoof", "test"):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True)
    Xe, truth, case = f["X"], list(f["truth"]), f["case"]
    Pp = np.zeros((len(Xe), len(CLASSES)), dtype=np.float32)
    if tag == "test":
        lda, mu = fit_proto(np.ones(len(Xt), bool)); Pp = proto_proba(lda, mu, Xe)
    else:
        for fd in json.load(open(SP720)):
            va = set(fd["val"]); sel = np.array([c in va for c in case])
            if not sel.any(): continue
            lda, mu = fit_proto(np.array([c not in va for c in ct]))
            Pp[sel] = proto_proba(lda, mu, Xe[sel])
    res = collections.defaultdict(list)
    for sd in range(5):
        g = np.load(A / f"proba_{tag}_s{sd}.npz", allow_pickle=True)
        Pr, cls = g["P"], list(g["classes"])
        assert cls == CLASSES, "클래스축 불일치"
        for w in WS:
            P = (1 - w) * Pr + w * Pp
            pred = [CLASSES[int(i)] for i in P.argmax(axis=1)]
            res[w].append(score(list(g["truth"]), pred))
    lab = "[292 OOF]" if tag == "trainoof" else "[test 83]"
    print(f"\n{lab}  병변 {len(Xe)}  (w=0 은 현행 RF)")
    print(f"  {'w':>5}{'top-1':>9}{'macroRec':>10}{'MCC(병변)':>11}")
    base = None
    for w in WS:
        m = np.array(res[w], dtype=float).mean(axis=0)
        if base is None: base = m
        d = "" if w == 0 else f"   macroRec {m[1]-base[1]:+.4f}"
        print(f"  {w:>5.1f}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>11.4f}{d}")
    if tag == "trainoof":
        best = max(WS[1:], key=lambda w: np.array(res[w]).mean(axis=0)[1])
        g_ = np.array(res[best]).mean(axis=0)[1] - base[1]
        print(f"\n  ★ 게이트: 최고 w={best} · macroRec {g_:+.4f} → "
              f"{'통과' if g_ >= 0.02 else '미달 — 중단'}")
