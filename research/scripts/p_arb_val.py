#!/usr/bin/env python
"""val 42 판정 — Q2 프로토타입 / Q3 인접평활 을 val 에서 재측정 (2026-08-26).

292 OOF 는 (a) in-sample 혈관(Dice 0.8863 vs test 0.8413) (b) 단일폴드 검출 이라
test 와 조건이 다르다. val 42 는 test 와 조건이 정확히 같고 292 와는 둘 다 다르다.
따라서 test+ / 292- 로 갈린 방법이 **혈관 조건 탓인지 test 노이즈인지** val 이 판정한다.

학습은 292 전체(=test 경로와 동일). val 은 어떤 폴드에도 안 들어갔으므로 누수 없음.
"""
import json, os, sys, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
NONE = "__none__"
BETA, TAU = 0.5, 0.5
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
CLASSES = sorted(set(list(yt) + list(ym))); CI = {c: i for i, c in enumerate(CLASSES)}
fv = np.load(A / "feat_val.npz", allow_pickle=True)
Xv, tv = fv["X"], list(fv["truth"])
print(f"학습 {len(Xt)}병변(미러 후 {2*len(Xt)}) · 클래스축 {len(CLASSES)} · val {len(Xv)}병변", flush=True)


def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))


def table(name, keys, res, unit):
    print(f"\n[val 42] {name}  병변 {len(Xv)}  ({keys[0]} 은 현행)")
    print(f"  {unit:>6}{'top-1':>9}{'macroRec':>10}{'MCC(병변)':>11}")
    base = None
    for k in keys:
        m = np.array(res[k], dtype=float).mean(axis=0)
        if base is None: base = m
        d = "" if k == keys[0] else f"   macroRec {m[1]-base[1]:+.4f}  MCC {m[2]-base[2]:+.4f}"
        print(f"  {k:>6.2f}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>11.4f}{d}")


# ---------------- Q2 프로토타입 ----------------
def fit_proto():
    X = np.concatenate([Xt, Xm]); y = np.concatenate([yt, ym])
    keep = np.array([collections.Counter(y)[c] >= 2 for c in y])
    X, y = X[keep], y[keep]
    k = min(30, len(set(y)) - 1, X.shape[1])
    lda = LDA(n_components=k, solver="eigen", shrinkage="auto").fit(X, y)
    Z = lda.transform(X)
    return lda, {c: Z[y == c].mean(axis=0) for c in set(y)}


def proto_proba(lda, mu, X, T=1.0):
    Z = lda.transform(X)
    cs = list(mu); M = np.stack([mu[c] for c in cs])
    d2 = ((Z[:, None, :] - M[None]) ** 2).sum(axis=2)
    p = np.exp(-0.5 * (d2 - d2.min(axis=1, keepdims=True)) / T); p /= p.sum(axis=1, keepdims=True)
    out = np.zeros((len(X), len(CLASSES)), dtype=np.float32)
    for j, c in enumerate(cs): out[:, CI[c]] = p[:, j]
    return out


def rf_proba(seed):
    X = np.concatenate([Xt, Xm]); y = np.concatenate([yt, ym])
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=1, class_weight="balanced",
                                 random_state=seed, n_jobs=4).fit(X, y)
    pri = np.array([collections.Counter(y)[c] for c in clf.classes_], dtype=float)
    P = clf.predict_proba(Xv); hi = P.max(axis=1) >= TAU
    Pc = P.copy(); Pc[~hi] = Pc[~hi] / (pri ** BETA)
    Pc /= np.maximum(Pc.sum(axis=1, keepdims=True), 1e-12)
    out = np.zeros((len(Xv), len(CLASSES)), dtype=np.float32)
    for j, c in enumerate(clf.classes_): out[:, CI[c]] = Pc[:, j]
    return out


lda, mu = fit_proto(); Pp = proto_proba(lda, mu, Xv)
WS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
res = collections.defaultdict(list)
for sd in range(5):
    Pr = rf_proba(sd)
    for w in WS:
        P = (1 - w) * Pr + w * Pp
        res[w].append(score(tv, [CLASSES[int(i)] for i in P.argmax(axis=1)]))
    print(f"  프로토타입 시드{sd} 완료", flush=True)
table("Q2 프로토타입 혼합", WS, res, "w")

# ---------------- Q3 인접 분절 평활 ----------------
def parse(c):
    m = re.match(r"^([LR])-(\d+)\.(\d+)", c)
    if m: return m.group(1), int(m.group(2)), int(m.group(3))
    m = re.match(r"^(\d+)\.(\d+)", c)
    if m: return "", int(m.group(1)), int(m.group(2))
    return None


NB = collections.defaultdict(list)
for a in CLASSES:
    pa = parse(a)
    if not pa: continue
    for b in CLASSES:
        if a != b:
            pb = parse(b)
            if pb and pa[0] == pb[0] and pa[1] == pb[1] and abs(pa[2] - pb[2]) == 1:
                NB[a].append(b)


def fit_adj(eps, seed):
    X = np.concatenate([Xt, Xm]); y = np.concatenate([yt, ym])
    Xa, ya, wa = [X], [y], [np.ones(len(y))]
    if eps > 0:
        ex_X, ex_y = [], []
        for i, c in enumerate(y):
            for nb in NB.get(c, []):
                ex_X.append(X[i]); ex_y.append(nb)
        if ex_X:
            Xa.append(np.array(ex_X)); ya.append(np.array(ex_y, dtype=object))
            wa.append(np.full(len(ex_X), eps))
    X = np.concatenate(Xa); y = np.concatenate(ya); w = np.concatenate(wa)
    return RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=seed, n_jobs=4).fit(X, y, sample_weight=w)


EPS = [0.0, 0.05, 0.1, 0.2, 0.4]
res2 = collections.defaultdict(list)
for sd in range(3):
    for e in EPS:
        res2[e].append(score(tv, list(fit_adj(e, sd).predict(Xv))))
    print(f"  인접평활 시드{sd} 완료", flush=True)
table("Q3 인접 분절 평활", EPS, res2, "eps")

import json as _j
_j.dump({"proto":{str(k):[list(map(float,v)) for v in res[k]] for k in WS},
         "adj":{str(k):[list(map(float,v)) for v in res2[k]] for k in EPS}},
        open(A/"arb_val_perseed.json","w"))
print("\n[저장] arb_val_perseed.json")

# 시드별 짝지은 검정
def paired(name, keys, r):
    b=np.array(r[keys[0]],dtype=float)
    print(f"\n[{name}] 시드별 짝지은 Δ (기준 {keys[0]})")
    for k in keys[1:]:
        m=np.array(r[k],dtype=float); d=m-b
        for j,mn in enumerate(("top-1","macroRec","MCC")):
            dd=d[:,j]; sd=dd.std(ddof=1)
            t=dd.mean()/(sd/np.sqrt(len(dd))) if sd>1e-12 else float("nan")
            print(f"  {k:>5} {mn:9s} Δ{dd.mean():+.4f}  t={t:+6.2f}  {int((dd>0).sum())}/{len(dd)}  seeds " + " ".join(f"{x:+.3f}" for x in dd))
paired("Q2 프로토타입", WS, res)
paired("Q3 인접평활", EPS, res2)
print("\n완료")
