#!/usr/bin/env python
"""Q3 — 인접 분절 라벨 평활 (2026-08-26).

근거: 라벨오류의 58.5% 가 '같은 혈관 위 인접 분절' 혼동이고, 정확도가 학습 샘플 수의
      함수다(1-4개 0.19 / 20+개 0.84). 인접 분절끼리 표본을 빌려주면 희소 클래스가
      이웃의 통계를 쓴다.

RF 는 소프트 라벨을 못 받으므로 **이웃 라벨 복사본을 낮은 sample_weight 로 추가**한다.
인접 정의: 같은 대분류 · 같은 좌우 · 소분류 번호 차이 1  (예: 3.3 <-> 3.4)

eps 는 292 OOF 에서만 고른다.
★ 게이트: macroRec +0.02 미만이면 중단.
"""
import json, os, sys, collections, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
NONE = "__none__"
ft = np.load(A / "feat_train.npz", allow_pickle=True)
Xt, Xm, yt, ym, ct = ft["X"], ft["Xm"], ft["y"], ft["ym"], ft["case"]
CLASSES = sorted(set(list(yt) + list(ym))); CI = {c: i for i, c in enumerate(CLASSES)}


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
        if a == b: continue
        pb = parse(b)
        if pb and pa[0] == pb[0] and pa[1] == pb[1] and abs(pa[2] - pb[2]) == 1:
            NB[a].append(b)
print(f"인접쌍이 있는 클래스 {len(NB)}/{len(CLASSES)} · 평균 이웃 {np.mean([len(v) for v in NB.values()]):.1f}")


def fit(mask, eps, seed):
    X = np.concatenate([Xt[mask], Xm[mask]]); y = np.concatenate([yt[mask], ym[mask]])
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


def score(truth, pred):
    yt_ = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = [np.mean([pred[i] == c for i, t in enumerate(truth) if t == c]) for c in present]
    return top1, float(np.mean(rec)), matthews_corrcoef(yt_, list(pred))


EPS = [0.0, 0.05, 0.1, 0.2, 0.4]
for tag in ("trainoof", "test"):
    f = np.load(A / f"feat_{tag}.npz", allow_pickle=True)
    Xe, truth, case = f["X"], list(f["truth"]), f["case"]
    res = collections.defaultdict(list)
    for sd in range(3):
        for e in EPS:
            pred = np.empty(len(Xe), dtype=object)
            if tag == "test":
                clf = fit(np.ones(len(Xt), bool), e, sd); pred[:] = clf.predict(Xe)
            else:
                for fd in json.load(open(SP720)):
                    va = set(fd["val"]); sel = np.array([c in va for c in case])
                    if not sel.any(): continue
                    clf = fit(np.array([c not in va for c in ct]), e, sd)
                    pred[sel] = clf.predict(Xe[sel])
            res[e].append(score(truth, list(pred)))
        print(f"  {tag} 시드{sd} 완료", flush=True)
    lab = "[292 OOF]" if tag == "trainoof" else "[test 83]"
    print(f"\n{lab}  병변 {len(Xe)}  (eps=0 은 현행)")
    print(f"  {'eps':>6}{'top-1':>9}{'macroRec':>10}{'MCC(병변)':>11}")
    base = None
    for e in EPS:
        m = np.array(res[e], dtype=float).mean(axis=0)
        if base is None: base = m
        d = "" if e == 0 else f"   macroRec {m[1]-base[1]:+.4f}"
        print(f"  {e:>6.2f}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>11.4f}{d}")
    if tag == "trainoof":
        best = max(EPS[1:], key=lambda e: np.array(res[e]).mean(axis=0)[1])
        g = np.array(res[best]).mean(axis=0)[1] - base[1]
        print(f"\n  ★ 게이트: 최고 eps={best} · macroRec {g:+.4f} → "
              f"{'통과' if g >= 0.02 else '미달 — 중단'}")
