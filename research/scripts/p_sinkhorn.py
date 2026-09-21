#!/usr/bin/env python
"""Q1b — 전역 배정(Sinkhorn) 조기 게이트 (2026-08-26).

지금은 병변마다 독립적으로 argmax(p / prior^β) 를 뽑는다. β 보정은 **병변별 근사**다.
전역 배정은 split 전체를 동시에 놓고 열(클래스) 주변분포를 목표치에 맞춘다.

  근거: test 에서 GT 등장 36클래스 중 **28개만 예측**되고 히스토그램 L1 오차가 45%다.
  pjh 문서가 소진했다는 후처리 5축은 전부 병변별 재배분이라("group 재분배는 항등식")
  이 축과 겹치지 않는다.

  P' = P * v^alpha  (v 는 Sinkhorn 열 스케일)  alpha=0 현행 / alpha=1 완전 정합

목표 주변분포는 **학습 사전확률**을 병변수로 스케일한 것. test 의 진짜 분포는 모른다.
alpha 는 292 OOF 에서만 고른다.

★ 게이트: macro-recall 이 alpha=0 대비 +0.02 미만이면 코드 통합 없이 중단.
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.metrics import matthews_corrcoef
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
NONE = "__none__"
ft = np.load(A / "feat_train.npz", allow_pickle=True)
prior = collections.Counter(list(ft["y"]) + list(ft["ym"]))


def sinkhorn_v(P, q, iters=200):
    """행 주변분포 1, 열 주변분포 q 를 맞추는 열 스케일 v."""
    n, C = P.shape
    v = np.ones(C)
    for _ in range(iters):
        u = 1.0 / np.maximum((P * v).sum(axis=1), 1e-30)
        v = q / np.maximum((P * u[:, None]).sum(axis=0), 1e-30)
    return v


def score(truth, pred):
    yt = [t if t else NONE for t in truth]
    present = sorted({t for t in truth if t})
    top1 = np.mean([p == t for t, p in zip(truth, pred) if t])
    rec = []
    for c in present:
        idx = [i for i, t in enumerate(truth) if t == c]
        rec.append(np.mean([pred[i] == c for i in idx]))
    return top1, float(np.mean(rec)), matthews_corrcoef(yt, list(pred)), len(set(pred))


ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
for tag in ("trainoof", "test"):
    res = collections.defaultdict(list)
    ncls = None
    for sd in range(5):
        f = np.load(A / f"proba_{tag}_s{sd}.npz", allow_pickle=True)
        P, cls, truth = f["P"], list(f["classes"]), list(f["truth"])
        n = len(P)
        pr = np.array([prior.get(c, 0) for c in cls], dtype=float)
        q = pr / pr.sum() * n
        v = sinkhorn_v(P, q)
        for a in ALPHAS:
            Pa = P * (v ** a)
            pred = [cls[int(i)] for i in Pa.argmax(axis=1)]
            res[a].append(score(truth, pred))
        ncls = len({t for t in truth if t})
    lab = "[292 OOF]" if tag == "trainoof" else "[test 83]"
    print(f"\n{lab}  병변 {n} · GT 등장 클래스 {ncls}")
    print(f"  {'alpha':>6}{'top-1':>9}{'macroRec':>10}{'MCC(병변)':>11}{'예측클래스수':>12}")
    base = None
    for a in ALPHAS:
        m = np.array(res[a], dtype=float).mean(axis=0)
        if base is None: base = m
        mark = ""
        if a > 0:
            d = m[1] - base[1]
            mark = f"   macroRec {d:+.4f}" + ("  ★" if d >= 0.02 else "")
        print(f"  {a:>6.2f}{m[0]:>9.3f}{m[1]:>10.3f}{m[2]:>11.4f}{m[3]:>12.1f}{mark}")
    if tag == "trainoof":
        best = max(ALPHAS[1:], key=lambda a: np.array(res[a]).mean(axis=0)[1])
        gain = np.array(res[best]).mean(axis=0)[1] - base[1]
        print(f"\n  ★ 게이트: 최고 alpha={best} · macroRec {gain:+.4f} → "
              f"{'통과 — Q1c 진행' if gain >= 0.02 else '미달 — 중단'}")
