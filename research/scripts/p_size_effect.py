#!/usr/bin/env python
"""집합 크기 효과 검정 (2026-08-26).

Q6/Q7 의 모든 방법이 test 83 에서 + 이고 val 42 에서 - 다. 방향이 이렇게 일관되면
방법의 문제가 아니라 **평가집합의 구조 차이**를 의심해야 한다.

  test  병변 87 / 존재클래스 36 = 클래스당 2.42
  val   병변 45 / 존재클래스 33 = 클래스당 1.36

MCC_i ≈ TP/√((TP+FP)(TP+FN)) 라 TP=1·FP=0 인 클래스는 FP 하나에 1.000→0.707 로 무너진다.
클래스당 인스턴스가 적을수록 FP 벌점이 커진다. 즉 **작은 평가집합이 다중라벨에 더 가혹**하다.

검정: test 83 의 케이스별 원자료를 42개씩 재표집해 같은 Δ 를 다시 낸다.
크기 효과가 맞으면 42케이스 부분표본에서 Δ 가 줄거나 음수로 간다.
"""
import json, math, sys, os, glob
import numpy as np

A = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
RNG = np.random.default_rng(0)
KEYS = ["TP", "FP", "FN", "TN"]


def load(tag, sp):
    d = json.load(open(f"{A}/c5_percase_{sp}_{tag}.json"))
    return d["per_case"], d["present"]


def cov_mcc(pc, idx, present, eps=1e-6):
    S = {k: np.zeros(53) for k in KEYS}
    for i in idx:
        c = pc[i]
        for k in KEYS:
            for j in range(1, 53):
                S[k][j] += c[f"{k}_{j}"]
    out = []
    for j in present:
        tp, fp, fn, tn = S["TP"][j], S["FP"][j], S["FN"][j], S["TN"][j]
        num = tp * tn - fn * fp
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
        out.append(num / den)
    return float(np.mean(out)) if out else 0.0


def run(mtag, btag, sp, label):
    fb = sorted(glob.glob(f"{A}/c5_percase_{sp}_{btag}_s*.json"))
    fm = sorted(glob.glob(f"{A}/c5_percase_{sp}_{mtag}_s*.json"))
    sb = {int(f.split('_s')[-1][:-5]) for f in fb}
    sm = {int(f.split('_s')[-1][:-5]) for f in fm}
    sds = sorted(sb & sm)
    if not sds:
        print(f"  {label}: 자료 없음"); return
    full, sub42, sub20 = [], [], []
    for sd in sds:
        pcb, pres = load(f"{btag}_s{sd}", sp)
        pcm, _ = load(f"{mtag}_s{sd}", sp)
        n = min(len(pcb), len(pcm))
        allid = np.arange(n)
        full.append(cov_mcc(pcm, allid, pres) - cov_mcc(pcb, allid, pres))
        for tgt, acc in ((42, sub42), (20, sub20)):
            if n <= tgt: continue
            ds = [cov_mcc(pcm, s, pres) - cov_mcc(pcb, s, pres)
                  for s in (RNG.choice(n, tgt, replace=False) for _ in range(200))]
            acc.append(float(np.mean(ds)))
    f = np.array(full)
    print(f"  {label:20s} 전체({len(pcb)}케이스) Δ{f.mean():+.4f}", end="")
    if sub42: print(f"   42케이스 재표집 Δ{np.mean(sub42):+.4f}", end="")
    if sub20: print(f"   20케이스 Δ{np.mean(sub20):+.4f}", end="")
    print(f"   [{len(sds)}시드]")


print("[집합 크기 효과 — test 83 을 잘라서 같은 Δ 재측정]")
print("  크기 효과가 맞으면 잘라낼수록 Δ 가 작아지거나 음수가 된다.\n")
for mt, lb in (("v3f_test", "vote3f"), ("tk2_test", "tk2 무조건"),
               ("tk2t_test", "tk2 τ<0.5"), ("mg70_test", "마진>0.7"),
               ("gA_test", "gA 희소≤4"), ("gB_test", "gB ICA·0.5"), ("gC_test", "gC ICA·0.7")):
    run(mt, "e11_hyb_ov", "test", lb)

print("\n[대조: val 42 는 원래 42케이스라 자를 게 없다 — 참고용 전체값]")
for mt, lb in (("v3f_val", "vote3f"), ("tk2_val", "tk2 무조건"), ("mg70_val", "마진>0.7"),
               ("gB_val", "gB ICA·0.5"), ("gC_val", "gC ICA·0.7")):
    run(mt, "v2_base", "val", lb)
