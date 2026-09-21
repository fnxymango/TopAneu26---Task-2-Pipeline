#!/usr/bin/env python3
"""두 실행 태그를 같은 자로 비교한다.  사용: cmp_tags.py <기준태그> <후보태그> [이름1] [이름2]
   출력: 양 기준(official÷52 / covered_gt) 지표, 케이스별 TP/FP/FN 변화, 케이스단위 부트스트랩."""
import json, os, sys
import numpy as np
A = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
BASE, CAND = sys.argv[1], sys.argv[2]
NB, rng = 2000, np.random.default_rng(20260904)
NB1 = sys.argv[3] if len(sys.argv) > 3 else BASE
NC1 = sys.argv[4] if len(sys.argv) > 4 else CAND

def ev(tag, sp):
    return [json.load(open(f"{A}/c5_eval_{sp}_{tag}_{sp}_s{s}.json")) for s in range(5)
            if os.path.exists(f"{A}/c5_eval_{sp}_{tag}_{sp}_s{s}.json")]

def counts(tag, sp, sd):
    d = json.load(open(f"{A}/c5_percase_{sp}_{tag}_{sp}_s{sd}.json"))
    C = np.zeros((len(d["per_case"]), 52, 3))
    for i, c in enumerate(d["per_case"]):
        for k in range(1, 53):
            C[i, k-1] = (c[f"TP_{k}"], c[f"FP_{k}"], c[f"FN_{k}"])
    return C, d["present"]

def mcc(C, idx, cols):
    s = C[idx].sum(0); N = len(idx)
    tp, fp, fn = s[:, 0], s[:, 1], s[:, 2]; tn = N - (tp + fn)
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return np.where(den > 0, (tp*tn - fp*fn)/np.where(den > 0, den, 1), 0.0)[cols].mean()

print(f"# {NC1}  vs  {NB1}\n")
for sp, n in (("test", 83), ("val", 41)):
    o, w = ev(BASE, sp), ev(CAND, sp)
    if not o or not w:
        print(f"## {sp}: 결과 부족 (기준 {len(o)} · 후보 {len(w)})\n"); continue
    print(f"## {sp} {n}건  (시드 5개 평균)\n")
    print(f"예측 blob  {NB1} {o[0]['n_lesions_predicted']} → {NC1} {w[0]['n_lesions_predicted']}\n")
    print("| 기준 | 지표 | " + NB1 + " | " + NC1 + " | Δ | 시드양수 |")
    print("|---|---|---|---|---|---|")
    for key, lab in (("official_div52", "official÷52"), ("adjusted_div_present", "covered_gt")):
        for met in ("MCC", "PRECISION", "RECALL"):
            a = np.array([x[key][met] for x in o]); b = np.array([x[key][met] for x in w]); d = b-a
            star = " **" if met == "MCC" else " "
            print(f"| {lab} | {met} |{star}{a.mean():.4f}{star.strip()} |{star}{b.mean():.4f}{star.strip()} |{star}{d.mean():+.4f}{star.strip()} | {int((d>0).sum())}/5 |")
        six = lambda arr: np.mean([[x[key][m] for m in ("PRECISION","RECALL","MCC","DICE","VOLSIM")]+[1-x[key]["HD95"]] for x in arr])
        print(f"| {lab} | 6지표 | {six(o):.4f} | {six(w):.4f} | {six(w)-six(o):+.4f} | — |")
    # 케이스별 TP/FP/FN
    try:
        po = np.mean([counts(BASE, sp, s)[0] for s in range(5)], axis=0)
        pw = np.mean([counts(CAND, sp, s)[0] for s in range(5)], axis=0)
        ids = json.load(open(f"{A}/c5_percase_{sp}_{BASE}_{sp}_s0.json"))["case_ids"]
        so, sw = po.sum((0,1)), pw.sum((0,1))
        ch = int((np.abs(pw-po).sum((1,2)) > 1e-9).sum())
        print(f"\n총 TP {so[0]:.1f} → {sw[0]:.1f} ({sw[0]-so[0]:+.1f}) · "
              f"FP {so[1]:.1f} → {sw[1]:.1f} ({sw[1]-so[1]:+.1f}) · "
              f"FN {so[2]:.1f} → {sw[2]:.1f} ({sw[2]-so[2]:+.1f}) · 달라진 케이스 {ch}/{len(ids)}")
    except Exception as e:
        print(f"\n(케이스별 집계 생략: {e})")
    # 부트스트랩
    try:
        ds = []
        for sd in range(5):
            Co, pres = counts(BASE, sp, sd); Cn, _ = counts(CAND, sp, sd)
            cols = np.array([p-1 for p in pres]); N = len(Co)
            for _ in range(NB//5):
                idx = rng.integers(0, N, N)
                ds.append(mcc(Cn, idx, cols) - mcc(Co, idx, cols))
        ds = np.array(ds); lo, hi = np.percentile(ds, [2.5, 97.5])
        print(f"부트스트랩 {len(ds)}회 · 95% CI [{lo:+.4f}, {hi:+.4f}] · Δ>0 {100*(ds>0).mean():.1f}%\n")
    except Exception as e:
        print(f"(부트스트랩 생략: {e})\n")
