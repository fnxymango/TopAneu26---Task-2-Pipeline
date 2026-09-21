#!/usr/bin/env python3
"""한 실행 태그의 지표를 그대로 기록한다.  사용: record_metrics.py <태그> [제목]"""
import json, os, sys
import numpy as np
A = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
TAG = sys.argv[1]
TITLE = sys.argv[2] if len(sys.argv) > 2 else TAG
def L(sp, sd):
    p = f"{A}/c5_eval_{sp}_{TAG}_{sp}_s{sd}.json"
    return json.load(open(p)) if os.path.exists(p) else None
print(f"# {TITLE}\n")
for sp, n in (("test", 83), ("val", 41)):
    seeds = [s for s in range(5) if L(sp, s)]
    if not seeds:
        print(f"## {sp}: 결과 없음\n"); continue
    o = L(sp, 3) or L(sp, seeds[0])
    off = "clf_seed=3" if L(sp, 3) else f"clf_seed={seeds[0]}"
    print(f"## {sp} {n}건 — 공식값 ({off})\n")
    print("| 기준 | Precision | Recall | MCC | Dice | VolSim | HD95 | 6지표 |")
    print("|---|---|---|---|---|---|---|---|")
    for key, lab in (("official_div52", "official ÷52"),
                     ("adjusted_div_present", f"covered_gt ÷{o['n_present_classes_in_split']}")):
        d = o[key]
        six = np.mean([d["PRECISION"], d["RECALL"], d["MCC"], d["DICE"], d["VOLSIM"], 1-d["HD95"]])
        print(f"| {lab} | {d['PRECISION']:.4f} | {d['RECALL']:.4f} | **{d['MCC']:.4f}** | "
              f"{d['DICE']:.4f} | {d['VOLSIM']:.4f} | {d['HD95']:.4f} | {six:.4f} |")
    print(f"\n예측 blob {o['n_lesions_predicted']} · 케이스 {o['n_cases']} · GT존재클래스 {o['n_present_classes_in_split']}\n")
    print(f"### 시드 변동폭 (seed {min(seeds)}~{max(seeds)}, n={len(seeds)})\n")
    print("| 기준 | 지표 | 평균 | 표준편차 | 최소 | 최대 |")
    print("|---|---|---|---|---|---|")
    for key, lab in (("official_div52", "official÷52"), ("adjusted_div_present", "covered_gt")):
        for met in ("MCC", "PRECISION", "RECALL"):
            v = np.array([L(sp, s)[key][met] for s in seeds])
            print(f"| {lab} | {met} | {v.mean():.4f} | {v.std(ddof=1):.4f} | {v.min():.4f} | {v.max():.4f} |")
    print()
