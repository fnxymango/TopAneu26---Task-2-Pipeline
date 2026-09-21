#!/usr/bin/env python3
"""iw_headroom.py <tag> — 공식 (케이스×클래스) 지표의 FN/FP 를 '검출 탓' 과 '분류 탓' 으로 가른다.

분류만 손볼 때 되찾을 수 있는 상한을 재는 게 목적이다.
  FN 의 출처   미검출 병변  = 분류로 못 되찾음
               검출됐는데 오분류 = 분류로 되찾을 수 있음
  FP 의 출처   오분류가 만든 가짜 클래스 = 분류로 없앨 수 있음
               환각 blob 에 붙은 클래스   = 검출/필터 쪽
"""
import json, sys, collections
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; D=f"{R}/experiments/D1_newdata"
NAME={int(k):v for k,v in json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["location_classes"].items()}
for TAG in sys.argv[1:]:
    d=json.load(open(f"{D}/intweak_{TAG}.json"))
    L=d["lesions"]; FP=d["fps"]; PC=d["per_case"]
    S=5
    # 케이스별 GT 클래스 집합, 검출된 GT 클래스 집합
    gtcls=collections.defaultdict(set); detcls=collections.defaultdict(set)
    okcls=collections.defaultdict(lambda: collections.defaultdict(set))  # seed -> case -> 맞힌 클래스
    for x in L:
        key=(x["split"],x["case"])
        gtcls[key].add(x["cls"])
        if x["detected"]: detcls[key].add(x["cls"])
        for si,p in enumerate(x["preds"]):
            if p==x["cls"]: okcls[si][key].add(x["cls"])
    fn_undet=fn_mis=0
    for key,g in gtcls.items():
        for si in range(S):
            missed=g-okcls[si][key]
            for c in missed:
                fn_undet += (c not in detcls[key])
                fn_mis   += (c in detcls[key])
    fp_mis=sum(1 for f in FP if f["on_gt"]>0); fp_hal=len(FP)-fp_mis
    tp=sum(p["n_tp"] for p in PC)/S; fp=sum(p["n_fp"] for p in PC)/S; fn=sum(p["n_fn"] for p in PC)/S
    print(f"\n## {TAG} — 시드평균 (케이스×클래스)\n")
    print(f"| | 수 | 분류로 개선 가능 | 불가(검출/환각 탓) |"); print("|---|---|---|---|")
    print(f"| TP | {tp:.1f} | — | — |")
    print(f"| FN | {fn:.1f} | {fn_mis/S:.1f} (오분류) | {fn_undet/S:.1f} (미검출) |")
    print(f"| FP | {fp:.1f} | {fp_mis/S:.1f} (오분류) | {fp_hal/S:.1f} (환각) |")
    print(f"\n분류를 완벽히 고치면: TP {tp:.1f} → {tp+fn_mis/S:.1f} · FN {fn:.1f} → {fn_undet/S:.1f} · FP {fp:.1f} → {fp_hal/S:.1f}")
    print(f"→ **분류가 만드는 오류는 시드당 FN {fn_mis/S:.1f} + FP {fp_mis/S:.1f} = {(fn_mis+fp_mis)/S:.1f}개**")
    print(f"   (검출/환각이 만드는 것은 FN {fn_undet/S:.1f} + FP {fp_hal/S:.1f} = {(fn_undet+fp_hal)/S:.1f}개)")
    # 클래스별 기여
    print(f"\n### 오분류가 만드는 오류 상위 클래스 (FN 쪽 · 시드합계)\n")
    cc=collections.Counter()
    for x in L:
        if not x["detected"]: continue
        cc[x["name"]] += sum(1 for p in x["preds"] if p!=x["cls"])
    print("| GT 클래스 | 오분류 시행 | 검출병변 |"); print("|---|---|---|")
    n_les=collections.Counter(x["name"] for x in L if x["detected"])
    for nm,c in cc.most_common(12):
        if c: print(f"| {nm} | {c} | {n_les[nm]} |")
    print(f"\n### 가짜로 만들어지는 클래스 상위 (FP 쪽 · 오분류 기인만 · 시드합계)\n")
    fc=collections.Counter(f["name"] for f in FP if f["on_gt"]>0)
    print("| 잘못 붙은 클래스 | 횟수 |"); print("|---|---|")
    for nm,c in fc.most_common(12): print(f"| {nm} | {c} |")
