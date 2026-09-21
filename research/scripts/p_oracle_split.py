#!/usr/bin/env python
"""오라클 분해 (2026-08-25) — 팀원 exp_55 분석과 같은 축으로 우리 test 83 을 가른다.

핵심 셈법: 한 번의 오분류는 벌점을 두 곳에서 받는다.
    라벨 틀림 -> 정답 :  FP -1, FN -1, TP +1   (3중 개선)
    환각 제거        :  FP -1                 (1중 개선)
그래서 blob 수가 적어도 라벨 교정이 더 큰 효과를 낸다. 우리 몫이 얼마인지 실측한다.

병변 단위 (case, 정답, 예측) 를 뽑아 macro-MCC 를 **재구성**한 뒤
  (a) 현행  (b) 라벨 전부 교정  (c) 환각 전부 제거  (d) 둘 다
네 시나리오를 같은 재구성기로 비교한다. 재구성 기준값이 실측과 맞는지 먼저 확인한다.
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d9xx_lib as L
import c5_location_v2 as C5

ST = np.ones((3, 3, 3), bool)


def macro_mcc(recs, present, n_aneu_by_case):
    """공식 채점기와 같은 셈법: 케이스별 tp=min(pred,gt) 등을 클래스별로 합산 후 MCC 평균."""
    by = collections.defaultdict(lambda: collections.Counter())
    for r in recs:
        if r["truth"]:
            by[r["case"]]["gt_" + r["truth"]] += 1
        if r["pred"]:
            by[r["case"]]["pd_" + r["pred"]] += 1
    tot = {c: [0, 0, 0, 0] for c in present}          # tp fp fn tn
    for case, cnt in by.items():
        na = n_aneu_by_case.get(case, 0)
        for c in present:
            g = cnt.get("gt_" + c, 0); p = cnt.get("pd_" + c, 0)
            tp = min(p, g); fp = max(0, p - g); fn = max(0, g - p)
            tn = na - (tp + fn)
            t = tot[c]; t[0] += tp; t[1] += fp; t[2] += fn; t[3] += tn
    vals = []
    for c in present:
        tp, fp, fn, tn = tot[c]
        num = tp * tn - fn * fp
        den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        vals.append(num / den if den > 0 else 0.0)
    return float(np.mean(vals))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "e11_hyb_ov"
    sd = sys.argv[2] if len(sys.argv) > 2 else "0"
    recs = json.load(open(os.path.join(
        os.environ["TOPANEU_ROOT"], "code/sblee/nnunet/analysis",
        f"p_lesion_{tag}_s{sd}.json")))
    present = sorted({r["truth"] for r in recs if r["truth"]})
    n_by_case = {}
    for r in recs:
        n_by_case.setdefault(r["case"], 0)
    for r in recs:
        if r["truth"]:
            n_by_case[r["case"]] += 1

    cur = [dict(r) for r in recs]
    fix = [dict(r) for r in recs]
    for r in fix:
        if r["truth"] and r["pred"]:
            r["pred"] = r["truth"]                      # 라벨 교정
    noh = [dict(r) for r in recs if r["truth"] or not r["pred"]]   # 환각 제거
    both = [dict(r) for r in fix if r["truth"] or not r["pred"]]

    n_ok = sum(1 for r in recs if r["truth"] and r["pred"] == r["truth"])
    n_bad = sum(1 for r in recs if r["truth"] and r["pred"] and r["pred"] != r["truth"])
    n_hal = sum(1 for r in recs if not r["truth"] and r["pred"])
    n_mis = sum(1 for r in recs if r["truth"] and not r["pred"])
    print(f"[분해] 맞힘 {n_ok} · 라벨틀림 {n_bad} · 환각 {n_hal} · 미검출 {n_mis} "
          f"· GT {n_ok+n_bad+n_mis} · 예측 {n_ok+n_bad+n_hal}")
    print(f"  분모 클래스 {len(present)}\n")
    base = macro_mcc(cur, present, n_by_case)
    print(f"  {'시나리오':<28}{'cov.MCC':>10}{'Δ':>10}")
    print(f"  {'현행 (재구성)':<28}{base:>10.4f}{'':>10}")
    for lab, rs in (("+ 라벨 전부 교정", fix), ("+ 환각 전부 제거", noh),
                    ("+ 둘 다", both)):
        v = macro_mcc(rs, present, n_by_case)
        print(f"  {lab:<28}{v:>10.4f}{v-base:>+10.4f}")


if __name__ == "__main__":
    main()
