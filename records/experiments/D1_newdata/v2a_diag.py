#!/usr/bin/env python3
"""V2-A 사후 진단 (판정 아님) — 주 지표가 검출 실패로 희석됐는가.

주 지표(케이스 존재기반 적중)는 분모에 **검출기가 놓친 ICA 원위 병변**을 포함한다. 그런 병변은
혈관 정보가 아무리 좋아도 못 맞히므로 Δ 가 희석된다. 여기서는 병변 단위로
  (a) 검출됨(예측 마스크가 병변과 겹침) 여부
  (b) 검출된 병변에서 예측 라벨(겹친 복셀 다수결) == GT 클래스
를 두 팔에서 센다. 판정규칙은 v2a_report.py 에 고정된 것을 따르고 이 표는 원인 설명용이다.
"""
import json, os, re, collections
import numpy as np, nibabel as nib
from scipy import ndimage
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
H = f"{R}/experiments/H1_patchfilter"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
LOC = {int(k): v for k, v in S["location_classes"].items()}
DIST = {k for k, v in LOC.items() if re.match(r"^(?:[RL]-)?3\.[2-7]\s", v)}

def one(arg):
    sp, c = arg
    gp = f"{R}/dataset/TopAneu/location_masks/{c}.nii.gz"
    if not os.path.exists(gp): return []
    g = np.asanyarray(nib.load(gp).dataobj)
    ks = [int(k) for k in np.unique(g) if int(k) in DIST]
    if not ks: return []
    preds = {}
    for tag in ("v2a36_pf", "v2a40_pf"):
        for sd in range(5):
            p = f"{H}/pred/{tag}_{sp}_s{sd}/{c}.nii.gz"
            if os.path.exists(p): preds[(tag, sd)] = np.asanyarray(nib.load(p).dataobj)
    out = []
    for k in ks:
        lab, n = ndimage.label(g == k)
        for i in range(1, n + 1):
            m = ndimage.binary_dilation(lab == i, iterations=2)
            for (tag, sd), pr in preds.items():
                v = pr[m]; v = v[v > 0]
                det = v.size > 0
                ok = det and collections.Counter(v.tolist()).most_common(1)[0][0] == k
                out.append((sp, tag, det, ok, LOC[k]))
    return out

if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [(sp, c) for sp in ("test", "val") for c in S["splits"][sp]]
    rows = []
    with mp.Pool(8) as pool:
        for r in pool.imap_unordered(one, jobs): rows += r
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for sp, tag, det, ok, _ in rows:
        for key in ((sp, tag), ("합계", tag)):
            a = agg[key]; a[0] += 1; a[1] += det; a[2] += ok
    print("| split | 팔 | 병변×시드 | 검출됨 | 검출된 것 중 분류 정답 |")
    print("|---|---|---|---|---|")
    for sp in ("test", "val", "합계"):
        for tag in ("v2a36_pf", "v2a40_pf"):
            n, d, o = agg[(sp, tag)]
            print(f"| {sp} | {tag[:5]} | {n} | {d} ({d/max(n,1):.0%}) | {o}/{d} = {o/max(d,1):.1%} |")
    by = collections.defaultdict(lambda: [0, 0])
    for sp, tag, det, ok, name in rows:
        if det:
            key = (re.sub(r'^[RL]-', '', name)[:24], tag); by[key][0] += 1; by[key][1] += ok
    print("\n| 클래스 | 36 정답 | 40 정답 |\n|---|---|---|")
    for nm in sorted({k[0] for k in by}):
        a, b = by[(nm, "v2a36_pf")], by[(nm, "v2a40_pf")]
        print(f"| {nm} | {a[1]}/{a[0]} | {b[1]}/{b[0]} |")
