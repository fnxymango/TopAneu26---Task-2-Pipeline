#!/usr/bin/env python3
"""arc 사후 분해 (2026-09-15 · 판정 불변) — 합계 +2.4%p 를 '살아난 병변 vs 새로 틀린 병변' 으로 나눈다.

arc_cv.py 와 같은 표본·같은 CV(층화 5폴드 × 시드 0~19 · RF 500 · 미러 포함)로 두 팔을 짝지어 돌리고,
원본 행(미러 제외)마다 base 정오 × arc 정오 교차표를 센다. 호길이 있음/없음(커버리지)으로 나눈다.
호길이 없는 행은 두 팔 입력이 같다(0 블록) → 그 행의 뒤집힘은 RF 난수 잡음의 기준선이다.
train 만 · test·val 안 봄.
"""
import json, os, sys, collections
import numpy as np
from multiprocessing import Pool
sys.argv = [sys.argv[0]]
import arc_cv as AC
from arc_cv import R, L, C5, A, add_block
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def build():
    rows = json.load(open(f"{R}/code/sblee/nnunet/analysis/e11_feat_hyb_ov.json"))
    S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
    NAME = {int(k): v for k, v in S["location_classes"].items()}
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names()); C5.USE_POS = True
    packs, Xb, Xa, y, has, orig = {}, [], [], [], [], []
    for r in rows:
        c = r["case"]
        if c not in packs:
            packs[c] = A.load_pack(f"{R}/experiments/D1_newdata/arc/train/{c}.npz")
        arc = A.lookup(packs[c], np.array(r["_cen"], float))
        for mir in (False, True):
            v = C5.row_to_vec(r, ves_axis, mirror=mir)
            Xb.append(v); Xa.append(add_block(v, arc, AC.ARC_W))
            y.append(C5.mirror_name(r["gt_loc"]) if mir else r["gt_loc"]); has.append(arc[7] > 0); orig.append(not mir)
    Xb, Xa, y, has, orig = map(np.array, (Xb, Xa, y, has, orig))
    ICA = {v for v in NAME.values() if any(f"3.{i}" in v for i in (2, 3, 4, 5, 6, 7))}
    keep = np.array([c for c in np.unique(y) if (y == c).sum() >= 5]); m = np.isin(y, keep)
    return Xb[m], Xa[m], y[m], has[m], orig[m], np.isin(y[m], sorted(ICA))


def run(seed):
    Xb, Xa, y, *_ = build()
    out = {}
    for nm, X in (("base", Xb), ("arc", Xa)):
        pred = np.empty_like(y)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            clf = RandomForestClassifier(n_estimators=500, max_features=0.3, class_weight="balanced", random_state=seed, n_jobs=1)
            clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
        out[nm] = pred == y
    return out


def main():
    Xb, Xa, y, has, orig, ica = build()
    with Pool(8) as p:
        res = p.map(run, range(20))
    print("# arc 사후 분해 — 살아난 병변 vs 새로 틀린 병변 (train CV · 시드 20 평균 · 원본 행만)\n")
    print("| 묶음 | 병변 | base 오답 | 살아남(오답→정답) | 새로 틀림(정답→오답) | 순증 | 살아난 비율(오답 중) |\n|---|---|---|---|---|---|---|")
    for gname, g in (("ICA 3.2~3.7 · 호길이 있음", ica & has), ("ICA 3.2~3.7 · 호길이 없음(입력 동일 → 잡음 기준)", ica & ~has),
                     ("ICA 밖 · 호길이 있음", ~ica & has), ("ICA 밖 · 호길이 없음(입력 동일 → 잡음 기준)", ~ica & ~has)):
        sel = g & orig
        n = sel.sum()
        bw = np.mean([(~r["base"][sel]).sum() for r in res])
        fx = np.mean([(~r["base"][sel] & r["arc"][sel]).sum() for r in res])
        br = np.mean([(r["base"][sel] & ~r["arc"][sel]).sum() for r in res])
        print(f"| {gname} | {n} | {bw:.1f} | {fx:.1f} | {br:.1f} | {fx - br:+.1f} | {fx / bw if bw else 0:.0%} |")


if __name__ == "__main__":
    main()
