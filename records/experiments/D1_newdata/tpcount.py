#!/usr/bin/env python3
"""tpcount.py — 두 구성의 **병변 단위** 적중을 비교한다.

왜 공식 지표 대신 이걸 1차 판정으로 쓰나: 공식 지표는 클래스 평균이라 희소 클래스
하나가 죽으면 0.01~0.02 가 통째로 빠지고, 우리 노이즈 바닥(학습표 6줄 차이)이
e2e MCC 로 ±0.02~0.03 이다(2026-09-11 실측). 병변 몇 개짜리 변화는 공식 지표로
판정할 수 없다. 병변 단위 적중 수는 그 변화를 직접 센다.

적중 정의: GT 클래스가 예측에 등장하면 적중(공식 eval 의 존재기반 TP 와 같은 규약).
사용: tpcount.py <기준태그> <후보태그>
"""
import json, os, sys, collections
import numpy as np, nibabel as nib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
H = f"{R}/experiments/H1_patchfilter"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
NAME = {int(k): v for k, v in S["location_classes"].items()}


def tally(tag):
    hit, tot = collections.Counter(), collections.Counter()
    for sp in ("test", "val"):
        for c in S["splits"][sp]:
            gp = f"{R}/dataset/TopAneu/location_masks/{c}.nii.gz"
            if not os.path.exists(gp):
                continue
            gt = np.asanyarray(nib.load(gp).dataobj)
            cls = sorted({int(x) for x in np.unique(gt) if x})
            if not cls:
                continue
            for sd in range(5):
                pp = f"{H}/pred/{tag}_{sp}_s{sd}/{c}.nii.gz"
                if not os.path.exists(pp):
                    continue
                pc = {int(x) for x in np.unique(np.asanyarray(nib.load(pp).dataobj)) if x}
                for k in cls:
                    tot[k] += 1
                    if k in pc:
                        hit[k] += 1
    return hit, tot


def main():
    a, b = sys.argv[1], sys.argv[2]
    ha, ta = tally(a); hb, tb = tally(b)
    A, B = sum(ha.values()), sum(hb.values()); T = sum(ta.values())
    print(f"기준 `{a}` {A}/{T} · 후보 `{b}` {B}/{T} · **Δ {B-A:+d}**\n")
    rows = [(hb.get(k, 0) - ha.get(k, 0), NAME[k], ha.get(k, 0), hb.get(k, 0), ta[k])
            for k in set(ta) | set(tb) if ta.get(k, 0)]
    ch = [r for r in rows if r[0]]
    if not ch:
        print("클래스별 변화 없음"); return 0
    print("| 클래스 | 기준 | 후보 | 기회 | Δ |")
    print("|---|---|---|---|---|")
    for d, n, x, y, t in sorted(ch):
        print(f"| {n} | {x}/{t} | {y}/{t} | {t} | **{d:+d}** |")
    up = sum(d for d, *_ in ch if d > 0); dn = sum(-d for d, *_ in ch if d < 0)
    print(f"\n새로 맞힌 {up} · 새로 틀린 {dn} · 순 {up-dn:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
