#!/usr/bin/env python3
"""detcmp.py — 검출기끼리만 비교한다. 분류기를 전혀 거치지 않는다.

왜 필요한가: 새 eval 의 MCC 는 검출과 분류가 섞인 값이라 검출기 우열을 직접 말해주지 못한다.
게다가 분류기는 시드마다 흔들려(H2 ③ val 산포가 평균의 15배) 검출 차이를 덮어버린다.
여기서 재는 두 값은 **시드와 무관**하다 — 분류기를 안 돌리기 때문이다.

  커버리지 : GT 병변을 3회 팽창시킨 영역이 검출 마스크와 겹치면 '검출'. perclass.py 와 같은 규약.
             분류기는 검출이 덮은 병변에만 손댈 수 있으므로, 여기서 놓친 것은 분류로 못 되찾는다.
  거짓 blob : 어떤 GT 병변과도 겹치지 않는 연결성분. 분류기가 지워야 할 짐의 크기다.

사용: detcmp.py b1ff e9ff [e9f3P5ff ...]
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; E = f"{R}/experiments"; P = f"{E}/_c1_realpred"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
NAME = {int(k): v for k, v in S["location_classes"].items()}; SP = S["splits"]
TAGS = sys.argv[1:]


def one(arg):
    sp, case = arg
    gp = f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"
    if not os.path.exists(gp):
        return None
    dps = {t: f"{P}/aneu_{sp}_{t}/{case}.nii.gz" for t in TAGS}
    dps = {t: v for t, v in dps.items() if os.path.exists(v)}
    if not dps:
        return None
    gi = nib.load(gp); gt = np.asanyarray(gi.dataobj)
    vx = float(np.prod(gi.header.get_zooms()[:3]))
    dm = {t: np.asanyarray(nib.load(v).dataobj) > 0 for t, v in dps.items()}
    les, fps = [], {}
    gtb = gt > 0
    gtd = ndimage.binary_dilation(gtb, iterations=3)      # 거짓 판정도 같은 관대함으로
    for t, m in dm.items():
        lab, n = ndimage.label(m)
        # GT(팽창) 와 한 복셀도 안 겹치는 성분만 거짓
        hit = set(np.unique(lab[gtd & m]).tolist()) - {0}
        fps[t] = int(n - len(hit))
    for c in sorted({int(x) for x in np.unique(gt) if x}):
        lab, n = ndimage.label(gt == c)
        for i in range(1, n + 1):
            m = lab == i
            if m.sum() < 3:
                continue
            d3 = ndimage.binary_dilation(m, iterations=3)
            les.append(dict(split=sp, case=case, cls=c, name=NAME[c], vox=int(m.sum()),
                            dia=float(2 * (3 * (m.sum() * vx) / (4 * np.pi)) ** (1 / 3)),
                            **{f"det_{t}": int(bool((d3 & dm[t]).sum())) for t in dm}))
    return les, dict(split=sp, case=case, **{f"fp_{t}": v for t, v in fps.items()})


def main():
    jobs = [(sp, c) for sp in ("test", "val") for c in SP[sp]]
    les, cas = [], []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(one, jobs, chunksize=2):
            if r:
                les += r[0]; cas.append(r[1])
    json.dump(dict(lesions=les, cases=cas, tags=TAGS),
              open(f"{E}/D1_newdata/detcmp.json", "w"), ensure_ascii=False, indent=1)

    print(f"# 검출기 단독 비교 · 분류기 미개입 · 시드 무관\n")
    for sp in ("test", "val"):
        L = [r for r in les if r["split"] == sp]; C = [r for r in cas if r["split"] == sp]
        avail = [t for t in TAGS if L and f"det_{t}" in L[0]]
        print(f"### {sp} · 병변 {len(L)} · 케이스 {len(C)}\n")
        print("| 검출기 | 커버 | 커버율 | 거짓blob/케이스 | 총 거짓blob |")
        print("|---|---|---|---|---|")
        for t in avail:
            cov = sum(r[f"det_{t}"] for r in L)
            fp = sum(r[f"fp_{t}"] for r in C if f"fp_{t}" in r)
            print(f"| {t} | {cov}/{len(L)} | {cov/len(L)*100:.1f}% | {fp/max(len(C),1):.2f} | {fp} |")
        print()
        if len(avail) >= 2:
            a, b = avail[0], avail[1]
            only_a = [r for r in L if r[f"det_{a}"] and not r[f"det_{b}"]]
            only_b = [r for r in L if r[f"det_{b}"] and not r[f"det_{a}"]]
            print(f"**{a} 만 잡음 {len(only_a)} · {b} 만 잡음 {len(only_b)}**  (서로 다르게 본 병변)\n")
            for nm, g in (("작음(<3mm)", lambda r: r["dia"] < 3), ("큼(≥3mm)", lambda r: r["dia"] >= 3)):
                sub = [r for r in L if g(r)]
                if not sub: continue
                row = " · ".join(f"{t} {sum(r[f'det_{t}'] for r in sub)}/{len(sub)}" for t in avail)
                print(f"- {nm}: {row}")
            print()


if __name__ == "__main__":
    raise SystemExit(main())
