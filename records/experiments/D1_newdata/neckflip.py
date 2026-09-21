#!/usr/bin/env python3
"""목 기준에서 살릴 것 — '낭 최근접 혈관이 반대쪽' 인 검출 병변이 test/val 에 몇 개인가.

배경: V1-D/V1-E 로 목 기준을 피처로 쓰는 두 방식(전역 교체·36차원 추가)이 둘 다 미채택됐다.
원인은 학습표 271행 중 목과 낭의 최근접 혈관이 다른 행이 **13개(4.8%)** 뿐이고,
좌우가 뒤집힌 행은 **2개(0.7%)** 라 RF 가 배울 표본이 없다는 것.
그래서 피처가 아니라 **라벨 없이 판정되는 조건부 룰**로 살릴 수 있는지 상금부터 잰다.

트리거(라벨 불필요): 병변의 낭 기준 최근접 혈관과 목 기준 최근접 혈관이 **좌우가 반대**.
  좌측 병변의 가장 가까운 혈관이 우측일 수는 없다 — 라벨을 안 봐도 틀린 게 확실하다.

이 스크립트는 룰을 적용하지 않는다. 검출된 병변에서 트리거가 몇 번 켜지는지만 센다.
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
E = f"{R}/experiments"; P = f"{E}/_c1_realpred"
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
VN = {int(v): k for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}
LOC = {int(k): v for k, v in S["location_classes"].items()}
MAX_R = 10.0


def neck_of(mc, vc):
    nk = mc & (vc > 0)
    if not nk.any():
        return mc
    lab, n = ndimage.label(nk, structure=np.ones((3, 3, 3)))
    if n > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        nk = lab == int(cnt.argmax())
    return nk


def dmap(ref, vc, spacing):
    d = ndimage.distance_transform_edt(~ref, sampling=spacing)
    out = {}
    for c in np.unique(vc):
        c = int(c)
        if c == 0 or c not in VN:
            continue
        x = float(d[vc == c].min())
        if x <= MAX_R:
            out[VN[c]] = x
    return out


def side(n):
    return n[0] if n and n[0] in "RL" else "M"


def one(arg):
    sp_name, cid = arg
    ap = f"{P}/aneu_{sp_name}_b1ff/{cid}.nii.gz"
    vp = f"{P}/vespp_{sp_name}/{cid}.nii.gz"
    gp = f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz"
    if not (os.path.exists(ap) and os.path.exists(vp)):
        return []
    ai = nib.load(ap); a = np.asanyarray(ai.dataobj)
    v = np.asanyarray(nib.load(vp).dataobj)
    if a.shape != v.shape:
        return []
    spacing = np.array(ai.header.get_zooms()[:3], float)
    gt = np.asanyarray(nib.load(gp).dataobj) if os.path.exists(gp) else None
    lab, n = ndimage.label(a > 0, structure=np.ones((3, 3, 3)))
    out = []
    for i in range(1, n + 1):
        mc = lab == i
        nv = int(mc.sum())
        if nv < 10:
            continue
        ds = dmap(mc, v, spacing)
        ref = neck_of(mc, v)
        dn = dmap(ref, v, spacing)
        if not ds or not dn:
            continue
        a1 = min(ds.items(), key=lambda kv: kv[1])[0]
        b1 = min(dn.items(), key=lambda kv: kv[1])[0]
        g = ""
        if gt is not None:
            u = collections.Counter(int(x) for x in gt[mc] if x)
            if u:
                g = LOC.get(u.most_common(1)[0][0], "")
        out.append(dict(split=sp_name, case=cid, n_vox=nv, n_neck=int(ref.sum()),
                        sac_near=a1, neck_near=b1, gt=g,
                        changed=a1 != b1,
                        flipped=side(a1) != side(b1) and "M" not in (side(a1), side(b1))))
    return out


def main():
    import multiprocessing as mp
    jobs = [(sp, c) for sp in ("test", "val") for c in S["splits"][sp]]
    res = []
    with mp.Pool(4) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, jobs), 1):
            res.extend(rows)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}  병변 {len(res)}", flush=True)
    json.dump(res, open(f"{R}/experiments/D1_newdata/neckflip.json", "w"), ensure_ascii=False)

    print(f"\n검출 병변 {len(res)}개 (test {sum(r['split']=='test' for r in res)} · val {sum(r['split']=='val' for r in res)})\n")
    for sp in ("test", "val"):
        s = [r for r in res if r["split"] == sp]
        ch = [r for r in s if r["changed"]]; fl = [r for r in s if r["flipped"]]
        print(f"{sp}: 최근접 혈관 바뀜 {len(ch)}/{len(s)} = {len(ch)/max(len(s),1):.1%} · "
              f"**좌우 뒤집힘 {len(fl)}**")
    fl = [r for r in res if r["flipped"]]
    if fl:
        print("\n| split | GT 클래스 | n_vox | 목/낭 | 낭 최근접 | 목 최근접 | 목이 옳은 쪽인가 |")
        print("|---|---|---|---|---|---|---|")
        for r in sorted(fl, key=lambda x: -x["n_vox"]):
            gs = side(r["gt"]) if r["gt"] else "?"
            ok = "-" if gs in ("?", "M") else ("예" if side(r["neck_near"]) == gs else "아니오")
            print(f"| {r['split']} | {r['gt'] or 'GT없음(오탐)'} | {r['n_vox']:,} | "
                  f"{r['n_neck']/max(r['n_vox'],1):.3f} | {r['sac_near']} | {r['neck_near']} | {ok} |")


if __name__ == "__main__":
    main()
