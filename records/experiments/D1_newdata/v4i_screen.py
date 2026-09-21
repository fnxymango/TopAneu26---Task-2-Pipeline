#!/usr/bin/env python3
"""V4-I 스크리닝 — ICA 원위 기하 룰을 **예측 혈관(vespp_train)** 조건으로 다시 잰다. test·val 안 봄.

deep_ica(GT 혈관)에서 M1 룰 τ2.0 55.3% vs RF 48.4%. 그러나 거리 임계 룰은 GT 표에서 정하면 추론에서
안 걸린다(V1-B). 그래서 축(t·구획)과 접촉거리(dist_mm) **둘 다 예측 혈관**에서 재고 τ·방식을 여기서 고른다.
RF 순위는 e2e 와 같은 하이브리드표 RF 의 OOF(deep_ica_oof.json).

── 사전 고정 (결과 보기 전 · 2026-09-14) ─────────────────────────────────────
 후보: τ ∈ {0.5, 1, 2, 3} × 방식 {M1 전면 대체, HC 접합(3.4·3.5·3.7)은 RF 유지·나머지 룰}
 대상: RF 1등이 ICA 원위(3.2~3.7)일 때만 룰 적용 (e2e 와 같은 발동 조건)
 선택: ICA 원위 train 병변 OOF 정확도 최고 (동률이면 망침 적은 쪽 → τ 작은 쪽)
 e2e 진행 관문: 정확도 ≥ RF + 3%p ∧ 고침 ≥ 2 × 망침. 미달이면 e2e 안 건다.
 ⚠ 발동 조건상 RF 1등이 ICA 원위가 아닌 병변(다른 영역으로 틀린 것)은 룰이 못 건드린다 — 그대로 센다.
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
sys.path.insert(0, D)
import deep_ica as DI
R, A, IL = DI.R, DI.A, DI.IL
VES = f"{R}/experiments/_c1_realpred/vespp_train"


def geo_case(cid, rows):
    im = nib.load(f"{VES}/{cid}.nii.gz")
    ves = np.asanyarray(im.dataobj); sp = np.array(im.header.get_zooms()[:3], float)
    im2 = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    assert im2.shape == ves.shape, cid
    loc = np.asanyarray(im2.dataobj)
    axes = {s: IL.side_axis(ves, sp, s) for s in ("R", "L")}
    out = {}
    for r in rows:
        k = DI.NAME2ID.get(r["gt_loc"])
        lab, n = ndimage.label(loc == k, structure=np.ones((3, 3, 3)))
        cen = np.array(r["_cen"]); best = None
        for i in range(1, n + 1):
            dd = float(np.linalg.norm(np.argwhere(lab == i).mean(0) * sp - cen))
            if best is None or dd < best[0]:
                best = (dd, i)
        if best is None or best[0] > 3.0:
            continue
        les = lab == best[1]; per = {}
        for s, ax in axes.items():
            if ax is None: continue
            lc = les[ax["sl"]]
            if not lc.any(): continue
            d = ndimage.distance_transform_edt(~lc, sampling=sp)
            dmin = float(np.nanmin(np.where(ax["body"], d, np.inf)))
            if dmin > 5.0: continue
            near = ax["body"] & (d <= dmin + 1.0)
            per[s] = dict(dmin=dmin, t=float(np.nanmedian(ax["t"][near])), bt=ax["bt"], len=ax["len_mm"])
        if per:
            s = min(per, key=lambda z: per[z]["dmin"])
            out[json.dumps([r["case"], r["lesion_mask_idx"]])] = dict(side=s, **per[s])
    return out


def main():
    hyb = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    pv = [r for r in json.load(open(f"{A}/c10_feat_train_predves_NEW.json")) if r.get("gt_loc")]
    assert [(r["case"], r["lesion_mask_idx"]) for r in hyb] == [(r["case"], r["lesion_mask_idx"]) for r in pv]
    oof = DI.oof(hyb)
    idx = [i for i, r in enumerate(pv) if DI.key(r["gt_loc"]) in DI.SEG]
    cache = f"{D}/v4i_geo_pv.json"
    if os.path.exists(cache):
        geo = json.load(open(cache))
    else:
        by = collections.defaultdict(list)
        for i in idx: by[pv[i]["case"]].append(pv[i])
        import multiprocessing as mp
        geo = {}
        with mp.Pool(8) as pool:
            for o in pool.starmap(geo_case, list(by.items())): geo.update(o)
        json.dump(geo, open(cache, "w"))
    gk = lambda i: json.dumps([pv[i]["case"], pv[i]["lesion_mask_idx"]])
    have = [i for i in idx if gk(i) in geo]
    seg_ok = sum(DI.geo_segment(geo[gk(i)]) == DI.SEG[DI.key(pv[i]["gt_loc"])] for i in have)
    side_ok = sum(geo[gk(i)]["side"] == pv[i]["gt_loc"][0] for i in have)
    print(f"# V4-I 스크리닝 — 예측 혈관 조건 ICA 원위 룰\n\nICA 원위 train 병변 {len(idx)} · 예측혈관 축 측정 {len(have)} · "
          f"쪽 정답 {side_ok} · 구획 정답 {seg_ok}/{len(have)} = {seg_ok/max(len(have),1):.0%}\n")
    # 접촉거리 분포 비교 (GT 표 vs 예측혈관 표) — 자기 클래스 분지
    BR = {"3.2": "OA", "3.4": "Pcom", "3.5": "AChA"}
    print("| 클래스 | 병변 | 분지 | GT표 거리 중앙 | 예측혈관표 거리 중앙 | 예측혈관에서 분지 없음 |\n|---|---|---|---|---|---|")
    for c, b in BR.items():
        ii = [i for i in idx if DI.key(pv[i]["gt_loc"]) == c]
        s = lambda t, i: (t[i]["dist_mm"] or {}).get(f"{t[i]['gt_loc'][0]}-{b}")
        gh = [s(hyb, i) for i in ii]; gp = [s(pv, i) for i in ii]
        med = lambda v: f"{np.median([x for x in v if x is not None]):.2f}" if any(x is not None for x in v) else "-"
        print(f"| {c} | {len(ii)} | {b} | {med(gh)} | {med(gp)} | {sum(x is None for x in gp)} |")
    print()
    tot = sum(len(oof[str(i)]) for i in idx)
    base = sum(s["names"][:1] == [pv[i]["gt_loc"]] for i in idx for s in oof[str(i)])
    print(f"M0 RF 1등: {base}/{tot} = {base/tot:.1%}\n")
    print("| 방식 | τ | 정답 | 정확도 | Δ%p | 고침 | 망침 | 발동 |\n|---|---|---|---|---|---|---|---|")
    res = []
    for mode in ("M1", "HC"):
        for tau in (0.5, 1.0, 2.0, 3.0):
            ok = fx = br = fire = 0
            for i in idx:
                g = geo.get(gk(i)); gt = pv[i]["gt_loc"]
                for s in oof[str(i)]:
                    old = s["names"][0] if s["names"] else None
                    new = old
                    if g is not None and DI.key(old) in DI.SEG and not (mode == "HC" and DI.key(old) in ("3.4", "3.5", "3.7")):
                        new = DI.rule_class(pv[i], g, tau)
                        fire += 1
                    ok += new == gt; fx += (new == gt) and (old != gt); br += (old == gt) and (new != gt)
            res.append((ok, -br, -tau, mode, tau, fx, br, fire))
            print(f"| {mode} | {tau} | {ok} | {ok/tot:.1%} | {100*(ok-base)/tot:+.1f} | {fx} | {br} | {fire} |")
    ok, _, _, mode, tau, fx, br, fire = max(res)
    gate = (ok - base) / tot >= 0.03 and fx >= 2 * br
    print(f"\n선택: {mode} τ{tau} · {ok/tot:.1%} (Δ {100*(ok-base)/tot:+.1f}%p) · 고침 {fx} 망침 {br}")
    print(f"**관문(≥+3%p ∧ 고침 ≥ 2×망침) → {'통과 · e2e 진행' if gate else '미달 · e2e 안 건다'}**")
    json.dump(dict(mode=mode, tau=tau, gate=gate, acc=ok/tot, base=base/tot, fixed=fx, broken=br),
              open(f"{D}/v4i_rule.json", "w"), indent=1)


if __name__ == "__main__":
    main()
