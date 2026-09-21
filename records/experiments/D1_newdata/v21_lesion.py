#!/usr/bin/env python3
"""V2-1 — 병변 단위 일관성 측정 (큐 2단계) · 같은 코드로 4단계(예측 혈관)도 잰다.

왜: 복셀 Dice 는 경계가 몇 복셀 어긋났는지를 잴 뿐, **분류기에게 필요한 정보**를 재지 않는다.
분류기가 알아야 하는 건 "이 병변이 C6·C7·terminus 중 어디에 붙어 있나" 이고, 그게 병변 라벨
(3.2/3.3 → C6 · 3.4/3.5/3.6 → C7 · 3.7 → terminus)과 맞는지가 곧 이 축의 상금이다.

무엇을 기록하나 (병변마다)
  lesion_t    병변에서 가장 가까운 ICA-C6-C7 본체 복셀들(최근접+1mm, 5mm 이내)의 t 중앙값
  manual_seg  같은 복셀들에서 수작업 fine GT 의 다수결 구획 (GT 혈관 모드에서만)
  expected    병변 클래스가 요구하는 구획
  bt          그 쪽 분지 위치(t_OA·t_Pcom·t_AChA·t_M1·t_A1A2)
규칙별 판정은 여기서 하지 않는다 — v22_select.py 가 저장된 lesion_t·bt 로 싸게 계산한다.

사용: v21_lesion.py gt    → v21_lesions_gt.json · v21_sides_gt.json · v21_cache/
      v21_lesion.py pred  → v21_lesions_pred.json · v21_sides_pred.json   (vespp_* 예측 혈관)
대상: ~/TopAneu-26/labeling 의 train·val 케이스만. **test 는 열지 않는다.**
"""
import json, os, sys, glob, re, collections
import numpy as np, nibabel as nib
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ica_split_lib as IL

R = IL.R
D = f"{R}/experiments/D1_newdata"
LAB = os.path.dirname(IL.FINE_JSON)
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
SPLIT = {c: sp for sp, cs in S["splits"].items() for c in cs}
LOC = {int(k): v for k, v in S["location_classes"].items()}
EXPECT = {"3.2": 0, "3.3": 0, "3.4": 1, "3.5": 1, "3.6": 1, "3.7": 2}
NEAR_MAX_MM = 5.0


def cls_key(name):
    m = re.match(r"^(?:[RL]-)?(\d+\.\d+)\s", name)
    return m.group(1) if m else None


def one(arg):
    cid, mode = arg
    sp = SPLIT[cid]
    vdir = (f"{R}/dataset/TopAneu/vessel_masks" if mode == "gt"
            else f"{R}/experiments/_c1_realpred/vespp_{sp}")
    vp = f"{vdir}/{cid}.nii.gz"; fp = f"{LAB}/{cid}.nii.gz"
    lp = f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz"
    if not all(os.path.exists(p) for p in (vp, fp, lp)):
        return [], [], []
    im = nib.load(vp)
    ves = np.asanyarray(im.dataobj)
    spacing = np.array(im.header.get_zooms()[:3], float)
    fine = np.asanyarray(nib.load(fp).dataobj)
    loc = np.asanyarray(nib.load(lp).dataobj)
    if not (ves.shape == fine.shape == loc.shape):
        return [], [], []
    sides, lesions, fails = [], [], []
    axes = {}
    for side in ("R", "L"):
        ax = IL.side_axis(ves, spacing, side)
        if ax is None:
            fails.append(dict(case=cid, split=sp, side=side, why="축 실패(덩어리 없음/연결 안 됨)"))
            continue
        axes[side] = ax
        sl, body, t = ax["sl"], ax["body"], ax["t"]
        ids = IL.NEW_ID[side]
        fc = fine[sl]
        segf = np.select([fc == ids[0], fc == ids[1], fc == ids[2]], [0, 1, 2], default=-1)
        rec = dict(case=cid, split=sp, side=side, len_mm=ax["len_mm"], bt=ax["bt"])
        if mode == "gt":
            for key, k in (("gt_c6_end", 0), ("gt_c7_end", 1)):
                m = body & (segf == k)
                rec[key] = float(np.nanmax(t[m])) if m.sum() >= 3 else None
            # 강건한 경계: 앞 구획과 뒤 구획 복셀을 가장 잘 가르는 t (오분류 복셀 수 최소).
            # max(t) 는 원위 끝에 흩어진 라벨 복셀 하나로 1.0 이 된다(V2-0b center2_ct_172 R).
            for key, k in (("gt_c6_cut", 0), ("gt_c7_cut", 1)):
                a = np.sort(t[body & (segf <= k) & (segf >= 0)])
                b = np.sort(t[body & (segf > k)])
                if len(a) < 3 or len(b) < 3:
                    rec[key] = None; continue
                cand = np.unique(np.concatenate([a, b]))
                err = (len(a) - np.searchsorted(a, cand, side="left")) + np.searchsorted(b, cand, side="left")
                rec[key] = float(cand[int(np.argmin(err))])
            rec["inverted"] = bool(rec["gt_c6_cut"] is not None and rec["gt_c7_cut"] is not None
                                   and rec["gt_c6_cut"] >= rec["gt_c7_cut"])
            os.makedirs(f"{D}/v21_cache", exist_ok=True)
            m = body & (segf >= 0)
            np.save(f"{D}/v21_cache/{cid}_{side}_t.npy", t[m].astype(np.float32))
            np.save(f"{D}/v21_cache/{cid}_{side}_g.npy", segf[m].astype(np.int8))
        sides.append(rec)
    for k in [int(x) for x in np.unique(loc) if x]:
        name = LOC.get(k, "")
        ck = cls_key(name)
        if ck not in EXPECT:
            continue
        side = name[0]
        lab, n = ndimage.label(loc == k, structure=np.ones((3, 3, 3)))
        for i in range(1, n + 1):
            rec = dict(case=cid, split=sp, side=side, cls=name, comp=i, key=ck, expected=EXPECT[ck])
            ax = axes.get(side)
            if ax is None:
                rec["why"] = "그 쪽 축 없음"; lesions.append(rec); continue
            sl, body, t = ax["sl"], ax["body"], ax["t"]
            les = (lab == i)[sl]
            if not les.any():
                rec["why"] = "병변이 크롭 밖"; lesions.append(rec); continue
            d = ndimage.distance_transform_edt(~les, sampling=spacing)
            dmin = float(np.nanmin(np.where(body, d, np.inf)))
            if dmin > NEAR_MAX_MM:
                rec["why"] = f"본체까지 {dmin:.1f}mm"; rec["dist_mm"] = dmin; lesions.append(rec); continue
            near = body & (d <= dmin + 1.0)
            rec["dist_mm"] = dmin
            rec["lesion_t"] = float(np.nanmedian(t[near]))
            rec["bt"] = ax["bt"]
            if mode == "gt":
                ids = IL.NEW_ID[side]; fc = fine[sl][near]
                votes = collections.Counter(int(s) for s in
                                            np.select([fc == ids[0], fc == ids[1], fc == ids[2]],
                                                      [0, 1, 2], default=-1) if s >= 0)
                rec["manual_seg"] = votes.most_common(1)[0][0] if votes else None
            lesions.append(rec)
    return sides, lesions, fails


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gt"
    assert mode in ("gt", "pred")
    import multiprocessing as mp
    cases = sorted({os.path.basename(f)[:-7] for f in glob.glob(f"{LAB}/*.nii.gz")})
    cases = [c for c in cases if SPLIT.get(c) in ("train", "val")]      # test 는 열지 않는다
    print(f"[v21 {mode}] fine GT train·val {len(cases)}케이스", flush=True)
    S_, L_, F_ = [], [], []
    with mp.Pool(int(os.environ.get("V2_WORKERS", "10"))) as pool:
        for i, (s, l, f) in enumerate(pool.imap_unordered(one, [(c, mode) for c in cases]), 1):
            S_ += s; L_ += l; F_ += f
            if i % 5 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  side {len(S_)}  병변 {len(L_)}", flush=True)
    json.dump(S_, open(f"{D}/v21_sides_{mode}.json", "w"))
    json.dump(L_, open(f"{D}/v21_lesions_{mode}.json", "w"))
    json.dump(F_, open(f"{D}/v21_fails_{mode}.json", "w"))
    ok = sum("lesion_t" in r for r in L_)
    print(f"[v21 {mode}] 완료 · side {len(S_)} · 병변 {len(L_)} (위치 측정 {ok}) · 축 실패 {len(F_)}")


if __name__ == "__main__":
    main()
