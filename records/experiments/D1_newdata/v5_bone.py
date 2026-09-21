#!/usr/bin/env python3
"""V5-F3 — 뼈 피처(CTA). train 원본 CT 가 09-10 정리로 지워져 HDD 의 nnU-Net 전처리본(Dataset720 3d_fullres)에서 뽑는다.

전처리본 = 원본을 (크롭 없음 확인) 목표 간격으로 리샘플 + 케이스별 z-score(클리핑 없음). HU 는 케이스별 1차변환이므로
두 기준점으로 되돌린다:  공기 봉우리 최빈 → −1000 HU · 연조직(뇌) 봉우리 최빈 → 40 HU (퍼센타일은 FOV 밖 패딩에 속는다 · to_hu 참고).
  python v5_bone.py check   원본 CT 가 남은 test·val CT 케이스로 복원 정확도만 잰다(영상만 · 라벨 안 봄)
                            관문: 뼈 마스크(HU>700) Dice 중앙 ≥ 0.90 ∧ 최저 ≥ 0.80 (센터 전부). 미달이면 F3 를 스크리닝에서 뺀다.
  python v5_bone.py train   train 271행 중 CT 행의 F3 = [최단 뼈 거리 mm · 반경 8mm 구 안 뼈 비율 · CT 플래그]
                            MR 행은 [-1,-1,0]. 출력 v5_bone_train.json
"""
import json, os, sys, collections, pickle
import numpy as np, nibabel as nib, blosc2
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
PRE = "/mnt/hdd/sblee/topaneu_archive/nnUNet_preprocessed/Dataset720_TopAneuBinary417/nnUNetPlans_3d_fullres"
ST = np.ones((3, 3, 3), bool)
BONE_HU = 700.0


def load_pre(cid, ref_shape_xyz, ref_sp_xyz):
    meta = pickle.load(open(f"{PRE}/{cid}.pkl", "rb"))
    bb = meta["bbox_used_for_cropping"]; sbc = tuple(meta["shape_before_cropping"])
    assert sbc[::-1] == tuple(ref_shape_xyz), (cid, sbc, ref_shape_xyz)
    assert all(b[0] == 0 and b[1] == s for b, s in zip(bb, sbc)), (cid, "크롭됨")
    z = blosc2.open(f"{PRE}/{cid}.b2nd", mode="r")[:][0].transpose(2, 1, 0)      # zyx → xyz
    sp = np.asarray(ref_sp_xyz) * np.asarray(ref_shape_xyz) / np.asarray(z.shape)
    return z, sp


def to_hu(z):
    """공기·연조직 두 봉우리로 1차변환을 되돌린다. 퍼센타일이 아니라 **봉우리 최빈**을 쓴다 —
    center4 일부는 FOV 밖이 −2048 로 패딩돼 0.5 퍼센타일이 공기가 아니라 패딩을 잡았다(뼈 Dice 0.33)."""
    s = z[::2, ::2, ::2].ravel()
    p5, p25, zt = (float(v) for v in np.percentile(s, [5, 25, 99.9]))
    def mode(lo, hi):
        h, e = np.histogram(s[(s > lo) & (s < hi)], bins=200)
        return float((e[h.argmax()] + e[h.argmax() + 1]) / 2)
    zs = mode(p25, zt)                                   # 연조직(뇌) 봉우리
    za = mode(p5 - 0.1 * (zs - p5), p5 + 0.5 * (zs - p5))  # 공기 봉우리
    return -1000.0 + (z - za) * (1040.0 / (zs - za))


def pre_index(idx_xyz, scale):
    return np.clip(np.round((idx_xyz + 0.5) * scale - 0.5).astype(int), 0, None)


def bone_feats(hu, sp, les_pre_idx):
    cv = les_pre_idx.mean(0)
    half = np.ceil(25.0 / sp).astype(int)
    lo = np.maximum(np.round(cv).astype(int) - half, 0); hi = np.minimum(np.round(cv).astype(int) + half + 1, hu.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    bone = hu[sl] > BONE_HU
    lab, n = ndimage.label(bone, structure=ST)
    minvox = int(np.ceil(60.0 / np.prod(sp)))          # 60 mm^3 미만 조각(석회화·잡음) 제외
    if n:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        bone = np.isin(lab, np.where(cnt >= minvox)[0])
    if not bone.any():
        return [25.0, 0.0, 1.0]
    db = ndimage.distance_transform_edt(~bone, sampling=sp)
    li = les_pre_idx - lo
    ok = np.all((li >= 0) & (li < np.array(bone.shape)), axis=1)
    dmin = float(db[tuple(li[ok].T)].min()) if ok.any() else 25.0
    g = np.stack(np.meshgrid(*[np.arange(s) for s in bone.shape], indexing="ij"), -1).reshape(-1, 3)
    ball = np.linalg.norm((g - (cv - lo)) * sp, axis=1) <= 8.0
    return [dmin, float(bone.ravel()[ball].mean()), 1.0]


def check_case(cid):
    im = nib.load(f"{R}/dataset/TopAneu/images/{cid}_0000.nii.gz")
    raw = np.asanyarray(im.dataobj).astype(np.float32); sp0 = np.array(im.header.get_zooms()[:3], float)
    z, sp = load_pre(cid, raw.shape, sp0)
    hu = to_hu(z)
    scale = np.array(z.shape) / np.array(raw.shape)
    # 원본 HU 를 전처리 격자로 최근접 표본 → 같은 격자에서 뼈 마스크 비교
    gi = [np.clip(np.round((np.arange(n) + 0.5) / s - 0.5).astype(int), 0, m - 1) for n, s, m in zip(z.shape, scale, raw.shape)]
    raw_on = raw[np.ix_(*gi)]
    a = raw_on > BONE_HU; b = hu > BONE_HU
    dice = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    # 연조직·혈관 대표값 오차
    return cid, float(dice), float(np.median(hu[raw_on > 500] - raw_on[raw_on > 500])) if (raw_on > 500).any() else 0.0


def train_case(cid, rows):
    li = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    loc = np.asanyarray(li.dataobj); sp0 = np.array(li.header.get_zooms()[:3], float)
    out = {}
    if "_ct_" not in cid:
        return {f"{cid}|{r['lesion_mask_idx']}": [-1.0, -1.0, 0.0] for r in rows}
    z, sp = load_pre(cid, loc.shape, sp0)
    hu = to_hu(z)
    scale = np.array(z.shape) / np.array(loc.shape)
    lab, _ = ndimage.label(loc > 0, structure=ST)
    for r in rows:
        idx = np.argwhere(lab == r["lesion_mask_idx"])
        assert np.linalg.norm(idx.mean(0) * sp0 - np.array(r["_cen"])) < 0.05
        out[f"{cid}|{r['lesion_mask_idx']}"] = bone_feats(hu, sp, np.unique(pre_index(idx, scale), axis=0))
    return out


def main():
    import multiprocessing as mp
    mode = sys.argv[1]
    S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
    if mode == "check":
        cs = sorted(c for sp in ("val", "test") for c in S[sp] if "_ct_" in c
                    and os.path.exists(f"{R}/dataset/TopAneu/images/{c}_0000.nii.gz") and os.path.exists(f"{PRE}/{c}.pkl"))
        with mp.Pool(6) as p:
            res = p.map(check_case, cs)
        for c, d, e in res:
            print(f"  {c}  뼈 Dice {d:.3f}  (HU>500 복셀 복원오차 중앙 {e:+.0f} HU)")
        med = float(np.median([d for _, d, _ in res])); worst = float(min(d for _, d, _ in res))
        ok = med >= 0.90 and worst >= 0.80
        print(f"\n뼈 Dice 중앙 {med:.3f} → 최저 {worst:.3f} → 관문(중앙≥0.90 ∧ 최저≥0.80) {'통과' if ok else '미달 — F3 제외'}")
        json.dump(dict(dice_median=med, dice_min=worst, gate=ok, n=len(res)), open(f"{D}/v5_bone_check.json", "w"))
    else:
        assert json.load(open(f"{D}/v5_bone_check.json"))["gate"], "복원 관문 미달"
        rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["case"]].append(r)
        res = {}
        with mp.Pool(int(os.environ.get("V5_WORKERS", "6"))) as p:
            for o in p.starmap(train_case, list(by.items()), chunksize=1):
                res.update(o)
        assert len(res) == len(rows)
        json.dump(res, open(f"{D}/v5_bone_train.json", "w"))
        ct = [v for v in res.values() if v[2] == 1.0]
        print(f"완료 {len(res)}행 · CT {len(ct)} · 최단 뼈거리 중앙 {np.median([v[0] for v in ct]):.1f}mm")


if __name__ == "__main__":
    main()
