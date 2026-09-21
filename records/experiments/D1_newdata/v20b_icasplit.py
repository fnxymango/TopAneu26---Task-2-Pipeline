#!/usr/bin/env python3
"""V2-0b — V2-0 의 축 버그를 고치고 분지 기준을 해부학적으로 바로잡아 재측정한다.

V2-0 에서 무엇이 잘못됐나
  (1) **seed 배치 버그.** 축의 근위 끝점을 'C1-C5 에 가장 가까운 복셀'로 잡았는데, 그게
      덩어리 **한가운데**에 떨어지면 측지거리가 양방향으로 커져 t 가 의미를 잃는다.
      실측: fine GT 42 side 중 4개(전부 center2 R측)에서 C6끝 t > C7끝 t 가 나왔다.
      전체로는 6/588 = 1% 지만 fine GT 표본의 10% 를 때려 Dice 비교를 오염시켰다.
      → **이중 스윕**으로 고친다. 덩어리 안에서 임의점→최원점 e1, e1→최원점 e2 를 구하면
        e1·e2 는 반드시 **끝점**이다. 둘 중 C1-C5 에 가까운 쪽을 seed 로 쓴다.
        seed 가 한가운데 떨어지는 일이 구조적으로 불가능해진다.

  (2) **분지를 잘못 골랐다.** terminus 시작을 AChA 로 잡았는데 해부학적으로 틀렸다.
      C7(교통 분절)은 Pcom 기시부에서 시작해 **종말 분지부**에서 끝난다. 종말 분지부는
      A1 과 M1 이 갈라지는 자리다. AChA 는 경계가 아니라 C7 **안쪽**에 있는 분지다.
      V2-0 의 라벨 없는 전수조사가 이 교과서적 사실과 일치했다(train 579 side):
         Pcom 0.587 · AChA 0.738 · A1 0.845 · M1 0.904
      수작업 경계(축 정상 38 side): C6끝 0.544 · C7끝 0.878
         → C6|C7 ↔ Pcom  차이 0.043 · C7|terminus ↔ A1  차이 0.033
      ⚠ 이 재정의는 **해부학 정의에서 나온 것**이지 Dice 를 보고 고른 게 아니다.
        (V2-0 의 AChA 선택이 내 실수였고, 후보는 Pcom·AChA·A1·M1 넷뿐이다.)

세 분할
  (A) jskim   고정 0.40 / 0.75                              ← 누수 있음. 대조군
  (B) branch  C6=[0,t_Pcom) · C7=[t_Pcom,t_A1) · terminus=[t_A1,1]
              **절단별 폴백** — 그 분지가 없을 때만 그 절단만 백분위로 대체
  (C) train   백분위. 절단점을 train fine GT 에서만 재추정(축 정상 side 만)

── 판정규칙 (V2-0 과 동일 · 결과 보기 전 고정) ──────────────────────────
  비교는 **train·val fine GT 에서만**. test 10케이스는 열지 않는다.
  1. B 평균 Dice ≥ A 평균 Dice → B 채택
  2. B 가 A 보다 0.05 이상 낮으면 → C
  3. B 에서 **두 절단 다 분지로 결정된** side 비율 < 40% → C
  4. C 절단점이 0.40/0.75 와 ±0.05 초과 차이 → 누수 영향 실재, A 사용 금지
  추가: 축 뒤집힘이 0 이어야 한다. 1개라도 남으면 측정 실패로 보고 판정하지 않는다.
───────────────────────────────────────────────────────────────────────────
"""
import json, os, glob, collections
import numpy as np, nibabel as nib
from scipy import ndimage
from skimage.graph import MCP_Geometric

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
LAB = "/home/sblee/TopAneu-26/labeling"   # 2026-09-14 이동 + 9건 추가: train 22 / val 8 / test 10
D = f"{R}/experiments/D1_newdata"
VID = {k: int(v) for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
SPLIT = {c: sp for sp, cs in S["splits"].items() for c in cs}
FINE = json.load(open(f"{LAB}/vessel_mapping_fine.json"))["labels"]
F_C6 = {"R": FINE["R-ICA-C6"], "L": FINE["L-ICA-C6"]}
F_C7 = {"R": FINE["R-ICA-C7"], "L": FINE["L-ICA-C7"]}
F_TM = {"R": FINE["R-ICA-C7-terminus"], "L": FINE["L-ICA-C7-terminus"]}
TOUCH_MM = 2.0


def geo(blob, spacing, seed):
    mcp = MCP_Geometric(np.where(blob, 1.0, np.inf),
                        sampling=tuple(float(x) for x in spacing))
    g, _ = mcp.find_costs([tuple(seed)])
    return np.where(blob, g, np.nan)


def axis_t(ves, spacing, side):
    """이중 스윕으로 끝점을 찾아 축을 세운다."""
    blob = ves == VID[f"{side}-ICA-C6-C7"]
    if blob.sum() < 30:
        return None
    lab, n = ndimage.label(blob, structure=np.ones((3, 3, 3)))
    if n > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        blob = lab == int(cnt.argmax())
    idx = np.argwhere(blob)
    # 1스윕: 임의점 → 최원점 e1 (반드시 끝점)
    g0 = geo(blob, spacing, idx[0])
    if not np.isfinite(np.nanmax(g0)):
        return None
    e1 = np.unravel_index(np.nanargmax(g0), blob.shape)
    # 2스윕: e1 → 최원점 e2 (반대쪽 끝점)
    g1 = geo(blob, spacing, e1)
    e2 = np.unravel_index(np.nanargmax(g1), blob.shape)
    prox = ves == VID[f"{side}-ICA-C1-C5"]
    if prox.any():
        d = ndimage.distance_transform_edt(~prox, sampling=spacing)
        seed = e1 if d[e1] <= d[e2] else e2
    else:
        seed = e1
    g = geo(blob, spacing, seed)
    mx = np.nanmax(g)
    if not np.isfinite(mx) or mx <= 0:
        return None
    return g / mx, blob, float(mx)


def branch_t(ves, spacing, blob, t, name):
    vid = VID.get(name)
    if vid is None:
        return None
    br = ves == vid
    if br.sum() < 5:
        return None
    d = ndimage.distance_transform_edt(~br, sampling=spacing)
    sel = blob & (d <= TOUCH_MM)
    if sel.sum() < 3:
        return None
    return float(np.nanmedian(t[sel]))


def one(cid):
    vp = f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz"
    fp = f"{LAB}/{cid}.nii.gz"
    if not (os.path.exists(vp) and os.path.exists(fp)):
        return []
    im = nib.load(vp)
    ves = np.asanyarray(im.dataobj)
    spacing = np.array(im.header.get_zooms()[:3], float)
    fine = np.asanyarray(nib.load(fp).dataobj)
    if fine.shape != ves.shape:
        return []
    out = []
    for side in ("R", "L"):
        a = axis_t(ves, spacing, side)
        if a is None:
            continue
        t, blob, length = a
        rec = dict(case=cid, split=SPLIT.get(cid, "?"), side=side,
                   n_vox=int(blob.sum()), len_mm=length)
        for nm, key in ((f"{side}-OA", "t_OA"), (f"{side}-Pcom", "t_Pcom"),
                        (f"{side}-AChA", "t_AChA"), (f"{side}-M1", "t_M1"),
                        (f"{side}-A1A2", "t_A1")):
            rec[key] = branch_t(ves, spacing, blob, t, nm)
        for key, ids in (("gt_c6_end", F_C6[side]), ("gt_c7_end", F_C7[side])):
            m = blob & (fine == ids)
            rec[key] = float(np.nanmax(t[m])) if m.sum() >= 3 else None
        os.makedirs(f"{D}/v20b_cache", exist_ok=True)
        np.save(f"{D}/v20b_cache/{cid}_{side}_t.npy", t[blob].astype(np.float32))
        np.save(f"{D}/v20b_cache/{cid}_{side}_g.npy",
                np.select([fine[blob] == F_C6[side], fine[blob] == F_C7[side],
                           fine[blob] == F_TM[side]], [0, 1, 2], default=-1).astype(np.int8))
        out.append(rec)
    return out


def main():
    import multiprocessing as mp
    cases = sorted({os.path.basename(f)[:-7] for f in glob.glob(f"{LAB}/*.nii.gz")})
    cases = [c for c in cases if SPLIT.get(c) in ("train", "val")]   # test 는 열지 않는다
    print(f"fine GT 중 train·val {len(cases)}케이스 (test 는 제외)", flush=True)
    res = []
    with mp.Pool(8) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, cases), 1):
            res.extend(rows)
            if i % 5 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  side {len(res)}", flush=True)
    json.dump(res, open(f"{D}/v20b_icasplit.json", "w"))
    print(f"측정 완료 side {len(res)}")


if __name__ == "__main__":
    main()
