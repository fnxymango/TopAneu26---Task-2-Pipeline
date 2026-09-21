#!/usr/bin/env python3
"""V2-0 — ICA C6/C7/terminus 분할 기준을 무엇으로 삼을 것인가.

배경
  2026-09-11 16:45 김지수 님이 ~/labeling/ 에 40클래스 fine 라벨 31건을 올렸다.
  전수 확인 결과 **순수한 세분할**이다 — 기존 ICA-C6-C7 덩어리를 축을 따라 3등분했고
  ICA 밖은 복셀 하나도 안 바뀐다(4케이스 대조, 복셀 총수 완전 일치).
  즉 새로 들어온 정보는 파일이 아니라 **절단점 두 개(0.40 / 0.75)** 뿐이고,
  나머지는 기존 GT 에서 결정론적으로 재현된다 → 292케이스를 우리가 직접 만들 수 있다.

문제
  그 절단점은 20케이스(40 side)에서 캘리브레이션됐고 **그중 10건이 우리 공식 test** 다
  (vessel_mapping_fine.json 의 test_leakage_do_not_train 목록이 우리 test 와 정확히 일치).
  마스크를 안 써도 **숫자에 test 정보가 들어갔다** — PROJECT_RULES.md 1장 위반이고
  C60 §19(라벨 보고 임계 고르면 292 로도 과적합)와 같은 구조다.

그래서 무엇을 재나 — 세 가지 분할을 만들어 비교한다
  (A) jskim   고정 백분위 0.40 / 0.75                      ← 누수 있음. 대조군
  (B) branch  분지 기시부 기준. C6=[시작,Pcom), C7=[Pcom,AChA), terminus=[AChA,끝]
              OA/Pcom/AChA/M1/A1A2 는 이미 36클래스에 있다 → **캘리브레이션이 없어 누수 불가**
  (C) train   백분위인데 절단점을 **train 케이스에서만** 재추정

부수 산출(라벨 불필요): train 전수에서 t_OA · t_Pcom · t_AChA 의 분포.
  0.40/0.75 가 실제 분지 위치와 얼마나 맞는지가 여기서 바로 보인다.

축 정의
  ICA-C6-C7 덩어리의 최대 연결성분에서, ICA-C1-C5 에 가장 가까운 복셀을 근위 끝점으로 잡고
  덩어리 **안쪽 측지거리**(MCP_Geometric, 복셀간격 반영)를 최댓값으로 나눈 t∈[0,1].
  분지 b 의 위치 t_b = b 에 닿는(≤2mm) 덩어리 복셀들의 t 중앙값.

── 판정규칙 (결과 보기 전에 고정) ────────────────────────────────────────
  비교 대상은 **train 7 + val 3 = 10케이스** 뿐이다. test 10건은 절대 보지 않는다
  (수작업 fine GT 가 있어도 본 순간 우리도 오염된다).
  1. (B) 분지기준의 평균 Dice(C6·C7·terminus) ≥ (A) jskim 의 평균 Dice
     → 분지기준 채택. 누수 없는 기준이 성능도 같거나 낫다는 뜻.
  2. (B) 가 (A) 보다 **0.05 이상** 낮으면 → (C) train 재추정 백분위를 쓴다.
  3. 분지가 잡힌 side 비율 < 40% → 분지기준은 주력이 될 수 없다. (C) 로 간다.
  4. (C) 의 절단점이 0.40/0.75 와 ±0.05 를 넘게 다르면 → 누수 영향이 실재한 것으로 기록하고
     이후 모든 실험에서 (A) 를 쓰지 않는다.
  어느 경우든 최종 분할 규칙 하나를 정해 V2-A(상한 실험)로 넘어간다.
───────────────────────────────────────────────────────────────────────────
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage
from skimage.graph import MCP_Geometric

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
LAB = "/home/sblee/labeling"
D = f"{R}/experiments/D1_newdata"
VID = {k: int(v) for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
SPLIT = {c: sp for sp, cs in S["splits"].items() for c in cs}
FINE = json.load(open(f"{LAB}/vessel_mapping_fine.json"))["labels"]
TOUCH_MM = 2.0

# fine GT 에서 쓰는 id
F_C6 = {"R": FINE["R-ICA-C6"], "L": FINE["L-ICA-C6"]}
F_C7 = {"R": FINE["R-ICA-C7"], "L": FINE["L-ICA-C7"]}
F_TM = {"R": FINE["R-ICA-C7-terminus"], "L": FINE["L-ICA-C7-terminus"]}


def axis_t(ves, spacing, side):
    """ICA-C6-C7 덩어리의 정규화 측지좌표 t 와 덩어리 마스크. 못 만들면 None."""
    blob = ves == VID[f"{side}-ICA-C6-C7"]
    if blob.sum() < 30:
        return None
    lab, n = ndimage.label(blob, structure=np.ones((3, 3, 3)))
    if n > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        blob = lab == int(cnt.argmax())
    prox = ves == VID[f"{side}-ICA-C1-C5"]
    if not prox.any():
        return None
    d = ndimage.distance_transform_edt(~prox, sampling=spacing)
    idx = np.argwhere(blob)
    seed = idx[np.argmin(d[blob])]
    mcp = MCP_Geometric(np.where(blob, 1.0, np.inf),
                        sampling=tuple(float(x) for x in spacing))
    g, _ = mcp.find_costs([tuple(seed)])
    g = np.where(blob, g, np.nan)
    mx = np.nanmax(g)
    if not np.isfinite(mx) or mx <= 0:
        return None
    return g / mx, blob, float(mx)


def branch_t(ves, spacing, blob, t, name):
    """분지 name 이 덩어리에 닿는 지점의 t 중앙값. 안 닿으면 None."""
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
    if not os.path.exists(vp):
        return []
    im = nib.load(vp)
    ves = np.asanyarray(im.dataobj)
    spacing = np.array(im.header.get_zooms()[:3], float)
    fine = None
    fp = f"{LAB}/{cid}.nii.gz"
    if os.path.exists(fp):
        f = np.asanyarray(nib.load(fp).dataobj)
        if f.shape == ves.shape:
            fine = f
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
        if fine is not None:
            # 수작업 경계를 t 로 환산 — C6 최대 t 와 C7 최대 t
            for key, ids in (("gt_c6_end", F_C6[side]), ("gt_c7_end", F_C7[side])):
                m = blob & (fine == ids)
                rec[key] = float(np.nanmax(t[m])) if m.sum() >= 3 else None
            rec["has_fine"] = True
            # Dice 계산용 복셀수
            rec["fine_n"] = dict(c6=int((blob & (fine == F_C6[side])).sum()),
                                 c7=int((blob & (fine == F_C7[side])).sum()),
                                 tm=int((blob & (fine == F_TM[side])).sum()))
            np.save(f"{D}/v20_cache/{cid}_{side}_t.npy", t[blob].astype(np.float32))
            np.save(f"{D}/v20_cache/{cid}_{side}_g.npy",
                    np.select([fine[blob] == F_C6[side], fine[blob] == F_C7[side],
                               fine[blob] == F_TM[side]], [0, 1, 2], default=-1).astype(np.int8))
        else:
            rec["has_fine"] = False
        out.append(rec)
    return out


def main():
    os.makedirs(f"{D}/v20_cache", exist_ok=True)
    import multiprocessing as mp
    cases = S["splits"]["train"] + S["splits"]["val"] + S["splits"]["test"]
    res = []
    with mp.Pool(10) as pool:
        for i, rows in enumerate(pool.imap_unordered(one, cases), 1):
            res.extend(rows)
            if i % 40 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}  side {len(res)}", flush=True)
    json.dump(res, open(f"{D}/v20_icasplit.json", "w"))
    print(f"\n측정 완료 side {len(res)}\n")


if __name__ == "__main__":
    main()
