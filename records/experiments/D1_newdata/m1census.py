#!/usr/bin/env python3
"""M1 길이 전수조사 — 5.2 early bifurcation 과 5.3 M1-M2 junction 을 가르는 축이 실재하는가.

왜 지금 하나
  5.2 와 5.3 은 해부학적으로 같은 사건(M1 이 M2 로 갈라짐)이고 차이는 **그 갈라짐이 M1 의
  어디쯤인가** 뿐이다. 그런데 현재 피처는 '종말부까지 거리'를 **두개 크기**로 나눈다.
  분모가 틀렸다 — 기준은 그 환자의 **M1 길이**여야 한다.
  기존 실측: 종말부까지 정규화거리 5.2 = 0.395±0.114 · 5.3 = 0.686±0.158,
  5.2 표본의 57% 가 5.3 범위 안에 들어간다.

설계 — 왜 과적합이 아닌가
  나쁜 설계: 병변 14개(5.2) vs 51개(5.3) 에서 임계를 탐색       ← 라벨을 보고 고름
  이 설계:   train 292케이스 × 좌우 = 584개 M1 길이로 모집단 분포를 먼저 만들고,
             병변의 M1 길이를 그 분포 안의 **백분위**로 환산한다.
             분포는 라벨과 무관하게 정해지므로 과적합할 대상이 없다.
  판정은 임계 탐색이 아니라 **AUC**(임계 무관 지표)로 한다.
  ⚠ 선례 C60 §19 — 292 전수로도 **라벨을 보고** 임계를 고르면 과적합한다(train +4 → test 0).

M1 길이 정의
  같은 쪽 M1 마스크의 최대 연결성분에서, ICA-C6-C7 에 가장 가까운 복셀을 근위 끝점으로 잡고
  마스크 **안쪽 측지거리**(skimage MCP_Geometric, 복셀 간격 반영)의 최댓값을 길이로 쓴다.
  직선 bbox 대각(구 m1len.py)은 사행이 심한 M1 에서 길이를 과소평가한다.

── 판정규칙 (결과 보기 전에 고정 · PLAN.md §2 그대로) ────────────────────
  1. AUC ≥ 0.90            단일 축으로 갈린다
  2. 겹침 구간 표본 < 20%   회색지대가 좁다
                           (겹침 = 5.2 최대와 5.3 최소가 만드는 구간에 든 표본 비율)
  3. 좌우 각각 성립         한쪽만 되면 우연 — R·L 각각 AUC ≥ 0.90
  셋 다 만족 → V1-A(호길이 피처) 착수.  하나라도 미달 → **이 축을 닫는다.**
───────────────────────────────────────────────────────────────────────────
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage
from skimage.graph import MCP_Geometric

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
A = f"{R}/code/sblee/nnunet/analysis"
VID = {k: int(v) for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}
S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
TRAIN = S["splits"]["train"]


def m1_length(ves, spacing, side):
    """side='R-' or 'L-'. 측지길이 mm, 부피 mm3, 복셀수를 돌려준다. 없으면 None."""
    m = ves == VID[f"{side}M1"]
    if m.sum() < 10:
        return None
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3)))
    if n > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        m = lab == int(cnt.argmax())
    ica = ves == VID[f"{side}ICA-C6-C7"]
    if not ica.any():
        return None
    # 근위 끝점 = ICA 종말부에 가장 가까운 M1 복셀
    d_ica = ndimage.distance_transform_edt(~ica, sampling=spacing)
    idx = np.argwhere(m)
    seed = idx[np.argmin(d_ica[m])]
    cost = np.where(m, 1.0, np.inf)
    mcp = MCP_Geometric(cost, sampling=tuple(float(x) for x in spacing))
    dist, _ = mcp.find_costs([tuple(seed)])
    dist = np.where(m, dist, -np.inf)
    L = float(dist.max())
    if not np.isfinite(L) or L <= 0:
        return None
    return L, float(m.sum() * np.prod(spacing)), int(m.sum())


def one_case(cid):
    """케이스 하나에서 좌우 M1 길이. 측지거리 계산이 케이스당 ~15초라 병렬로 돈다."""
    p = f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz"
    if not os.path.exists(p):
        return []
    im = nib.load(p)
    ves = np.asanyarray(im.dataobj)
    sp = np.array(im.header.get_zooms()[:3], float)
    out = []
    for side in ("R-", "L-"):
        r = m1_length(ves, sp, side)
        if r:
            out.append((f"{cid}|{side}", dict(len_mm=r[0], vol_mm3=r[1], n_vox=r[2])))
    return out


def main():
    out = f"{R}/experiments/D1_newdata/m1census.json"
    if os.path.exists(out) and "--force" not in sys.argv:
        cen = json.load(open(out))
        print(f"[census] 기존 결과 재사용 {len(cen)}건", flush=True)
        return cen
    import multiprocessing as mp
    cen = {}
    with mp.Pool(10) as pool:
        for i, rows in enumerate(pool.imap_unordered(one_case, TRAIN), 1):
            for k, v in rows:
                cen[k] = v
            if i % 25 == 0 or i == len(TRAIN):
                print(f"  {i}/{len(TRAIN)}  측정 {len(cen)}", flush=True)
    tmp = out + ".tmp"
    json.dump(cen, open(tmp, "w")); os.replace(tmp, out)
    return cen


if __name__ == "__main__":
    main()
