#!/usr/bin/env python3
"""ICA-C6-C7 덩어리를 C6 / C7 / C7-terminus 로 나누는 공용 라이브러리 (V2 계열).

v20b_icasplit.py 의 축 정의를 그대로 옮기되 두 가지를 더했다.
  1. **bbox 크롭.** 거리변환·측지거리를 덩어리 주변(여유 20mm)에서만 계산한다.
     V2-0/0b 는 전체 볼륨에서 분지마다 EDT 를 돌려 케이스당 수 분이 걸렸다.
     V2-A 는 415케이스 전부를 분할해야 하므로 이게 없으면 몇 시간이 걸린다.
  2. **분할 마스크 생성.** 최대 연결성분 밖의 ICA-C6-C7 조각은 가장 가까운 본체 복셀의
     t 를 물려받는다(조각을 버리면 혈관 복셀 수가 바뀌어 기존 GT 와 달라진다).

축: 이중 스윕으로 양 끝점을 구하고 ICA-C1-C5 에 가까운 끝을 seed 로 쓴다(V2-0b 수정).
t = 덩어리 안쪽 측지거리 / 최댓값 ∈ [0,1].
분지 위치 t_b = 분지 b 에 2mm 이내로 닿는 덩어리 복셀들의 t 중앙값.

규칙(rule dict)
  kind="pct"     c1, c2 고정
  kind="branch"  c1 = t_Pcom + off1 (없으면 fallback), c2 = t_A1 + off2 (없으면 fallback)
                 fallback: fb1/fb2 가 있으면 선형식, 없으면 c1_pct/c2_pct
"""
import json, os
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
VID = {k: int(v) for k, v in json.load(
    open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"].items()}
FINE_JSON = "/home/sblee/TopAneu-26/labeling/vessel_mapping_fine.json"
FINE = json.load(open(FINE_JSON))["labels"]
NEW_ID = {"R": (FINE["R-ICA-C6"], FINE["R-ICA-C7"], FINE["R-ICA-C7-terminus"]),
          "L": (FINE["L-ICA-C6"], FINE["L-ICA-C7"], FINE["L-ICA-C7-terminus"])}
TOUCH_MM = 2.0
PAD_MM = 20.0
BRANCHES = ("OA", "Pcom", "AChA", "M1", "A1A2")


def _crop(mask, spacing):
    idx = np.argwhere(mask)
    pad = np.ceil(PAD_MM / np.asarray(spacing)).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, np.array(mask.shape))
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _geo(blob, spacing, seed):
    mcp = MCP_Geometric(np.where(blob, 1.0, np.inf), sampling=tuple(float(x) for x in spacing))
    g, _ = mcp.find_costs([tuple(int(x) for x in seed)])
    return np.where(blob, g, np.nan)


def side_axis(ves, spacing, side):
    """덩어리 전체(조각 포함) 마스크, 본체의 t, 분지 t, 크롭 슬라이스. 실패 시 None."""
    full = ves == VID[f"{side}-ICA-C6-C7"]
    if full.sum() < 30:
        return None
    sl = _crop(full, spacing)
    v = ves[sl]; full_c = full[sl]
    lab, n = ndimage.label(full_c, structure=np.ones((3, 3, 3)))
    body = full_c
    if n > 1:
        cnt = np.bincount(lab.ravel()); cnt[0] = 0
        body = lab == int(cnt.argmax())
    idx = np.argwhere(body)
    g0 = _geo(body, spacing, idx[0])
    if not np.isfinite(np.nanmax(g0)):
        return None
    e1 = np.unravel_index(np.nanargmax(g0), body.shape)
    g1 = _geo(body, spacing, e1)
    e2 = np.unravel_index(np.nanargmax(g1), body.shape)
    prox = v == VID[f"{side}-ICA-C1-C5"]
    seed = e1
    if prox.any():
        d = ndimage.distance_transform_edt(~prox, sampling=spacing)
        seed = e1 if d[e1] <= d[e2] else e2
    g = _geo(body, spacing, seed)
    mx = np.nanmax(g)
    if not np.isfinite(mx) or mx <= 0:
        return None
    t = g / mx
    bt = {}
    for b in BRANCHES:
        vid = VID.get(f"{side}-{b}")
        br = (v == vid) if vid is not None else None
        if br is None or br.sum() < 5:
            bt[b] = None; continue
        dd = ndimage.distance_transform_edt(~br, sampling=spacing)
        sel = body & (dd <= TOUCH_MM)
        bt[b] = float(np.nanmedian(t[sel])) if sel.sum() >= 3 else None
    # 방향 검사(라벨 불필요): 종말 분지 M1·A1 은 원위라 t 가 커야 한다. 평균이 0.5 미만이면
    # seed 가 원위 끝에 잡힌 것이므로 뒤집는다. (C1-C5 라벨이 이상하게 붙은 케이스 대비 —
    # 실측 center2_ct_172 R: GT 축에서 C7-Pcom 병변 t=0.256, 예측 축에서 0.811)
    term = [bt[k] for k in ("M1", "A1A2") if bt.get(k) is not None]
    flipped = bool(term and np.mean(term) < 0.5)
    if flipped:
        t = 1.0 - t
        bt = {k: (None if v is None else 1.0 - v) for k, v in bt.items()}
    return dict(sl=sl, full=full_c, body=body, t=t, len_mm=float(mx), bt=bt, flipped=flipped)


def cuts(rule, bt):
    """rule 과 분지 t 로 (c1, c2, 두 절단 모두 분지로 결정됐는가)."""
    if rule["kind"] == "pct":
        return rule["c1"], rule["c2"], False
    p, a = bt.get("Pcom"), bt.get("A1A2")
    full1 = p is not None; full2 = a is not None
    if full1:
        c1 = p + rule.get("off1", 0.0)
    elif rule.get("fb1") and all(bt.get(k) is not None for k in rule["fb1"]["x"]):
        f = rule["fb1"]; c1 = f["b0"] + sum(w * bt[k] for k, w in zip(f["x"], f["w"]))
    else:
        c1 = rule["c1_pct"]
    if full2:
        c2 = a + rule.get("off2", 0.0)
    elif rule.get("fb2") and all(bt.get(k) is not None for k in rule["fb2"]["x"]):
        f = rule["fb2"]; c2 = f["b0"] + sum(w * bt[k] for k, w in zip(f["x"], f["w"]))
    else:
        c2 = rule["c2_pct"]
    if not (0.0 < c1 < c2 < 1.0):
        c1, c2 = rule["c1_pct"], rule["c2_pct"]
        full1 = full2 = False
    return float(c1), float(c2), bool(full1 and full2)


def seg_of_t(t, c1, c2):
    return np.where(t < c1, 0, np.where(t < c2, 1, 2))


def split_volume(ves, spacing, rule):
    """36클래스 혈관 마스크 → 40클래스. ICA-C6-C7 밖은 한 복셀도 안 바꾼다."""
    out = ves.astype(np.int16).copy()
    info = {}
    for side in ("R", "L"):
        ax = side_axis(ves, spacing, side)
        if ax is None:
            info[side] = None      # 덩어리가 없거나 축 실패 → 원래 id(4/6) 그대로 = C6 로 남는다
            continue
        c1, c2, full = cuts(rule, ax["bt"])
        t = ax["t"]; body = ax["body"]; fullm = ax["full"]
        # 본체 밖 조각: 가장 가까운 본체 복셀의 t 를 물려받는다
        if (fullm & ~body).any():
            _, ind = ndimage.distance_transform_edt(~body, sampling=spacing, return_indices=True)
            tt = t[tuple(ind)]
        else:
            tt = t
        seg = seg_of_t(np.where(fullm, tt, 0.0), c1, c2)
        ids = NEW_ID[side]
        sub = out[ax["sl"]]
        for k in (0, 1, 2):
            sub[fullm & (seg == k)] = ids[k]
        info[side] = dict(c1=c1, c2=c2, branch_full=full, len_mm=ax["len_mm"], bt=ax["bt"])
    return out, info


# 40클래스 이름표 (C6 은 id 4/6 을 이어받는다)
def ves40_names():
    return {int(v): k for k, v in FINE.items() if int(v) != 0}
