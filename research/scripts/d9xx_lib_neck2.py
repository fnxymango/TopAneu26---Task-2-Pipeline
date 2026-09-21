"""D9xx 위치분류 공용 로직 — feature 추출 + signature 규칙 + lookup table.

oracle_location_assign_417.py와 동일한 feature-extraction/rule 로직을 재사용하되,
GT 대신 예측(D720 aneurysm / D800 vessel)에도 적용할 수 있도록 일반화.

signature(=nearest vessel class, 필요시 반경 R 이웃집합)로 (train split GT에서 만든)
lookup table을 찾아 위치클래스를 예측한다. D720(fold0 base)/D800(fold0)를 그대로 쓴다는
사용자 지시(2026-08-09)에 따라 이 모듈은 모델 종류를 가리지 않고 "예측된 두 마스크"만 받는다.
"""
import json, os, collections
from pathlib import Path

import numpy as np
from scipy import ndimage

TOPANEU_ROOT = Path(os.environ.get("TOPANEU_ROOT", os.path.expanduser("~/topaneu_sblee")))
DATA = TOPANEU_ROOT / "dataset" / "TopAneu"
RAW = TOPANEU_ROOT / "nnunet" / "nnUNet_raw"
VES_RAW = RAW / "Dataset800_TopAneuVessel417"
SPLIT_JSON = DATA / "dataset_split.json"

MAX_R = 10.0


def load_split():
    return json.load(open(SPLIT_JSON))


def official_location_names():
    """official_id(int) -> name, name -> official_id(int)."""
    d = load_split()
    id2name = {int(k): v.strip() for k, v in d["location_classes"].items()}
    name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id


def vessel_dense_names():
    """D800 dense id(int, 1..36) -> name."""
    lab = json.load(open(VES_RAW / "dataset.json"))["labels"]
    return {int(v): k for k, v in lab.items() if int(v) != 0}


def case_ids_by_split():
    d = load_split()
    return d["splits"]["train"], d["splits"]["val"], d["splits"]["test"]


NECK_DIST = int(os.environ.get("TOPANEU_NECKBLOCK", "1"))   # 1이면 neck_dist 블록을 같이 뽑는다


def neck_of(mc, vc):
    """낭 ∩ 혈관 = 목 근사. 비면 낭 전체로 되돌린다.

    V1-D 와 같은 정의지만 **쓰임이 다르다.** V1-D 는 dist_mm 자체를 목 기준으로
    갈아끼웠고(전역 교체), 여기 V1-E 는 기존 낭 기준 dist_mm 을 **그대로 두고**
    목 기준 거리를 neck_dist 로 따로 내보낸다. 상위(c5_neck2)가 이를 36차원
    블록으로 덧붙이므로 RF 가 클래스별로 어느 쪽을 볼지 스스로 고른다.

    최대 연결성분만 남기는 이유: 낭이 다른 혈관을 살짝 스친 조각까지 포함하면
    기준점이 튄다(실측: 99,025복셀 낭의 목이 481복셀=0.5% 로 잡혀 가장 가까운
    혈관이 R-AICA 로 튐).
    """
    nk = mc & (vc > 0)
    if not nk.any():
        return mc
    lab, n = ndimage.label(nk, structure=np.ones((3, 3, 3)))
    if n > 1:
        cnt = np.bincount(lab.ravel())
        cnt[0] = 0
        nk = lab == int(cnt.argmax())
    return nk


def extract_lesion_features(loc_arr, ves_arr, spacing, ves_names, loc_id_to_name=None):
    """loc_arr: 병변 마스크(라벨 or 이진), ves_arr: vessel 마스크(dense id).
    loc_id_to_name이 주어지면 loc_arr의 값을 이름으로 변환해 gt_loc를 채운다(GT용).
    없으면(예측 aneurysm은 보통 이진 0/1) gt_loc는 None으로 둔다(추론용)."""
    rows = []
    lesions, n = ndimage.label(loc_arr > 0, structure=np.ones((3, 3, 3)))
    for lid in range(1, n + 1):
        m = lesions == lid
        nvox = int(m.sum())
        if nvox == 0:
            continue
        gt_loc = None
        if loc_id_to_name is not None:
            vals, cnts = np.unique(loc_arr[m], return_counts=True)
            gt_id = int(vals[np.argmax(cnts)])
            gt_loc = loc_id_to_name.get(gt_id)

        idx = np.argwhere(m)
        lo = idx.min(0); hi = idx.max(0) + 1
        marg = np.ceil(MAX_R / spacing).astype(int) + 1
        lo2 = np.maximum(lo - marg, 0); hi2 = np.minimum(hi + marg, np.array(loc_arr.shape))
        sl = tuple(slice(a, b) for a, b in zip(lo2, hi2))
        mc, vc = m[sl], ves_arr[sl]

        inside = vc[mc]
        ov = collections.Counter(int(x) for x in inside if x > 0)
        overlap = {ves_names[k]: int(v) for k, v in ov.items() if k in ves_names}

        dist = ndimage.distance_transform_edt(~mc, sampling=spacing)
        dmap = {}
        for c in np.unique(vc):
            if c == 0 or int(c) not in ves_names:
                continue
            sel = (vc == c) & (~mc)
            if not sel.any():
                dmap[ves_names[int(c)]] = 0.0
                continue
            d = float(dist[sel].min())
            if d <= MAX_R:
                dmap[ves_names[int(c)]] = round(d, 3)

        # ── V1-E 추가분 · 목 기준 거리 (기존 dist_mm 은 손대지 않는다) ──────────
        nmap = {}
        n_neck = nvox
        if NECK_DIST:
            ref = neck_of(mc, vc)
            n_neck = int(ref.sum())
            ndist = ndimage.distance_transform_edt(~ref, sampling=spacing)
            for c in np.unique(vc):
                if c == 0 or int(c) not in ves_names:
                    continue
                # 기준집합과 겹치면 0. 원본 [A] 의 `& (~mc)` 방식은 목이 작을 때
                # 무너진다(V1-D 실측: 목이 L-VA 안인데 L-VA 의 목 밖 복셀이 14.9mm
                # 떨어져 MAX_R 에 걸려 통째로 비었다).
                d = float(ndist[vc == c].min())
                if d <= MAX_R:
                    nmap[ves_names[int(c)]] = round(d, 3)

        rows.append(dict(
            lesion_mask_idx=lid, n_vox=nvox, n_neck=n_neck,
            gt_loc=gt_loc,
            overlap=overlap, dist_mm=dmap, neck_dist=nmap,
        ))
    return rows, lesions


def nearest_class(r):
    if not r["dist_mm"]:
        return None
    md = min(r["dist_mm"].values())
    tied = [c for c, d in r["dist_mm"].items() if d <= md + 1e-6]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda c: r["overlap"].get(c, 0))


def inside_major(r):
    if r["overlap"]:
        return max(r["overlap"], key=r["overlap"].get)
    return nearest_class(r)


def partners(r, R, exclude):
    return frozenset(c for c, d in r["dist_mm"].items() if d <= R and c != exclude)


RULES = {
    "V0_centroid": None,  # centroid 규칙은 별도 계산 필요, 여기선 미지원(오라클 전용)
    "V1_surface_nearest": lambda r: (nearest_class(r),),
    "V3_inside_major": lambda r: (inside_major(r),),
}
for _R in (1.0, 2.0, 3.0, 5.0):
    RULES[f"V2_nearest_r{_R:g}"] = (lambda R: lambda r: (nearest_class(r), partners(r, R, nearest_class(r))))(_R)
    RULES[f"V4_inside_major_r{_R:g}"] = (lambda R: lambda r: (inside_major(r), partners(r, R, inside_major(r))))(_R)


def build_lookup(rows, sig_fn):
    """(signature, backoff=nearest_class) -> Counter(location_name) 두 단계 테이블 + 전역 최빈값."""
    tab = collections.defaultdict(collections.Counter)
    btab = collections.defaultdict(collections.Counter)
    glob = collections.Counter()
    for r in rows:
        if r["gt_loc"] is None:
            continue
        sig = sig_fn(r)
        back = nearest_class(r)
        tab[sig][r["gt_loc"]] += 1
        btab[back][r["gt_loc"]] += 1
        glob[r["gt_loc"]] += 1
    glob_major = glob.most_common(1)[0][0] if glob else None
    return {
        "exact": {repr(k): dict(v) for k, v in tab.items()},
        "backoff": {repr(k): dict(v) for k, v in btab.items()},
        "global_major": glob_major,
    }


def predict_location(r, sig_fn, lookup):
    sig = repr(sig_fn(r))
    cand = lookup["exact"].get(sig)
    if not cand:
        back = repr(nearest_class(r))
        cand = lookup["backoff"].get(back)
    if not cand:
        return lookup["global_major"]
    return max(cand, key=cand.get)
