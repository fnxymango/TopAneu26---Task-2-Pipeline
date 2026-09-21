#!/usr/bin/env python3
"""bpfalse.py — 가설 1 검증: 분류기가 보는 분기점이 '거짓'이면 그 병변이 더 많이 틀리는가.

분류기(c5)의 피처 [C] 는 병변에서 각 분기점 노드까지의 근접도 1/(1+거리mm) 다(반경 15mm).
그 분기점이 예측 혈관마스크에서 나온 것이므로, 혈관 모델이 없는 분지를 그리면
없는 분기점이 생기고 분류기는 가짜 랜드마크를 기준으로 판정하게 된다.

비교 대상:
  예측 분기점  experiments/_c4_bpgraph/{vespp_test, val_pred}/   (V5 후처리 혈관 → C4)
  GT 분기점    experiments/_c4_bpgraph/all_ref/                  (GT 혈관마스크 → C4)

정의
  '거짓 분기점' = 예측 노드 중, 같은 클래스쌍을 가진 GT 노드가 MATCH_MM 안에 없는 것
  '누락 분기점' = GT 노드 중, 같은 클래스쌍을 가진 예측 노드가 MATCH_MM 안에 없는 것
  병변 기준 반경 NEAR_MM 안의 노드만 센다 (c5 의 BP_MAX_R 과 같은 15mm).

추가로 작은 분지(Pcom·AChA·OA)가 병변 주변에서 **GT 에 없는데 예측에는 있는지**를 본다.
ICA 3.2/3.4/3.5 는 각각 OA·Pcom·AChA 기시부로 정의되는 클래스라 이게 직접 원인이 된다.

출력: bpfalse.json  (병변마다 n_ok(0~5) 와 함께)
"""
import json, os, collections
import numpy as np
import nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
E = f"{R}/experiments"
P = f"{E}/_c1_realpred"
BP = f"{E}/_c4_bpgraph"
D = f"{E}/D1_newdata"

NEAR_MM = 15.0      # c5 의 BP_MAX_R
MATCH_MM = 5.0      # 예측 노드와 GT 노드를 같은 것으로 볼 거리
TAG = "b1on_pf"     # 제출본 구성

S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
SPLITS = S["splits"]
NAME = {int(k): v for k, v in S["location_classes"].items()}
VL = json.load(open(f"{R}/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json"))["labels"]
VID = {k: int(v) for k, v in VL.items()}
SMALL = {"R-Pcom": VID["R-Pcom"], "L-Pcom": VID["L-Pcom"],
         "R-AChA": VID["R-AChA"], "L-AChA": VID["L-AChA"],
         "R-OA": VID["R-OA"], "L-OA": VID["L-OA"]}

iw = {(x["split"], x["case"], x["cls"], round(x["dia"], 4)): x
      for x in json.load(open(f"{D}/intweak_{TAG}.json"))["lesions"]}

def nodes_of(path):
    if not os.path.exists(path):
        return []
    d = json.load(open(path))
    out = []
    for n in d.get("nodes", []):
        if not n.get("valid", True):
            continue
        out.append((frozenset(n["class_ids"]), np.array(n["centroid_mm"], float)))
    return out

recs = []
for sp in ("test", "val"):
    pbp = f"{BP}/vespp_test" if sp == "test" else f"{BP}/val_pred"
    for case in SPLITS[sp]:
        lp = f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"
        gvp = f"{R}/dataset/TopAneu/vessel_masks/{case}.nii.gz"
        pvp = f"{P}/vespp_{sp}/{case}.nii.gz"
        if not all(os.path.exists(x) for x in (lp, gvp, pvp)):
            continue
        pn = nodes_of(f"{pbp}/{case}.json")
        gn = nodes_of(f"{BP}/all_ref/{case}.json")
        if not pn and not gn:
            continue
        li = nib.load(lp)
        loc = np.asanyarray(li.dataobj)
        zoom = np.array(li.header.get_zooms()[:3], float)
        vx = float(np.prod(zoom))
        gv = np.asanyarray(nib.load(gvp).dataobj)
        pv = np.asanyarray(nib.load(pvp).dataobj)
        if gv.shape != loc.shape or pv.shape != loc.shape:
            continue

        # 예측 노드마다 '거짓인가' 판정 (같은 클래스쌍의 GT 노드가 MATCH_MM 안에 있나)
        pfalse = []
        for cs, c in pn:
            ok = any(cs == cs2 and float(np.linalg.norm(c - c2)) <= MATCH_MM for cs2, c2 in gn)
            pfalse.append(not ok)
        gmiss = []
        for cs, c in gn:
            ok = any(cs == cs2 and float(np.linalg.norm(c - c2)) <= MATCH_MM for cs2, c2 in pn)
            gmiss.append(not ok)

        for cl in sorted({int(x) for x in np.unique(loc) if x}):
            lab, n = ndimage.label(loc == cl)
            for i in range(1, n + 1):
                m = lab == i
                if m.sum() < 3:
                    continue
                dia = float(2 * (3 * (m.sum() * vx) / (4 * np.pi)) ** (1 / 3))
                key = (sp, case, cl, round(dia, 4))
                rec = iw.get(key)
                if rec is None or not rec["detected"]:
                    continue
                ctr_vox = np.argwhere(m).mean(0)
                ctr = ctr_vox * zoom                      # C4 의 centroid_mm 과 같은 규약
                dp = [float(np.linalg.norm(ctr - c)) for _, c in pn]
                dg = [float(np.linalg.norm(ctr - c)) for _, c in gn]
                near = [j for j, d in enumerate(dp) if d <= NEAR_MM]
                nearg = [j for j, d in enumerate(dg) if d <= NEAR_MM]
                nearest = int(np.argmin(dp)) if dp else None

                # 병변 주변(팽창 3회 + 5mm 반경)의 작은 분지 유무
                d3 = ndimage.binary_dilation(m, iterations=3)
                rad = np.ceil(5.0 / zoom).astype(int)
                sl = tuple(slice(max(0, int(a.min()) - r), min(s, int(a.max()) + r + 1))
                           for a, r, s in zip(np.argwhere(d3).T, rad, loc.shape))
                gvs, pvs = gv[sl], pv[sl]
                spurious = [nm for nm, vid in SMALL.items()
                            if (pvs == vid).sum() >= 10 and (gvs == vid).sum() < 10]
                missing = [nm for nm, vid in SMALL.items()
                           if (gvs == vid).sum() >= 10 and (pvs == vid).sum() < 10]

                recs.append(dict(
                    split=sp, case=case, cls=cl, name=NAME[cl], dia=dia, n_ok=rec["n_ok"],
                    n_bp_near=len(near),
                    n_false_near=sum(1 for j in near if pfalse[j]),
                    n_miss_near=sum(1 for j in nearg if gmiss[j]),
                    nearest_mm=round(min(dp), 2) if dp else None,
                    nearest_false=int(pfalse[nearest]) if nearest is not None else None,
                    spurious_branch=spurious, missing_branch=missing))

json.dump(recs, open(f"{D}/bpfalse.json", "w"), ensure_ascii=False, indent=1)
print(f"저장 {len(recs)}병변 → {D}/bpfalse.json")
