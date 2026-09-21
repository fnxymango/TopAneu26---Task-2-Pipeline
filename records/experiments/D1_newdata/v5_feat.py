#!/usr/bin/env python3
"""V5 — 한 번도 안 써본 병변 피처 세 묶음을 train 학습표 271행에 대해 뽑는다 (스크리닝 재료).

기존 112차원(혈관 근접 36 · 중첩 36 · 분기점 34 · 좌표 6)에 없는 정보만 고른다.
  F1 돌출 방향 (5)  목(병변∩혈관 인접) → 돔(병변 중심) 단위벡터를 C10 랜드마크 좌표계(x=R→L, y=→BA tip)로
                    표현한 3성분 + 모혈관 국소 축과의 |cos| + 유효 플래그.
                    근거: ICA 원위 분지 동맥류는 돌출 방향이 정형적이다(Pcom 후외측·하방, AChA 상외측, OA 상방).
                    목 '위치'(V1-D/E)는 닫혔지만 '방향'은 쓴 적이 없다. 방향은 무차원이라 거리 부풀림(V1-B)에 둔감하다.
  F2 굵기 (3)       목 근처 모혈관 반경(mm) · 케이스 큰 혈관(BA·ICA-C6-C7) 반경 대비 비 · 목 근처 다른 혈관의 최대 반경 / 모혈관 반경.
                    근거: M1↔M2 · A1↔A2 · VA↔PICA · 태아형 Pcom 은 굵기가 다르다. 현 피처에 굵기 정보가 0.
  F3 뼈 (3)         CTA 만: 병변 최단 뼈 거리(mm) · 반경 8mm 구 안 뼈 비율 · CT 플래그. MR 은 −1,−1,0.
                    근거: 3.1 infraclinoid ↔ 3.3 C6 경계는 혈관이 아니라 전상돌기·경막륜(뼈)이 정한다.
미러: F1 의 x 성분만 부호 반전. F2·F3 는 좌우 무관.

학습표와 같은 방식(PROJECT_RULES.md 0-2): 위치 계열이므로 GT 병변 + GT 혈관 + GT 분기점 그래프(all_ref). 뼈는 원본 영상.
출력: v5_feat_train.json  {"case|lesion_idx": {"f1": [...], "f2": [...]}}
F3 는 train 원본 CT 가 09-10 정리로 지워져 v5_bone.py 가 HDD 전처리본(Dataset720 3d_fullres)에서 따로 뽑는다.
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"
A = f"{R}/code/sblee/nnunet/analysis"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
os.environ.setdefault("TOPANEU_ROOT", R)
ST = np.ones((3, 3, 3), bool)
BONE_HU, BONE_MIN_VOX = 700.0, 500



def vessel_ids():
    import d9xx_lib as L
    names = L.vessel_dense_names()          # {id: name}
    return {n: i for i, n in names.items()}


def frame(nodes):
    import c10_landmark_coords as C10
    lm = C10.find_landmarks(nodes)
    f = C10.frame_from_landmarks(lm)
    return None if f is None else f[1]      # 3x3 축행렬 (행 = x,y,z)


def crop(center_vox, half_vox, shape):
    lo = np.maximum(center_vox - half_vox, 0); hi = np.minimum(center_vox + half_vox + 1, shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), lo


def case_feats(cid, rows):
    import c5_location_v2 as C5
    nm2id = vessel_ids()
    gi = nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz")
    loc = np.asanyarray(gi.dataobj); sp = np.array(gi.header.get_zooms()[:3], float)
    ves = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz").dataobj)
    assert ves.shape == loc.shape, cid
    is_ct = "_ct_" in cid
    lab, n = ndimage.label(loc > 0, structure=ST)
    Fm = frame(C5.load_bp(f"{R}/experiments/_c4_bpgraph/all_ref", cid))
    big = np.isin(ves, [nm2id[k] for k in ("BA", "R-ICA-C6-C7", "L-ICA-C6-C7") if k in nm2id])
    out = {}
    half = np.ceil(20.0 / sp).astype(int)
    for r in rows:
        m_full = lab == r["lesion_mask_idx"]
        idx = np.argwhere(m_full)
        cen_vox = idx.mean(0)
        err = float(np.linalg.norm(cen_vox * sp - np.array(r["_cen"])))
        assert err < 0.05, (cid, r["lesion_mask_idx"], err)
        sl, lo = crop(np.round(cen_vox).astype(int), half + (idx.max(0) - idx.min(0)) // 2, np.array(loc.shape))
        m = m_full[sl]; v = np.where(m, 0, ves[sl]); vb = v > 0     # GT 혈관 라벨이 낭까지 덮는다(예: 6355복셀 중 4036) → 병변 밖 혈관만
        cen = cen_vox * sp
        # ── 목 · 모혈관 ─────────────────────────────────────────────
        dv, ind = ndimage.distance_transform_edt(~vb, sampling=sp, return_indices=True)
        dl = dv[m]
        neck = m & (dv <= max(float(dl.min()), 0.0) + 0.6)
        nidx = np.argwhere(neck)
        cn = (nidx.mean(0) + lo) * sp
        # 목 복셀의 최근접 혈관 복셀 라벨 최빈 = 모혈관
        near_lab = v[tuple(ind[:, neck])]
        near_lab = near_lab[near_lab > 0]
        parent = int(np.bincount(near_lab).argmax()) if near_lab.size else 0
        # 돔 = 혈관에서 가장 먼 병변 끝(최대거리의 80% 이상 복셀 중심). 중심-중심은 분기부 병변(사방이 혈관)에서
        # 목과 돔이 겹쳐 방향이 사라졌다(초판 무효 79/271 중 5.3·4.1·5.2 가 41).
        dmx = float(dl.max())
        tip = (np.argwhere(m & (dv >= 0.8 * dmx)).mean(0) + lo) * sp if dmx >= 0.5 else cen
        u = tip - cn; nu = float(np.linalg.norm(u))
        f1 = [0.0, 0.0, 0.0, 0.0, 0.0]
        if nu >= 0.3 and parent and Fm is not None:
            u = u / nu
            ux, uy, uz = (Fm @ u).tolist()
            pvox = np.argwhere(v == parent) + lo
            pc = pvox * sp
            sel = pc[np.linalg.norm(pc - cn, axis=1) <= 6.0]
            cosa = 0.0
            if len(sel) >= 20:
                w, V = np.linalg.eigh(np.cov((sel - sel.mean(0)).T))
                cosa = abs(float(np.dot(V[:, -1], u)))
            f1 = [ux, uy, uz, cosa, 1.0]
        # ── 굵기 ──────────────────────────────────────────────────
        din = ndimage.distance_transform_edt(vb, sampling=sp)
        vc = np.argwhere(vb) + lo
        dist_n = np.linalg.norm(vc * sp - cn, axis=1)
        vlab = v[vb]; vr = din[vb]
        f2 = [-1.0, -1.0, -1.0]
        selp = (vlab == parent) & (dist_n <= 3.0)
        if parent and selp.any():
            rp = float(vr[selp].max())
            f2[0] = rp
            f2[2] = float(vr[(vlab != parent) & (dist_n <= 4.0)].max() / rp) if ((vlab != parent) & (dist_n <= 4.0)).any() else 0.0
        out[f"{cid}|{r['lesion_mask_idx']}"] = dict(f1=f1, f2=f2, parent=parent, cn=cn.tolist(), neck_vox=int(neck.sum()))
    # 케이스 큰 혈관 반경 (전체 볼륨 EDT 는 무거우므로 큰 혈관 bbox 에서만)
    if big.any():
        bi = np.argwhere(big); bl = np.maximum(bi.min(0) - 3, 0); bh = np.minimum(bi.max(0) + 4, np.array(ves.shape))
        bs = tuple(slice(a, b) for a, b in zip(bl, bh))
        rref = float(np.percentile(ndimage.distance_transform_edt(ves[bs] > 0, sampling=sp)[big[bs]], 95))
    else:
        rref = -1.0
    for k, o in out.items():
        o["f2"][1] = o["f2"][0] / rref if (o["f2"][0] > 0 and rref > 0) else -1.0
    return out


def main():
    rows = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["case"]].append(r)
    import multiprocessing as mp
    res = {}
    with mp.Pool(int(os.environ.get("V5_WORKERS", "10"))) as pool:
        for i, o in enumerate(pool.starmap(case_feats, list(by.items()), chunksize=1), 1):
            res.update(o)
    assert len(res) == len(rows), (len(res), len(rows))
    json.dump(res, open(f"{D}/v5_feat_train.json", "w"))
    f1v = sum(o["f1"][4] for o in res.values())
    print(f"완료 {len(res)}행 · F1 유효 {int(f1v)} · F2 모혈관 반경 유효 {sum(o['f2'][0] > 0 for o in res.values())}")


if __name__ == "__main__":
    main()
