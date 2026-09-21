#!/usr/bin/env python
"""피처 벡터 내보내기 (2026-08-26) — Q2(프로토타입)·Q3 공용.

row_to_vec 출력은 **시드와 무관**하다(시드는 RF 학습에만 들어간다). 그래서 추출은 한 번만
하면 되고, 시드별 모델은 이 행렬 위에서 초 단위로 다시 적합할 수 있다.
Q1a 가 시드마다 재추출하는 건 낭비였다 — 여기서는 한 번만 돈다.

출력 analysis/feat_{split}.npz
  X (n,D) · truth (n,) · case (n,) · lesion (n,)
학습쪽은 feat_train.npz (GT 병변 268, 미러 전)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L, c5_location_v2 as C5

ST = np.ones((3, 3, 3), bool)
A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
BP = L.TOPANEU_ROOT / "experiments" / "_c4_bpgraph"
C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
id2name, _ = L.official_location_names()
ves_names = L.vessel_dense_names()
ves_axis, _ = C5.build_feature_axes(ves_names)
train_rows = json.load(open(A / "e11_feat_hyb_ov.json"))
train_ids, _, test_ids = L.case_ids_by_split()

# 학습 피처 (GT 병변) — 미러 포함 확장은 소비쪽에서 한다
Xt = np.array([C5.row_to_vec(r, ves_axis, mirror=False) for r in train_rows])
Xm = np.array([C5.row_to_vec(r, ves_axis, mirror=True) for r in train_rows])
np.savez_compressed(A / "feat_train.npz", X=Xt, Xm=Xm,
                    y=np.array([r["gt_loc"] for r in train_rows], dtype=object),
                    ym=np.array([C5.mirror_name(r["gt_loc"]) for r in train_rows], dtype=object),
                    case=np.array([r["case"] for r in train_rows], dtype=object))
print(f"[저장] feat_train  X{Xt.shape}", flush=True)


def dump(tag, ids, vdir, bdir, adir):
    if (A / f"feat_{tag}.npz").exists():
        print(f"  feat_{tag} 이미 있음"); return
    X, tr, cs, ls = [], [], [], []
    for i, cid in enumerate(ids, 1):
        vp = P / vdir / f"{cid}.nii.gz"; ap = adir / f"{cid}.nii.gz"
        gp = L.DATA / "location_masks" / f"{cid}.nii.gz"
        if not vp.exists() or not ap.exists() or not gp.exists(): continue
        vi = nib.load(vp); ves = np.asanyarray(vi.dataobj)
        sp = np.array(vi.header.get_zooms()[:3], dtype=float)
        nodes = C5.load_bp(str(BP / bdir), cid)
        gt = np.asanyarray(nib.load(gp).dataobj)
        pred = np.asanyarray(nib.load(ap).dataobj)
        glab, gn = ndi.label(gt > 0, structure=ST)
        gcls = {j: id2name.get(int(np.bincount(gt[glab == j].ravel()).argmax()), None)
                for j in range(1, gn + 1)}
        rows, lesions = C5.extract_case_rows(pred, ves, sp, ves_names, nodes, None)
        for r in rows:
            m = lesions == r["lesion_mask_idx"]
            hits = np.bincount(glab[m].ravel())
            truth = ""
            if len(hits) >= 2 and hits[1:].max() > 0:
                truth = gcls.get(int(hits[1:].argmax()) + 1) or ""
            X.append(C5.row_to_vec(r, ves_axis, mirror=False))
            tr.append(truth); cs.append(cid); ls.append(r["lesion_mask_idx"])
        if i % 40 == 0: print(f"    {tag} {i}/{len(ids)}", flush=True)
    np.savez_compressed(A / f"feat_{tag}.npz", X=np.array(X),
                        truth=np.array(tr, dtype=object), case=np.array(cs, dtype=object),
                        lesion=np.array(ls, dtype=np.int32))
    print(f"[저장] feat_{tag}  n={len(X)}", flush=True)


dump("test", test_ids, "vespp_test", "vespp_test", P / "aneu_test_probavgf")
_, val_ids, _ = L.case_ids_by_split()
dump("val", val_ids, "vespp_val", "val_pred", P / "aneu_val_probavgf")
dump("trainoof", train_ids, "vespp_train", "vespp_train", P / "aneu_train_ooff")
print("완료")
