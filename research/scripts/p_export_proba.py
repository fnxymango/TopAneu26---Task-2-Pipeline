#!/usr/bin/env python
"""병변별 확률행렬 내보내기 (2026-08-26). Q1(전역배정)·Q2(프로토타입) 공용 재료.

T16 설정 그대로 RF 를 태우되 argmax 가 아니라 **β/τ 보정 후 확률 벡터**를 저장한다.
  train : 292 OOF (Dataset720 5폴드, 자기를 안 본 폴드로만 예측 — 누수 없음)
  test  : 268 전부로 학습 후 예측

출력 analysis/proba_{split}_s{seed}.npz
  P      (n, C) 보정 후 확률   classes (C,) 클래스명
  truth  (n,)  GT 라벨 (환각이면 "")   case (n,)  lesion_idx (n,)
"""
import json, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L, c5_location_v2 as C5

ST = np.ones((3, 3, 3), bool)
A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
BP = L.TOPANEU_ROOT / "experiments" / "_c4_bpgraph"
SP720 = L.TOPANEU_ROOT / "nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
BETA = 0.5
id2name, _ = L.official_location_names()
ves_names = L.vessel_dense_names()
ves_axis, _ = C5.build_feature_axes(ves_names)
train_rows = json.load(open(A / "e11_feat_hyb_ov.json"))
train_ids, _, test_ids = L.case_ids_by_split()


def proba(model, r):
    v = C5.row_to_vec(r, model["ves_axis"], mirror=False)
    if np.linalg.norm(v) == 0: return None
    p = model["clf"].predict_proba(v.reshape(1, -1))[0]
    b = BETA if float(p.max()) < C5.CONF_TAU else C5.CONF_BETA_HI
    if b > 0: p = p / (model["pri"] ** b)
    return p / max(p.sum(), 1e-12)


def collect(ids, vdir, bdir, adir, model):
    """예측 blob 마다 확률과 GT 라벨."""
    out = []
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
            pv = proba(model, r)
            if pv is None: continue
            out.append((cid, r["lesion_mask_idx"], truth, pv))
        if i % 40 == 0: print(f"    {i}/{len(ids)}", flush=True)
    return out


def save(tag, sd, model, data):
    cls = list(model["clf"].classes_)
    np.savez_compressed(A / f"proba_{tag}_s{sd}.npz",
                        P=np.array([d[3] for d in data], dtype=np.float32),
                        classes=np.array(cls, dtype=object),
                        truth=np.array([d[2] for d in data], dtype=object),
                        case=np.array([d[0] for d in data], dtype=object),
                        lesion=np.array([d[1] for d in data], dtype=np.int32))
    print(f"  [저장] proba_{tag}_s{sd}  n={len(data)} · C={len(cls)}", flush=True)


for sd in [int(x) for x in (sys.argv[1:] or ["0", "1", "2", "3", "4"])]:
    os.environ["CLF_SEED"] = str(sd)
    # ---- test : 268 전부로 학습 ----
    if not (A / f"proba_test_s{sd}.npz").exists():
        m = C5.fit_model(train_rows, ves_axis, kind="rf", mirror=True, balance=True)
        print(f"[시드{sd}] test 수집", flush=True)
        save("test", sd, m, collect(test_ids, "vespp_test", "vespp_test",
                                    P / "aneu_test_probavgf", m))
    # ---- train : 292 OOF ----
    if not (A / f"proba_trainoof_s{sd}.npz").exists():
        folds = json.load(open(SP720))
        allc = collections.OrderedDict()
        m_any = None
        for k, f in enumerate(folds):
            va = set(f["val"])
            if not va & set(train_ids): continue
            fit_rows = [r for r in train_rows if r["case"] not in va]
            mk = C5.fit_model(fit_rows, ves_axis, kind="rf", mirror=True, balance=True)
            m_any = mk
            ids = [c for c in train_ids if c in va]
            print(f"[시드{sd}] OOF fold{k} · 학습 {len(fit_rows)} · 예측 {len(ids)}케이스", flush=True)
            for rec in collect(ids, "vespp_train", "vespp_train", P / "aneu_train_ooff", mk):
                allc[(rec[0], rec[1])] = rec
        if m_any is not None:
            save("trainoof", sd, m_any, list(allc.values()))
print("완료")
