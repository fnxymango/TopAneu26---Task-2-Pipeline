#!/usr/bin/env python
"""e2e 병변단위 기록 (2026-08-25) — 오라클 분해용.

E16 은 **GT 마스크**를 병변 소스로 써서 "검출이 완벽했다면" 경로를 본다.
여기서는 실제 파이프라인과 같이 **예측 blob** 을 소스로 쓴다:
    예측 blob -> GT 와 겹치면 truth = 그 GT 클래스, 안 겹치면 truth = None (환각)
    GT 성분 중 어느 예측과도 안 겹친 것 -> pred = None (미검출)
T16 설정(중첩만 예측혈관 학습 · rf · β0.5 · τ0.5)을 그대로 쓴다.
"""
import collections, json, os, sys
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
_, _, test_ids = L.case_ids_by_split()

for sd in [int(x) for x in (sys.argv[1:] or ["0"])]:
    os.environ["CLF_SEED"] = str(sd)
    model = C5.fit_model(train_rows, ves_axis, kind="rf", mirror=True, balance=True)
    recs = []
    for i, cid in enumerate(test_ids, 1):
        vp = P / "vespp_test" / f"{cid}.nii.gz"; ap = P / "aneu_test_probavgf" / f"{cid}.nii.gz"
        if not vp.exists() or not ap.exists():
            continue
        vi = nib.load(vp); ves = np.asanyarray(vi.dataobj)
        sp = np.array(vi.header.get_zooms()[:3], dtype=float)
        nodes = C5.load_bp(str(BP / "vespp_test"), cid)
        gt = np.asanyarray(nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz").dataobj)
        pred = np.asanyarray(nib.load(ap).dataobj)
        glab, gn = ndi.label(gt > 0, structure=ST)
        gcls = {j: id2name.get(int(np.bincount(gt[glab == j].ravel()).argmax()), None)
                for j in range(1, gn + 1)}
        rows, lesions = C5.extract_case_rows(pred, ves, sp, ves_names, nodes, None)
        hit = set()
        for r in rows:
            m = lesions == r["lesion_mask_idx"]
            md = ndi.binary_dilation(m, structure=ST, iterations=2)
            ov = collections.Counter(glab[md & (glab > 0)].ravel().tolist())
            j = ov.most_common(1)[0][0] if ov else None
            if j: hit.add(j)
            recs.append({"case": cid, "truth": gcls.get(j) if j else None,
                         "pred": C5.predict_one(model, r, 0.5)})
        for j in range(1, gn + 1):
            if j not in hit:
                recs.append({"case": cid, "truth": gcls.get(j), "pred": None})
        if i % 30 == 0: print(f"  시드{sd} {i}/{len(test_ids)}", flush=True)
    p = A / f"p_lesion_e11_hyb_ov_s{sd}.json"
    json.dump(recs, open(p, "w"), ensure_ascii=False)
    print(f"[저장] 시드{sd} {len(recs)}건 -> {p}", flush=True)
