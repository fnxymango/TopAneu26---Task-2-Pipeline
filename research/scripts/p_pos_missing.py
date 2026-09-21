#!/usr/bin/env python
"""pos(랜드마크 좌표) 결측 병변의 정확도 진단 (2026-08-26).

landmark_frame 은 분기점 그래프에서 BAtip·Rterm·Lterm 세 랜드마크를 다 찾아야 성립한다.
학습 268병변 중 9개(3.4%)에서 실패하고, 실패 케이스 6개 중 5개가 center5_mr 이다.
실패하면 pos 6차원이 전부 0 으로 들어가는데, 이는 '모름'과 '원점'을 구분하지 못하는
인코딩이고 학습 샘플에 9개뿐이라 RF 가 그 의미를 배울 수도 없다.

여기서는 **실제로 못 맞히고 있는지** 만 잰다. 잘 맞히고 있으면 고칠 게 없다.
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
C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
id2name, _ = L.official_location_names()
ves_names = L.vessel_dense_names()
ves_axis, _ = C5.build_feature_axes(ves_names)
train_rows = json.load(open(A / "e11_feat_hyb_ov.json"))
_, _, test_ids = L.case_ids_by_split()

agg = collections.defaultdict(lambda: [0, 0])   # haspos -> [맞힘, 전체]
per_case = collections.Counter()
for sd in (0, 1, 2):
    os.environ["CLF_SEED"] = str(sd)
    model = C5.fit_model(train_rows, ves_axis, kind="rf", mirror=True, balance=True)
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
        for r in rows:
            m = lesions == r["lesion_mask_idx"]
            hits = np.bincount(glab[m].ravel())
            if len(hits) < 2 or hits[1:].max() == 0:
                continue
            truth = gcls.get(int(hits[1:].argmax()) + 1)
            if truth is None:
                continue
            hp = bool(r.get("pos"))
            p = C5.predict_one(model, r, 0.5)
            agg[hp][1] += 1
            if p == truth: agg[hp][0] += 1
            if sd == 0 and not hp: per_case[cid] += 1
        if i % 30 == 0: print(f"  시드{sd} {i}/{len(test_ids)}", flush=True)

print(f"\n[test 83 · 3시드 · GT 와 겹친 예측 병변만]")
print(f"  {'pos':<10}{'병변(시드합)':>12}{'맞힘':>7}{'정확도':>9}")
for k in (True, False):
    ok, n = agg[k]
    if not n: continue
    print(f"  {'있음' if k else '없음':<10}{n:>12}{ok:>7}{ok/n:>9.3f}")
tot = sum(v[1] for v in agg.values())
print(f"\n  pos 결측 비율 {agg[False][1]/tot*100:.1f}%  (시드당 {agg[False][1]/3:.1f}병변 / {tot/3:.1f})")
if per_case:
    print(f"  결측 케이스: {dict(per_case)}")
