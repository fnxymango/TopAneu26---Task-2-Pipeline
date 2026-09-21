#!/usr/bin/env python
"""③ top-k 상한 곡선 (2026-08-25).

분류기가 정답을 몇 등 안에 넣는지 본다. top-1 은 못 맞혀도 top-2/3 안에 있다면
**재순위(re-ranking) 여지가 있다**는 뜻이고, top-5 밖으로 새어나가면 피처에 정보가
없다는 뜻이라 재순위로는 못 고친다 — ① 곁가지 신뢰도 피처의 기대치를 미리 가늠한다.

T16 설정(β0.5 τ0.5, 중첩만 예측혈관 학습)을 그대로 쓴다. 예측 blob 을 소스로 쓰므로
GT 와 겹치는 병변(=truth 가 있는 것)만 채점한다. 환각/미검출은 순위가 정의되지 않는다.
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
BETA = 0.5

ranks_all = []
for sd in [int(x) for x in (sys.argv[1:] or ["0", "1", "2"])]:
    os.environ["CLF_SEED"] = str(sd)
    model = C5.fit_model(train_rows, ves_axis, kind="rf", mirror=True, balance=True)
    clf = model["clf"]; cls_ = clf.classes_
    ranks = []
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
                continue                                   # 환각 — 순위 정의 안 됨
            truth = gcls.get(int(hits[1:].argmax()) + 1)
            if truth is None:
                continue
            v = C5.row_to_vec(r, ves_axis, mirror=False)
            if np.linalg.norm(v) == 0:
                continue
            p = clf.predict_proba(v.reshape(1, -1))[0]
            b = BETA if float(p.max()) < C5.CONF_TAU else C5.CONF_BETA_HI
            if b > 0:
                p = p / (model["pri"] ** b)
            order = np.argsort(-p)
            where = np.where(cls_[order] == truth)[0]
            ranks.append(int(where[0]) + 1 if len(where) else 999)
        if i % 30 == 0:
            print(f"  시드{sd} {i}/{len(test_ids)}", flush=True)
    ranks_all.append(ranks)
    print(f"  시드{sd} 채점 {len(ranks)}병변", flush=True)

print(f"\n[top-k 상한 곡선 · test 83 · 시드 {len(ranks_all)}판 평균]")
n = np.mean([len(r) for r in ranks_all])
print(f"  채점 대상 병변 {n:.1f}개 (GT 와 겹친 예측 blob 만)")
print(f"  {'k':>4}{'누적정확도':>12}{'추가회수':>10}{'오라클 Δ 상한':>15}")
prev = 0.0
for k in (1, 2, 3, 5, 10, 52):
    acc = np.mean([np.mean([x <= k for x in r]) for r in ranks_all])
    add = (acc - prev) * n
    print(f"  {k:>4}{acc:>12.3f}{add:>10.1f}{add*0.0178:>+15.4f}")
    prev = acc
miss = np.mean([np.mean([x > 52 for x in r]) for r in ranks_all])
print(f"\n  52등 밖(사실상 후보에 없음) {miss*100:.1f}%")
