"""E16 — 52클래스 중 어디가 약한가, 그 약한 것들의 공통 특징은 무엇인가 (2026-08-19).

지금까지의 진단은 "희소 클래스가 약하다" 수준에서 멈췄다. 그런데 오늘 측정으로
병변 1개가 test cov.MCC 를 0.028 움직인다는 게 확인됐으니, **어느 클래스를 고쳐야
얼마가 오르는지**를 클래스 단위로 계산할 수 있다.

세 층으로 나눠 본다 — 손실이 어디서 나는지 층을 갈라야 처방이 갈린다:
  (a) 검출이 못 찾음        -> 분류기가 손댈 수 없다 (검출 개선 소관)
  (b) 검출은 했는데 오분류  -> 분류기 소관
  (c) 위양성을 그 클래스로 잘못 뱉음 -> 정밀도 손실

그리고 약한 클래스들의 공통 속성을 찾는다:
  표본 수 / 좌우쌍 / 같은 혈관가족 안 형제 수 / 혼동 상대 / sac 크기 / 분기점 근접도
"""
import collections, json, sys
import numpy as np, nibabel as nib
from scipy import ndimage
sys.path.insert(0, "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/scripts")
import d9xx_lib as L, c5_location_v2 as C5

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
BP = L.TOPANEU_ROOT / "experiments" / "_c4_bpgraph"
C5.USE_POS = True; C5.CONF_TAU = 0.5; C5.CONF_BETA_HI = 0.0
id2name, name2id = L.official_location_names()
ves_names = L.vessel_dense_names()
ves_axis, _ = C5.build_feature_axes(ves_names)
train_rows = json.load(open(A / "c10_feat_train.json"))
prior = collections.Counter(r["gt_loc"] for r in train_rows if r.get("gt_loc"))
_, _, test_ids = L.case_ids_by_split()

recs = []          # (case, truth, pred, detected, nvox, bp_min)
for sd in range(3):
    import os; os.environ["CLF_SEED"] = str(sd)
    model = C5.fit_model(train_rows, ves_axis, kind="rf", mirror=True, balance=True)
    for i, cid in enumerate(test_ids, 1):
        vp = P / "vespp_test" / f"{cid}.nii.gz"; ap = P / "aneu_test_probavgf" / f"{cid}.nii.gz"
        if not vp.exists(): continue
        vi = nib.load(vp); ves = np.asanyarray(vi.dataobj)
        sp = np.array(vi.header.get_zooms()[:3], dtype=float)
        nodes = C5.load_bp(str(BP / "vespp_test"), cid)
        gt = np.asanyarray(nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz").dataobj)
        det = np.asanyarray(nib.load(ap).dataobj) > 0 if ap.exists() else np.zeros_like(gt, bool)
        rows, lesions = C5.extract_case_rows(gt, ves, sp, ves_names, nodes, id2name)
        lab, n = ndimage.label(gt > 0, structure=np.ones((3, 3, 3)))
        for r in rows:
            m = lesions == r["lesion_mask_idx"]
            bpm = [x for x in (r.get("bp_mm") or []) if x is not None]
            recs.append({"seed": sd, "case": cid, "truth": r["gt_loc"],
                         "pred": C5.predict_one(model, r, 0.5),
                         "det": bool((det & m).any()), "nvox": int(r.get("n_vox") or m.sum()),
                         "bp": float(min(bpm)) if bpm else None})
        if i % 30 == 0: print(f"  시드{sd} {i}/{len(test_ids)}", flush=True)
json.dump(recs, open(A / "e16_weak_classes.json", "w"), ensure_ascii=False)

# ── 클래스별 3층 분해 ─────────────────────────────────────────────
by = collections.defaultdict(list)
for r in recs: by[r["truth"]].append(r)
S = 3
print(f"\n{'클래스':<32}{'GT':>4}{'train':>6}{'검출':>7}{'분류':>7}{'최종':>7}{'손실원':>9}")
rows_out = []
for c, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    ngt = len(rs) / S
    ndet = sum(r["det"] for r in rs) / S
    nok = sum(r["det"] and r["pred"] == c for r in rs) / S
    ncls = sum(r["pred"] == c for r in rs) / S          # 검출 무관 분류 정확도
    det_r = ndet / ngt; cls_r = (nok / ndet) if ndet else 0.0
    lose = "검출" if det_r < 0.7 and det_r <= cls_r else ("분류" if cls_r < 0.7 else "-")
    rows_out.append((c, ngt, prior.get(c, 0), det_r, cls_r, nok / ngt, lose))
    print(f"{c:<32}{ngt:>4.0f}{prior.get(c,0):>6}{det_r:>7.2f}{cls_r:>7.2f}{nok/ngt:>7.2f}{lose:>9}")

# ── 약한 클래스의 공통 속성 ───────────────────────────────────────
weak = [r for r in rows_out if r[5] < 0.5]
strong = [r for r in rows_out if r[5] >= 0.5]
def summ(g, lab):
    if not g: return
    tr = np.array([x[2] for x in g], float)
    print(f"  {lab:<10} 클래스 {len(g):>2}개 · GT병변 {sum(x[1] for x in g):>4.0f} · "
          f"train표본 중앙값 {np.median(tr):>4.1f} · 좌우쌍 {sum(1 for x in g if x[0][:2] in ('R-','L-'))}/{len(g)}")
print(f"\n[최종 정확도 0.5 미만 = 약한 클래스]")
summ(weak, "약함"); summ(strong, "강함")
print(f"\n  약한 클래스가 test 에서 차지하는 GT 병변 {sum(x[1] for x in weak):.0f} / {sum(x[1] for x in rows_out):.0f}")
print(f"  전부 맞히면 cov.MCC 상승 여지 ≈ {len(weak)/36:.3f} (클래스당 1/36)")

# ── 혼동 상대: 같은 가족인가 ─────────────────────────────────────
fam = lambda n: n.split()[0].replace("R-", "").replace("L-", "")
mis = [r for r in recs if r["det"] and r["pred"] != r["truth"]]
same_fam = sum(1 for r in mis if fam(r["pred"]) == fam(r["truth"]))
mirror = sum(1 for r in mis if r["pred"] == C5.mirror_name(r["truth"]))
print(f"\n[오분류 {len(mis)/S:.0f}건의 상대]")
print(f"  같은 혈관가족 안 형제  {same_fam/S:>5.0f}  ({same_fam/max(len(mis),1):.1%})")
print(f"  좌우 반대편 같은 위치  {mirror/S:>5.0f}  ({mirror/max(len(mis),1):.1%})")
cc = collections.Counter((r["truth"], r["pred"]) for r in mis)
print(f"\n  가장 흔한 혼동 10쌍:")
for (t, p), n in cc.most_common(10):
    print(f"    {t:<30} -> {p:<30} {n/S:>4.1f}회")
