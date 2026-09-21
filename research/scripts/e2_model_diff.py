"""E2 — 두 분류기가 실제로 **몇 개의 병변에서 갈리는지** 세어본다 (2026-08-19).

동기: C36 에서 ExtraTrees 가 RF 를 test cov.MCC +0.0303 으로 이겼다. 그런데 test 는
병변 87 / 분모클래스 36 이라 희소 클래스 병변 1개가 cov.MCC 를 +0.0278 움직인다.
즉 +0.0303 은 **병변 2.9개분**이다. 이 크기의 차이는 지표만 봐서는
"모델이 더 낫다"와 "운 좋은 클래스 몇 개를 주웠다"를 구분할 수 없다.

그래서 지표 대신 원자료를 센다:
  - 두 모델이 서로 다른 라벨을 준 병변이 몇 개인가
  - 그중 ET 만 맞힌 것 / RF 만 맞힌 것 / 둘 다 틀린 것
  - 갈린 병변이 희소 클래스에 쏠려 있는가

승패 기록이 6-2 면 모델 차이고, 4-3 인데 MCC 만 벌어졌으면 클래스 운이다.

사용: python e2_model_diff.py --split test --models rf,et
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

import d9xx_lib as L
import c5_location_v2 as C5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--models", default="rf,et")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--conf-tau", type=float, default=0.5)
    a = ap.parse_args()

    A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
    P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
    BP = L.TOPANEU_ROOT / "experiments" / "_c4_bpgraph"
    C5.USE_POS = True
    C5.CONF_TAU = a.conf_tau; C5.CONF_BETA_HI = 0.0

    id2name, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()
    ves_axis, _ = C5.build_feature_axes(ves_names)
    train_rows = json.load(open(A / "c10_feat_train.json"))
    prior = collections.Counter(r["gt_loc"] for r in train_rows if r.get("gt_loc"))

    kinds = a.models.split(",")
    models = {k: C5.fit_model(train_rows, ves_axis, kind=k, mirror=True, balance=True)
              for k in kinds}
    print(f"[e2] 학습병변 {len(train_rows)} · 모델 {kinds} · β={a.beta} τ={a.conf_tau}", flush=True)

    tr_ids, val_ids, test_ids = L.case_ids_by_split()
    cids = val_ids if a.split == "val" else test_ids
    vdir = P / (f"vespp_{a.split}")
    bdir = BP / ("val_pred" if a.split == "val" else "vespp_test")
    adir = P / f"aneu_{a.split}_probavgf"

    recs = []
    for i, cid in enumerate(cids, 1):
        vp = vdir / f"{cid}.nii.gz"; app = adir / f"{cid}.nii.gz"
        if not vp.exists() or not app.exists():
            continue
        vi = nib.load(vp); ves = np.asanyarray(vi.dataobj)
        spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
        nodes = C5.load_bp(str(bdir), cid)
        gt = np.asanyarray(nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz").dataobj)
        src = np.asanyarray(nib.load(app).dataobj)
        rows, lesions = C5.extract_case_rows(src, ves, spacing, ves_names, nodes, None)
        for r in rows:
            m = lesions == r["lesion_mask_idx"]
            vals, cnts = np.unique(gt[m], return_counts=True)
            keep = [(v, c) for v, c in zip(vals, cnts) if v != 0]
            truth = id2name.get(int(max(keep, key=lambda x: x[1])[0])) if keep else None
            recs.append({"case": cid, "idx": r["lesion_mask_idx"], "truth": truth,
                         **{k: C5.predict_one(models[k], r, a.beta) for k in kinds}})
        if i % 20 == 0:
            print(f"  {i}/{len(cids)} 누적 병변 {len(recs)}", flush=True)

    k1, k2 = kinds[0], kinds[1]
    tp = [r for r in recs if r["truth"]]          # 검출이 덮은 GT 병변
    fp = [r for r in recs if not r["truth"]]      # 위양성
    diff = [r for r in tp if r[k1] != r[k2]]
    only1 = [r for r in diff if r[k1] == r["truth"]]
    only2 = [r for r in diff if r[k2] == r["truth"]]
    both_wrong = [r for r in diff if r[k1] != r["truth"] and r[k2] != r["truth"]]
    same_ok = [r for r in tp if r[k1] == r[k2] == r["truth"]]

    print(f"\n=== {a.split}: 예측병변 {len(recs)} (GT덮음 {len(tp)} · 위양성 {len(fp)}) ===")
    print(f"  둘 다 정답            {len(same_ok)}")
    print(f"  둘 다 오답(같은 라벨)  {len(tp)-len(same_ok)-len(diff)}")
    print(f"  **갈린 병변**          {len(diff)}")
    print(f"      {k1} 만 정답       {len(only1)}")
    print(f"      {k2} 만 정답       {len(only2)}")
    print(f"      둘 다 오답         {len(both_wrong)}")
    print(f"  위양성 중 라벨 갈림    {sum(1 for r in fp if r[k1]!=r[k2])}/{len(fp)}")

    print(f"\n  [승패] {k2} {len(only2)} : {len(only1)} {k1}"
          f"  (순증 {len(only2)-len(only1):+d}개)")
    if only1 or only2:
        print(f"\n  갈려서 정답이 바뀐 병변 (정답클래스 · train 표본수):")
        for r in sorted(only1+only2, key=lambda r: prior.get(r["truth"], 0)):
            w = k1 if r in only1 else k2
            print(f"    {r['truth']:<14} n={prior.get(r['truth'],0):<3} 승자={w:<4} "
                  f"{k1}={r[k1]} {k2}={r[k2]}  ({r['case']})")

    out = A / f"e2_model_diff_{a.split}_{k1}_{k2}.json"
    json.dump({"split": a.split, "models": kinds, "beta": a.beta, "conf_tau": a.conf_tau,
               "n_pred": len(recs), "n_covered": len(tp), "n_fp": len(fp),
               "n_diff": len(diff), f"only_{k1}": len(only1), f"only_{k2}": len(only2),
               "both_wrong": len(both_wrong), "records": recs},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
