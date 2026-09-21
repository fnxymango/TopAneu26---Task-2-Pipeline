#!/usr/bin/env python
"""혈관 in-sample / out-of-sample 격차 측정 (2026-08-25).

문제: V4-2 fold0 은 train = 정확히 그 292, val = 42, test 83 은 완전 미노출이다.
      그래서 분류기 **학습 피처의 혈관(vespp_train 292)은 in-sample** 이고
      **평가 피처의 혈관(vespp_test 83)은 out-of-sample** 이다.
      학습 때는 깨끗한 혈관 맥락을 보고 배우고 실전에서는 열화된 걸 받는다.

혈관 재학습(4폴드 × 34.6시간 ≈ 2.9일)은 태우지 않기로 했으므로,
**격차의 크기만 재서** 이 축을 닫을지 나중에 다시 볼지 판단한다.

작으면 닫는다. 크면 시간이 생겼을 때 우선순위를 다시 매길 근거가 된다.
곁가지(Pcom/AChA 등)를 따로 보는 이유는 거기가 라벨오류 58.5% 의 원인이기 때문이다.
"""
import json, os, sys, collections
import numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d9xx_lib as L

P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.DATA / "vessel_masks"
NAMES = {v: k for k, v in
         json.load(open(L.TOPANEU_ROOT / "dataset/TopAneu/vessel_mapping.json"))["labels"].items()}
SMALL = {8, 9, 10, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34}   # Pcom/Acom/SCA/AICA/PICA/AChA/OA
train_ids, val_ids, test_ids = L.case_ids_by_split()


def dice_stats(vdir, ids, tag):
    inter = collections.Counter(); psum = collections.Counter(); gsum = collections.Counter()
    gpres = collections.Counter(); ppres = collections.Counter(); n = 0
    for i, cid in enumerate(ids, 1):
        pf = P / vdir / f"{cid}.nii.gz"; gf = GT / f"{cid}.nii.gz"
        if not pf.exists() or not gf.exists():
            continue
        a = np.asanyarray(nib.load(str(pf)).dataobj)
        b = np.asanyarray(nib.load(str(gf)).dataobj)
        n += 1
        for c in range(1, 37):
            pm = a == c; gm = b == c
            ps = int(pm.sum()); gs = int(gm.sum())
            if ps: ppres[c] += 1
            if gs: gpres[c] += 1
            if ps or gs:
                psum[c] += ps; gsum[c] += gs; inter[c] += int((pm & gm).sum())
        if i % 60 == 0:
            print(f"  {tag} {i}/{len(ids)}", flush=True)
    per = {c: 2 * inter[c] / max(psum[c] + gsum[c], 1) for c in range(1, 37) if gpres[c]}
    return {"n": n, "per": per, "gpres": gpres, "ppres": ppres}


res = {}
for tag, vdir, ids, mode in (("train292", "vespp_train", train_ids, "in-sample"),
                             ("val42", "vespp_val", val_ids, "out-of-sample"),
                             ("test83", "vespp_test", test_ids, "out-of-sample")):
    res[tag] = dice_stats(vdir, ids, tag)
    res[tag]["mode"] = mode
    print(f"  {tag} 완료 {res[tag]['n']}케이스", flush=True)

print("\n[V4-2 fold0 혈관 품질 · 분할별]")
print(f"  {'분할':<10}{'모드':<15}{'n':>5}{'전체Dice':>10}{'큰혈관':>9}{'곁가지':>9}")
for tag in ("train292", "val42", "test83"):
    r = res[tag]; per = r["per"]
    allm = np.mean(list(per.values()))
    big = np.mean([v for c, v in per.items() if c not in SMALL])
    sml = np.mean([v for c, v in per.items() if c in SMALL])
    print(f"  {tag:<10}{r['mode']:<15}{r['n']:>5}{allm:>10.4f}{big:>9.4f}{sml:>9.4f}")

a = res["train292"]["per"]; b = res["test83"]["per"]
com = sorted(set(a) & set(b))
d_all = np.mean([a[c] - b[c] for c in com])
d_big = np.mean([a[c] - b[c] for c in com if c not in SMALL])
d_sml = np.mean([a[c] - b[c] for c in com if c in SMALL])
print(f"\n  in-sample 이득 (train292 - test83)   전체 {d_all:+.4f}  큰혈관 {d_big:+.4f}  곁가지 {d_sml:+.4f}")

print(f"\n[곁가지 과대등장 비율 (예측등장 / GT등장)]")
print(f"  {'라벨':<16}{'train292':>10}{'val42':>9}{'test83':>9}")
for c in sorted(SMALL):
    row = []
    for tag in ("train292", "val42", "test83"):
        g = res[tag]["gpres"][c]; p = res[tag]["ppres"][c]
        row.append(p / g if g else float("nan"))
    print(f"  {NAMES.get(c,str(c)):<16}{row[0]:>10.3f}{row[1]:>9.3f}{row[2]:>9.3f}")
json.dump({k: {"n": v["n"], "per": {str(c): x for c, x in v["per"].items()}, "mode": v["mode"]}
           for k, v in res.items()},
          open(L.TOPANEU_ROOT / "code/sblee/nnunet/analysis/ves_insample_gap.json", "w"))
print("\n저장: analysis/ves_insample_gap.json")
