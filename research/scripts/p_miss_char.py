#!/usr/bin/env python
"""미검출 병변 특성화 (2026-08-26).

오라클 분해에서 미검출 23개 = +0.2117 로 남은 가장 큰 웅덩이인데 한 번도 들여다본 적이 없다.
X5(전역 5폴드 교체)는 적중 +4 · 환각 +3 으로 상쇄돼 실패했다. 전역으로 문턱을 낮추는 대신
**놓친 병변이 (a) 애초에 신호가 없는지 (b) 문턱/후처리에 잘렸는지** 를 먼저 가른다.
(b) 라면 국소적으로만 되살릴 여지가 있다.

디스크에 이미 있는 검출 변형본을 강도 순으로 훑는다:
  probavgf (현행·후처리 후) < probavg (후처리 전) < vote4/3/2 < unionu80/u40/u20
"""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import d9xx_lib as L

P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
GT = L.TOPANEU_ROOT / "dataset" / "TopAneu" / "location_masks"
id2name, _ = L.official_location_names()
_, _, test_ids = L.case_ids_by_split()
LADDER = ["aneu_test_probavgf", "aneu_test_probavg", "aneu_test_vote4",
          "aneu_test_vote3", "aneu_test_vote2", "aneu_test_unionu80",
          "aneu_test_unionu40", "aneu_test_unionu20"]
SHORT = {d: d.replace("aneu_test_", "") for d in LADDER}

rows = []
for n, cid in enumerate(test_ids, 1):
    g = nib.load(str(GT / f"{cid}.nii.gz"))
    ga = np.asarray(g.dataobj)
    if not ga.any():
        continue
    sp = g.header.get_zooms()[:3]
    vox_mm3 = float(np.prod(sp))
    lab, k = ndi.label(ga > 0)
    preds = {}
    for d in LADDER:
        f = P / d / f"{cid}.nii.gz"
        preds[d] = (np.asarray(nib.load(str(f)).dataobj) > 0) if f.exists() else None
    for i in range(1, k + 1):
        m = lab == i
        nv = int(m.sum())
        cls_ids = [int(v) for v in np.unique(ga[m]) if v > 0]
        cname = id2name.get(cls_ids[0], f"?{cls_ids}") if cls_ids else "?"
        # 등가구 지름
        dia = 2.0 * (3.0 * nv * vox_mm3 / (4 * np.pi)) ** (1 / 3)
        hit = {}
        for d in LADDER:
            hit[SHORT[d]] = bool(preds[d] is not None and (preds[d] & m).any())
        rows.append(dict(case=cid, cls=cname, nvox=nv, mm3=nv * vox_mm3, dia=dia, **hit))
    if n % 20 == 0:
        print(f"  {n}/{len(test_ids)}", flush=True)

print(f"\nGT 병변 {len(rows)}개 / test {len(test_ids)}케이스\n")
cur = "probavgf"
miss = [r for r in rows if not r[cur]]
print(f"현행({cur}) 적중 {len(rows)-len(miss)} · 미검출 {len(miss)}\n")

print("[미검출 병변이 더 느슨한 검출본에서는 잡히나]")
print(f"  {'단계':<12}{'미검출 중 회수':>14}{'누적 회수':>10}")
rec = set()
for d in LADDER[1:]:
    s = SHORT[d]
    now = {j for j, r in enumerate(miss) if r[s]}
    rec |= now
    print(f"  {s:<12}{len(now):>14}{len(rec):>10}")
print(f"\n  → 어떤 단계에서도 안 잡히는(신호 자체가 없는) 병변: {len(miss)-len(rec)}")

print("\n[미검출 크기 분포]")
for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 10), (10, 999)]:
    a = [r for r in rows if lo <= r["dia"] < hi]
    b = [r for r in a if not r[cur]]
    if a:
        print(f"  지름 {lo:>2}-{hi if hi<999 else '∞':>3}mm  GT {len(a):>3}  미검출 {len(b):>3}  "
              f"미검출률 {len(b)/len(a):.2f}")

print("\n[미검출 클래스 상위]")
c = collections.Counter(r["cls"] for r in miss)
tot = collections.Counter(r["cls"] for r in rows)
for cl, v in c.most_common(12):
    print(f"  {cl:<28}{v:>3}/{tot[cl]:<3}  미검출률 {v/tot[cl]:.2f}")

with open(L.TOPANEU_ROOT / "experiments" / "miss_char_test.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print("\n[저장] experiments/miss_char_test.json")
