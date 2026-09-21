#!/usr/bin/env python
"""① 곁가지 존재 신뢰도 (2026-08-25).

진단: 예측 혈관에서 Pcom/AChA 가 GT 대비 1.31~1.46배 과대 등장한다(다른 혈관은 1.0~1.1).
      Pcom·AChA 는 해부학적으로 결손이 흔한 혈관이다. 없는 걸 그려놓으면 중첩/거리 피처가
      "곁가지가 붙어 있다"고 말하고 분류기는 junction 이라고 답한다.
      실제 최다 오답이 전부 "-> Pcom-junction" 이라 기전이 들어맞는다.

여기서는 케이스별·혈관클래스별 **예측 성분의 크기**를 재서 신뢰도를 만든다.
진짜 혈관은 굵고 이어진 관이고, 허깨비는 작고 끊긴 조각이다.

  conf[c] = clip( 최대연결성분_복셀[c] / 기준크기[c] , 0, 2 )
  기준크기[c] = 학습 292 케이스 중 그 클래스가 등장한 케이스의 **중앙값**

기준을 학습 케이스에서만 잡으므로 val/test 누수가 없다.
출력: vesconf_{train,test}.json (case -> {클래스명: conf}), vesconf_ref.json
"""
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
NAMES = {v: k for k, v in
         json.load(open(L.TOPANEU_ROOT / "dataset/TopAneu/vessel_mapping.json"))["labels"].items()}
ST = np.ones((3, 3, 3), bool)
train_ids, _, test_ids = L.case_ids_by_split()


def case_cc(vdir, cid):
    """클래스별 최대 연결성분 복셀 수.

    36번 전체 볼륨(36M 복셀)에 ndi.label 을 거는 건 케이스당 60초가 걸렸다.
    find_objects 로 클래스별 경계상자를 한 번에 얻어 **상자 안에서만** 라벨링한다.
    혈관 하나가 볼륨의 1% 미만이라 수십 배 빨라진다.
    """
    f = P / vdir / f"{cid}.nii.gz"
    if not f.exists():
        return None
    a = np.asanyarray(nib.load(str(f)).dataobj)
    if a.dtype != np.int32:
        a = a.astype(np.int32, copy=False)
    boxes = ndi.find_objects(a, max_label=36)
    out = {}
    for c in range(1, 37):
        sl = boxes[c - 1]
        if sl is None:
            continue
        sub = a[sl] == c
        lab, k = ndi.label(sub, structure=ST)
        if not k:
            continue
        out[NAMES.get(c, str(c))] = int(np.bincount(lab.ravel())[1:].max())
    return out


raw = {}
for split, ids, vdir in (("train", train_ids, "vespp_train"), ("test", test_ids, "vespp_test")):
    d = {}
    for i, cid in enumerate(ids, 1):
        r = case_cc(vdir, cid)
        if r is not None:
            d[cid] = r
        if i % 60 == 0:
            print(f"  {split} {i}/{len(ids)}", flush=True)
    raw[split] = d
    print(f"  {split} 완료 {len(d)}케이스", flush=True)

# 기준크기는 학습 케이스에서만
by = collections.defaultdict(list)
for cid, d in raw["train"].items():
    for k, v in d.items():
        by[k].append(v)
ref = {k: float(np.median(v)) for k, v in by.items() if v}
json.dump(ref, open(A / "vesconf_ref.json", "w"))

for split in ("train", "test"):
    conf = {cid: {k: round(min(2.0, v / max(ref.get(k, 1.0), 1.0)), 4) for k, v in d.items()}
            for cid, d in raw[split].items()}
    json.dump(conf, open(A / f"vesconf_{split}.json", "w"))

print("\n[클래스별 기준크기(학습 중앙값) 와 등장률]")
print(f"  {'라벨':<16}{'기준복셀':>9}{'학습등장':>9}{'test등장':>9}{'test conf중앙':>14}")
for k in sorted(ref, key=lambda x: -ref[x]):
    tv = [d[k] / max(ref[k], 1.0) for d in raw["test"].values() if k in d]
    print(f"  {k:<16}{ref[k]:>9.0f}{len(by[k]):>9}{len(tv):>9}"
          f"{(np.median(tv) if tv else 0):>14.3f}")
print("\n저장: vesconf_train.json / vesconf_test.json / vesconf_ref.json")
