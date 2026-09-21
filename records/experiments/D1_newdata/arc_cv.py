#!/usr/bin/env python3
"""arc_cv.py — ICA 호길이 블록이 쓸모 있는지 **교차검증으로 선별만** 한다.

이 축은 이미 한 번 닫혔다 — C34(분기점 두 개 사이 상대 호위치, 4차원)가
C60 §18 에서 e2e test −0.0228 · val −0.0135 로 기각됐고 "축 종결" 로 적혔다.
이번 구현은 두 가지가 다르다:
  (1) 좌표 앵커가 **고정 해부 지점**(C1-C5 경계=0, 종말부=1)이라 환자 간 비교가 된다.
      C34 의 t 는 '가장 가까운 분기점 두 개 사이 몇 %' 라 병변마다 구간이 달랐다.
  (2) OA·Pcom·AChA 까지의 **부호 있는 차이**가 들어간다. 이게 트리가 스스로 못 만드는
      비교량이다. C34 에는 이 항이 아예 없었다.
  (3) ICA 밖 병변은 블록이 전부 0 이다. row_to_vec 은 블록마다 재정규화하므로
      0 블록은 벡터를 바꾸지 않는다 → 다른 클래스에 잡음을 안 넣는다. C34 는 전 혈관에 얹었다.

**이건 선별이지 채택 판정이 아니다.** CV 상한은 믿지 않는다(PROJECT_RULES.md 2장).
여기서 통과해도 5시드 e2e test·val 동시 개선이 있어야 채택이다.

── 판정규칙 (결과 보기 전에 고정) ───────────────────────────────────────
  주: ICA C6/C7 세부분절(3.2~3.7) 정답률이 base 대비 **+3%p 이상** 오른다.
  부: 전체 macro-recall 이 base 대비 **−1%p 이내**(다른 클래스를 망치지 않는다).
  둘 다 만족해야 e2e 로 간다. 하나라도 어기면 **축을 닫고** POSTMORTEM 에 기록한다.
  반복 20회(시드 0~19) 평균으로 본다 — 268행이라 한 번 돌려선 잡음에 속는다.
────────────────────────────────────────────────────────────────────────
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
sys.path.insert(0, f"{R}/code/sblee"); sys.path.insert(0, f"{R}/code/sblee/nnunet")
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts"); sys.path.insert(0, f"{R}/experiments/D1_newdata")
os.environ.setdefault("TOPANEU_ROOT", R)

import d9xx_lib as L
import c5_location_v2 as C5
import arc_feat as A
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

ARC_W = float(os.environ.get("ARC_W", "0.5"))     # pos/geo 와 같은 블록 가중치


def add_block(v, raw, w):
    """row_to_vec 과 같은 방식 — 블록별 정규화 후 가중치, 그 뒤 전체 재정규화."""
    a = np.asarray(raw, float)
    na = np.linalg.norm(a)
    add = a / na * w if na > 0 else np.zeros(len(a))
    v = np.concatenate([v, add])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    rows = json.load(open(f"{R}/code/sblee/nnunet/analysis/e11_feat_hyb_ov.json"))
    S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
    NAME = {int(k): v for k, v in S["location_classes"].items()}
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    C5.USE_POS = True

    packs, Xb, Xa, y = {}, [], [], []
    nz = 0
    for r in rows:
        c = r["case"]
        if c not in packs:
            packs[c] = A.load_pack(f"{R}/experiments/D1_newdata/arc/train/{c}.npz")
        arc = A.lookup(packs[c], np.array(r["_cen"], float))
        if arc[7] > 0:
            nz += 1
        v = C5.row_to_vec(r, ves_axis)
        Xb.append(v); Xa.append(add_block(v, arc, ARC_W)); y.append(r["gt_loc"])
        # 좌우 미러 증강 — arc 는 '그 혈관을 따라 몇 %' 라 좌우 불변이므로 그대로 붙인다
        vm = C5.row_to_vec(r, ves_axis, mirror=True)
        Xb.append(vm); Xa.append(add_block(vm, arc, ARC_W))
        y.append(C5.mirror_name(r["gt_loc"]))
    Xb = np.array(Xb); Xa = np.array(Xa); y = np.array(y)
    print(f"행 {len(y)} (미러 포함) · base {Xb.shape[1]}차원 → +arc {Xa.shape[1]}차원")
    print(f"ICA 호길이가 잡힌 병변 {nz}/{len(rows)} = {nz/len(rows)*100:.0f}%\n")

    ICA = {v for v in NAME.values() if any(f"3.{i}" in v for i in (2, 3, 4, 5, 6, 7))}
    keep = np.array([c for c in np.unique(y) if (y == c).sum() >= 5])   # 5-fold 층화 최소
    m = np.isin(y, keep)
    print(f"CV 대상: 표본 3개 이상인 {len(keep)}클래스 · {m.sum()}행 (나머지는 층화 불가)\n")
    Xb, Xa, y = Xb[m], Xa[m], y[m]

    res = {}
    for nm, X in (("base", Xb), ("base+arc", Xa)):
        ic, mr = [], []
        for seed in range(20):
            skf = StratifiedKFold(5, shuffle=True, random_state=seed)
            pred = np.empty_like(y)
            for tr, te in skf.split(X, y):
                clf = RandomForestClassifier(n_estimators=500, max_features=0.3,
                                             class_weight="balanced", random_state=seed, n_jobs=1)
                clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
            im = np.isin(y, sorted(ICA))
            ic.append((pred[im] == y[im]).mean())
            mr.append(recall_score(y, pred, average="macro", zero_division=0))
        res[nm] = (np.mean(ic), np.std(ic), np.mean(mr), np.std(mr))
        print(f"{nm:9s}  ICA정답률 {np.mean(ic)*100:5.1f}% (±{np.std(ic)*100:.1f})  "
              f"macro-recall {np.mean(mr)*100:5.1f}% (±{np.std(mr)*100:.1f})")

    dI = (res["base+arc"][0] - res["base"][0]) * 100
    dM = (res["base+arc"][2] - res["base"][2]) * 100
    print(f"\nΔICA {dI:+.1f}%p · Δmacro-recall {dM:+.1f}%p")
    ok = dI >= 3.0 and dM >= -1.0
    print(f"판정: {'통과 → e2e 로 간다' if ok else '미달 → 축을 닫는다'} "
          f"(기준 ΔICA ≥ +3.0%p ∧ Δmacro ≥ −1.0%p)")


if __name__ == "__main__":
    raise SystemExit(main())
