#!/usr/bin/env python3
"""noise_cv.py — 학습표 몇 줄이 바뀌면 점수가 얼마나 흔들리는가 (노이즈 바닥)

왜 필요한가
-----------
구 학습표 → 개정판 학습표로 바꾸니 test MCC 가 −0.0240 이었다. 그런데 두 표의 차이는
**병변 6개**뿐이다(268 vs 271, 공통 258줄 중 240줄은 숫자까지 동일).
그리고 **5시드는 학습표를 안 바꾼다** — 다섯 번 다 같은 표를 본다. 그래서 −0.0240 에는
"어느 6줄이었나" 의 오차막대가 없다. 점추정 두 개를 비교한 것뿐이다.

여기서 재는 것: 개정판 표에서 **무작위 k줄을 빼고** CV 를 돌리기를 N 번 반복해,
"k줄 차이가 만드는 점수 폭" 의 분포를 얻는다. 실제 관측된 두 표의 CV 차이가
그 폭 안에 들어가면 **판정 자체가 성립하지 않는다**.

주의: CV 는 e2e 가 아니다(검출기·c7·패치필터를 안 거친다). 여기서 얻는 건 **상대적
흔들림의 크기**이지 e2e 점수가 아니다. 채택 판정에 쓰지 않는다 — 해석의 기준선일 뿐이다.
"""
import json, os, sys, warnings, collections
warnings.filterwarnings("ignore")
import numpy as np
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
for p in (f"{R}/code/sblee", f"{R}/code/sblee/nnunet", f"{R}/code/sblee/nnunet/scripts"):
    sys.path.insert(0, p)
os.environ.setdefault("TOPANEU_ROOT", R)
import d9xx_lib as L
import c5_location_v2 as C5
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score

A = f"{R}/code/sblee/nnunet/analysis"
K = int(os.environ.get("NOISE_K", "6"))        # 두 표의 실제 차이와 같은 줄 수
N = int(os.environ.get("NOISE_N", "60"))       # 반복
SEED_CV = 0


def vecs(rows, ves_axis):
    X, y = [], []
    for r in rows:
        X.append(C5.row_to_vec(r, ves_axis)); y.append(r["gt_loc"])
        X.append(C5.row_to_vec(r, ves_axis, mirror=True)); y.append(C5.mirror_name(r["gt_loc"]))
    return np.array(X), np.array(y)


def score(rows, ves_axis, seed=SEED_CV):
    X, y = vecs(rows, ves_axis)
    keep = np.array([c for c in np.unique(y) if (y == c).sum() >= 5])
    m = np.isin(y, keep); X, y = X[m], y[m]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.empty_like(y)
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, max_features=0.3,
                                     class_weight="balanced", random_state=seed, n_jobs=1)
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    return recall_score(y, pred, average="macro", zero_division=0), (pred == y).mean()


def main():
    C5.USE_POS = True
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    old = json.load(open(f"{A}/e11_feat_hyb_ov.json"))
    new = json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json"))
    mo, ao = score(old, ves_axis); mn, an = score(new, ves_axis)
    print(f"구 학습표   {len(old)}행 · macro-recall {mo*100:5.2f}% · 정답률 {ao*100:5.2f}%")
    print(f"개정판 표   {len(new)}행 · macro-recall {mn*100:5.2f}% · 정답률 {an*100:5.2f}%")
    print(f"실제 관측 차이  Δmacro {(mn-mo)*100:+.2f}%p · Δ정답률 {(an-ao)*100:+.2f}%p\n")

    rng = np.random.default_rng(0)
    ds = []
    for i in range(N):
        idx = rng.choice(len(new), len(new) - K, replace=False)
        sub = [new[j] for j in idx]
        m, a = score(sub, ves_axis)
        ds.append((m - mn) * 100)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N} 진행", flush=True)
    ds = np.array(ds)
    print(f"\n## {K}줄을 무작위로 뺐을 때 macro-recall 변화 ({N}회)\n")
    print(f"  평균 {ds.mean():+.2f}%p · 표준편차 {ds.std():.2f}%p")
    print(f"  최소 {ds.min():+.2f}  ·  5% {np.percentile(ds,5):+.2f}  ·  중앙 {np.percentile(ds,50):+.2f}"
          f"  ·  95% {np.percentile(ds,95):+.2f}  ·  최대 {ds.max():+.2f}")
    obs = (mo - mn) * 100
    inside = (np.abs(ds) >= abs(obs)).mean() * 100
    print(f"\n  관측된 구−개정판 차이: {obs:+.2f}%p")
    print(f"  무작위 {K}줄 교체가 그만큼 이상 흔드는 비율: **{inside:.0f}%**")
    print(f"  → {'노이즈 범위 안 — 판정 불가' if inside >= 10 else '노이즈보다 큼 — 실재 가능'}")


if __name__ == "__main__":
    raise SystemExit(main())
