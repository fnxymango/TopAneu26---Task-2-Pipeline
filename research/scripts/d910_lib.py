"""D910 방법론(weighted-kNN, 신규): D900(signature+3단계 hard-fallback lookup)의 대안.

문제의식(2026-08-10): D900의 predict_location()은 exact match가 없으면 backoff(nearest_class만),
그것도 없으면 무조건 global_major(=가장 흔한 단일 클래스, 예: R-5.3 M1-M2 junction) 하나로 찍는다.
official eval.py는 52(or 존재-클래스수)개 클래스를 "균등 평균"하므로, 이 하드 폴백이 발동될 때마다
- 진짜 클래스: FN 확정
- global_major 클래스: 근거 없는 FP 하나 추가
가 반복돼서 특히 표본이 적은 롱테일 클래스의 Precision/Dice를 구조적으로 깎아먹는다(사용자 지적).

D910 = 같은 lesion feature(dist_mm: 반경 MAX_R 이내 vessel class -> mm거리)를 soft membership
벡터(1/(1+거리))로 바꾸고, train lesion 전체와의 코사인 유사도로 weighted k-NN 투표.
exact match 개념이 없어 하드 폴백이 필요 없고, 유사도가 전부 0인(=반경 내 아무 vessel도 없는)
경우에만 "예측 포기"(global_major로 억지로 찍지 않음 -> FN만 발생, FP는 안 늘림).

MG-HGLNet(coronary, dynamic prototype) 등 학습기반 coarse-to-fine 문헌의 정신을 룰베이스로
축소구현한 것 — 각 train lesion이 자기 자신의 micro-prototype 역할을 함.
"""
import numpy as np

import d9xx_lib as L  # 공용 인프라(feature 추출/이름 매핑 등) 재사용


def _vecs(rows, cidx, D):
    V = np.zeros((len(rows), D))
    for i, r in enumerate(rows):
        for c, d in r["dist_mm"].items():
            j = cidx.get(c)
            if j is not None:
                V[i, j] = 1.0 / (1.0 + d)
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return V / norm


def load_knn_index(path):
    z = np.load(path, allow_pickle=True)
    classes = list(z["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    return {"V": z["V"], "y": list(z["y"]), "cidx": cidx, "D": len(classes),
            "k": int(z["k"]), "power": float(z["power"])}


def build_knn_index(rows, k=5, power=1.0):
    """rows: gt_loc 있는(train split) lesion feature 목록. dist_mm에 등장하는 모든 vessel
    class를 축으로 하는 코사인 유사도 공간을 만든다."""
    all_classes = sorted(set(c for r in rows for c in r["dist_mm"].keys()))
    cidx = {c: i for i, c in enumerate(all_classes)}
    V = _vecs(rows, cidx, len(all_classes))
    y = [r["gt_loc"] for r in rows]
    return {"V": V, "y": y, "cidx": cidx, "D": len(all_classes), "k": k, "power": power}


def predict_location_knn(r, index):
    """반환값 None = 예측 포기(반경 내 vessel 없음/train과 겹치는 클래스 없음).
    D900의 global_major 강제와 달리, 여기서는 확신 없으면 아무것도 안 찍어서
    official 지표에서 근거없는 FP를 만들지 않는다."""
    v = np.zeros(index["D"])
    for c, d in r["dist_mm"].items():
        j = index["cidx"].get(c)
        if j is not None:
            v[j] = 1.0 / (1.0 + d)
    n = np.linalg.norm(v)
    if n == 0:
        return None
    v = v / n
    sims = index["V"] @ v
    order = np.argsort(-sims)
    acc = {}
    used = 0
    for j in order:
        s = sims[j]
        if s <= 0:
            break
        yj = index["y"][j]
        acc[yj] = acc.get(yj, 0.0) + s ** index["power"]
        used += 1
        if used >= index["k"]:
            break
    if not acc:
        return None
    return max(acc, key=acc.get)
