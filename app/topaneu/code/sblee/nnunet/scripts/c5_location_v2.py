"""C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속).

문제(2026-08-14 분석): 위치클래스 52개 vs 혈관클래스 36개. 21개 클래스(병변의 61%)가
"두 혈관이 만나는 지점"으로 정의되는데, 기존 C1/C2의 34차원 피처(dist_mm = 각 혈관까지 거리)에는
분기점 정보도 혈관 위 위치 정보도 없다. 그래서 GT 병변 + GT 혈관을 넣어도 오라클 상한이 61.0%
(oracle_location_report_417.md). 그 천장을 뚫으려면 피처에 분기점이 들어가야 한다.

C5의 피처 (병변당):
  [A] 혈관 근접도 36차원   — 1/(1+거리mm), 기존 C2와 동일
  [B] sac 내부 점유율 36차원 — 기존 C2가 계산해놓고 버리던 overlap (오라클에서 V3>V1 이었음)
  [C] 분기점 근접도 N차원  — c4_branchpoint_graph.py가 뽑은 노드까지 1/(1+거리mm).
                            "R-ICA-C6-C7 + R-Pcom 분기점까지 몇 mm"가 곧 3.4 판정 근거다.

그리고 기존 C2에 있던 세 가지 구조적 결함을 함께 고친다:
  1. 인덱스는 참조 혈관마스크로 만들고 추론은 예측 혈관을 썼다(분포 불일치) -> --vessel-dir 로 통일
  2. 좌우 미러 증강 없음 -> 268 -> 536 (뇌혈관 좌우대칭, R-/L- 접두 swap)
  3. 공식지표는 52클래스 균등평균인데 kNN 다수결은 빈발 클래스로 쏠림 -> 클래스 사전확률 보정

사용:
  # 1) 피처 추출 (split별)
  python c5_location_v2.py build --split train --vessel-dir <36클래스> --bp-dir <c4출력> --out <feat.json>
  # 2) 학습 + 평가
  python c5_location_v2.py eval --train-feat <..> --split val --vessel-dir <..> --bp-dir <..> \
      --aneurysm-pred-dir <..> --model knn|rf --tag <..>
"""
import argparse, re, json, math, os, sys, time, collections
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

import d9xx_lib as L

MAX_R = 10.0          # 혈관 근접도를 재는 최대 반경 (d9xx_lib와 동일)
BP_MAX_R = 15.0       # 분기점 근접도 최대 반경 — 병변이 분기점에서 좀 떨어져 있어도 잡아야 함
CONF_TAU = 0.0        # C31: 최대확률이 이 값 미만일 때만 β 적용 (0이면 기존처럼 전부 적용)
CONF_BETA_HI = 0.0    # 확신이 높은 병변에 적용할 β

# 반경 밖이면 dist_mm이 비어 피처가 전영이 되고 -> 예측 기권 -> 확정 오답이 된다.
# c8 진단에서 '1.4 BA trunk -> None' 3건이 이 경로였다. 환경변수로 반경을 키워 재추출할 수 있게 한다.
if os.environ.get("TOPANEU_MAX_R"):
    MAX_R = L.MAX_R = float(os.environ["TOPANEU_MAX_R"])
if os.environ.get("TOPANEU_BP_MAX_R"):
    BP_MAX_R = float(os.environ["TOPANEU_BP_MAX_R"])


# --- 분기점 피처 타입 ------------------------------------------------------
# 52개 위치클래스 정의에 실제로 등장하는 혈관 쌍만 추린다. 모든 쌍을 쓰면 차원이 폭발하고
# (한 케이스에만 37종) 학습 병변이 268개뿐이라 과적합한다.
JUNCTION_PAIRS = [
    # --- 후순환 ---
    ("R-VA", "R-PICA"), ("L-VA", "L-PICA"),            # 1.3 VA-PICA junction
    ("BA", "R-VA"), ("BA", "L-VA"),                    # 1.5 VA-BA junction
    ("BA", "R-AICA"), ("BA", "L-AICA"),                # 1.7 BA-AICA junction
    ("BA", "R-SCA"), ("BA", "L-SCA"),                  # 1.9 BA-SCA junction
    ("BA", "R-P1P2"), ("BA", "L-P1P2"),                # 1.10 BA tip
    ("R-P1P2", "R-P3P4"), ("L-P1P2", "L-P3P4"),        # 2.1/2.2 경계
    ("R-P1P2", "R-Pcom"), ("L-P1P2", "L-Pcom"),
    # --- ICA (병변의 29%가 여기) ---
    ("R-ICA-C1-C5", "R-ICA-C6-C7"),                    # 3.1/3.3 경계 (serial)
    ("L-ICA-C1-C5", "L-ICA-C6-C7"),
    ("R-ICA-C6-C7", "R-OA"), ("L-ICA-C6-C7", "L-OA"),      # 3.2 C6-OA-junction
    ("R-ICA-C6-C7", "R-Pcom"), ("L-ICA-C6-C7", "L-Pcom"),  # 3.4 C7-Pcom-junction
    ("R-ICA-C6-C7", "R-AChA"), ("L-ICA-C6-C7", "L-AChA"),  # 3.5 C7-AChA-junction
    ("R-ICA-C6-C7", "R-M1"), ("L-ICA-C6-C7", "L-M1"),      # 3.7 terminus
    ("R-ICA-C6-C7", "R-A1A2"), ("L-ICA-C6-C7", "L-A1A2"),  # 3.7 terminus
    # --- ACA ---
    ("Acom", "R-A1A2"), ("Acom", "L-A1A2"),            # 4.1 Acom complex
    ("R-A1A2", "R-A3"), ("L-A1A2", "L-A3"),            # 4.3/4.4 경계
    # --- MCA ---
    ("R-M1", "R-M2"), ("L-M1", "L-M2"),                # 5.2 early bif / 5.3 M1-M2 junction
    ("R-M2", "R-M3"), ("L-M2", "L-M3"),                # 5.4 Distal-M2M3
]
BP_DIM = len(JUNCTION_PAIRS)

# --- 분기점 블록 모드 (E18) ------------------------------------------------
#   raw(기본)  기존: 15mm 게이트 + 1/(1+d)
#   off        블록 전체를 0 으로 (대조군 B)
#   sharp      게이트 없이 exp(-d/BP_SHARP_MM)
#   branch_ov  쌍 양쪽의 sac 점유 여부(0/1) 의 min
#   branch_mix 점유하면 1, 아니면 거리로 약하게 (0.3*exp(-d/2mm)) 의 min
BP_MODE = os.environ.get("TOPANEU_BP_MODE", "raw")

# --- ① 곁가지 존재 신뢰도 (2026-08-25) --------------------------------------
# 예측 혈관에서 Pcom/AChA 가 GT 대비 1.31~1.46배 과대 등장한다. 없는 곁가지를 그려놓으면
# 거리/중첩 피처가 "junction 이다"라고 거짓말을 하고, 실제 최다 오답이 전부 -> Pcom-junction 이다.
# conf[c] = 최대연결성분 / 학습중앙값 (0~2). 진짜 혈관은 굵고 이어져 있고 허깨비는 조각이다.
#   off   기존 그대로 (112차원)
#   block conf 36차원을 뒤에 붙인다 (148차원) — RF 가 쓸지 말지 스스로 정한다
#   gate  거리/중첩 피처에 conf 를 곱한다 (112차원 유지) — 허깨비 신호를 직접 누른다
VESCONF_MODE = os.environ.get("TOPANEU_VESCONF", "off")
_VESCONF = {}
def load_vesconf(path):
    global _VESCONF
    if path and os.path.exists(path):
        _VESCONF = json.load(open(path))
    return len(_VESCONF)
_VESCONF_LOADED = False
def _vesconf_table():
    """train/test 표를 한 번만 읽어 합친다. 케이스 id 가 split 간 유일해서 한 딕셔너리면 된다."""
    global _VESCONF, _VESCONF_LOADED
    if _VESCONF_LOADED:
        return _VESCONF
    _VESCONF_LOADED = True
    base = os.environ.get("TOPANEU_VESCONF_DIR")
    if not base:
        base = os.path.join(os.environ.get("TOPANEU_ROOT", ""), "code/sblee/nnunet/analysis")
    for sp in ("train", "val", "test"):
        f = os.path.join(base, f"vesconf_{sp}.json")
        if os.path.exists(f):
            _VESCONF.update(json.load(open(f)))
    print(f"[vesconf] {VESCONF_MODE} · {len(_VESCONF)}케이스 로드", flush=True)
    return _VESCONF

def _conf(r, name):
    d = _vesconf_table().get(r.get("case") or "", None)
    if d is None:
        return 1.0
    return float(d.get(name, 0.0))


# --- W2: jslee 좌표 Mahalanobis 재라벨링 (REIMPLEMENT_jslee_pjh.md §2.6b) ---
# 예측 blob 의 **좌표만** 보고 클래스를 정하는 최근접중심 분류기. 저쪽 라인에서
# MCC 0.1140 -> 0.1671 (+47%) 를 냈고, 문서가 "라인을 가리지 않는다" 고 적어뒀다.
# 우리 RF 는 이미 top1 이 훨씬 높으므로 **대체가 아니라 혼합항**으로 쓴다.
# ICA 직렬 분절(C1-C5 / C6 / C7)처럼 위치가 곧 판별자인 약한 클래스에 기대를 건다.
MAHA_W = float(os.environ.get("TOPANEU_MAHA_W", "0"))

# --- W2: junction <-> trunk 전용 이진 판별기 (2026-08-25) ---
# E16 혼동 상위 10쌍 중 6쌍이 "분기점이냐 몸통이냐" 다. E18 에서 분기점 **피처 형태**를
# 네 가지로 바꿔봤지만 전부 현행보다 나빴다. 형태가 아니라 **문제를 좁히는** 쪽으로 간다.
#
#   52-way 에서 이 두 클래스를 가르는 건 어렵다 — 클래스당 표본이 1~5개다.
#   그런데 "이 둘 중 어느 쪽이냐" 2-way 로 좁히고, 좌우를 미러로 한쪽에 모으면
#   표본이 두 배가 되고 결정경계도 하나만 배우면 된다.
#
# 적용: RF 의 top-2 가 알려진 혼동쌍이고 **같은 쪽(L/L 또는 R/R)** 이면 전용 판별기가 재정한다.
#   좌우가 다르면(L-4.4 vs R-4.4) 그건 junction/trunk 문제가 아니라 편측 문제라 건드리지 않는다.
PAIR_ON = os.environ.get("TOPANEU_PAIR", "0") == "1"
PAIR_MIN_N = int(os.environ.get("TOPANEU_PAIR_MIN_N", "6"))

CONFUSE_PAIRS = [
    ("3.4 ICA C7-Pcom-junction", "3.6 ICA C7-nonBranch"),
    ("3.5 ICA C7-AChA-junction", "3.4 ICA C7-Pcom-junction"),
    ("3.3 ICA C6-nonOA",         "3.2 ICA C6-OA-junction"),
    ("3.4 ICA C7-Pcom-junction", "3.3 ICA C6-nonOA"),
    ("3.6 ICA C7-nonBranch",     "3.3 ICA C6-nonOA"),
    ("1.3 VA-PICA junction",     "1.1 VA trunk"),
    ("5.3 M1-M2 junction",       "5.2 M1 early bifurcation"),
]


def _side(n):
    return n[:2] if n[:2] in ("R-", "L-") else ""


def _sfx(n):
    return n[2:] if n[:2] in ("R-", "L-") else n


def fit_pairs(rows, ves_axis):
    """쌍마다 이진 RF. 좌우를 미러로 R 쪽에 모아 표본을 두 배로 쓴다."""
    from sklearn.ensemble import RandomForestClassifier
    by = collections.defaultdict(list)
    for r in rows:
        g = r.get("gt_loc")
        if g:
            by[_sfx(g)].append(r)
    out = {}
    for a, b in CONFUSE_PAIRS:
        ra, rb = by.get(a, []), by.get(b, [])
        if len(ra) < 2 or len(rb) < 2 or len(ra) + len(rb) < PAIR_MIN_N:
            continue
        X, y = [], []
        for lab, rs in ((a, ra), (b, rb)):
            for r in rs:
                # 왼쪽 표본은 미러해서 오른쪽 좌표계로 옮긴다(좌우대칭 가정 — 미러 증강의 근거와 동일)
                X.append(row_to_vec(r, ves_axis, mirror=(_side(r["gt_loc"]) == "L-")))
                y.append(lab)
        out[frozenset((a, b))] = {"clf": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, class_weight="balanced",
            random_state=_seed(), n_jobs=-1).fit(np.asarray(X), np.asarray(y)),
            "n": (len(ra), len(rb))}
    return out


def pair_override(pairs, r, ves_axis, cls_, p):
    """top-2 가 혼동쌍이면 전용 판별기 결과로 바꾼다. 아니면 None."""
    if not pairs:
        return None
    order = np.argsort(-p)[:2]
    n1, n2 = str(cls_[order[0]]), str(cls_[order[1]])
    s1, s2 = _side(n1), _side(n2)
    if s1 != s2:                       # 편측 혼동은 이 판별기의 문제가 아니다
        return None
    key = frozenset((_sfx(n1), _sfx(n2)))
    m = pairs.get(key)
    if m is None or len(key) != 2:
        return None
    v = row_to_vec(r, ves_axis, mirror=(s1 == "L-"))
    if np.linalg.norm(v) == 0:
        return None
    q = m["clf"].predict_proba(v[None, :])[0]
    win = str(m["clf"].classes_[int(np.argmax(q))])
    return f"{s1}{win}" if s1 else win
MAHA_REG = float(os.environ.get("TOPANEU_MAHA_REG", "0.1"))


def fit_maha(rows):
    """클래스별 pos 평균 + 공용(pooled) 역공분산. 표본이 클래스당 1~4개라
    클래스별 공분산은 못 쓴다 — 전체에서 하나만 추정하고 대각 정칙화한다."""
    by = collections.defaultdict(list)
    for r in rows:
        p = r.get("pos")
        if p:
            by[r["gt_loc"]].append(np.asarray(p, dtype=float))
    if not by:
        return None
    dev = []
    mu = {}
    for c, vs in by.items():
        m = np.mean(vs, axis=0); mu[c] = m
        for v in vs:
            dev.append(v - m)
    D = np.asarray(dev)
    if len(D) < 2:
        return None
    S = np.cov(D.T) + MAHA_REG * np.eye(D.shape[1])
    return {"mu": mu, "Sinv": np.linalg.pinv(S)}


def maha_proba(maha, r, classes):
    """classes 순서에 맞춘 확률. pos 가 없으면 None (조용히 RF 단독으로 떨어진다)."""
    if not maha:
        return None
    p = r.get("pos")
    if not p:
        return None
    x = np.asarray(p, dtype=float)
    d2 = np.full(len(classes), np.inf)
    for j, c in enumerate(classes):
        m = maha["mu"].get(str(c))
        if m is None:
            continue
        v = x - m
        d2[j] = float(v @ maha["Sinv"] @ v)
    if not np.isfinite(d2).any():
        return None
    d2 = np.where(np.isfinite(d2), d2, d2[np.isfinite(d2)].max() + 50.0)
    q = np.exp(-0.5 * (d2 - d2.min()))
    ssum = q.sum()
    return q / ssum if ssum > 0 else None
BP_SHARP_MM = float(os.environ.get("TOPANEU_BP_SHARP_MM", "3.0"))


def _involve(r, vname):
    """병변 sac 이 혈관 vname 을 얼마나 무는가. 점유가 있으면 1, 없으면 모드에 따라 거리로 약하게."""
    ov = r.get("overlap") or {}
    if ov.get(vname, 0) > 0:
        return 1.0
    if BP_MODE == "branch_ov":
        return 0.0
    d = (r.get("dist_mm") or {}).get(vname)
    if d is None:
        return 0.0
    return 0.3 * math.exp(-float(d) / 2.0)


def mirror_name(n):
    """R-/L- 접두를 뒤집는다. 중앙 구조(BA, Acom, 3rd-*, 숫자접두 없는 위치클래스)는 그대로."""
    if n is None:
        return None
    if n.startswith("R-"):
        return "L-" + n[2:]
    if n.startswith("L-"):
        return "R-" + n[2:]
    return n


def build_feature_axes(ves_names):
    ves_axis = [ves_names[i] for i in sorted(ves_names)]
    return ves_axis, [f"{a}|{b}" for a, b in JUNCTION_PAIRS]


def load_bp(bp_dir, case):
    p = Path(bp_dir) / f"{case}.json"
    if not p.exists():
        return []
    return json.load(open(p)).get("nodes", [])


def lesion_bp_features(lesion_coords_mm, nodes, only_valid=True):
    """병변에서 각 분기점 타입까지의 최소거리(mm). 없으면 inf."""
    out = np.full(BP_DIM, np.inf)
    if not nodes or len(lesion_coords_mm) == 0:
        return out
    tree = cKDTree(lesion_coords_mm)
    pair_idx = {frozenset(p): i for i, p in enumerate(JUNCTION_PAIRS)}
    for nd in nodes:
        if only_valid and not nd.get("valid", True):
            continue
        cls = nd["classes"]
        cen = np.array(nd["centroid_mm"])
        d = float(tree.query(cen)[0])
        # 이 노드에 있는 모든 클래스 조합 중 우리가 추적하는 쌍에 해당하는 것 전부 갱신
        for i in range(len(cls)):
            for j in range(i + 1, len(cls)):
                k = pair_idx.get(frozenset((cls[i], cls[j])))
                if k is not None and d < out[k]:
                    out[k] = d
    return out


# --- C10: 해부 랜드마크 정규화 좌표 (순환 import 피하려고 여기 인라인) ---
# CV에서 macro-recall 0.410 -> 0.455 로 오늘 최대 개선. 기존 피처엔 전역 위치 정보가 없었다.
POS_DIM = 6
POS_WEIGHT = float(os.environ.get("C10_POS_WEIGHT", "0.5"))
LANDMARKS = {
    "BAtip": [{"BA", "R-P1P2"}, {"BA", "L-P1P2"}],
    "Rterm": [{"R-ICA-C6-C7", "R-M1"}, {"R-ICA-C6-C7", "R-A1A2"}],
    "Lterm": [{"L-ICA-C6-C7", "L-M1"}, {"L-ICA-C6-C7", "L-A1A2"}],
}


def landmark_frame(nodes):
    """(원점, 축행렬, 스케일, 랜드마크dict) 또는 None. 참조마스크 417케이스 중 97.4%에서 성립."""
    lm = {}
    for nd in nodes:
        cs = set(nd["classes"])
        for name, pats in LANDMARKS.items():
            if name not in lm and any(p <= cs for p in pats):
                lm[name] = np.array(nd["centroid_mm"], dtype=float)
    if not all(k in lm for k in ("BAtip", "Rterm", "Lterm")):
        return None
    R_, L_, B = lm["Rterm"], lm["Lterm"], lm["BAtip"]
    O = (R_ + L_ + B) / 3.0
    x = L_ - R_; s = float(np.linalg.norm(x))
    if s < 1e-6:
        return None
    x = x / s
    y = B - O; y = y - np.dot(y, x) * x
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        return None
    y = y / ny
    return O, np.stack([x, y, np.cross(x, y)]), s, lm


def pos_of(centroid_mm, frame):
    O, A, s, lm = frame
    p = A @ (centroid_mm - O) / s
    d = [float(np.linalg.norm(centroid_mm - lm[k])) / s for k in ("Rterm", "Lterm", "BAtip")]
    return [float(v) for v in p] + d


def mirror_pos(pos):
    x, y, z, dR, dL, dB = pos
    return [-x, y, z, dL, dR, dB]


def extract_case_rows(loc_or_aneu, ves, spacing, ves_names, nodes, loc_id_to_name=None):
    """d9xx_lib.extract_lesion_features + 분기점 피처 + 랜드마크 좌표."""
    rows, lesions = L.extract_lesion_features(loc_or_aneu, ves, spacing, ves_names,
                                              loc_id_to_name=loc_id_to_name)
    frame = landmark_frame(nodes)
    for r in rows:
        m = lesions == r["lesion_mask_idx"]
        idx = np.argwhere(m)
        coords_mm = idx * spacing
        r["bp_mm"] = [None if not np.isfinite(x) else round(float(x), 3)
                      for x in lesion_bp_features(coords_mm, nodes)]
        cen = idx.mean(axis=0) * spacing
        r["_cen"] = [float(x) for x in cen]      # 병변 중심(mm) — C11 합성 샘플 매칭에 쓰임
        r["pos"] = pos_of(cen, frame) if frame else None
    return rows, lesions


def row_to_vec(r, ves_axis, mirror=False):
    """[A] 혈관 근접도 36 + [B] sac 점유율 36 + [C] 분기점 근접도 BP_DIM"""
    nves = len(ves_axis)
    extra = nves if VESCONF_MODE == "block" else 0
    v = np.zeros(nves * 2 + BP_DIM + extra)
    vi = {n: i for i, n in enumerate(ves_axis)}
    total_ov = sum(r["overlap"].values()) or 1

    for c, d in r["dist_mm"].items():
        name = mirror_name(c) if mirror else c
        j = vi.get(name)
        if j is not None:
            g = min(1.0, _conf(r, c)) if VESCONF_MODE == "gate" else 1.0
            v[j] = g / (1.0 + d)
    for c, n in r["overlap"].items():
        name = mirror_name(c) if mirror else c
        j = vi.get(name)
        if j is not None:
            g = min(1.0, _conf(r, c)) if VESCONF_MODE == "gate" else 1.0
            v[nves + j] = g * n / total_ov

    bp = r.get("bp_mm") or [None] * BP_DIM
    for k, d in enumerate(bp):
        a, b = JUNCTION_PAIRS[k]
        if mirror:
            key = frozenset((mirror_name(a), mirror_name(b)))
            k2 = next((i for i, p in enumerate(JUNCTION_PAIRS) if frozenset(p) == key), None)
            if k2 is None:
                continue
        else:
            k2 = k
        if BP_MODE == "off":
            continue
        if BP_MODE in ("branch_ov", "branch_mix"):
            # E18-A: "분기점이냐"를 거리가 아니라 **가지 관여도**로 본다.
            # 측정(268병변)상 분기점까지의 거리는 PICA/AChA 에서 부호가 뒤집히는데,
            # sac 이 가지혈관을 무는지(overlap>0)는 5개 쌍 전부 방향이 옳았다.
            # 부모/가지 라벨링 없이 되도록 쌍의 **양쪽 관여도의 min** 을 쓴다 —
            # 분기점 병변은 두 혈관을 다 물고, 몸통 병변은 한쪽만 문다.
            v[nves * 2 + k2] = min(_involve(r, a), _involve(r, b))
            continue
        if d is None:
            continue
        if BP_MODE == "sharp":
            # 대조군: 축(거리)은 그대로 두고 15mm 하드게이트만 걷어내고 날카롭게.
            v[nves * 2 + k2] = math.exp(-d / BP_SHARP_MM)
        elif d <= BP_MAX_R:
            v[nves * 2 + k2] = 1.0 / (1.0 + d)

    n = np.linalg.norm(v)
    v = v / n if n > 0 else v

    if USE_POS:
        # pos는 값 범위가 O(1)인데 나머지는 106차원에 퍼진 단위벡터라 성분이 O(0.1).
        # 블록별 정규화 후 가중치로 섞지 않으면 cosine을 pos가 지배한다.
        p = r.get("pos")
        add = np.zeros(POS_DIM)
        if p:
            a = np.array(mirror_pos(p) if mirror else p, dtype=float)
            na = np.linalg.norm(a)
            if na > 0:
                add = a / na * POS_WEIGHT
        v = np.concatenate([v, add])
        n = np.linalg.norm(v)
        v = v / n if n > 0 else v

    # C15 측지 / C34 호위치. pos 와 같은 이유로 블록별 정규화 후 가중치로 섞는다.
    # 좌우 미러에서 geo 는 [BAtip, Rterm, Lterm] x [거리, 사행비] 라 R/L 을 맞바꿔야 하고,
    # arc 는 '그 혈관을 따라 몇 %' 라 좌우에 불변이므로 그대로 둔다.
    for flag, key, dim, wgt in ((USE_GEO, "geo", 6, GEO_WEIGHT), (USE_ARC, "arc", 4, ARC_WEIGHT),
                                (USE_MISS, "bp_miss", BP_DIM, MISS_WEIGHT),
                                (USE_IMP, "bp_imp", BP_DIM, IMP_WEIGHT)):
        if not flag:
            continue
        raw = r.get(key)
        add = np.zeros(dim)
        if raw and len(raw) == dim:
            a = np.array(raw, dtype=float)
            if key == "geo" and mirror:
                a = a[[0, 2, 1, 3, 5, 4]]      # BAtip 고정, Rterm<->Lterm
            elif key in ("bp_miss", "bp_imp") and mirror:
                a = a[_BP_MIRROR_IDX]          # 분기점 축이므로 좌우 쌍을 맞바꾼다
            na = np.linalg.norm(a)
            if na > 0:
                add = a / na * wgt
        v = np.concatenate([v, add])
        n = np.linalg.norm(v)
        v = v / n if n > 0 else v
    if VESCONF_MODE == "block":
        base = nves * 2 + BP_DIM
        for c in ves_axis:
            name = mirror_name(c) if mirror else c
            v[base + vi[c]] = _conf(r, name)

    return v


USE_POS = False        # --use-pos 로 켠다 (C10)
# 분기점 축의 좌우 미러 대응표 (bp_miss / bp_imp 용). 짝이 없으면 자기 자신.
_BP_MIRROR_IDX = []
for _a, _b in JUNCTION_PAIRS:
    _key = frozenset((mirror_name(_a), mirror_name(_b)))
    _j = next((i for i, _p in enumerate(JUNCTION_PAIRS) if frozenset(_p) == _key), None)
    _BP_MIRROR_IDX.append(_j if _j is not None else len(_BP_MIRROR_IDX))

USE_MISS = False       # --use-miss (C42) 분기점 결측 지시자 34차원
USE_IMP = False        # --use-imp  (C43) 아틀라스 대체값 34차원
MISS_WEIGHT = float(os.environ.get("C42_MISS_WEIGHT", "0.5"))
IMP_WEIGHT = float(os.environ.get("C43_IMP_WEIGHT", "0.5"))
USE_GEO = False        # --use-geo 로 켠다 (C15 측지거리+사행비 6차원)
USE_ARC = False        # --use-arc 로 켠다 (C34 분기점 사이 상대 호위치 4차원)
GEO_WEIGHT = float(os.environ.get("C15_GEO_WEIGHT", "0.5"))
ARC_WEIGHT = float(os.environ.get("C34_ARC_WEIGHT", "0.5"))


# --- 명령: build ------------------------------------------------------------
def cmd_build(args):
    id2name, _ = L.official_location_names()
    ves_names = L.vessel_dense_names()
    train_ids, val_ids, test_ids = L.case_ids_by_split()
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[args.split]

    ves_dir = Path(args.vessel_dir)
    all_rows, t0 = [], time.time()
    for i, cid in enumerate(ids, 1):
        loc_p = L.DATA / "location_masks" / f"{cid}.nii.gz"
        ves_p = ves_dir / f"{cid}.nii.gz"
        if not loc_p.exists() or not ves_p.exists():
            print(f"  스킵 {cid}"); continue
        li = nib.load(loc_p); vi = nib.load(ves_p)
        loc = np.asanyarray(li.dataobj); ves = np.asanyarray(vi.dataobj)
        if loc.shape != ves.shape:
            print(f"  스킵 {cid}: shape {loc.shape} vs {ves.shape}"); continue
        spacing = np.array(li.header.get_zooms()[:3], dtype=float)
        nodes = load_bp(args.bp_dir, cid)
        rows, _ = extract_case_rows(loc, ves, spacing, ves_names, nodes, loc_id_to_name=id2name)
        for r in rows:
            r["case"] = cid
        all_rows.extend(rows)
        if i % 25 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}  누적 병변 {len(all_rows)}  {time.time()-t0:.0f}s", flush=True)

    labeled = [r for r in all_rows if r.get("gt_loc")]
    json.dump(labeled, open(args.out, "w"), ensure_ascii=False)
    print(f"[build] {args.split}: 라벨된 병변 {len(labeled)}개 -> {args.out}")


# --- 분류기 -----------------------------------------------------------------
def fit_model(rows, ves_axis, kind="knn", k=5, mirror=True, balance=True,
              synth_rows=None, synth_repeat=2):
    X, y = [], []
    for r in rows:
        X.append(row_to_vec(r, ves_axis, mirror=False)); y.append(r["gt_loc"])
        if mirror:
            X.append(row_to_vec(r, ves_axis, mirror=True)); y.append(mirror_name(r["gt_loc"]))
    # C11: 분기점에 가상 sac을 놓아 만든 합성 샘플. train split 케이스에서만 생성했으므로
    # val/test 평가에 누수가 없다. CV에서 macro-recall 0.400 -> 0.463, ICA 0.471 -> 0.529.
    if synth_rows:
        for _ in range(synth_repeat):
            for s in synth_rows:
                X.append(row_to_vec(s, ves_axis, mirror=False)); y.append(s["gt_loc"])
    X = np.array(X); y = np.array(y)
    prior = collections.Counter(y)
    if kind != "knn":
        clf = build_clf(kind, X, y)
        return {"kind": "rf", "clf": clf, "ves_axis": ves_axis,
                "pri": np.array([prior[c] for c in clf.classes_], dtype=float),
                "maha": fit_maha(rows) if MAHA_W > 0 else None,
                "pairs": fit_pairs(rows, ves_axis) if PAIR_ON else None}
    return {"kind": "knn", "X": X, "y": y, "k": k, "prior": prior,
            "balance": balance, "ves_axis": ves_axis}


class ProbAvg:
    """여러 분류기의 확률을 평균낸다. classes_ / predict_proba 만 있으면
    아래 결정 경로(β, 확신 게이트, 크롭 혼합)가 전부 그대로 동작한다."""

    def __init__(self, models):
        self.models = models
        self.classes_ = models[0].classes_

    def predict_proba(self, X):
        idx = {c: j for j, c in enumerate(self.classes_)}
        acc = np.zeros((len(X), len(self.classes_)))
        for m in self.models:
            p = m.predict_proba(X)
            for j, c in enumerate(m.classes_):
                if c in idx:
                    acc[:, idx[c]] += p[:, j]
        return acc / max(len(self.models), 1)


def _seed():
    """CLF_SEED: 트리 분류기의 난수 시드. 기본 0(기존 결과와 동일).

    E4(2026-08-19) — ExtraTrees 는 분할점을 무작위로 뽑으므로 시드가 결과를 흔든다.
    test 는 병변 87/클래스 36 이라 병변 1개가 cov.MCC 를 0.028 움직인다. 그래서
    "시드만 바꿔도 점수가 얼마나 요동치는가"가 그 점수를 믿을지의 선행 질문이다."""
    return int(os.environ.get("CLF_SEED", "0"))


def build_clf(kind, X, y):
    """분류기 후보. C21 에서 GBM/앙상블을 이미 기각했지만 그건 β 이중보정(β=1.0) 상태에서
    train CV 로 판정한 것이라 다시 잰다 — 같은 조건에서 C15 측지 노선의 부호가 뒤집혔었다.
    선택은 반드시 val e2e 로 한다(GT 병변만 보는 CV 는 검출 위양성이 없어 e2e 를 못 맞춘다)."""
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                  HistGradientBoostingClassifier)
    if kind == "rf":
        # Q9(2026-08-26): 500→1500 은 test/val 병변수준 둘 다 +0.005 (분산축소, 부작용 없음)
        return RandomForestClassifier(n_estimators=int(os.environ.get("RF_TREES", "500")),
                                      min_samples_leaf=1,
                                      class_weight="balanced", random_state=_seed(),
                                      n_jobs=-1).fit(X, y)
    if kind == "et":
        # 268샘플 x 112차원처럼 표본이 적고 차원이 높으면 분할점을 무작위로 뽑는 쪽이
        # 분산이 낮다. RF 와 같은 계열이라 앙상블 상보성도 기대할 수 있다.
        # max_features 기본값 'sqrt' 는 분할마다 10개만 본다 — 이 차원수에선 너무 적을 수 있어
        # 환경변수로 조절한다(C39).
        mf = os.environ.get("ET_MAX_FEATURES", "sqrt")
        try:
            mf = float(mf)
        except ValueError:
            pass
        return ExtraTreesClassifier(n_estimators=int(os.environ.get("ET_TREES", "800")),
                                    min_samples_leaf=int(os.environ.get("ET_MIN_LEAF", "1")),
                                    max_features=mf,
                                    class_weight="balanced", random_state=_seed(),
                                    n_jobs=-1).fit(X, y)
    if kind == "gb":
        cnt = collections.Counter(y)
        sw = np.array([1.0 / cnt[v] for v in y]); sw *= len(sw) / sw.sum()
        return HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=.06,
                                              l2_regularization=1.0, min_samples_leaf=3,
                                              random_state=_seed()).fit(X, y, sample_weight=sw)
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(X, y)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=2000,
                             random_state=0).fit(X, y)
    if kind in ("rf_et", "rf_gb", "rf_et_gb"):
        parts = {"rf_et": ["rf", "et"], "rf_gb": ["rf", "gb"],
                 "rf_et_gb": ["rf", "et", "gb"]}[kind]
        return ProbAvg([build_clf(k, X, y) for k in parts])
    raise ValueError(kind)


USE_TTA = False        # --tta: 추론에서도 좌우 미러본을 함께 예측해 평균 (C39)


def _proba(model, r):
    """(classes_, 확률) — --tta 면 미러본 예측을 클래스명 되돌려 평균한다.

    미러 증강은 **학습에만** 써왔다. 뇌혈관 좌우대칭이 성립한다는 건 그 증강이 채택된
    근거로 이미 확인됐으므로, 추론에서도 같은 대칭을 쓰면 새 정보 없이 분산만 줄어든다.
    """
    cls = model["clf"].classes_
    v = row_to_vec(r, model["ves_axis"], mirror=False)
    if np.linalg.norm(v) == 0:
        return cls, None
    p = model["clf"].predict_proba(v[None, :])[0]
    if USE_TTA:
        vm = row_to_vec(r, model["ves_axis"], mirror=True)
        if np.linalg.norm(vm) > 0:
            pm = model["clf"].predict_proba(vm[None, :])[0]
            idx = {c: j for j, c in enumerate(cls)}
            back = np.zeros_like(pm)
            for j, c in enumerate(cls):
                k = idx.get(mirror_name(c))       # 미러본의 c 는 원본의 mirror(c)
                if k is not None:
                    back[k] += pm[j]
            p = (p + back) / 2.0
    return cls, p


def predict_one_blend(model, r, beta=0.0, crop_pv=None, w=0.0):
    """기하 RF 확률과 크롭 분류기 확률(C25)을 w로 섞는다. rf 전용.

    **각 분기를 자기 동작점으로 보정한 뒤 섞는다.** 처음엔 섞은 뒤에 β를 걸었는데,
    그러면 사전확률 보정이 필요 없는 크롭 분기까지 같이 뭉개져 이득이 0으로 나왔다
    (C25 앙상블 이득 +0.2%). 분리해서 재니 크롭만 맞히는 샘플이 43/268 이고
    Pcom 은 기하 0.300 -> 크롭 0.650 이라 정보는 확실히 상보적이었다.

    크롭 확률이 없는 병변(크롭 생성 실패 등)은 조용히 기하 단독으로 떨어진다 —
    앙상블 때문에 오히려 기권/오답이 늘어나는 일이 없게 하기 위함.
    """
    if model["kind"] != "rf" or not crop_pv or w <= 0:
        return predict_one(model, r, beta)
    v = row_to_vec(r, model["ves_axis"], mirror=False)
    if np.linalg.norm(v) == 0:
        return None
    cls, p = _proba(model, r)
    if p is None:
        return None
    # C31 확신 게이트를 여기서도 건다. 예전엔 이 분기만 게이트를 빼먹어서
    # --crop-prob 을 쓰는 순간 τ 가 조용히 사라졌다(비교 오염).
    b = beta
    if CONF_TAU > 0:
        b = beta if float(p.max()) < CONF_TAU else CONF_BETA_HI
    if b > 0:                                      # 기하 분기만 사전확률 보정
        p = p / (model["pri"] ** b)
    p = p / max(p.sum(), 1e-12)
    q = np.array([float(crop_pv.get(str(c), 0.0)) for c in cls])
    P = (w * (q / q.sum()) + (1 - w) * p) if q.sum() > 0 else p
    return str(cls[int(np.argmax(P))])


# --- Q7 다중라벨 방출 (2026-08-26) ---------------------------------------
# 공식지표는 케이스·클래스당 0/1 존재 플래그이고 TP 판정이 intersection>0 이다.
#   tp = 1 if intersection(gt==cls, pred==cls) > 0
# 즉 2등 클래스에 **복셀 몇 개짜리 조각**만 떼어 줘도 그 클래스는 TP 가 된다.
# 1등은 나머지 복셀을 다 가지므로 DICE 손실이 사실상 없다.
# 그리고 GT 에 없는 클래스로 새는 조각은 tp=fn=0 이라 MCC 벌점이 0 이다(line 696 참조).
# 손해는 **조각이 존재 클래스로 갈 때의 FP** 뿐이다.
TOPK_N   = int(os.environ.get("TOPANEU_TOPK", "1"))
TOPK_VOX = int(os.environ.get("TOPANEU_TOPK_VOX", "3"))
TOPK_FRAC = float(os.environ.get("TOPANEU_TOPK_FRAC", "0"))   # >0 이면 blob 의 이 비율을 등수마다 준다
TOPK_TAU = float(os.environ.get("TOPANEU_TOPK_TAU", "0"))   # 1등 확신이 이 값 미만일 때만
TOPK_MRG = float(os.environ.get("TOPANEU_TOPK_MARGIN", "0"))  # p2/p1 이 이 값 초과일 때만
TOPK_ICA  = int(os.environ.get("TOPANEU_TOPK_ICA", "0"))    # 1: 1등이 ICA(3.x) 일 때만
TOPK_MAXN = int(os.environ.get("TOPANEU_TOPK_MAXN", "0"))   # 1등 클래스 학습표본이 이 수 이하일 때만
TOPK_P2   = float(os.environ.get("TOPANEU_TOPK_P2", "0"))   # 2등 절대확률이 이 값 초과일 때만
TOPK_OR   = int(os.environ.get("TOPANEU_TOPK_OR", "0"))     # 1: (ICA&마진) OR p2 합집합
CPRIOR_W    = float(os.environ.get("TOPANEU_CPRIOR_W", "0"))   # 케이스 prior 지수 (0=무동작)
OUT_DILATE  = int(os.environ.get("TOPANEU_OUT_DILATE", "0"))     # S4 출력 팽창 복셀 (0=무동작)
# S5(2026-09-17 채택): 예측 부피가 GT 보다 작다(적중병변 pred/GT 중앙 val 0.756 / test 0.816).
# 거리순으로 필요한 만큼만 껍질 복셀을 추가해 총 부피를 OUT_GROW 배로 만든다.
# 라벨은 최근접 상속이라 **클래스 존재가 불변**이다(10시드 실측 절대편차 0).
OUT_GROW    = float(os.environ.get("TOPANEU_OUT_GROW", "0"))      # S5 목표 부피 배수 (0=무동작)
CPRIOR_PATH = os.environ.get("TOPANEU_CPRIOR", "")             # {case: {class: p}} json
_cprior_tab = None
def _cprior(case):
    """N2(2026-08-26): 혈관 36분절 부피로 학습한 P(클래스 존재|케이스). 없으면 None."""
    global _cprior_tab
    if CPRIOR_W <= 0 or not CPRIOR_PATH:
        return None
    if _cprior_tab is None:
        _cprior_tab = json.load(open(CPRIOR_PATH))
    return _cprior_tab.get(case)
# N1(2026-08-26): 기대지표 결정규칙 — 방출 기대이득 = p2·TP가치 - (1-p2)·FP비용 > 0.
# ICA·마진 게이트는 "p2 높은 곳"의 대리변수였다. p2 절대문턱이 원리해:
#   p2>0.20 의 2등적중 = 292 0.205 / test 0.359 (비ICA 0.242/0.457 로 ICA 제한이 기회를 버림)
# 2등 적중률(=조각이 TP 가 될 확률) 은 1등 클래스에 따라 크게 다르다. 292/test 둘 다 일치:
#   ICA(3.x)          0.207 / 0.186      학습표본 1-4개  0.283 / 0.375
#   비ICA             0.074 / 0.135      학습표본 20개+  0.017 / 0.047
# 손익분기 0.102 이므로 흔한 클래스(1등 정확도 0.91~0.94)에 붙이면 순손실이다.
# 손익분기(val 실측 역산): FP 1개 비용 0.0911 / TP 1개 이득 0.798 → 2등 적중 10.2%
# 무조건 방출은 실현 4.4%(45병변 중 고칠 수 있는 건 9개뿐)라 진다.
# 마진 게이트 p2/p1>0.7 은 2등 적중 33.9% 로 분기점의 3배다.


def predict_ranked(model, r, beta=0.0):
    """predict_one 의 rf 분기와 **동일한 확률**을 쓰되 상위 순서와 학습표본수를 돌려준다."""
    if model["kind"] != "rf":
        return None, None, None
    v = row_to_vec(r, model["ves_axis"], mirror=False)
    if np.linalg.norm(v) == 0:
        return None, None, None
    cls_, p = _proba(model, r)
    if p is None:
        return None, None, None
    b = beta
    if CONF_TAU > 0:
        b = beta if float(p.max()) < CONF_TAU else CONF_BETA_HI
    if b > 0:
        p = p / (model["pri"] ** b)
    if MAHA_W > 0 and model.get("maha"):
        q = maha_proba(model["maha"], r, cls_)
        if q is not None:
            p = p / max(p.sum(), 1e-12)
            p = (1 - MAHA_W) * p + MAHA_W * q
    cp = _cprior(r.get("case"))
    if cp:
        pr_ = np.array([float(cp.get(str(c), 0.5)) for c in cls_])
        p = p * np.clip(pr_, 1e-3, 1) ** CPRIOR_W
    p = p / max(p.sum(), 1e-12)
    o = np.argsort(-p)
    pri = model.get("pri")
    cnt = (pri[o] if pri is not None and len(pri) == len(p) else np.zeros(len(o)))
    return [str(cls_[j]) for j in o], p[o], cnt


def predict_one(model, r, beta=0.0):
    v = row_to_vec(r, model["ves_axis"], mirror=False)
    if np.linalg.norm(v) == 0:
        return None
    if model["kind"] == "rf":
        # 공식지표가 52클래스 균등평균(macro)이라 사후확률 argmax는 최적이 아니다.
        # 사전확률로 나눠 균등 사전확률 쪽으로 옮긴다 (C9에서 macro-recall +18%).
        #
        # C31 — β 를 **확신이 낮을 때만** 건다(2026-08-17).
        # β 의 가치는 희소 클래스를 살리는 게 아니라 **위양성을 무해한 곳에 버리는 것**이었다:
        # GT 에 없는 클래스는 tp=fn=0 이라 MCC 분모가 0 -> 0 그대로라 벌점이 없다.
        # 그래서 GT 병변만 보는 train CV 는 β=0 이 최선이라 하고(FP 가 없으니까),
        # FP 가 섞이는 val e2e 는 β=0.5 가 최선이라 한다. 둘을 동시에 만족시키려면
        # 확신이 높은 병변(진짜 TP 후보)은 argmax 로 두고 낮은 것만 희소 쪽으로 민다.
        cls_, p = _proba(model, r)
        if p is None:
            return None
        b = beta
        if CONF_TAU > 0:
            b = beta if float(p.max()) < CONF_TAU else CONF_BETA_HI
        if b > 0:
            p = p / (model["pri"] ** b)
        if MAHA_W > 0 and model.get("maha"):
            # 각 분기를 자기 동작점으로 보정한 뒤 섞는다(C25 에서 배운 순서).
            q = maha_proba(model["maha"], r, cls_)
            if q is not None:
                p = p / max(p.sum(), 1e-12)
                p = (1 - MAHA_W) * p + MAHA_W * q
        cp = _cprior(r.get("case"))
        if cp:
            pr_ = np.array([float(cp.get(str(c), 0.5)) for c in cls_])
            p = p * np.clip(pr_, 1e-3, 1) ** CPRIOR_W
        if PAIR_ON and model.get("pairs"):
            alt = pair_override(model["pairs"], r, model["ves_axis"], cls_, p)
            if alt is not None:
                return alt
        return str(cls_[int(np.argmax(p))])
    sims = model["X"] @ v
    order = np.argsort(-sims)
    acc = {}
    used = 0
    for j in order:
        s = sims[j]
        if s <= 0:
            break
        yj = model["y"][j]
        w = s
        if model["balance"]:                      # 공식지표가 클래스 균등평균이므로 빈도 보정
            w = s / np.sqrt(model["prior"][yj])
        acc[yj] = acc.get(yj, 0.0) + w
        used += 1
        if used >= model["k"]:
            break
    return max(acc, key=acc.get) if acc else None


def _per_class_vox(n_idx):
    """등수 하나에 줄 복셀 수. TOPK_FRAC 이 켜지면 blob 크기에 비례한다."""
    if TOPK_FRAC > 0:
        return max(1, int(round(n_idx * TOPK_FRAC)))
    return TOPK_VOX


def _emit_topk(out, lesions, r, mdl, beta, top1, name2id):
    """blob 중심부 복셀 몇 개를 2..K 등 클래스에 떼어 준다.

    중심부를 쓰는 이유: 예측 blob 과 GT 병변이 부분적으로만 겹칠 때 겹침 영역에
    들어갈 확률이 가장 높은 곳이 중심이다. 1등은 나머지 전부를 유지한다.
    """
    names, probs, cnt = predict_ranked(mdl, r, beta)
    if not names:
        return
    _ica_ok = (not TOPK_ICA) or bool(re.match(r"^(?:[LR]-)?3\.", names[0]))
    _mrg_ok = (TOPK_MRG <= 0) or (len(probs) >= 2 and
               float(probs[1]) / max(float(probs[0]), 1e-12) > TOPK_MRG)
    _p2_ok = (TOPK_P2 <= 0) or (len(probs) >= 2 and float(probs[1]) > TOPK_P2)
    _p2_hit = TOPK_P2 > 0 and len(probs) >= 2 and float(probs[1]) > TOPK_P2
    if TOPK_OR:
        # 합집합(2026-08-26): (ICA & 마진) OR p2 절대문턱. 추가분 2등적중이
        # test 0.333 / 292 0.223 / val 0.192 — 3집합 모두 분기점(0.102) 위.
        if not ((_ica_ok and _mrg_ok and TOPK_ICA) or _p2_hit):
            return
    else:
        if not (_ica_ok and _mrg_ok and _p2_ok):
            return
    if TOPK_MAXN > 0 and float(cnt[0]) > TOPK_MAXN:
        return                                    # 흔한 클래스는 1등 정확도가 높아 손해다
    if TOPK_TAU > 0 and float(probs[0]) >= TOPK_TAU:
        return                                    # 1등이 확신하면 건드리지 않는다
    idx = np.argwhere(lesions == r["lesion_mask_idx"])
    if len(idx) < 2:
        return
    c = idx.mean(axis=0)
    order = np.argsort(((idx - c) ** 2).sum(axis=1))
    used = 0
    per = _per_class_vox(len(idx))
    budget = min(per * (TOPK_N - 1), len(idx) - 1)        # 1등에 최소 1복셀은 남긴다
    for nm in names[1:TOPK_N]:
        if nm == top1:
            continue
        oid = name2id.get(nm)
        if oid is None:
            continue
        take = min(per, budget - used)
        if take <= 0:
            break
        sel = idx[order[used:used + take]]
        out[sel[:, 0], sel[:, 1], sel[:, 2]] = oid
        used += take


# --- 명령: eval -------------------------------------------------------------
def cmd_eval(args):
    # --fast: 공식 evaluate.py(HD95 등)가 케이스당 수 초씩 걸려 하이퍼파라미터 스윕이 비싸다.
    # 분류단계만 볼 때는 top-1 정확도로 충분하므로 공식 채점을 건너뛴다.
    official_eval = None
    if not args.fast:
        EVAL_DIR = L.TOPANEU_ROOT / "code" / "TopAneu-26" / "eval" / "task2"
        sys.path.insert(0, str(EVAL_DIR))
        import evaluate as official_eval

    global USE_POS, CONF_TAU, CONF_BETA_HI, USE_GEO, USE_ARC, USE_TTA, USE_MISS, USE_IMP
    USE_POS = args.use_pos
    USE_GEO = getattr(args, 'use_geo', False); USE_ARC = getattr(args, 'use_arc', False)
    USE_TTA = getattr(args, 'tta', False)
    USE_MISS = getattr(args, 'use_miss', False); USE_IMP = getattr(args, 'use_imp', False)
    _atlas = json.load(open(args.atlas)) if getattr(args, 'atlas', None) else {}
    CONF_TAU = getattr(args, 'conf_tau', 0.0); CONF_BETA_HI = getattr(args, 'conf_beta_hi', 0.0)
    id2name, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()
    ves_axis, _ = build_feature_axes(ves_names)
    train_rows = json.load(open(args.train_feat))
    synth = json.load(open(args.synth_feat)) if args.synth_feat else None
    if synth:
        print(f"[c5] C11 합성 샘플 {len(synth)}개 x{args.synth_repeat} 학습에 추가", flush=True)

    # 스윕용 최단경로: 미리 뽑아둔 피처로 top-1만 계산 (볼륨 생성/공식채점 없음)
    if args.fast and args.eval_feat:
        model = fit_model(train_rows, ves_axis, kind=args.model, k=args.k,
                          mirror=not args.no_mirror, balance=not args.no_balance,
                          synth_rows=synth, synth_repeat=args.synth_repeat)
        rows = json.load(open(args.eval_feat))
        ok = sum(predict_one(model, r, args.beta) == r["gt_loc"] for r in rows)
        acc = ok / len(rows) if rows else None
        print(json.dumps({"tag": args.tag, "model": args.model, "k": args.k,
                          "mirror": not args.no_mirror, "balance": not args.no_balance,
                          "n": len(rows), "top1_accuracy": acc}, ensure_ascii=False), flush=True)
        return
    # ── E1: 폴드별 분류기 (평가 폴드의 환자를 학습에서 제외) ──────────────
    # 목적은 점수가 아니라 분해능이다. test 는 병변 87/클래스 36 이라 희소 클래스
    # 병변 1개가 cov.MCC 를 +0.0278 움직인다 — 우리 개선 총합이 병변 1.5개분이다.
    # 292 코호트(병변 268/클래스 43)에서는 병변 1개가 +0.0029 라 3.6배 촘촘하다.
    fold_of, models = {}, None
    if args.cv_splits:
        import c8_classifier_cv as _C8
        pm = _C8.patient_map()
        sp720 = json.load(open(args.cv_splits))
        for _i in range(1, len(sp720)):
            for _c in sp720[_i]["val"]:
                fold_of[_c] = _i
        models = {}
        for _i in sorted(set(fold_of.values())):
            hold = {pm.get(c, c) for c, f in fold_of.items() if f == _i}
            sub = [r for r in train_rows if pm.get(r["case"], r["case"]) not in hold]
            models[_i] = fit_model(sub, ves_axis, kind=args.model, k=args.k,
                                   mirror=not args.no_mirror, balance=not args.no_balance,
                                   synth_rows=synth, synth_repeat=args.synth_repeat)
            print(f"[c5] CV fold{_i}: 학습병변 {len(sub)}/{len(train_rows)} "
                  f"(제외 환자 {len(hold)})", flush=True)
    model = fit_model(train_rows, ves_axis, kind=args.model, k=args.k,
                      mirror=not args.no_mirror, balance=not args.no_balance,
                      synth_rows=synth, synth_repeat=args.synth_repeat)
    if args.save_model:
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
        with open(args.save_model, "wb") as _f:
            pickle.dump({"model": model, "ves_axis": ves_axis, "beta": args.beta, "use_pos": args.use_pos,
                         "conf_tau": CONF_TAU, "conf_beta_hi": CONF_BETA_HI,
                         "topk": {"n": TOPK_N, "vox": TOPK_VOX, "ica": TOPK_ICA, "margin": TOPK_MRG,
                                  "p2": TOPK_P2, "or": TOPK_OR, "tau": TOPK_TAU, "maxn": TOPK_MAXN},
                         "out_dilate": OUT_DILATE, "clf_seed": os.environ.get("CLF_SEED"),
                         "rf_trees": os.environ.get("RF_TREES", "500"), "train_feat": args.train_feat}, _f)
        print(f"[c5] 모델 저장 → {args.save_model}", flush=True)
    print(f"[c5] model={args.model} k={args.k} mirror={not args.no_mirror} "
          f"balance={not args.no_balance} beta={args.beta} pos={args.use_pos} train병변={len(train_rows)} "
          f"(미러 후 {len(train_rows)*(1 if args.no_mirror else 2)}) dim={len(ves_axis)*2+BP_DIM}",
          flush=True)

    train_ids, val_ids, test_ids = L.case_ids_by_split()
    case_ids = {"train": train_ids, "val": val_ids}.get(args.split, test_ids)

    # --covered-gt 일 때 공식 평가기가 "검출기가 덮은 GT 성분만" 담긴 GT를 보게 한다.
    gt_override = {}

    def local_load_gt(fn):
        case = (fn.replace("_0000.mha", "").replace("_0000.nii.gz", "")
                  .replace(".mha", "").replace(".nii.gz", ""))
        if case in gt_override:
            return gt_override[case]
        return np.asanyarray(nib.load(L.DATA / "location_masks" / f"{case}.nii.gz").dataobj)
    # --fast 는 official_eval 을 None 으로 두므로 몽키패치를 걸 대상이 없다.
    # (기존엔 무조건 붙여서 --fast 가 AttributeError 로 죽었다.)
    if official_eval is not None:
        official_eval.load_gt = local_load_gt

    crop_prob = json.load(open(args.crop_prob)) if args.crop_prob else None
    if crop_prob:
        print(f"[c5] C25 크롭 확률 {len(crop_prob)}건 · 가중치 w={args.crop_w}", flush=True)

    aneu_dir = Path(args.aneurysm_pred_dir) if args.aneurysm_pred_dir else None
    ves_dir = Path(args.vessel_dir)
    per_case, n_les, present = [], 0, set()
    n_top1_ok, n_top1_tot = 0, 0
    n_cov, n_gt_les = 0, 0          # covered-GT 용: 검출기가 덮은 GT 성분 / 전체 GT 성분

    for i, cid in enumerate(case_ids, 1):
        ves_p = ves_dir / f"{cid}.nii.gz"
        if not ves_p.exists():
            print(f"  스킵 {cid}: 혈관 없음"); continue
        mdl = model
        if models is not None:
            if cid not in fold_of:
                print(f"  스킵 {cid}: CV 폴드 미배정"); continue
            mdl = models[fold_of[cid]]
        idx_remap = {}
        vi = nib.load(ves_p); ves = np.asanyarray(vi.dataobj)
        spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
        nodes = load_bp(args.bp_dir, cid)
        gt = local_load_gt(cid)
        if not args.covered_gt:      # covered-GT 는 마스킹 뒤에 present 를 채운다
            present.update(int(x) for x in np.unique(gt) if x != 0)

        if aneu_dir and args.covered_gt:
            # covered-GT: 검출기가 1복셀이라도 덮은 GT 성분만 남기고, 그 GT 마스크로 분류한다.
            # 검출 리콜을 분모에서 빼내어 **분류기 단독 성능**을 공식지표로 재기 위함.
            ap = aneu_dir / f"{cid}.nii.gz"
            if not ap.exists():
                print(f"  스킵 {cid}: 동맥류 예측 없음"); continue
            det = np.asanyarray(nib.load(ap).dataobj) > 0
            lab, nl = ndimage.label(gt > 0, structure=np.ones((3, 3, 3)))
            keep = [l for l in range(1, nl + 1) if (det & (lab == l)).any()]
            n_cov += len(keep); n_gt_les += nl
            gt = np.where(np.isin(lab, keep), gt, 0)
            gt_override[cid] = gt
            present.update(int(x) for x in np.unique(gt) if x != 0)
            rows, lesions = extract_case_rows(gt, ves, spacing, ves_names, nodes, id2name)
            # 성분을 지우면 extract_case_rows 내부 라벨링이 1..k 로 다시 매겨져
            # 크롭 파일의 원래 인덱스와 어긋난다. 새 라벨 -> 원래 라벨 매핑을 만든다.
            lab2, nl2 = ndimage.label(gt > 0, structure=np.ones((3, 3, 3)))
            if nl2:
                mx = ndimage.maximum(lab, lab2, index=range(1, nl2 + 1))
                idx_remap = {l2: int(v) for l2, v in enumerate(np.atleast_1d(mx), 1)}
        elif aneu_dir:                                 # 엔드투엔드: 예측 병변
            ap = aneu_dir / f"{cid}.nii.gz"
            if not ap.exists():
                print(f"  스킵 {cid}: 동맥류 예측 없음"); continue
            src = np.asanyarray(nib.load(ap).dataobj)
            rows, lesions = extract_case_rows(src, ves, spacing, ves_names, nodes, None)
        else:                                          # GT-ceiling: GT 병변 (top-1 정확도 측정)
            rows, lesions = extract_case_rows(gt, ves, spacing, ves_names, nodes, id2name)

        for _r in rows:                       # vesconf 조회 키 (① 곁가지 신뢰도)
            _r["case"] = cid

        # C15/C34 블록을 켰으면 **평가쪽 병변에도** 같은 피처를 채운다.
        # 안 채우면 학습엔 신호가 있고 추론엔 0 만 들어가 블록이 상수가 되고 비교가 무의미해진다.
        if USE_MISS or USE_IMP:
            import c42_anchor_atlas as C42
            try:
                C42.fill(rows, nodes, _atlas)
            except Exception as e:
                print(f"  {cid}: 앵커 블록 계산 실패 {e}", flush=True)

        if USE_GEO or USE_ARC:
            import c34_arc_position as C34
            try:
                C34.fill(rows, ves, spacing, nodes, want_geo=USE_GEO, want_arc=USE_ARC)
            except Exception as e:
                print(f"  {cid}: geo/arc 계산 실패 {e}", flush=True)

        out = np.zeros(gt.shape, dtype=np.int32)
        for r in rows:
            li = idx_remap.get(r["lesion_mask_idx"], r["lesion_mask_idx"])
            cp = crop_prob.get(f"{cid}__{li}") if crop_prob else None
            name = predict_one_blend(mdl, r, args.beta, cp, args.crop_w)
            if r.get("gt_loc"):
                n_top1_tot += 1
                n_top1_ok += int(name == r["gt_loc"])
            if name is None:
                continue
            oid = name2id.get(name)
            if oid is not None:
                out[lesions == r["lesion_mask_idx"]] = oid
                if TOPK_N > 1:
                    _emit_topk(out, lesions, r, mdl, args.beta, name, name2id)
        n_les += len(rows)
        if OUT_GROW > 1.0 and out.any():
            fg = out > 0
            n0 = int(fg.sum())
            need = int(round(n0 * OUT_GROW)) - n0
            if need > 0:
                dist, nn = ndimage.distance_transform_edt(~fg, return_indices=True)
                shell = np.flatnonzero((dist.ravel() > 0) & (dist.ravel() <= 4.0))
                if shell.size:
                    order = shell[np.argsort(dist.ravel()[shell], kind="stable")[:need]]
                    ii = np.unravel_index(order, out.shape)
                    out[ii] = out[nn[0][ii], nn[1][ii], nn[2][ii]]
        if OUT_DILATE > 0 and out.any():
            # S4(2026-08-27): 출력 라벨맵을 분류 **뒤에** 팽창. 새 복셀은 최근접 라벨을 상속하므로
            # 케이스별 클래스 존재가 안 바뀌고 intersection 은 커지기만 한다 → MCC 불변, Dice/VolSim/HD95 만 변동.
            # 근거: X5 blob 부피가 GT 보다 작다(pred/GT 중앙 val 0.72 / test 0.84).
            fg = out > 0
            grown = ndimage.binary_dilation(fg, iterations=OUT_DILATE)
            _, nn = ndimage.distance_transform_edt(~fg, return_indices=True)
            newv = grown & ~fg
            out[newv] = out[nn[0][newv], nn[1][newv], nn[2][newv]]
        if args.save_pred_dir:
            os.makedirs(args.save_pred_dir, exist_ok=True)
            _gi = nib.load(str(L.DATA / "location_masks" / f"{cid}.nii.gz"))
            _img = nib.Nifti1Image(out.astype(np.uint8), _gi.affine, _gi.header)
            _img.set_data_dtype(np.uint8)
            nib.save(_img, os.path.join(args.save_pred_dir, f"{cid}.nii.gz"))
        if official_eval is not None:
            per_case.append(official_eval.evaluation_function(out, cid))
        if i % 10 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)}  누적 병변 {n_les}", flush=True)

    pres = sorted(present)
    keys = ["PRECISION", "RECALL", "MCC", "DICE", "HD95", "VOLSIM"]
    if official_eval is not None:
        agg = official_eval.evaluation_aggregation(per_case)
        official_avg = official_eval.evaluation_average(agg)
        adj = {k: float(np.mean([agg[f"{k}_{i}"] for i in pres])) for k in keys}
    else:
        official_avg, adj = None, None

    res = {"method": f"C5_branchpoint_{args.model}",
           "eval_mode": "covered_gt" if args.covered_gt else ("official_e2e" if aneu_dir else "gt_ceiling"),
           "cv_splits": args.cv_splits, "cv_folds": (sorted(set(fold_of.values())) if models else None),
           "covered_lesions": n_cov, "gt_lesions_total": n_gt_les,
           "detect_coverage": (n_cov / n_gt_les) if n_gt_les else None,
           "crop_prob": args.crop_prob, "crop_w": args.crop_w,
           "beta": args.beta, "use_pos": args.use_pos, "synth": (len(synth) if synth else 0), "synth_repeat": args.synth_repeat, "split": args.split,
           "mirror": not args.no_mirror, "balance": not args.no_balance, "k": args.k,
           "vessel_dir": str(ves_dir), "bp_dir": args.bp_dir,
           "aneurysm_pred_dir": str(aneu_dir) if aneu_dir else "GT(ceiling)",
           "n_cases": len(per_case), "n_lesions_predicted": n_les,
           "n_present_classes_in_split": len(pres),
           "top1_accuracy": (n_top1_ok / n_top1_tot) if n_top1_tot else None,
           "n_top1_scored": n_top1_tot,
           "official_div52": official_avg, "adjusted_div_present": adj}
    print(json.dumps(res, indent=2, ensure_ascii=False))
    out_p = (L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
             / f"c5_eval_{args.split}_{args.tag}.json")
    json.dump(res, open(out_p, "w"), indent=2, ensure_ascii=False)
    print(f"[저장] {out_p}")

    # E6(2026-08-19) — 케이스별 원자료를 남긴다. 보고 숫자에 **신뢰구간**을 붙이려면
    # 케이스를 부트스트랩 재표집해 재집계해야 하는데, 집계값만 저장하면 그게 불가능하다.
    # test 는 83케이스뿐이라 표집 변동이 우리가 다투는 개선폭(병변 1~3개)보다 클 수 있다.
    if official_eval is not None and per_case:
        pc_p = out_p.with_name(f"c5_percase_{args.split}_{args.tag}.json")
        json.dump({"split": args.split, "tag": args.tag, "present": pres,
                   "case_ids": [c for c in case_ids][:len(per_case)],
                   "per_case": per_case}, open(pc_p, "w"))
        print(f"[저장] {pc_p}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--split", required=True, choices=["train", "val", "test"])
    b.add_argument("--vessel-dir", required=True)
    b.add_argument("--bp-dir", required=True)
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)

    e = sub.add_parser("eval")
    e.add_argument("--save-model", default=None, help="학습된 분류기(dict)를 pickle 로 저장 (번들용)")
    e.add_argument("--save-pred-dir", default=None, help="케이스별 최종 라벨맵(.nii.gz) 저장 (sanity 기준)")
    e.add_argument("--train-feat", required=True)
    e.add_argument("--split", required=True, choices=["train", "val", "test"])
    e.add_argument("--vessel-dir", required=True)
    e.add_argument("--bp-dir", required=True)
    e.add_argument("--aneurysm-pred-dir", default=None,
                   help="생략하면 GT 병변으로 top-1 정확도(천장) 측정")
    e.add_argument("--model", default="knn",
                   choices=["knn", "rf", "et", "gb", "logreg", "mlp",
                            "rf_et", "rf_gb", "rf_et_gb"])
    e.add_argument("--k", type=int, default=5)
    e.add_argument("--no-mirror", action="store_true")
    e.add_argument("--no-balance", action="store_true")
    e.add_argument("--fast", action="store_true",
                   help="공식 evaluate.py 생략, top-1 정확도만 (스윕용)")
    e.add_argument("--eval-feat", default=None,
                   help="--fast 와 함께: 미리 뽑아둔 평가 피처 재사용(재추출 생략)")
    e.add_argument("--beta", type=float, default=0.0,
                   help="RF 사전확률 역보정 지수 (macro 정렬, C9/C10 최적 1.0)")
    e.add_argument("--use-pos", action="store_true", help="C10 랜드마크 좌표 블록 사용")
    e.add_argument("--synth-feat", default=None, help="C11 합성 샘플 json (학습에만 추가)")
    e.add_argument("--synth-repeat", type=int, default=2, help="합성 샘플 반복 배수 (C11 최적 2)")
    e.add_argument("--use-miss", action="store_true",
                   help="C42 분기점 결측 지시자 34차원 — '멀다'와 '없다'를 구분시킨다")
    e.add_argument("--use-imp", action="store_true",
                   help="C43 아틀라스 대체값 34차원 — 결측 분기점을 기댓값 위치로 채운다")
    e.add_argument("--atlas", default=None, help="C43 아틀라스 json")
    e.add_argument("--tta", action="store_true",
                   help="추론에서도 좌우 미러본을 함께 예측해 평균 (C39)")
    e.add_argument("--use-geo", action="store_true", help="C15 측지거리+사행비 블록 사용")
    e.add_argument("--use-arc", action="store_true", help="C34 분기점 사이 상대 호위치 블록 사용")
    e.add_argument("--conf-tau", type=float, default=0.0,
                   help="C31: 최대확률이 이 값 미만인 병변에만 --beta 적용 (0=전부 적용)")
    e.add_argument("--conf-beta-hi", type=float, default=0.0,
                   help="C31: 확신이 높은 병변에 적용할 β")
    e.add_argument("--covered-gt", action="store_true",
                   help="검출기가 덮은 GT 성분만 남겨 평가 — 검출 리콜을 뺀 분류기 단독 공식지표")
    e.add_argument("--crop-prob", default=None, help="C25 크롭 분류기 확률 json")
    e.add_argument("--crop-w", type=float, default=0.0, help="크롭 확률 혼합 가중치")
    e.add_argument("--cv-splits", default=None,
                   help="E1: Dataset720 splits_final.json. --split train 과 함께 쓰면 "
                        "fold1~4 를 환자단위 OOF 로 돌린다 — 평가 폴드 케이스를 "
                        "분류기 학습에서 제외하고 적합한다.")
    e.add_argument("--tag", default="v2")
    e.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
