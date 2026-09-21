"""C34 — 분기점 사이의 상대 호(arc) 위치. C15 측지 노선의 정식화 수정 (2026-08-18).

C15(중심선 측지거리 + 사행비)는 기각됐다(macro-recall 0.400 -> 0.368). 그런데 그때
평가 조건이 지금과 다르다 — β 이중보정(β=1.0)이 걸려 있었고 학습이 268병변이었다.
게다가 핸드오프 문서(C22)에 **정식화 자체가 틀렸다**고 적어뒀다:

    C15 는 랜드마크 3점(BAtip/Rterm/Lterm)까지의 **절대** 측지거리를 썼다.
    옳은 형태는 **분기점 사이의 상대 위치**다 (저쪽 --arc-train 이 채택된 형태).

왜 상대 위치가 맞나 — 오늘 클래스별 진단에서 무너지는 것들이 전부 "같은 혈관을 따라
어디쯤인가"로 갈리는 클래스다:
    3.1 infraclinoid C1-C5  vs  3.3 C6-nonOA        (ICA 를 따라 근위/원위)
    5.1 M1 trunk  vs  5.2 early bif  vs  5.3 M1-M2  (M1 을 따라)
    4.2 A1  vs  4.3 A2  vs  4.4 A3                  (ACA 를 따라)
랜드마크까지의 절대거리는 두개 크기와 혈관 길이에 같이 흔들리지만, **그 혈관을 감싸는
두 분기점 사이에서 몇 % 지점인가**는 스케일 불변이고 구간 정의와 직접 대응한다.

피처 (병변당 4차원, 표본이 311개뿐이라 작게 유지):
    t        = d1 / (d1 + d2)      두 경계 분기점 사이 정규화 위치 [0,1]
    d1/s, d2/s                     각 경계까지 측지거리 (랜드마크 스케일로 정규화)
    (d1+d2)/s                      그 구간의 길이

경계 분기점은 **병변이 올라탄 혈관 클래스를 포함하는** 분기점 노드 중 측지거리 최소 2개로
잡는다. 그래야 "이 혈관을 따라"가 된다 — 아무 분기점이나 쓰면 다른 혈관의 분기점이 잡힌다.

사용:
  python c34_arc_position.py --feat <c10_feat_*.json> --vessel-dir <..> --bp-dir <..> --out <..>
"""
import argparse, collections, json, os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

import d9xx_lib as L
import c5_location_v2 as C5
import c15_geodesic_coords as C15

ARC_DIM = 4


# C15 기각 원인(2026-08-17 규명): Dijkstra 도달 불가를 0.0 으로 인코딩해서
# "랜드마크에 딱 붙어 있음"(거리 0)과 같은 값이 됐다. 측지 슬롯이 하나라도 0인 병변이
# 259개 중 101개(39%), BA tip 만 89건(34%) — 블록 전체가 잡음이 됐다.
# 트리 기반 분류기는 문턱으로 자르므로 음수 센티넬이면 유효값(≥0)과 깨끗이 갈린다.
MISSING = -1.0

# S7(2026-09-02): arc 가 접혀 있었다. 기존 구현은 후보 분기점 거리를 **정렬**해서
#   cand.sort(); d1,d2 = cand[0],cand[1];  t = d1/(d1+d2)
# 를 쓰는데, 정렬 때문에 항상 d1 <= d2 → **t <= 0.5 로 접힌다**(실측 max 0.5000, 초과 0개).
# 그래서 구간의 20% 지점과 80% 지점이 같은 t 를 받고, 근위/원위 구분이 정의상 소거됐다.
# 그런데 3.1↔3.3, 5.1↔5.2↔5.3, 4.2↔4.3 이 전부 근위/원위 문제다.
# 수정: 랜드마크까지의 측지거리로 방향을 갈라 **양쪽에서 각각** 최근접 분기점을 잡는다.
#       t = d_prox/(d_prox+d_dist) 가 [0,1] 전체를 쓰게 된다.
ARC_SIGNED = int(os.environ.get("TOPANEU_ARC_SIGNED", "0"))


def _ref_field(adj, tree, lm):
    """랜드마크 하나를 기준으로 한 측지거리장 — 근위/원위 방향을 정하는 데 쓴다."""
    for k in ("BAtip", "Rterm", "Lterm"):
        if k in lm:
            _, i = tree.query(lm[k])
            return C15.dijkstra(adj, int(i))
    return None


def _arc_signed(d, gref, src, node_idx, node_cls, v, scale):
    """방향을 가른 호위치. 한쪽이 비면 None 을 돌려 호출부가 MISSING 을 넣게 한다."""
    if gref is None or not np.isfinite(gref[src]):
        return None
    pool = [k for k in range(len(node_idx)) if v and v in node_cls[k]]
    if len(pool) < 2:
        pool = list(range(len(node_idx)))
    gl = gref[src]
    prox, dist = [], []
    for k in pool:
        i = node_idx[k]
        if not (np.isfinite(d[i]) and np.isfinite(gref[i])):
            continue
        (prox if gref[i] <= gl else dist).append(float(d[i]))
    if not prox or not dist:
        return None
    d1, d2 = min(prox), min(dist)
    tot = d1 + d2
    return [float(d1 / tot) if tot > 1e-6 else 0.0,
            float(min(d1 / scale, 3.0)), float(min(d2 / scale, 3.0)),
            float(min(tot / scale, 3.0))]


def dominant_vessel(r):
    """병변이 올라탄 혈관 = sac 점유율 최대, 없으면 최근접."""
    ov = r.get("overlap") or {}
    if ov:
        return max(ov.items(), key=lambda kv: kv[1])[0]
    dm = r.get("dist_mm") or {}
    return min(dm.items(), key=lambda kv: kv[1])[0] if dm else None


def fill(rows_of_case, ves, spacing, nodes, want_geo=False, want_arc=True):
    """배열에서 직접 geo/arc 를 채운다 — 평가 경로(c5 eval)가 이걸 쓴다.

    평가할 때도 test 병변에 같은 피처가 있어야 한다. 없으면 학습엔 신호가 있고
    추론엔 0 만 들어가서 블록이 상수가 되고 비교 자체가 무의미해진다.
    골격 그래프는 케이스당 한 번만 만든다(비싸다).
    """
    g = C15.skeleton_graph(ves, spacing)
    if g is None:
        return
    coords, adj, tree = g
    fr = C5.landmark_frame(nodes)
    scale = fr[2] if fr else 100.0
    lm = fr[3] if fr else {}

    node_idx, node_cls = [], []
    for nd in nodes:
        if not nd.get("valid", True):
            continue
        _, i = tree.query(np.array(nd["centroid_mm"]))
        node_idx.append(int(i)); node_cls.append(set(nd["classes"]))

    lm_idx = {}
    for k in ("BAtip", "Rterm", "Lterm"):
        if k in lm:
            _, i = tree.query(lm[k]); lm_idx[k] = int(i)

    _gref = _ref_field(adj, tree, lm) if ARC_SIGNED else None

    for r in rows_of_case:
        cen = np.array(r.get("_cen") or [0, 0, 0], dtype=float)
        _, src = tree.query(cen)
        d = C15.dijkstra(adj, int(src))

        if want_geo:
            out = []
            for k in ("BAtip", "Rterm", "Lterm"):
                if k in lm_idx and np.isfinite(d[lm_idx[k]]):
                    gd = float(d[lm_idx[k]]); eu = float(np.linalg.norm(cen - lm[k]))
                    out += [gd / scale, min(gd / eu, 5.0) if eu > 1e-6 else 0.0]
                else:
                    out += [MISSING, MISSING]
            r["geo"] = out

        if want_arc:
            v = dominant_vessel(r)
            if ARC_SIGNED:
                _a = _arc_signed(d, _gref, int(src), node_idx, node_cls, v, scale)
                r["arc"] = _a if _a is not None else [MISSING] * ARC_DIM
                continue
            cand = [d[node_idx[k]] for k in range(len(node_idx)) if v and v in node_cls[k]]
            cand = [x for x in cand if np.isfinite(x)]
            if len(cand) < 2:
                cand = [x for x in (d[i] for i in node_idx) if np.isfinite(x)]
            if len(cand) < 2:
                r["arc"] = [MISSING] * ARC_DIM
                continue
            cand.sort()
            d1, d2 = cand[0], cand[1]
            tot = d1 + d2
            r["arc"] = [float(d1 / tot) if tot > 1e-6 else 0.0,
                        float(min(d1 / scale, 3.0)), float(min(d2 / scale, 3.0)),
                        float(min(tot / scale, 3.0))]


def case_arc(cid, ves_dir, bp_dir, rows_of_case, want_geo=False):
    """케이스 하나의 병변들에 arc(+geo) 피처를 채운다."""
    vp = Path(ves_dir) / f"{cid}.nii.gz"
    if not vp.exists():
        return
    vi = nib.load(vp)
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    nodes = C5.load_bp(bp_dir, cid)
    fill(rows_of_case, ves, spacing, nodes, want_geo=want_geo, want_arc=True)
    return


def _unused_case_arc(cid, ves_dir, bp_dir, rows_of_case):
    vp = Path(ves_dir) / f"{cid}.nii.gz"
    vi = nib.load(vp)
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    g = C15.skeleton_graph(ves, spacing)
    if g is None:
        return
    coords, adj, tree = g
    nodes = C5.load_bp(bp_dir, cid)
    fr = C5.landmark_frame(nodes)
    scale = fr[2] if fr else 100.0          # 두개 크기 정규화 (없으면 상수)

    # 분기점 노드를 중심선 인덱스로 스냅
    node_idx, node_cls = [], []
    for nd in nodes:
        if not nd.get("valid", True):
            continue
        _, i = tree.query(np.array(nd["centroid_mm"]))
        node_idx.append(int(i)); node_cls.append(set(nd["classes"]))
    if not node_idx:
        return

    _lm = fr[3] if fr else {}
    _gref = _ref_field(adj, tree, _lm) if ARC_SIGNED else None

    for r in rows_of_case:
        cen = np.array(r["_cen"], dtype=float)
        _, src = tree.query(cen)
        d = C15.dijkstra(adj, int(src))
        v = dominant_vessel(r)
        if ARC_SIGNED:
            _a = _arc_signed(d, _gref, int(src), node_idx, node_cls, v, scale)
            r["arc"] = _a if _a is not None else [MISSING] * ARC_DIM
            continue
        # 그 혈관을 포함하는 분기점만 후보로. 없으면 전체 분기점으로 완화한다.
        cand = [d[node_idx[k]] for k in range(len(node_idx)) if v and v in node_cls[k]]
        cand = [x for x in cand if np.isfinite(x)]
        if len(cand) < 2:
            cand = [d[i] for i in node_idx]
            cand = [x for x in cand if np.isfinite(x)]
        if len(cand) < 2:
            r["arc"] = [MISSING] * ARC_DIM
            continue
        cand.sort()
        d1, d2 = cand[0], cand[1]
        tot = d1 + d2
        r["arc"] = [float(d1 / tot) if tot > 1e-6 else 0.0,
                    float(min(d1 / scale, 3.0)), float(min(d2 / scale, 3.0)),
                    float(min(tot / scale, 3.0))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--bp-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-geo", action="store_true", help="C15 측지 6차원도 같이 채운다")
    a = ap.parse_args()

    rows = json.load(open(a.feat))
    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)
    done = 0
    for i, (cid, rs) in enumerate(sorted(by_case.items()), 1):
        try:
            case_arc(cid, a.vessel_dir, a.bp_dir, rs, want_geo=a.with_geo)
        except Exception as e:
            print(f"  스킵 {cid}: {e}", flush=True)
        done += sum(1 for r in rs if r.get("arc"))
        if i % 25 == 0 or i == len(by_case):
            print(f"  {i}/{len(by_case)} 케이스 · arc 채운 병변 {done}", flush=True)
    json.dump(rows, open(a.out, "w"), ensure_ascii=False)
    got = sum(1 for r in rows if r.get("arc"))
    nz = sum(1 for r in rows if r.get("arc") and any(r["arc"]))
    print(f"[c34] {len(rows)}행 중 arc {got} (비영 {nz}) -> {a.out}")


if __name__ == "__main__":
    main()
