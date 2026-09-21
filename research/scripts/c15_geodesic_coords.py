"""C15 — 혈관 트리 위 측지거리(geodesic) 좌표.

문제 재정의: 52클래스는 무순서 집합이 아니라 **혈관 트리 위의 경로**다
(ICA C1-C5 -> C6 -> C7 -> terminus -> M1 -> M2 -> M3). 지금 피처의 거리는 전부
**유클리드**인데, 구간을 가르는 것은 "혈관을 따라 얼마나 갔는가"다.
사행이 심한 ICA siphon에서 유클리드 3mm와 혈관따라 15mm는 완전히 다른 위치이고,
3.2/3.3/3.4/3.6이 뒤섞이는 원인일 수 있다.

여기서는 c4의 중심선을 그래프로 보고 **중심선을 따라간 거리**를 잰다:
  - 병변에서 가장 가까운 중심선 voxel을 시작점으로
  - 26-이웃 그래프에서 Dijkstra (간선 가중치 = 실제 mm 거리)
  - 각 랜드마크(BA tip, R/L ICA terminus)와 주요 분기점까지의 측지거리
유클리드 거리와의 **비율**(tortuosity)도 넣는다 — 사행 정도가 구간의 특징이 된다.

C10 좌표가 "머리 안 어디"라면 C15는 "혈관을 따라 어디"다. 상보적일 것으로 본다.

사용:
  python c15_geodesic_coords.py --feat <c10_feat_train.json> --vessel-dir <..> --bp-dir <..>
"""
import argparse, json, collections, heapq
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8

try:
    from skimage.morphology import skeletonize
    def skel3d(b): return skeletonize(b, method="lee").astype(bool)
except Exception:
    from skimage.morphology import skeletonize_3d
    def skel3d(b): return skeletonize_3d(b).astype(bool)

OFFS = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                 if (a, b, c) != (0, 0, 0)], dtype=int)
GEO_TARGETS = ["BAtip", "Rterm", "Lterm"]
GEO_DIM = 2 * len(GEO_TARGETS)      # 측지거리 3 + 사행비 3


def skeleton_graph(ves, spacing):
    skel = skel3d(ves > 0)
    coords = np.argwhere(skel)
    if len(coords) == 0:
        return None
    idx = {tuple(c): i for i, c in enumerate(coords)}
    adj = [[] for _ in coords]
    for i, c in enumerate(coords):
        for o in OFFS:
            j = idx.get(tuple(c + o))
            if j is not None and j > i:
                w = float(np.linalg.norm(o * spacing))
                adj[i].append((j, w)); adj[j].append((i, w))
    return coords, adj, cKDTree(coords * spacing)


def dijkstra(adj, src):
    d = np.full(len(adj), np.inf)
    d[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        for v, w in adj[u]:
            nd = du + w
            if nd < d[v]:
                d[v] = nd
                heapq.heappush(pq, (nd, v))
    return d


def case_geo(cid, ves_dir, bp_dir, rows_of_case):
    ves_p = Path(ves_dir) / f"{cid}.nii.gz"
    bp_p = Path(bp_dir) / f"{cid}.json"
    if not ves_p.exists() or not bp_p.exists():
        return
    nodes = json.load(open(bp_p))["nodes"]
    fr = C5.landmark_frame(nodes)
    if fr is None:
        return
    _, _, scale, lm = fr
    vi = nib.load(ves_p)
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    g = skeleton_graph(ves, spacing)
    if g is None:
        return
    coords, adj, tree = g

    # 랜드마크마다 한 번씩만 Dijkstra (병변 수보다 랜드마크가 적다)
    dmaps = {}
    for k in GEO_TARGETS:
        src = int(tree.query(lm[k])[1])
        dmaps[k] = dijkstra(adj, src)

    for r in rows_of_case:
        cen = np.array(r["_cen"], dtype=float)
        s = int(tree.query(cen)[1])
        feats = []
        for k in GEO_TARGETS:
            gd = float(dmaps[k][s])
            eu = float(np.linalg.norm(cen - lm[k]))
            feats.append(gd / scale if np.isfinite(gd) else 0.0)          # 정규화 측지거리
            feats.append(min(gd / eu, 5.0) if (np.isfinite(gd) and eu > 1e-6) else 0.0)  # 사행비
        r["geo"] = feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--bp-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    C5.USE_POS = True
    rows = [r for r in json.load(open(args.feat)) if r.get("gt_loc")]
    if not rows[0].get("_cen"):
        raise SystemExit("피처에 _cen(병변 중심)이 없습니다 — c10 피처를 최신 c5로 다시 빌드하세요")
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())

    cache = Path(args.cache) if args.cache else Path(args.feat).parent / "c15_geo_rows.json"
    if cache.exists():
        rows = json.load(open(cache))
        print(f"[c15] 캐시 로드 {len(rows)}")
    else:
        by_case = collections.defaultdict(list)
        for r in rows:
            by_case[r["case"]].append(r)
        for i, (cid, rs) in enumerate(by_case.items(), 1):
            case_geo(cid, args.vessel_dir, args.bp_dir, rs)
            if i % 25 == 0 or i == len(by_case):
                print(f"  {i}/{len(by_case)} 케이스", flush=True)
        json.dump(rows, open(cache, "w"), ensure_ascii=False)
    have = sum(1 for r in rows if r.get("geo"))
    print(f"[c15] geo 보유 {have}/{len(rows)} ({have/len(rows)*100:.1f}%)\n")

    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])

    def vec(r, use_geo, mirror=False):
        v = C5.row_to_vec(r, ves_axis, mirror=mirror)
        if not use_geo:
            return v
        g = r.get("geo") or [0.0] * GEO_DIM
        g = np.array(g, dtype=float)
        if mirror:                       # 좌우 반전 시 R/L terminus 항목 swap
            g = g.copy(); g[[2, 3, 4, 5]] = g[[4, 5, 2, 3]]
        ng = np.linalg.norm(g)
        g = g / ng * 0.5 if ng > 0 else g
        out = np.concatenate([v, g]); n = np.linalg.norm(out)
        return out / n if n > 0 else out

    print(f"{'실험':<24}{'top-1':>8}{'macroRec':>10}{'ICA':>8}")
    for use_geo in (False, True):
        pred = np.empty(len(rows), dtype=object)
        for tr, te in GroupKFold(n_splits=args.folds).split(np.zeros(len(rows)), y, groups):
            X, Y = [], []
            for i in tr:
                X.append(vec(rows[i], use_geo)); Y.append(y[i])
                X.append(vec(rows[i], use_geo, True)); Y.append(C5.mirror_name(y[i]))
            X = np.array(X); Y = np.array(Y)
            clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(X, Y)
            prior = collections.Counter(Y)
            pri = np.array([prior[c] for c in clf.classes_], dtype=float)
            P = clf.predict_proba(np.array([vec(rows[i], use_geo) for i in te])) / pri
            pred[te] = clf.classes_[np.argmax(P, axis=1)]
        t1 = float(np.mean(y == pred))
        mr, _ = C8.macro_recall(y, pred)
        ica = [(a, b) for a, b in zip(y, pred) if C8.group_of(a) == "3"]
        print(f"{('geo 포함' if use_geo else 'geo 없음(기준선)'):<24}{t1:>8.3f}{mr:>10.3f}"
              f"{float(np.mean([a==b for a,b in ica])):>8.3f}")
    print("[c15] 완료")


if __name__ == "__main__":
    main()
