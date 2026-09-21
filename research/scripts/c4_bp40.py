# 자동생성: experiments/D1_newdata/v2a_make40.py · 원본 c4_branchpoint_graph.py md5 e7aa6543fd6d48781b3238c0ad9ca345
"""C4 — 혈관 중심선에서 분기점 그래프 추출.

배경(2026-08-14): 위치클래스 52개 중 21개(병변의 61%)가 "두 혈관이 만나는 지점"으로
정의된다(VA-PICA junction, BA tip, ICA C7-Pcom-junction, Acom complex, M1-M2 junction ...).
36클래스 혈관 분할을 아무리 잘게 쪼개도 이건 표현되지 않으므로, 중심선 위상에서
분기점을 직접 뽑아 위치 판정의 1차 근거로 쓴다.

설계 결정(사용자 2026-08-14): 분기점은 "두 혈관 라벨이 접촉하는 voxel 영역"이 아니라
**중심선 위상**으로 잡는다. 대신 잡음 대응을 후처리에 맡긴다 —
  - 마스크 수준 파편: V5 후처리(close→prune→adj→endpoint)가 이미 처리 (클래스별 min_cc_vox)
  - 중심선 수준 스퍼: 여기서 **가지 길이 기준**으로 잘라낸다.
    컴포넌트 크기 필터로는 스퍼를 못 잡는다 — 본체에 붙어 있어서 별도 컴포넌트가 아니다.
    스퍼 하나 = 가짜 분기점 하나이므로 이 단계가 없으면 위상 방식 자체가 성립하지 않는다.

파이프라인:
  1) 36클래스 마스크 → 이진화 → Lee thinning (챌린지 공식 metric과 동일 관례)
  2) 중심선 voxel에 원 혈관 라벨 되붙임
  3) 스퍼 pruning — endpoint에서 분기 voxel까지 걸어가 길이 < SPUR_MM 이면 제거, 안정될 때까지 반복
  4) 클래스 전이점 추출 — 인접한 중심선 voxel의 클래스가 다른 곳
  5) 전이 voxel을 연결성분으로 묶어 노드화, 노드 반경 JUNCTION_R_MM 안의 클래스 집합을 수집
  6) V5 postproc_params.json 의 neighbors 로 해부학적으로 불가능한 쌍 기각

출력: <out>/<case>.json
  {"case":..., "spacing":[...], "nodes":[{"classes":["R-VA","R-PICA"], "class_ids":[23,29],
    "centroid_vox":[...], "centroid_mm":[...], "n_transition_vox":N, "valid":true}, ...]}

사용:
  python c4_branchpoint_graph.py --vessel-dir <36클래스 예측 폴더> --out <출력폴더> [--cases a,b,c] [--jobs 4]
"""
import argparse, json, os, sys, time
from pathlib import Path
from collections import deque

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

try:
    from skimage.morphology import skeletonize
    def skel3d(b): return skeletonize(b, method="lee").astype(bool)
except Exception:                                            # skimage 구버전
    from skimage.morphology import skeletonize_3d
    def skel3d(b): return skeletonize_3d(b).astype(bool)

TOPANEU_ROOT = Path(os.environ.get("TOPANEU_ROOT", os.path.expanduser("~/topaneu_sblee")))
VES_RAW = TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset800_TopAneuVessel417"
POSTPROC_PARAMS = (TOPANEU_ROOT / "experiments" / "V5_vessel_classweighted_postproc_417"
                   / "postproc_params.json")

SPUR_MM = 2.0          # 이보다 짧은 중심선 곁가지는 잡음으로 보고 제거
JUNCTION_R_MM = 2.0    # 노드 주변 이 반경 안의 클래스를 그 분기점의 구성 혈관으로 본다
                       # (오라클 리포트에서 반경 2mm 이웃집합이 최고 61.0% — 3/5mm는 하락)

ST26 = np.ones((3, 3, 3), dtype=bool)
_OFFS = np.array([(dz, dy, dx)
                  for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dz, dy, dx) != (0, 0, 0)], dtype=np.int8)


def vessel_names():
    """[V2-A] 40클래스 fine 매핑"""
    lab = json.load(open("/home/sblee/TopAneu-26/labeling/vessel_mapping_fine.json"))["labels"]
    return {int(v): k for k, v in lab.items() if int(v) != 0}


def adjacency_table():
    """V5 후처리 파라미터에 이미 들어 있는 클래스별 이웃 목록을 재사용.
    이게 그대로 '어떤 혈관 쌍이 실제로 만날 수 있는가' 테이블이다."""
    if not POSTPROC_PARAMS.exists():
        print(f"  [경고] {POSTPROC_PARAMS} 없음 — 인접성 검증 생략", file=sys.stderr)
        return None
    p = json.load(open(POSTPROC_PARAMS))
    adj = {}
    for k, v in p["classes"].items():
        adj[int(k)] = set(int(n) for n in v["neighbors"])
    # [V2-A] 기존 ICA-C6-C7(4/6)을 C6(4/6)·C7(37/38)·terminus(39/40)로 나눴다.
    # 세 구획 모두 기존 4/6 의 이웃을 물려받고 서로 이웃이다(허용적으로 — 검증은 엉뚱한 노드만 거른다).
    for old, trio in ((4, (4, 37, 39)), (6, (6, 38, 40))):
        base = set(adj.get(old, set()))
        for a in trio:
            adj[a] = set(base) | (set(trio) - {a})
            for nb in base:
                adj.setdefault(nb, set()).add(a)
    return adj


def _neighbor_index(coords, shape):
    """중심선 voxel 좌표 배열 -> 각 voxel의 26-이웃 인덱스 리스트."""
    idx_vol = np.full(shape, -1, dtype=np.int32)
    idx_vol[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(len(coords))
    nbrs = []
    for c in coords:
        loc = c[None, :] + _OFFS
        ok = np.all((loc >= 0) & (loc < np.array(shape)), axis=1)
        loc = loc[ok]
        j = idx_vol[loc[:, 0], loc[:, 1], loc[:, 2]]
        nbrs.append(j[j >= 0])
    return nbrs, idx_vol


def prune_spurs(skel, spacing, spur_mm=SPUR_MM):
    """중심선에서 길이 spur_mm 미만의 곁가지를 제거. 제거로 새 endpoint가 생길 수 있어
    변화가 없을 때까지 반복한다. 반환: (정리된 skel, 제거된 voxel 수)."""
    removed_total = 0
    for _ in range(20):                                  # 안전장치 — 보통 2~4회에 수렴
        coords = np.argwhere(skel)
        if len(coords) == 0:
            break
        nbrs, _ = _neighbor_index(coords, skel.shape)
        deg = np.array([len(n) for n in nbrs])
        ends = np.where(deg == 1)[0]
        if len(ends) == 0:
            break

        drop = set()
        for e in ends:
            path = [e]
            prev, cur = -1, e
            length = 0.0
            while True:
                nxt = [j for j in nbrs[cur] if j != prev]
                if len(nxt) != 1:                        # 분기(>=2) 또는 막다른 길(0)에 도달
                    break
                step = (coords[nxt[0]] - coords[cur]) * spacing
                length += float(np.linalg.norm(step))
                if length >= spur_mm:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
                if deg[cur] >= 3:
                    break
            if length < spur_mm:                         # 분기점에 닿기 전에 끝난 짧은 가지
                drop.update(path)
        if not drop:
            break
        bad = coords[sorted(drop)]
        skel = skel.copy()
        skel[bad[:, 0], bad[:, 1], bad[:, 2]] = False
        removed_total += len(drop)
    return skel, removed_total


def extract_nodes(ves, spacing, adj, ves_names, spur_mm=SPUR_MM, jr_mm=JUNCTION_R_MM):
    binary = ves > 0
    if not binary.any():
        return [], {"n_skel": 0, "n_spur_removed": 0}

    skel = skel3d(binary)
    skel, n_spur = prune_spurs(skel, spacing, spur_mm)

    cls = np.where(skel, ves, 0)                          # 중심선 voxel의 혈관 클래스
    coords = np.argwhere(skel)
    if len(coords) == 0:
        return [], {"n_skel": 0, "n_spur_removed": n_spur}
    nbrs, _ = _neighbor_index(coords, skel.shape)
    cvals = cls[coords[:, 0], coords[:, 1], coords[:, 2]]

    # 4) 클래스 전이 voxel — 인접 중심선 voxel의 클래스가 다른 곳
    is_trans = np.zeros(len(coords), dtype=bool)
    for i, ns in enumerate(nbrs):
        if len(ns) == 0 or cvals[i] == 0:
            continue
        if np.any((cvals[ns] != cvals[i]) & (cvals[ns] > 0)):
            is_trans[i] = True
    if not is_trans.any():
        return [], {"n_skel": int(len(coords)), "n_spur_removed": n_spur}

    # 5) 전이 voxel을 연결성분으로 묶어 노드화
    tvol = np.zeros(skel.shape, dtype=bool)
    tc = coords[is_trans]
    tvol[tc[:, 0], tc[:, 1], tc[:, 2]] = True
    lab, k = ndi.label(tvol, structure=ST26)

    tree = None
    if len(coords) > 0:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords * spacing)

    nodes = []
    for cid in range(1, k + 1):
        mem = np.argwhere(lab == cid)
        cen_vox = mem.mean(axis=0)
        cen_mm = cen_vox * spacing
        # 노드 반경 안의 모든 중심선 voxel의 클래스 집합 = 이 분기점의 구성 혈관
        near = tree.query_ball_point(cen_mm, jr_mm)
        ids = sorted(set(int(x) for x in cvals[near] if x > 0))
        if len(ids) < 2:
            continue
        # 6) 인접성 검증 — 구성 혈관 쌍이 해부학적으로 만날 수 있어야 한다
        valid = True
        if adj is not None:
            valid = all(b in adj.get(a, set()) or a in adj.get(b, set())
                        for i, a in enumerate(ids) for b in ids[i + 1:])
        nodes.append({
            "class_ids": ids,
            "classes": [ves_names.get(i, f"?{i}") for i in ids],
            "centroid_vox": [round(float(x), 2) for x in cen_vox],
            "centroid_mm": [round(float(x), 3) for x in cen_mm],
            "n_transition_vox": int(len(mem)),
            "valid": bool(valid),
        })
    nodes.sort(key=lambda n: -n["n_transition_vox"])
    return nodes, {"n_skel": int(len(coords)), "n_spur_removed": int(n_spur)}


def process_case(ves_path, out_dir, adj, ves_names, spur_mm, jr_mm):
    vi = nib.load(ves_path)
    ves = np.asanyarray(vi.dataobj).astype(np.int16)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    nodes, stats = extract_nodes(ves, spacing, adj, ves_names, spur_mm, jr_mm)
    case = ves_path.name.replace(".nii.gz", "")
    rec = {"case": case, "spacing": [float(x) for x in spacing],
           "spur_mm": spur_mm, "junction_r_mm": jr_mm,
           "n_nodes": len(nodes), "n_valid": sum(n["valid"] for n in nodes),
           **stats, "nodes": nodes}
    if out_dir:
        json.dump(rec, open(Path(out_dir) / f"{case}.json", "w"), indent=1, ensure_ascii=False)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vessel-dir", required=True, help="36클래스 혈관 마스크 폴더(.nii.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=None, help="쉼표구분 case id (없으면 폴더 전체)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--spur-mm", type=float, default=SPUR_MM)
    ap.add_argument("--junction-r-mm", type=float, default=JUNCTION_R_MM)
    args = ap.parse_args()

    ves_dir = Path(args.vessel_dir)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ves_names = vessel_names()
    adj = adjacency_table()

    if args.cases:
        paths = [ves_dir / f"{c}.nii.gz" for c in args.cases.split(",")]
    else:
        paths = sorted(ves_dir.glob("*.nii.gz"))
    if args.limit:
        paths = paths[:args.limit]
    print(f"[c4] {len(paths)}케이스  spur={args.spur_mm}mm  junction_r={args.junction_r_mm}mm", flush=True)

    t0 = time.time()
    for i, p in enumerate(paths, 1):
        if not p.exists():
            print(f"  스킵 {p.name}: 없음"); continue
        rec = process_case(p, out_dir, adj, ves_names, args.spur_mm, args.junction_r_mm)
        print(f"  [{i}/{len(paths)}] {rec['case']}  노드 {rec['n_nodes']}"
              f"(유효 {rec['n_valid']})  skel {rec['n_skel']}  스퍼제거 {rec['n_spur_removed']}"
              f"  {time.time()-t0:.0f}s", flush=True)
    print(f"[c4] 완료 {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
