#!/usr/bin/env python3
"""arc_feat.py — ICA 위 호길이(arc-length) 좌표를 만든다.

왜 필요한가
-----------
분류기가 가장 많이 틀리는 곳이 ICA C6/C7 세부분절이다(전체 오답의 40%). 그런데 혈관 라벨은
C6 와 C7 이 `ICA-C6-C7` 한 덩어리라, 세부분절은 곁가지(OA·Pcom·AChA) 까지의 거리로 역추론된다.

RF 가 이걸 못 쓰는 이유는 정보가 없어서가 아니라 **트리가 축 정렬 분할만 하기 때문**이다.
C6 냐 C7 이냐를 가르는 건 `d_OA < d_Pcom` 같은 **두 피처의 비교**인데, 트리는 그런 축을
스스로 만들지 못한다. 표본이 클래스당 2~4개뿐이라 분할을 겹쳐 쌓아 발견할 가능성도 없다.

그래서 비교를 **축 자체로** 바꿔준다:

    t = ICA 중심선을 따라 C1-C5 경계(0) → 종말부(1) 로 정규화한 위치

이 한 축 위에 여섯 분절이 해부학적 순서대로 단조 정렬된다.
   C6-OA → C6-nonOA → C7-Pcom → C7-AChA → C7-nonBranch → terminus
추가로 곁가지 부착점의 t 와의 **부호 있는 차이**를 준다. 이게 RF 가 못 만들던 비교다.

거리는 유클리드가 아니라 **혈관 안쪽 측지거리(geodesic)** 로 잰다. ICA 사이펀은 심하게 굽어
있어서 직선거리로는 C6 와 C7 이 공간적으로 붙어버린다.

없는 곁가지 처리
----------------
분할이 안 된 곁가지는 거리가 inf 가 되는데, 트리에게 inf 는 '아주 멀다' 다. **'없다' 와
'멀다' 는 다르다.** 그래서 존재 여부를 별도 이진 플래그로 뺀다.

산출: 케이스별 npz — ICA 복셀의 mm 좌표와 t 값. 임의 점의 t 는 최근접 조회로 얻는다.
"""
import json, os, sys
import numpy as np, nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
VES = {  # Dataset800 라벨 id
    "R": dict(c15=35, c67=4, OA=33, Pcom=8, AChA=31, M1=5, A1A2=11),
    "L": dict(c15=36, c67=6, OA=34, Pcom=9, AChA=32, M1=7, A1A2=12),
}
BR = ("OA", "Pcom", "AChA")


def _node_mm(nodes, a, b):
    """클래스 a,b 가 만나는 분기점의 mm 좌표. 여러 개면 n_transition_vox 가 가장 큰 것."""
    best, bn = None, -1
    for nd in nodes:
        cs = set(nd["classes"])
        if a in cs and b in cs:
            n = nd.get("n_transition_vox", 1)
            if n > bn:
                bn, best = n, np.array(nd["centroid_mm"], float)
    return best


def arc_field(ves_path, nodes, side):
    """한쪽 ICA 의 (복셀 mm 좌표, t) 와 곁가지 t 를 만든다. 실패하면 None."""
    V = VES[side]
    img = nib.load(ves_path)
    vol = np.asanyarray(img.dataobj)
    mask = (vol == V["c15"]) | (vol == V["c67"])
    if mask.sum() < 50:
        return None

    # 굽은 혈관이라 bbox 로 잘라도 안전하다 — 측지경로가 ICA 밖으로 나갈 수 없으므로.
    sl = ndimage.find_objects(mask.astype(np.uint8))[0]
    sl = tuple(slice(max(0, s.start - 2), s.stop + 2) for s in sl)
    sub = mask[sl]
    off = np.array([s.start for s in sl])

    # 좌표 규약 주의 — 분기점 그래프(c4)와 분류기는 affine 을 쓰지 않고 **복셀인덱스 × 간격**
    # 을 mm 로 쓴다(실측 확인: centroid_vox [143.5,140.5,63.5] · centroid_mm [54.81,53.66,28.58]).
    # affine 으로 역변환하면 ICA 밖 좌표가 나와 전부 실패한다. 반드시 같은 규약을 쓴다.
    zoom = np.array(img.header.get_zooms()[:3], dtype=float)   # mm/voxel

    def to_vox(mm):
        return np.round(np.asarray(mm, float) / zoom).astype(int) - off

    def nearest_on(sub_mask, v):
        idx = np.argwhere(sub_mask)
        if len(idx) == 0:
            return None
        return idx[np.argmin(((idx - v) ** 2).sum(1))]

    nm = lambda a, b: _node_mm(nodes, a, b)
    ica15, ica67 = f"{side}-ICA-C1-C5", f"{side}-ICA-C6-C7"
    start_mm = nm(ica15, ica67)
    end_mm = nm(ica67, f"{side}-M1")
    if end_mm is None:                      # 종말부는 M1 우선, 없으면 A1A2
        end_mm = nm(ica67, f"{side}-A1A2")
    if start_mm is None or end_mm is None:
        return None

    s = nearest_on(sub, to_vox(start_mm))
    e = nearest_on(sub, to_vox(end_mm))
    if s is None or e is None:
        return None

    # 측지거리: 마스크 안은 비용 1, 밖은 통과 불가. sampling 으로 mm 환산.
    cost = np.where(sub, 1.0, np.inf)
    mcp = MCP_Geometric(cost, sampling=tuple(float(z) for z in zoom))
    dist, _ = mcp.find_costs([tuple(s)])
    L = float(dist[tuple(e)])
    if not np.isfinite(L) or L < 5:
        return None

    # 부호: 시작점(C1-C5/C6-C7 경계)에서 **양쪽으로** 거리가 자라므로 그냥 나누면
    # 근위부 C1-C5 와 종말부 너머가 똑같이 큰 양수가 되어 뭉개진다. 라벨로 부호를 준다.
    #   C1-C5 → 음수 (근위) · C6-C7 → 양수, 종말부가 +1
    vox = np.argwhere(sub & np.isfinite(dist))
    t = dist[tuple(vox.T)] / L
    is15 = vol[sl][tuple(vox.T)] == V["c15"]
    t = np.where(is15, -t, t)
    mm = (vox + off) * zoom

    # 곁가지 부착점의 t — 같은 측지장에서 읽는다
    bt = {}
    for b in BR:
        p = nm(ica67, f"{side}-{b}")
        if p is None:
            bt[b] = None; continue
        v = nearest_on(sub, to_vox(p))
        d = dist[tuple(v)] if v is not None else np.inf
        bt[b] = float(d / L) if np.isfinite(d) else None
    return dict(mm=mm.astype(np.float32), t=t.astype(np.float32),
                branch_t={k: v for k, v in bt.items()}, length_mm=L)


def build_case(case, ves_dir, bp_dir, out_dir):
    vp = os.path.join(ves_dir, f"{case}.nii.gz")
    bp = os.path.join(bp_dir, f"{case}.json")
    op = os.path.join(out_dir, f"{case}.npz")
    if os.path.exists(op) or not (os.path.exists(vp) and os.path.exists(bp)):
        return case, "skip"
    nodes = json.load(open(bp)).get("nodes", [])
    packs, meta = {}, {}
    for side in ("R", "L"):
        try:
            f = arc_field(vp, nodes, side)
        except Exception as ex:
            f = None; meta[f"{side}_err"] = f"{type(ex).__name__}: {ex}"
        if f is None:
            continue
        packs[f"{side}_mm"] = f["mm"]; packs[f"{side}_t"] = f["t"]
        meta[side] = dict(branch_t=f["branch_t"], length_mm=f["length_mm"])
    tmp = op + ".tmp"                       # savez_compressed 는 .npz 를 덧붙인다
    np.savez_compressed(tmp, meta=json.dumps(meta, ensure_ascii=False), **packs)
    os.replace(tmp + ".npz", op)
    return case, ("ok:" + ",".join(sorted(k for k in meta if k in ("R", "L"))) or "empty")


ARC_DIM = 8   # t, (t-t_OA), (t-t_Pcom), (t-t_AChA), 존재 3개, ICA 여부 1개


def lookup(pack, centroid_mm, max_mm=12.0):
    """병변 중심의 ICA 호길이 피처. ICA 근처가 아니면 전부 0 + is_ica=0."""
    out = np.zeros(ARC_DIM, dtype=np.float32)
    if pack is None:
        return out
    best = None
    for side in ("R", "L"):
        if f"{side}_mm" not in pack["arrays"]:
            continue
        tree = pack["trees"][side]
        d, i = tree.query(centroid_mm)
        if d <= max_mm and (best is None or d < best[0]):
            best = (d, side, float(pack["arrays"][f"{side}_t"][i]))
    if best is None:
        return out
    _, side, t = best
    bt = pack["meta"].get(side, {}).get("branch_t", {})
    out[0] = t
    for k, b in enumerate(BR):
        v = bt.get(b)
        if v is None:
            out[1 + k] = 0.0; out[4 + k] = 0.0
        else:
            out[1 + k] = t - v; out[4 + k] = 1.0
    out[7] = 1.0
    return out


def load_pack(npz_path):
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != "meta"}
    meta = json.loads(str(z["meta"]))
    trees = {s: cKDTree(arrays[f"{s}_mm"]) for s in ("R", "L") if f"{s}_mm" in arrays}
    return dict(arrays=arrays, meta=meta, trees=trees)


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--ves-dir"); ap.add_argument("--bp-dir"); ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
    ids = S["splits"][a.split]
    if a.limit:
        ids = ids[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    ok = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build_case, c, a.ves_dir, a.bp_dir, a.out) for c in ids]
        for f in futs:
            c, st = f.result()
            if st.startswith("ok"):
                ok += 1
            if not st.startswith("ok") and st != "skip":
                print(f"  {c}: {st}", flush=True)
    print(f"[arc] {a.split} {ok}/{len(ids)} 생성 → {a.out}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
