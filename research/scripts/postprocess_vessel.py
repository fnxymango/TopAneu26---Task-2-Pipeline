#!/usr/bin/env python3
"""혈관 seg 후처리 — GT 통계 기반 (전역 largest-CC 금지, 클래스별 처리).

TopBrain/TopCoW 상위팀 레시피(docs/topbrain_top_methods_recipe.md §5):
  "전역 largest-CC 금지(라벨된 혈관 삭제됨). 클래스별 largest-CC + 인접성 강제."
단, 우리 GT 통계상 M3/M2/SCA/PICA 등은 정상적으로 여러 조각이라
클래스별 'largest만 남기기'는 오히려 파괴적 → 조각 크기 임계값을 GT에서 학습해 사용.

4단계 endpoint 재연결(2026-08-10 사용자 지시, TopCoW/CLAIM 계열 baseline 방식 —
"Circle of Willis Centerline Graphs" 논문: U-Net skeleton -> A* 재연결 -> graph)을
클래스별 skeleton-endpoint 매칭 + 최단연결로 축소 구현. 원논문은 raw intensity를
A* cost에 같이 쓰지만, 여기선 라벨 볼륨만 갖고 있어 순수 기하(거리) 기반 버전.

단계:
  1) close    : 클래스별 형태학적 closing (같은 클래스 조각 사이 ≤2*r mm 간극만 연결)
  2) prune    : GT에서 학습한 최소 조각 크기 미만 제거
  3) adjacency: GT에서 학습한 인접 가능 클래스와 안 붙은 고아 조각 제거
  4) endpoint : 같은 클래스의 남은 조각들 사이, skeleton 끝점(26-neighbor 1개)이
                max_gap_mm 이내면 최단경로(narrow A*, 다른 클래스 voxel 회피)로 연결

사용:
  postprocess_vessel.py fit                      # GT → postproc_params.json
  postprocess_vessel.py apply <in_dir> <out_dir> # 예측에 후처리 적용
  postprocess_vessel.py eval <pred_dir> [...]    # GT 대비 Dice/clDice/조각수 채점
"""
import sys, os, json, glob, heapq
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

TOPANEU_ROOT = os.environ.get("TOPANEU_ROOT", os.path.expanduser("~/topaneu_sblee"))
R = f'{TOPANEU_ROOT}/nnunet'
SCRIPTS = f'{TOPANEU_ROOT}/code/sblee/nnunet/scripts'
DS = f'{R}/nnUNet_preprocessed/Dataset800_TopAneuVessel417'
GT = f'{DS}/gt_segmentations'
PARAMS = f'{SCRIPTS}/postproc_params.json'
NAME = {v: k for k, v in json.load(open(f'{R}/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json'))['labels'].items() if v != 0}
NCLS = 36
ST = np.ones((3, 3, 3))           # 26-connectivity
DEFAULT_MAX_GAP_MM = 3.0          # endpoint 재연결 최대 간극 (보수적 기본값 — 오탐 연결 방지)
CLOSE_MM = 1.0                    # 반경 1mm closing → 최대 2mm 간극 연결

try:
    from skimage.morphology import skeletonize
    def skel3d(b): return skeletonize(b, method="lee").astype(bool)
except Exception:
    from skimage.morphology import skeletonize_3d
    def skel3d(b): return skeletonize_3d(b).astype(bool)


def split_ids(which):
    return json.load(open(f'{DS}/splits_final.json'))[0][which]


# ---------------------------------------------------------------- fit
def fit():
    ids = split_ids('train')
    sizes = {c: [] for c in range(1, NCLS + 1)}
    adj = {c: {} for c in range(1, NCLS + 1)}
    for n, cid in enumerate(ids, 1):
        g = sitk.GetArrayFromImage(sitk.ReadImage(f'{GT}/{cid}.nii.gz')).astype(np.int16)
        objs = ndi.find_objects(g)
        for c in range(1, NCLS + 1):
            sl = objs[c - 1]
            if sl is None:
                continue
            pad = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
            sub = g[pad]
            m = sub == c
            lab, k = ndi.label(m, structure=ST)
            sizes[c] += np.bincount(lab.ravel())[1:].tolist()
            # 인접: 이 클래스를 1voxel 부풀렸을 때 닿는 다른 라벨
            touch = sub[ndi.binary_dilation(m, ST) & ~m]
            for t, cnt in zip(*np.unique(touch, return_counts=True)):
                if t > 0:
                    adj[c][int(t)] = adj[c].get(int(t), 0) + int(cnt)
        if n % 20 == 0:
            print(f"  ...{n}/{len(ids)}")

    p = {'close_mm': CLOSE_MM, 'classes': {}}
    for c in range(1, NCLS + 1):
        s = np.array(sizes[c])
        # GT에 실제로 존재하는 조각 크기의 5%분위 → 이보다 작으면 가짜로 간주
        min_cc = int(max(5, np.percentile(s, 5))) if len(s) else 5
        # 전체 인접 접촉의 1% 이상 차지하는 클래스만 유효 이웃으로 채택(라벨 경계 잡음 제거)
        tot = sum(adj[c].values()) or 1
        nb = sorted([k for k, v in adj[c].items() if v / tot >= 0.01])
        p['classes'][str(c)] = {'name': NAME[c], 'n_cc_gt': len(s),
                                'min_cc_vox': min_cc,
                                'median_cc_vox': int(np.median(s)) if len(s) else 0,
                                'neighbors': nb, 'neighbor_names': [NAME[i] for i in nb]}
    json.dump(p, open(PARAMS, 'w'), indent=2)
    print(f"[fit] {PARAMS}")
    print(f"{'class':<14}{'GT조각수':>8}{'조각중앙':>9}{'최소조각':>9}  유효이웃")
    for c in range(1, NCLS + 1):
        d = p['classes'][str(c)]
        print(f"{d['name']:<14}{d['n_cc_gt']:>8}{d['median_cc_vox']:>9}{d['min_cc_vox']:>9}"
              f"  {','.join(d['neighbor_names'][:5])}")


# ---------------------------------------------------------------- apply
def close_class(m, spacing, r):
    """이방성 voxel을 고려한 mm 단위 closing (거리변환 기반)."""
    dil = ndi.distance_transform_edt(~m, sampling=spacing) <= r
    return ndi.distance_transform_edt(dil, sampling=spacing) > r


def _skeleton_endpoints(skel):
    """skeleton voxel 중 26-neighbor가 정확히 1개인 점(끝점) 좌표 배열."""
    nbr = ndi.convolve(skel.astype(np.uint8), ST, mode='constant') - skel.astype(np.uint8)
    return np.argwhere(skel & (nbr == 1))


def _astar_path(other_label, blocked_val, start, goal, spacing):
    """start->goal 최단경로. 이미 다른 클래스(blocked_val 중 자기 자신 제외)가 있는
    voxel은 큰 페널티를 줘서 되도록 피하되(다른 혈관 관통 방지), 완전 차단은 아님
    (막다른 길 방지). cost = 이동거리(mm) + 다른클래스 penalty."""
    shape = other_label.shape
    start, goal = tuple(start), tuple(goal)
    PEN = 50.0
    nbrs = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) != (0, 0, 0)]
    nbr_cost = [np.sqrt(((np.array(d) * spacing) ** 2).sum()) for d in nbrs]

    def h(p):
        return np.sqrt((((np.array(p) - np.array(goal)) * spacing) ** 2).sum())

    openq = [(h(start), 0.0, start)]
    gscore = {start: 0.0}
    came = {}
    visited = set()
    max_expand = 20000  # 안전장치 — 국소 탐색이라 이 정도면 충분히 큼
    n_expand = 0
    while openq and n_expand < max_expand:
        _, gc, cur = heapq.heappop(openq)
        if cur in visited:
            continue
        visited.add(cur)
        n_expand += 1
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for (dz, dy, dx), mc in zip(nbrs, nbr_cost):
            nb = (cur[0] + dz, cur[1] + dy, cur[2] + dx)
            if not (0 <= nb[0] < shape[0] and 0 <= nb[1] < shape[1] and 0 <= nb[2] < shape[2]):
                continue
            step = mc
            v = other_label[nb]
            if v != 0 and v != blocked_val:
                step += PEN
            ng = gc + step
            if ng < gscore.get(nb, np.inf):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(openq, (ng + h(nb), ng, nb))
    return None  # 탐색 실패(막힘/너무 멂) -> 연결 포기


def reconnect_endpoints(m, other_label_full, pad_offset, spacing, max_gap_mm, c):
    """같은 클래스(m, bool) 내 서로 다른 조각의 skeleton 끝점을 A*로 재연결.
    other_label_full/pad_offset: 원본 볼륨 좌표계에서 다른 클래스 배치를 보기 위함
    (패딩된 로컬 sub-array 좌표 <-> 전역 좌표 변환용)."""
    lab, k = ndi.label(m, structure=ST)
    if k < 2:
        return m
    skel = skel3d(m)
    endpoints = _skeleton_endpoints(skel)
    if len(endpoints) < 2:
        return m
    ep_lab = lab[tuple(endpoints.T)]

    parent = {i: i for i in range(1, k + 1)}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    pairs = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if ep_lab[i] == ep_lab[j]:
                continue
            d_mm = np.sqrt((((endpoints[i] - endpoints[j]) * spacing) ** 2).sum())
            if d_mm <= max_gap_mm:
                pairs.append((d_mm, i, j))
    pairs.sort(key=lambda t: t[0])

    out = m.copy()
    for d_mm, i, j in pairs:
        ci, cj = find(ep_lab[i]), find(ep_lab[j])
        if ci == cj:
            continue
        # 로컬(sub-array) 좌표로 A* — other_label_full은 sub-array와 같은 크기로 전달됨
        path = _astar_path(other_label_full, c, endpoints[i], endpoints[j], spacing)
        if path is None:
            continue
        for v in path:
            out[v] = True
        parent[ci] = cj
    return out


def postprocess(seg, spacing, p, stats=None, steps=('close', 'prune', 'adj', 'endpoint'),
                 close_mm=None, max_gap_mm=None):
    out = np.zeros_like(seg)
    objs = ndi.find_objects(seg)
    for c in range(1, NCLS + 1):
        sl = objs[c - 1]
        if sl is None:
            continue
        cp = p['classes'][str(c)]
        marg = 4
        pad = tuple(slice(max(s.start - marg, 0), min(s.stop + marg, seg.shape[i]))
                    for i, s in enumerate(sl))
        sub_seg = seg[pad]
        m = sub_seg == c

        # 1) closing — 같은 클래스 조각 사이 작은 간극만 메움
        if 'close' in steps:
            r = p['close_mm'] if close_mm is None else close_mm
            m2 = close_class(m, spacing, r)
            m2 = m | (m2 & (sub_seg == 0))   # 다른 클래스 침범 금지
        else:
            m2 = m

        # 2) prune — GT에서 학습한 최소 조각 크기 미만 제거.
        # ⚠ GT에서 늘 한 덩어리인 클래스(ICA 등)는 min_cc_vox가 수천이라 그대로 쓰면
        #   예측이 쪼개졌을 때 진짜 혈관 조각까지 지운다. 본체의 5%로 상한을 건다.
        lab, k = ndi.label(m2, structure=ST)
        if k == 0:
            continue
        cnt = np.bincount(lab.ravel())[1:]
        if 'prune' in steps:
            thr = max(5, min(cp['min_cc_vox'], 0.05 * cnt.max()))
            keep = np.where(cnt >= thr)[0] + 1
            if len(keep) == 0:                   # 전부 작으면 최대 조각만 살림
                keep = np.array([int(cnt.argmax()) + 1])
        else:
            keep = np.arange(1, k + 1)

        # 3) adjacency — 유효 이웃과 안 붙은 고아 조각 제거(최대 조각은 보존)
        nb = set(cp['neighbors'])
        biggest = int(cnt.argmax()) + 1
        final = np.zeros_like(m2)
        for j in keep:
            cm = lab == j
            if 'adj' in steps and j != biggest and nb:
                touch = set(int(t) for t in np.unique(sub_seg[ndi.binary_dilation(cm, ST) & ~cm]))
                touch.discard(0); touch.discard(c)
                if touch and not (touch & nb):
                    if stats is not None: stats['adj_removed'] += 1
                    continue
            final |= cm
        if stats is not None:
            stats['cc_removed'] += k - int(ndi.label(final, structure=ST)[1])

        # 4) endpoint — 남은 조각들 사이 skeleton 끝점을 A*로 재연결(같은 클래스 내에서만)
        if 'endpoint' in steps and final.any():
            n_before = ndi.label(final, structure=ST)[1]
            gap = p.get('max_gap_mm', DEFAULT_MAX_GAP_MM) if max_gap_mm is None else max_gap_mm
            final = reconnect_endpoints(final, sub_seg, pad, spacing, gap, c)
            if stats is not None:
                n_after = ndi.label(final, structure=ST)[1]
                stats['endpoint_bridged'] = stats.get('endpoint_bridged', 0) + (n_before - n_after)

        out[pad][final & (out[pad] == 0)] = c
    return out


def _apply_one(args):
    f, out_dir, steps, close_mm, max_gap_mm, p = args
    cid = os.path.basename(f)[:-7]
    img = sitk.ReadImage(f)
    seg = sitk.GetArrayFromImage(img).astype(np.int16)
    sp = np.array(img.GetSpacing())[::-1]              # z,y,x mm
    st = {'cc_removed': 0, 'adj_removed': 0, 'endpoint_bridged': 0}
    out = postprocess(seg, sp, p, st, steps, close_mm, max_gap_mm)
    o = sitk.GetImageFromArray(out); o.CopyInformation(img)
    sitk.WriteImage(o, f'{out_dir}/{cid}.nii.gz')
    b0, b1 = seg > 0, out > 0
    n0 = ndi.label(b0, structure=ST)[1]; n1 = ndi.label(b1, structure=ST)[1]
    return (f"  {cid}: vox {int(b0.sum()):>7}→{int(b1.sum()):>7}  덩어리 {n0:>3}→{n1:>3}"
            f"  (조각제거 {st['cc_removed']}, 인접위반제거 {st['adj_removed']}, "
            f"endpoint연결 {st['endpoint_bridged']})")


def apply(in_dir, out_dir, steps=('close', 'prune', 'adj', 'endpoint'), close_mm=None, max_gap_mm=None):
    p = json.load(open(PARAMS))
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f'{in_dir}/*.nii.gz'))
    jobs = [(f, out_dir, steps, close_mm, max_gap_mm, p) for f in files]
    nproc = max(1, min(8, len(jobs), (os.cpu_count() or 4) // 2))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        for line in ex.map(_apply_one, jobs, chunksize=1):
            print(line)
    print(f"[apply] {out_dir}  ({len(files)}케이스, 프로세스 {nproc}개)")


# ---------------------------------------------------------------- eval
def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def cldice(p, g):
    sp, sg = skel3d(p), skel3d(g)
    tprec = (sp & g).sum() / sp.sum() if sp.sum() else 0.0
    tsens = (sg & p).sum() / sg.sum() if sg.sum() else 0.0
    return 0.0 if tprec + tsens == 0 else 2 * tprec * tsens / (tprec + tsens)


def evaluate(pred_dir, tag=''):
    ids = split_ids('val')
    per = {c: [] for c in range(1, NCLS + 1)}
    cld, ncc = [], []
    for cid in ids:
        g = sitk.GetArrayFromImage(sitk.ReadImage(f'{GT}/{cid}.nii.gz')).astype(np.int16)
        p = sitk.GetArrayFromImage(sitk.ReadImage(f'{pred_dir}/{cid}.nii.gz')).astype(np.int16)
        for c in range(1, NCLS + 1):
            gm = g == c
            if gm.sum() == 0:
                continue
            per[c].append(dice(p == c, gm))
        cld.append(cldice(p > 0, g > 0))
        ncc.append(ndi.label(p > 0, structure=ST)[1])
    res = {NAME[c]: round(float(np.mean(v)), 4) for c, v in per.items() if v}
    mean = float(np.mean(list(res.values())))
    dead = [k for k, v in res.items() if v == 0]
    alive = [v for v in res.values() if v > 0]
    print(f"\n[eval{tag}] {pred_dir}")
    print(f"  mean Dice(36)  {mean:.4f}   |  살아있는 클래스만 {np.mean(alive):.4f} ({len(alive)}개)")
    print(f"  mean clDice    {np.mean(cld):.4f}")
    print(f"  케이스당 덩어리 중앙값 {int(np.median(ncc))}  (범위 {min(ncc)}~{max(ncc)})")
    print(f"  0점 클래스 {len(dead)}개: {', '.join(dead) if dead else '없음'}")
    return {'per_class': res, 'mean_dice': mean, 'mean_cldice': float(np.mean(cld)),
            'median_components': int(np.median(ncc)), 'dead': dead}


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'fit':
        fit()
    elif cmd == 'apply':
        a = sys.argv[4:]
        steps = tuple(a[a.index('--steps') + 1].split(',')) if '--steps' in a else ('close', 'prune', 'adj', 'endpoint')
        cmm = float(a[a.index('--close-mm') + 1]) if '--close-mm' in a else None
        gmm = float(a[a.index('--max-gap-mm') + 1]) if '--max-gap-mm' in a else None
        print(f"[apply] steps={steps} close_mm={cmm if cmm is not None else 'default'} "
              f"max_gap_mm={gmm if gmm is not None else DEFAULT_MAX_GAP_MM}")
        apply(sys.argv[2], sys.argv[3], steps, cmm, gmm)
    elif cmd == 'eval':
        out = {}
        for d in sys.argv[2:]:
            out[d] = evaluate(d)
        if len(sys.argv) > 3:
            a, b = list(out.values())[:2]
            print(f"\n[비교] mean Dice {a['mean_dice']:.4f} → {b['mean_dice']:.4f} "
                  f"({b['mean_dice']-a['mean_dice']:+.4f}) | "
                  f"clDice {a['mean_cldice']:.4f} → {b['mean_cldice']:.4f} "
                  f"({b['mean_cldice']-a['mean_cldice']:+.4f}) | "
                  f"덩어리 {a['median_components']} → {b['median_components']}")
