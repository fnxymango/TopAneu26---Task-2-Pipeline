#!/usr/bin/env python3
"""intweak.py <tag> [<det>] — 최종모델의 약점을 검출/분류로 분해한다.

  tag : H1_patchfilter/pred/<tag>_<split>_s<seed> 에 있는 52클래스 예측 (5시드 전부 읽는다)
  det : 검출 마스크 디렉터리 접미사 (_c1_realpred/aneu_<split>_<det>) · 기본 b1ff

무엇을 재나 — 세 층으로 분해한다.

  1) 검출층   GT 병변이 검출 마스크에 잡혔는가. 여기서 놓치면 분류기는 손도 못 댄다.
  2) 분류층   잡힌 병변에 올바른 52클래스가 붙었는가. 틀렸다면 **어디로** 틀렸는가.
  3) 환각층   GT 에 없는 (케이스×클래스)를 만들어냈는가. 공식 지표의 FP 가 여기서 나온다.

병변 단위 판정은 perclass.py 와 같은 규약이다: GT 병변을 3회 팽창시킨 영역에서
  검출  = 검출 마스크와 겹침
  예측  = 그 영역의 최빈 비영 라벨
5시드를 전부 읽어 병변마다 '5시드 중 몇 번 맞았나'를 센다. 시드 1판으로 클래스를 논하면
병변 1~2개 차이에 휘둘린다(PROJECT_RULES.md 6-1b).

오분류 분류(taxonomy)는 아래 ADJ 그래프를 쓴다. 인덱스 산술이 아니라 해부학적 연결이다 —
5.3 이 두 클래스(M1-M2 junction · Distal-M2M3)에 붙어 있고, ACA 는 A1 → Acom → A2 순서라
번호 순서와 해부 순서가 어긋나기 때문이다.
"""
import json, os, sys, collections
import numpy as np
import nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
E = f"{R}/experiments"
P = f"{E}/_c1_realpred"
H = f"{E}/H1_patchfilter"

TAG = sys.argv[1]
DET = sys.argv[2] if len(sys.argv) > 2 else "b1ff"
SEEDS = range(5)

S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
LC = S["location_classes"]                       # {"1": "R-1.1 VA trunk", ...}
SPLITS = S["splits"]
NAME = {int(k): v for k, v in LC.items()}
ID = {v: k for k, v in NAME.items()}

def side(nm):
    return nm[0] if nm[:2] in ("R-", "L-") else ""

def seg(nm):
    """'R-3.4 ICA C7-Pcom-junction' -> '3.4' ; 5.3 은 두 개라 뒷말로 갈라준다."""
    body = nm[2:] if nm[:2] in ("R-", "L-") else nm
    code = body.split()[0]
    if code == "5.3":
        code = "5.3j" if "M1-M2" in body else "5.3d"
    return code

def group(nm):
    return seg(nm).split(".")[0]

# 해부학적 인접 (분절 코드 기준, 방향 무시)
ADJ_PAIRS = [
    ("3.1", "3.2"), ("3.2", "3.3"), ("3.3", "3.4"), ("3.4", "3.5"), ("3.5", "3.6"), ("3.6", "3.7"),
    ("3.7", "5.1"), ("3.7", "4.2"),
    ("5.1", "5.2"), ("5.2", "5.3j"), ("5.3j", "5.3d"), ("5.1", "5.3j"),
    ("4.2", "4.1"), ("4.1", "4.3"), ("4.3", "4.4"), ("4.4", "4.5"), ("4.2", "4.3"),
    ("2.1", "2.2"), ("1.10", "2.1"),
    ("1.1", "1.3"), ("1.3", "1.2"), ("1.1", "1.5"), ("1.5", "1.4"),
    ("1.4", "1.7"), ("1.7", "1.6"), ("1.4", "1.9"), ("1.9", "1.8"), ("1.4", "1.10"),
]
ADJ = collections.defaultdict(set)
for a, b in ADJ_PAIRS:
    ADJ[a].add(b); ADJ[b].add(a)

UNSIDED = {"4.1", "1.4", "1.5", "1.10"}          # 정중부 — 좌우 개념이 없다

def kind(gt, pr):
    """오분류 유형."""
    if pr == 0:
        return "미할당"
    gn, pn = NAME[gt], NAME[pr]
    gs, ps = seg(gn), seg(pn)
    gd, pd = side(gn), side(pn)
    if gs == ps:
        return "좌우반전" if gd != pd else "정답"
    if ps in ADJ[gs]:
        same_side = (gd == pd) or (gs in UNSIDED) or (ps in UNSIDED)
        return "인접분절" if same_side else "인접분절+좌우반전"
    if group(gn) == group(pn):
        return "같은혈관군 원거리"
    return "다른혈관군"

def dominant(arr, mask):
    v, c = np.unique(arr[mask], return_counts=True)
    best, bn = 0, 0
    for x, n in zip(v, c):
        if x and n > bn:
            best, bn = int(x), int(n)
    return best

lesions, fps, per_case = [], [], []
for sp in ("test", "val"):
    dirs = [f"{H}/pred/{TAG}_{sp}_s{s}" for s in SEEDS]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        continue
    for case in SPLITS[sp]:
        gp = f"{R}/dataset/TopAneu/location_masks/{case}.nii.gz"
        dp = f"{P}/aneu_{sp}_{DET}/{case}.nii.gz"
        pps = [f"{d}/{case}.nii.gz" for d in dirs]
        if not os.path.exists(gp) or not all(os.path.exists(x) for x in pps):
            continue
        gi = nib.load(gp)
        gt = np.asanyarray(gi.dataobj)
        vx = float(np.prod(gi.header.get_zooms()[:3]))
        det = np.asanyarray(nib.load(dp).dataobj) if os.path.exists(dp) else np.zeros_like(gt)
        preds = [np.asanyarray(nib.load(x).dataobj) for x in pps]
        if any(p.shape != gt.shape for p in preds):
            continue

        gt_classes = {int(x) for x in np.unique(gt) if x}
        # ---- 병변 단위 ----
        covered_masks = []
        for c in sorted(gt_classes):
            lab, n = ndimage.label(gt == c)
            for i in range(1, n + 1):
                m = lab == i
                if m.sum() < 3:
                    continue
                d = ndimage.binary_dilation(m, iterations=3)
                covered_masks.append(d)
                got = [dominant(p, d) for p in preds]
                lesions.append(dict(
                    split=sp, case=case, cls=c, name=NAME[c],
                    dia=float(2 * (3 * (m.sum() * vx) / (4 * np.pi)) ** (1 / 3)),
                    detected=int(bool((d & (det > 0)).sum())),
                    n_seed=len(preds),
                    n_ok=sum(1 for g in got if g == c),
                    preds=got,
                    kinds=[kind(c, g) for g in got]))
        # ---- (케이스 × 클래스) 단위: 환각 ----
        union_gt = np.zeros_like(gt, bool)
        for d in covered_masks:
            union_gt |= d
        for si, p in enumerate(preds):
            pc = {int(x) for x in np.unique(p) if x}
            for c in pc - gt_classes:
                overlap = int(((p == c) & union_gt).sum())
                fps.append(dict(split=sp, case=case, seed=si, cls=c, name=NAME[c],
                                vox=int((p == c).sum()), on_gt=overlap))
            per_case.append(dict(split=sp, case=case, seed=si,
                                 n_gt_cls=len(gt_classes), n_pred_cls=len(pc),
                                 n_tp=len(gt_classes & pc), n_fp=len(pc - gt_classes),
                                 n_fn=len(gt_classes - pc)))

out = dict(tag=TAG, det=DET, n_lesion=len(lesions), lesions=lesions, fps=fps, per_case=per_case)
json.dump(out, open(f"{E}/D1_newdata/intweak_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(f"저장 {len(lesions)}병변 · FP기록 {len(fps)} · {E}/D1_newdata/intweak_{TAG}.json")
