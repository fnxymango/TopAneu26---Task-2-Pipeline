"""C10 — 해부 랜드마크 기준 정규화 좌표.

문제(c8/c9 진단): 현재 106차원 피처는 전부 "가장 가까운 혈관/분기점까지의 거리"다.
**병변이 머리 안 어디에 있는지는 0비트도 들어가지 않는다.** 그런데 동맥류 위치클래스는
해부학적으로 정형화돼 있다 — BA tip은 뇌간 앞 정중, Acom은 정중선, M1-M2는 측방.
이 전역 위치 정보의 부재가 남은 최대 공백.

해법: 아틀라스 등록 없이, c4가 이미 뽑아둔 분기점 중 세 개를 랜드마크로 삼아 좌표계를 만든다.
  BA tip        = {BA, R-P1P2} 또는 {BA, L-P1P2} 분기점
  R-terminus    = {R-ICA-C6-C7, R-M1} 또는 {R-ICA-C6-C7, R-A1A2}
  L-terminus    = {L-ICA-C6-C7, L-M1} 또는 {L-ICA-C6-C7, L-A1A2}
참조마스크 417케이스 중 **97.4%에서 세 개 모두 존재**(확인 2026-08-14).

좌표계:
  원점 O = 세 랜드마크의 무게중심
  x축   = R-terminus -> L-terminus 방향 (좌우축)
  y축   = O -> BA tip 을 x에 직교화 (전후축; BA tip은 ICA terminus보다 후방)
  z축   = x × y (상하축)
  스케일 = |R-terminus - L-terminus| (두개 크기 정규화)
-> 두개 크기·자세·모달리티에 불변인 6차원: 정규화 (x,y,z) + 세 랜드마크까지의 정규화 거리

미러 증강 시 x좌표 부호가 뒤집히고 R/L terminus까지의 거리가 swap된다(아래 mirror_pos).

출력: 기존 피처 json에 pos 필드를 추가한 새 json.
사용:
  python c10_landmark_coords.py --feat <c5_feat_train.json> --bp-dir <c4출력> --split train --out <..>
  python c10_landmark_coords.py --cv <출력.json>        # CV로 pos 블록 기여 검증
"""
import argparse, json, collections
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8
import c9_classifier_tune as C9

ST = np.ones((3, 3, 3), dtype=bool)
LANDMARKS = {
    "BAtip":  [{"BA", "R-P1P2"}, {"BA", "L-P1P2"}],
    "Rterm":  [{"R-ICA-C6-C7", "R-M1"}, {"R-ICA-C6-C7", "R-A1A2"}],
    "Lterm":  [{"L-ICA-C6-C7", "L-M1"}, {"L-ICA-C6-C7", "L-A1A2"}],
}
POS_DIM = 6


def find_landmarks(nodes):
    """랜드마크별 좌표(mm). 후보가 여럿이면 전이 voxel이 가장 큰 노드를 택한다
    (c4가 n_transition_vox 내림차순 정렬해 두었으므로 첫 매치가 최대)."""
    out = {}
    for nd in nodes:
        cs = set(nd["classes"])
        for name, pats in LANDMARKS.items():
            if name in out:
                continue
            if any(p <= cs for p in pats):
                out[name] = np.array(nd["centroid_mm"], dtype=float)
    return out


def frame_from_landmarks(lm):
    """(원점, 3x3 축행렬, 스케일). 셋 중 하나라도 없으면 None."""
    if not all(k in lm for k in ("BAtip", "Rterm", "Lterm")):
        return None
    R_, L_, B = lm["Rterm"], lm["Lterm"], lm["BAtip"]
    O = (R_ + L_ + B) / 3.0
    x = L_ - R_
    sx = float(np.linalg.norm(x))
    if sx < 1e-6:
        return None
    x = x / sx
    y = B - O
    y = y - np.dot(y, x) * x                 # x에 직교화
    ny = float(np.linalg.norm(y))
    if ny < 1e-6:
        return None
    y = y / ny
    z = np.cross(x, y)
    return O, np.stack([x, y, z]), sx


def pos_features(centroid_mm, frame, lm):
    O, A, s = frame
    p = A @ (centroid_mm - O) / s                       # 정규화 좌표 3
    d = [float(np.linalg.norm(centroid_mm - lm[k])) / s  # 랜드마크까지 정규화 거리 3
         for k in ("Rterm", "Lterm", "BAtip")]
    return [float(v) for v in p] + d


def mirror_pos(pos):
    """좌우 미러: x 부호 반전, R/L terminus 거리 swap."""
    if pos is None:
        return None
    x, y, z, dR, dL, dB = pos
    return [-x, y, z, dL, dR, dB]


def cmd_build(args):
    ids = {"train": 0, "val": 1, "test": 2}[args.split]
    ids = L.case_ids_by_split()[ids]
    rows = json.load(open(args.feat))
    by_case = collections.defaultdict(list)
    for r in rows:
        by_case[r["case"]].append(r)

    n_ok, n_miss = 0, 0
    for cid in ids:
        rs = by_case.get(cid)
        if not rs:
            continue
        bp = Path(args.bp_dir) / f"{cid}.json"
        if not bp.exists():
            n_miss += len(rs); continue
        lm = find_landmarks(json.load(open(bp))["nodes"])
        fr = frame_from_landmarks(lm)
        if fr is None:
            n_miss += len(rs); continue
        # 병변 중심 — 피처 추출과 동일한 라벨링을 재현해 lesion_mask_idx로 매칭
        li = nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz")
        loc = np.asanyarray(li.dataobj)
        spacing = np.array(li.header.get_zooms()[:3], dtype=float)
        lab, n = ndi.label(loc > 0, structure=ST)
        cents = {}
        for l in range(1, n + 1):
            cents[l] = np.argwhere(lab == l).mean(axis=0) * spacing
        for r in rs:
            c = cents.get(r["lesion_mask_idx"])
            if c is None:
                n_miss += 1; continue
            r["pos"] = pos_features(c, fr, lm)
            n_ok += 1

    json.dump(rows, open(args.out, "w"), ensure_ascii=False)
    print(f"[c10] pos 부여 {n_ok} / 실패 {n_miss} (전체 {len(rows)}) -> {args.out}")


# --- pos 블록을 포함한 피처 벡터 ---
# C9.row_vec 를 몽키패치해서 쓰므로 **패치 전 원본**을 여기 붙잡아 둔다.
# (안 그러면 row_vec_pos -> C9.row_vec(=patched) -> row_vec_pos 무한재귀)
_ORIG_ROW_VEC = C9.row_vec

# pos 블록은 값 범위가 O(1)인데 나머지 블록은 106차원에 퍼진 단위벡터라 성분이 O(0.1)이다.
# 그대로 붙이면 cosine 유사도를 pos가 지배한다 -> 블록별로 각각 정규화한 뒤 가중치로 섞는다.
POS_WEIGHT = float(__import__("os").environ.get("C10_POS_WEIGHT", "0.5"))


def row_vec_pos(r, ves_axis, blocks, tau, mirror=False):
    v = _ORIG_ROW_VEC(r, ves_axis, blocks, tau, mirror=mirror)
    if "pos" not in blocks:
        return v
    p = r.get("pos")
    p = mirror_pos(p) if (mirror and p) else p
    add = np.array(p, dtype=float) if p else np.zeros(POS_DIM)
    na = np.linalg.norm(add)
    if na > 0:
        add = add / na * POS_WEIGHT
    out = np.concatenate([v, add])
    n = np.linalg.norm(out)
    return out / n if n > 0 else out


def cmd_cv(args):
    rows = [r for r in json.load(open(args.cv)) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    have = sum(1 for r in rows if r.get("pos"))
    print(f"[c10-cv] 병변 {len(rows)} (pos 보유 {have}, {have/len(rows)*100:.1f}%)  "
          f"{args.folds}-fold 환자단위 CV\n")

    C9.row_vec = row_vec_pos            # cv_run이 이 함수를 쓴다 (원본은 _ORIG_ROW_VEC에 보존)
    print(f"  pos 블록 가중치 = {POS_WEIGHT}")

    print(f"{'실험':<32}{'top-1':>8}{'macroRec':>10}{'ICA':>8}{'기권':>7}")
    res = []
    for blocks, beta, model in (
            (("dist", "ov", "bp"), 0.75, "rf"),           # C9 최적 = 기준선
            (("dist", "ov", "bp", "pos"), 0.0, "rf"),
            (("dist", "ov", "bp", "pos"), 0.5, "rf"),
            (("dist", "ov", "bp", "pos"), 0.75, "rf"),
            (("dist", "ov", "bp", "pos"), 1.0, "rf"),
            (("dist", "ov", "pos"), 0.75, "rf"),
            (("pos",), 0.75, "rf"),                       # 위치만 — 단독 판별력 확인
            (("dist", "ov", "bp", "pos"), 0.75, "knn"),
    ):
        yt, yp, nab = C9.cv_run(rows, ves_axis, blocks, model, 5, 0.75, beta, args.tau, args.folds)
        t1, mr, ia = C9.score(yt, yp)
        label = f"{'+'.join(blocks)} {model} b={beta}"
        print(f"{label:<32}{t1:>8.3f}{mr:>10.3f}{ia:>8.3f}{nab:>7}")
        res.append({"blocks": list(blocks), "model": model, "beta": beta,
                    "top1": t1, "macro_recall": mr, "ica": ia, "abstain": nab})

    best = max(res, key=lambda r: r["macro_recall"])
    print(f"\n[최적 macro-recall] {'+'.join(best['blocks'])} {best['model']} b={best['beta']} "
          f"-> {best['macro_recall']:.3f} (top-1 {best['top1']:.3f}, ICA {best['ica']:.3f})")
    out = Path(args.cv).parent / "c10_cv_report.json"
    json.dump({"n": len(rows), "pos_coverage": have / len(rows), "trials": res, "best": best},
              open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat"); ap.add_argument("--bp-dir"); ap.add_argument("--split")
    ap.add_argument("--out"); ap.add_argument("--cv")
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--tau", type=float, default=2.0)
    args = ap.parse_args()
    if args.cv:
        cmd_cv(args)
    else:
        cmd_build(args)


if __name__ == "__main__":
    main()
