"""C42/C43 — 앵커 결측 지시자 + 랜드마크 아틀라스 기반 대체값 (2026-08-18).

문제(C41 진단): 52클래스 중 61%가 분기점으로 정의되는데, Pcom/AChA/OA 계열 7종은
참조 마스크에서도 검출률이 41~60%다. 그 케이스들에서 3.4/3.5/3.6 은 **완전히 동일한
피처**를 받으므로 어떤 분류기도 구분할 수 없다. 오늘 모델·결정규칙·피처를 다 바꿔봐도
안 움직인 이유가 이것이다.

두 블록을 만든다. 둘 다 학습(참조마스크)·추론(예측마스크) 양쪽에서 같은 방식으로
계산되므로 C25 크롭이 걸렸던 분포 불일치 함정에 걸리지 않는다.

  [MISS] 34차원 이진 — 그 분기점이 검출됐는가.
         지금 bp 블록은 결측이면 0 이고 '멀리 있어도' 0 에 가깝다. 분류기가
         **"멀다"와 "없다"를 구분하지 못한다.** 그 정보가 통째로 버려지고 있었다.

  [IMP]  34차원 — 검출된 케이스들에서 각 분기점의 **랜드마크 정규화 좌표 중앙값**을
         모아 아틀라스를 만들고, 결측 케이스에는 그 기댓값 위치까지의 거리를 채운다.
         랜드마크 3점은 98%+ 검출되므로 좌표계는 거의 항상 성립한다.
         "AChA 분기점까지 거리"가 결측 대신 "AChA 가 있어야 할 자리까지 거리"가 된다.

사용:
  python c42_anchor_atlas.py build --bp-dir <all_ref> --split train --out <atlas.json>
"""
import argparse, collections, json
from pathlib import Path

import numpy as np

import d9xx_lib as L
import c5_location_v2 as C5

MISS_DIM = C5.BP_DIM
IMP_DIM = C5.BP_DIM


def frame_of(nodes):
    return C5.landmark_frame(nodes)


def junction_positions(nodes):
    """검출된 분기점 타입 -> mm 좌표 (같은 타입 여럿이면 노드 크기 최대인 것)."""
    best = {}
    for nd in nodes:
        if not nd.get("valid", True):
            continue
        cs = set(nd["classes"])
        n = nd.get("n_transition_vox", 1)
        for k, p in enumerate(C5.JUNCTION_PAIRS):
            if set(p) <= cs and n >= best.get(k, (0, None))[0]:
                best[k] = (n, np.array(nd["centroid_mm"], dtype=float))
    return {k: v[1] for k, v in best.items()}


def build_atlas(bp_dir, case_ids):
    """검출된 케이스들의 랜드마크 정규화 좌표를 모아 타입별 중앙값을 낸다."""
    acc = collections.defaultdict(list)
    n_frame = 0
    for cid in case_ids:
        nodes = C5.load_bp(bp_dir, cid)
        fr = frame_of(nodes)
        if not fr:
            continue
        n_frame += 1
        O, A, s, _ = fr
        for k, p in junction_positions(nodes).items():
            acc[k].append((A @ (p - O)) / s)
    atlas = {str(k): {"q": np.median(np.stack(v), axis=0).tolist(), "n": len(v)}
             for k, v in acc.items() if len(v) >= 5}
    return atlas, n_frame


def fill(rows, nodes, atlas):
    """rows 에 bp_miss / bp_imp 를 채운다. atlas 키는 문자열 인덱스."""
    fr = frame_of(nodes)
    pos = junction_positions(nodes)
    O, A, s = (fr[0], fr[1], fr[2]) if fr else (None, None, None)
    for r in rows:
        bp = r.get("bp_mm") or [None] * C5.BP_DIM
        miss = [0.0 if (bp[k] is not None) else 1.0 for k in range(C5.BP_DIM)]
        imp = [0.0] * C5.BP_DIM
        cen = np.array(r.get("_cen") or [0, 0, 0], dtype=float)
        for k in range(C5.BP_DIM):
            if bp[k] is not None:                       # 검출됨 -> 실제 거리 (bp 블록과 동일)
                imp[k] = 1.0 / (1.0 + float(bp[k]))
            elif fr is not None and str(k) in atlas:    # 결측 -> 아틀라스 기댓값까지 거리
                q = np.array(atlas[str(k)]["q"], dtype=float)
                p_mm = O + s * (A.T @ q)
                imp[k] = 1.0 / (1.0 + float(np.linalg.norm(cen - p_mm)))
        r["bp_miss"] = miss
        r["bp_imp"] = imp


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--bp-dir", required=True)
    b.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    b.add_argument("--out", required=True)
    # 학습쪽 피처 파일에도 같은 블록을 채워 넣는다. 안 채우면 학습엔 신호가 있고
    # 추론엔 0 만 들어가(또는 그 반대) 블록이 상수가 되어 비교가 무의미해진다.
    ff = sub.add_parser("fillfeat")
    ff.add_argument("--feat", required=True)
    ff.add_argument("--bp-dir", required=True)
    ff.add_argument("--atlas", required=True)
    ff.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "fillfeat":
        atlas = json.load(open(a.atlas))
        rows = json.load(open(a.feat))
        by_case = collections.defaultdict(list)
        for r in rows:
            by_case[r["case"]].append(r)
        for cid, rs in by_case.items():
            fill(rs, C5.load_bp(a.bp_dir, cid), atlas)
        json.dump(rows, open(a.out, "w"), ensure_ascii=False)
        nm = sum(1 for r in rows if r.get("bp_miss"))
        miss_rate = float(np.mean([np.mean(r["bp_miss"]) for r in rows if r.get("bp_miss")]))
        print(f"[c42] {len(rows)}행 중 {nm} 채움 · 평균 결측률 {miss_rate*100:.1f}% -> {a.out}")
        return

    tr, va, te = L.case_ids_by_split()
    ids = {"train": tr, "val": va, "test": te, "all": tr + va + te}[a.split]
    atlas, nf = build_atlas(a.bp_dir, ids)
    json.dump(atlas, open(a.out, "w"), indent=1)
    rates = sorted(((int(k), v["n"] / max(nf, 1)) for k, v in atlas.items()), key=lambda x: x[1])
    print(f"[c42] {a.split} {len(ids)}케이스 중 랜드마크 성립 {nf} · 아틀라스 타입 {len(atlas)}/{C5.BP_DIM}")
    print(f"  검출률 낮은 5종 (이들이 대체값의 주 수혜자):")
    for k, r in rates[:5]:
        a_, b_ = C5.JUNCTION_PAIRS[k]
        print(f"    {a_} + {b_:<22} {r*100:>5.1f}%  (n={atlas[str(k)]['n']})")
    print(f"[저장] {a.out}")


if __name__ == "__main__":
    main()
