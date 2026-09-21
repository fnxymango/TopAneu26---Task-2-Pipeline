"""C6 — 동맥류 검출 다수결 합의 마스크 생성 (N모델 중 M표 이상).

배경(2026-08-14): C3 대조표에서 파이프라인이 민감도가 아니라 **위양성에 묶여 있음**이 확인됐다.
A6-2(adaptive)는 병변 민감도가 0.814로 A5-2(0.721)보다 높은데도 FP가 55->152로 3배라
엔드투엔드 지표는 오히려 졌다(val adj MCC 0.2164 vs 0.2651).

다수결은 정확히 이 지점을 겨냥한다 — 허위 검출은 여러 모델이 같은 자리에서 재현해야만 살아남는다.
목표는 "A6-2의 민감도 + A5-2 수준의 FP".

5-fold 3/5 를 위해 만들었지만 현재 학습된 fold가 fold_0 뿐이라, 우선 이미 디스크에 있는
A5-2 / A6-2 두 모델의 2/2 합의로 **가설 자체를 무료로 검증**한다.
(합의 필터가 효과가 없으면 5-fold도 효과가 없다 — 29 GPU시간을 쓰기 전에 알 수 있다.)

모드:
  voxel  — 복셀 단위 득표 >= M (표준 다수결). M=N이면 교집합. 병변이 깎일 수 있음.
  lesion — 합집합 연결성분 중 M개 이상 모델이 겹치는 것만 남기고, 그 안은 합집합 범위를 유지.
           위치분류는 성분 단위로 클래스를 매기므로 병변 범위 보존이 유리하다.

사용:
  python c6_consensus.py --in-dirs d1,d2[,d3...] --min-votes 2 --mode lesion --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

ST = np.ones((3, 3, 3), dtype=bool)


def consensus(masks, min_votes, mode):
    votes = np.zeros(masks[0].shape, dtype=np.uint8)
    for m in masks:
        votes += (m > 0).astype(np.uint8)

    if mode == "voxel":
        return (votes >= min_votes).astype(np.uint8)

    union = votes > 0
    lab, k = ndi.label(union, structure=ST)
    if k == 0:
        return np.zeros_like(votes)
    # 성분별 최대 득표수 — 그 성분에 M개 이상 모델이 실제로 걸쳤는지
    maxv = ndi.maximum(votes, lab, index=np.arange(1, k + 1))
    keep = np.zeros(k + 1, dtype=bool)
    keep[1:] = maxv >= min_votes
    return keep[lab].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dirs", required=True, help="쉼표구분 예측 폴더")
    ap.add_argument("--min-votes", type=int, required=True)
    ap.add_argument("--mode", default="lesion", choices=["voxel", "lesion"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.in_dirs.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name for p in dirs[0].glob("*.nii.gz"))
    print(f"[c6] {len(dirs)}모델 중 {args.min_votes}표 이상  mode={args.mode}  {len(cases)}케이스")

    n_les_in, n_les_out = 0, 0
    for i, fn in enumerate(cases, 1):
        paths = [d / fn for d in dirs]
        if not all(p.exists() for p in paths):
            print(f"  스킵 {fn}: 일부 모델 예측 없음"); continue
        imgs = [nib.load(p) for p in paths]
        masks = [np.asanyarray(im.dataobj) for im in imgs]
        if len({m.shape for m in masks}) != 1:
            print(f"  스킵 {fn}: shape 불일치"); continue

        res = consensus(masks, args.min_votes, args.mode)
        n_les_in += ndi.label(masks[0] > 0, structure=ST)[1]
        n_les_out += ndi.label(res > 0, structure=ST)[1]
        nib.save(nib.Nifti1Image(res.astype(np.int16), imgs[0].affine, imgs[0].header), out / fn)
        if i % 20 == 0 or i == len(cases):
            print(f"  {i}/{len(cases)}  성분 {n_les_in} -> {n_les_out}", flush=True)

    print(f"[c6] 완료 — 첫모델 성분 {n_les_in} -> 합의 후 {n_les_out} -> {out}")


if __name__ == "__main__":
    main()
