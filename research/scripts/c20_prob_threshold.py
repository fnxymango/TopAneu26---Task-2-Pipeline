"""C20 — 확률맵 임계 스윕 (검출 동작점 튜닝).

현재 최고 파이프라인(C16 5-fold 확률평균)의 동작점은 nnU-Net 기본 argmax(=0.5)로
**한 번도 튜닝된 적이 없다.** 목표가 MCC이므로 민감도/FP 교환점을 직접 고른다.
  - 임계를 낮추면 민감도 ↑ FP ↑  (recall 기여, precision 손해)
  - 임계를 높이면 반대
MCC는 두 방향 모두에 반응하므로 최적점이 0.5라는 보장이 없다.

--save_probabilities 로 저장된 npz(확률맵)를 읽어 임계별 마스크를 만들고,
c7 필터(혈관거리+크기)를 적용한 뒤 병변단위 지표를 낸다.

사용:
  python c20_prob_threshold.py --prob-dir <npz 폴더> --vessel-dir <..> --split val \
      --thresholds 0.3,0.4,0.5,0.6,0.7 --out-root <마스크 저장 루트>
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi

import d9xx_lib as L

ST = np.ones((3, 3, 3), dtype=bool)
LAB720 = L.TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset720_TopAneuBinary417" / "labelsTr"


def load_prob(npz_path):
    """nnU-Net --save_probabilities 산출물에서 전경(동맥류) 확률 채널을 꺼낸다."""
    z = np.load(npz_path)
    key = "probabilities" if "probabilities" in z else z.files[0]
    p = z[key]
    return p[1] if p.ndim == 4 and p.shape[0] >= 2 else p


def align_prob(prob, shape):
    """nnU-Net 은 확률맵을 (C, Z, Y, X) 로 쓰고 nibabel 라벨은 (X, Y, Z) 다.

    원래 여기서 np.argsort(np.argsort(shape)) 로 축을 맞추려 했는데 그건 축 길이가
    모두 다르고 정렬 순서가 우연히 일치할 때만 맞는 휴리스틱이라, 전 케이스에서
    IndexError 로 죽어 마스크가 하나도 안 만들어졌다(_c20_val 이 비어 있던 원인).
    실제 관계는 축 역순이므로 그것부터 시도하고, 안 맞으면 순열을 찾는다.
    """
    if prob.shape == shape:
        return prob
    t = prob.transpose(2, 1, 0)
    if t.shape == shape:
        return t
    import itertools
    for perm in itertools.permutations(range(3)):
        if tuple(prob.shape[i] for i in perm) == shape:
            return prob.transpose(perm)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob-dir", required=True)
    ap.add_argument("--vessel-dir", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--ref-dir", required=True, help="형상/affine 참조용 기존 예측 폴더")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--min-vox", type=int, default=5)
    ap.add_argument("--max-dist", type=float, default=3.0)
    args = ap.parse_args()

    ths = [float(x) for x in args.thresholds.split(",")]
    ids = L.case_ids_by_split()[{"train": 0, "val": 1, "test": 2}[args.split]]
    pdir, vdir, rdir = Path(args.prob_dir), Path(args.vessel_dir), Path(args.ref_dir)
    outs = {t: Path(args.out_root) / f"th{t:g}" for t in ths}
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    agg = {t: {"det": 0, "tot": 0, "fp": 0} for t in ths}
    for i, cid in enumerate(ids, 1):
        npz = pdir / f"{cid}.npz"
        ref = rdir / f"{cid}.nii.gz"
        vp = vdir / f"{cid}.nii.gz"
        gp = LAB720 / f"{cid}.nii.gz"
        if not (npz.exists() and ref.exists() and vp.exists() and gp.exists()):
            continue
        ri = nib.load(ref)
        prob = load_prob(npz)
        ves = np.asanyarray(nib.load(vp).dataobj)
        gt = np.asanyarray(nib.load(gp).dataobj) > 0
        spacing = np.array(ri.header.get_zooms()[:3], dtype=float)
        prob = align_prob(prob, gt.shape)
        if prob is None:
            print(f"  스킵 {cid}: 확률맵 축을 라벨 {gt.shape} 에 맞출 수 없음", flush=True)
            continue
        dist = (ndi.distance_transform_edt(~(ves > 0), sampling=spacing)
                if (ves > 0).any() else np.full(gt.shape, np.inf))
        glab, gn = ndi.label(gt, structure=ST)

        for t in ths:
            m = prob >= t
            lab, n = ndi.label(m, structure=ST)
            keep = np.zeros(n + 1, dtype=bool)
            found, fp = set(), 0
            for l in range(1, n + 1):
                sel = lab == l
                if sel.sum() < args.min_vox or dist[sel].min() > args.max_dist:
                    continue
                keep[l] = True
                g_ids = set(int(x) for x in np.unique(glab[sel]) if x > 0)
                if g_ids:
                    found |= g_ids
                else:
                    fp += 1
            agg[t]["det"] += len(found); agg[t]["tot"] += gn; agg[t]["fp"] += fp
            nib.save(nib.Nifti1Image(keep[lab].astype(np.int16), ri.affine, ri.header),
                     outs[t] / f"{cid}.nii.gz")
        if i % 10 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}", flush=True)

    print(f"\n=== {args.split} · c7 필터(min_vox {args.min_vox}, dist {args.max_dist}mm) 적용 후 ===")
    print(f"{'임계':>6}{'민감도':>16}{'FP 총계':>9}{'FP/case':>9}")
    res = []
    for t in ths:
        a = agg[t]
        se = a["det"] / a["tot"] if a["tot"] else 0
        print(f"{t:>6.2f}{se:>8.3f} ({a['det']:>3}/{a['tot']}){a['fp']:>9}{a['fp']/len(ids):>9.2f}")
        res.append({"threshold": t, "sensitivity": se, "detected": a["det"],
                    "total": a["tot"], "fp_total": a["fp"]})
    out = (L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
           / f"c20_prob_threshold_{args.split}.json")
    json.dump({"split": args.split, "grid": res}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
