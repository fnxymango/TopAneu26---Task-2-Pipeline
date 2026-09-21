"""D910(weighted-kNN) 인덱스 빌드 — train split GT lesion feature만 사용(leakage 방지).
d900_build_lookup.py와 동일 구조, build_lookup 대신 build_knn_index.

사용:
  python d910_build_index.py --k 5 --power 1.0
"""
import argparse, json
from pathlib import Path

import numpy as np
import nibabel as nib

import d9xx_lib as L
import d910_lib as L10

OUT_DIR = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    train_ids, _, _ = L.case_ids_by_split()
    id2name, _ = L.official_location_names()
    ves_names = L.vessel_dense_names()

    all_rows = []
    for i, cid in enumerate(train_ids, 1):
        loc_p = L.DATA / "location_masks" / f"{cid}.nii.gz"
        ves_p = L.VES_RAW / "labelsTr" / f"{cid}.nii.gz"
        if not loc_p.exists() or not ves_p.exists():
            print(f"  스킵 {cid}: 파일 없음")
            continue
        li = nib.load(loc_p); vi = nib.load(ves_p)
        loc = np.asanyarray(li.dataobj)
        ves = np.asanyarray(vi.dataobj)
        if loc.shape != ves.shape:
            print(f"  스킵 {cid}: shape 불일치 {loc.shape} vs {ves.shape}")
            continue
        spacing = np.array(li.header.get_zooms()[:3], dtype=float)
        rows, _ = L.extract_lesion_features(loc, ves, spacing, ves_names, loc_id_to_name=id2name)
        all_rows.extend(rows)
        if i % 50 == 0 or i == len(train_ids):
            print(f"  {i}/{len(train_ids)}  누적 병변 {len(all_rows)}", flush=True)

    labeled = [r for r in all_rows if r["gt_loc"] is not None]
    index = L10.build_knn_index(labeled, k=args.k, power=args.power)

    out = Path(args.out) if args.out else OUT_DIR / f"d910_index_k{args.k}_p{args.power:g}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, V=index["V"], y=np.array(index["y"], dtype=object),
             classes=np.array(list(index["cidx"].keys()), dtype=object),
             k=args.k, power=args.power)
    print(f"[저장] {out}  (train lesions={len(labeled)}, feature-dim={index['D']}, k={args.k}, power={args.power})")


if __name__ == "__main__":
    main()
