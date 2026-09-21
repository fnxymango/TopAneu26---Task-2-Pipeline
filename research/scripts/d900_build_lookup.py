"""D900 방법론(signature+3단계 lookup table, 기존 방식): train split GT로부터
signature -> 위치클래스 lookup table을 만든다. (2026-08-10 사용자 지시로 "D900"이라 명명
— weighted-kNN 신규 방식은 d910_lib.py/d910_build_index.py/d910_infer_eval.py, "D910".)

train split만 사용(=val/test는 절대 안 봄, leakage 방지).
위치 GT: dataset/TopAneu/location_masks/*.nii.gz (official value space)
vessel GT: nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr/*.nii.gz (dense id)

사용:
  python d900_build_lookup.py --rule V4_inside_major_r3
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import nibabel as nib

import d9xx_lib as L

OUT_DIR = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, choices=list(L.RULES.keys()))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    sig_fn = L.RULES[args.rule]
    if sig_fn is None:
        raise SystemExit(f"규칙 {args.rule}은 lookup 빌드 지원 안 함 (centroid 전용, 오라클 검증용)")

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

    lookup = L.build_lookup(all_rows, sig_fn)
    lookup["_meta"] = {
        "rule": args.rule,
        "n_train_cases": len(train_ids),
        "n_lesions": len(all_rows),
    }
    out = Path(args.out) if args.out else OUT_DIR / f"d9xx_lookup_{args.rule}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(lookup, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}  (train lesions={len(all_rows)}, exact sigs={len(lookup['exact'])}, "
          f"backoff sigs={len(lookup['backoff'])}, global_major={lookup['global_major']})")


if __name__ == "__main__":
    main()
