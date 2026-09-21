"""D910(weighted-kNN): 예측(또는 GT-ceiling용 GT) -> 위치할당 -> 공식 evaluate.py로 채점.
d900_infer_eval.py와 동일 구조, lookup 대신 d910 knn index 사용.

사용:
  python d910_infer_eval.py --split val \
      --aneurysm-pred-dir <D720 predict 출력 폴더> \
      --vessel-pred-dir <D800 predict 출력 폴더> \
      --index analysis/d910_index_k5_p1.npz
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import nibabel as nib

import d9xx_lib as L
import d910_lib as L10

EVAL_DIR = L.TOPANEU_ROOT / "code" / "TopAneu-26" / "eval" / "task2"
sys.path.insert(0, str(EVAL_DIR))
import evaluate as official_eval  # noqa: E402


def build_predicted_location_volume(loc_arr_shape, aneu_pred, ves_pred, spacing, ves_names, index, name2id):
    rows, lesions = L.extract_lesion_features(aneu_pred, ves_pred, spacing, ves_names, loc_id_to_name=None)
    out = np.zeros(loc_arr_shape, dtype=np.int32)
    for r in rows:
        name = L10.predict_location_knn(r, index)
        if name is None:
            continue
        oid = name2id.get(name)
        if oid is None:
            continue
        out[lesions == r["lesion_mask_idx"]] = oid
    return out, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--aneurysm-pred-dir", required=True)
    ap.add_argument("--vessel-pred-dir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--save-pred-dir", default=None)
    ap.add_argument("--tag", default=None, help="출력 파일명에 붙일 태그(기본: index 파일명)")
    args = ap.parse_args()

    index = L10.load_knn_index(args.index)
    tag = args.tag or Path(args.index).stem
    print(f"[knn-index] {args.index}  train_lesions={len(index['y'])} dim={index['D']} "
          f"k={index['k']} power={index['power']}")

    _, val_ids, test_ids = L.case_ids_by_split()
    case_ids = val_ids if args.split == "val" else test_ids

    id2name, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()

    aneu_dir = Path(args.aneurysm_pred_dir)
    ves_dir = Path(args.vessel_pred_dir)
    save_dir = Path(args.save_pred_dir) if args.save_pred_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    def local_load_gt(fn):
        case = fn.replace("_0000.mha", "").replace("_0000.nii.gz", "").replace(".mha", "").replace(".nii.gz", "")
        p = L.DATA / "location_masks" / f"{case}.nii.gz"
        img = nib.load(p)
        return np.asanyarray(img.dataobj)
    official_eval.load_gt = local_load_gt

    per_case_metrics = []
    n_lesions_total = 0
    present_classes_this_split = set()
    for i, cid in enumerate(case_ids, 1):
        aneu_p = aneu_dir / f"{cid}.nii.gz"
        ves_p = ves_dir / f"{cid}.nii.gz"
        if not aneu_p.exists() or not ves_p.exists():
            print(f"  스킵 {cid}: 예측 파일 없음")
            continue
        ai = nib.load(aneu_p); vi = nib.load(ves_p)
        aneu = np.asanyarray(ai.dataobj)
        ves = np.asanyarray(vi.dataobj)
        spacing = np.array(ai.header.get_zooms()[:3], dtype=float)

        pred_loc, n_lesions = build_predicted_location_volume(
            aneu.shape, aneu, ves, spacing, ves_names, index, name2id)
        n_lesions_total += n_lesions

        if save_dir:
            nib.save(nib.Nifti1Image(pred_loc.astype(np.int16), ai.affine, ai.header), save_dir / f"{cid}.nii.gz")

        gt = local_load_gt(cid)
        present_classes_this_split.update(int(x) for x in np.unique(gt) if x != 0)
        assert pred_loc.shape == gt.shape, f"{cid} shape mismatch"

        m = official_eval.evaluation_function(pred_loc, cid)
        per_case_metrics.append(m)
        if i % 10 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)} 처리, 누적 병변 {n_lesions_total}", flush=True)

    if not per_case_metrics:
        raise SystemExit("평가할 케이스가 없음")

    agg = official_eval.evaluation_aggregation(per_case_metrics)
    official_avg = official_eval.evaluation_average(agg)

    present = sorted(present_classes_this_split)
    adj = {"PRECISION": 0, "RECALL": 0, "MCC": 0, "DICE": 0, "HD95": 0, "VOLSIM": 0}
    for i in present:
        adj["PRECISION"] += agg[f"PRECISION_{i}"]
        adj["RECALL"] += agg[f"RECALL_{i}"]
        adj["MCC"] += agg[f"MCC_{i}"]
        adj["DICE"] += agg[f"DICE_{i}"]
        adj["HD95"] += agg[f"HD95_{i}"]
        adj["VOLSIM"] += agg[f"VOLSIM_{i}"]
    adj = {k: v / len(present) for k, v in adj.items()}

    result = {
        "method": "D910_weighted_knn",
        "index": str(args.index),
        "split": args.split,
        "n_cases": len(per_case_metrics),
        "n_lesions_predicted": n_lesions_total,
        "n_present_classes_in_split": len(present),
        "official_div52": official_avg,
        "adjusted_div_present": adj,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    out_path = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis" / f"d910_eval_{args.split}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[저장] {out_path}")


if __name__ == "__main__":
    main()
