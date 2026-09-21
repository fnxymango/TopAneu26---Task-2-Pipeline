"""D9xx: 예측(D720 aneurysm + D800 vessel) -> 위치할당 -> 공식 evaluate.py로 채점.

lookup table(d9xx_build_lookup.py 산출물)을 이용해 각 케이스의 병변 connected-component마다
위치클래스를 예측하고, 공식 eval/task2/evaluate.py의 evaluation_function/aggregation/average를
"그대로" 재사용해 점수를 낸다 (다만 load_gt는 로컬 경로를 보도록 monkeypatch).

점수는 두 가지로 보고:
  official : evaluate.py 그대로, N_CLASSES=52로 나눔
  adjusted : 이번에 평가하는 스플릿(val42 또는 test83)에 실제로 존재하는 클래스 수로만 나눔
             (2026-08-09 사용자 지시: "마지막엔 결국 test로 해야되니까" -> test83 기준 36을 기본값으로 채택)

사용:
  python d9xx_infer_eval.py --split test \
      --aneurysm-pred-dir <D720 predict 출력 폴더> \
      --vessel-pred-dir <D800 predict 출력 폴더> \
      --lookup analysis/d9xx_lookup_V4_inside_major_r3.json
"""
import argparse, json, os, sys, collections
from pathlib import Path

import numpy as np
import nibabel as nib

import d9xx_lib as L

EVAL_DIR = L.TOPANEU_ROOT / "code" / "TopAneu-26" / "eval" / "task2"
sys.path.insert(0, str(EVAL_DIR))
import evaluate as official_eval  # noqa: E402


def build_predicted_location_volume(loc_arr_shape, aneu_pred, ves_pred, spacing, ves_names, sig_fn, lookup, name2id):
    rows, lesions = L.extract_lesion_features(aneu_pred, ves_pred, spacing, ves_names, loc_id_to_name=None)
    out = np.zeros(loc_arr_shape, dtype=np.int32)
    for r in rows:
        name = L.predict_location(r, sig_fn, lookup)
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
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--save-pred-dir", default=None, help="예측 위치볼륨을 nii.gz로 저장할 폴더(선택)")
    # 출력 파일명 태그. 같은 rule로 입력만 바꿔 여러 번 돌릴 때(GT-ceiling vs 실제예측 등)
    # 결과가 서로 덮어써지는 것을 막는다. d910_infer_eval.py와 동일한 인터페이스.
    ap.add_argument("--tag", default=None, help="출력 파일명에 붙일 태그(기본: lookup의 rule명)")
    args = ap.parse_args()

    lookup = json.load(open(args.lookup))
    rule = lookup["_meta"]["rule"]
    sig_fn = L.RULES[rule]
    print(f"[lookup] rule={rule} exact_sigs={len(lookup['exact'])} global_major={lookup['global_major']}")

    _, val_ids, test_ids = L.case_ids_by_split()
    case_ids = val_ids if args.split == "val" else test_ids

    id2name, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()

    aneu_dir = Path(args.aneurysm_pred_dir)
    ves_dir = Path(args.vessel_pred_dir)
    save_dir = Path(args.save_pred_dir) if args.save_pred_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # official_eval.load_gt를 로컬 location_masks를 보도록 monkeypatch
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
            print(f"  스킵 {cid}: 예측 파일 없음 (aneu={aneu_p.exists()}, ves={ves_p.exists()})")
            continue
        ai = nib.load(aneu_p); vi = nib.load(ves_p)
        aneu = np.asanyarray(ai.dataobj)
        ves = np.asanyarray(vi.dataobj)
        spacing = np.array(ai.header.get_zooms()[:3], dtype=float)

        pred_loc, n_lesions = build_predicted_location_volume(
            aneu.shape, aneu, ves, spacing, ves_names, sig_fn, lookup, name2id)
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

    # adjusted: 이 스플릿에 실제 존재하는 클래스만으로 평균
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
        "method": "C1_signature_lookup",
        "rule": rule,
        "split": args.split,
        "n_cases": len(per_case_metrics),
        "n_lesions_predicted": n_lesions_total,
        "n_present_classes_in_split": len(present),
        "official_div52": official_avg,
        "adjusted_div_present": adj,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    tag = args.tag or rule
    out_path = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis" / f"d9xx_eval_{args.split}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"[저장] {out_path}")


if __name__ == "__main__":
    main()
