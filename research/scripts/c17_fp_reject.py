"""C17 — 학습된 위양성 기각기.

현행 FP 필터는 손규칙 두 개뿐이다(성분 크기 >= N voxel, 혈관까지 거리 <= R mm).
c7 스윕에서 이것만으로 FP를 55->12 로 줄였지만, 남은 FP는 전부 혈관 위에 붙어 있어
규칙으로는 더 못 줄인다. 그런데 우리에겐 이미 C5 피처 기계가 있다 —
혈관 근접도 36 + sac 점유율 36 + 분기점 근접도 34 + 랜드마크 좌표 6.
검출 후보마다 이 피처를 뽑아 **TP/FP 이진 분류기**를 학습하면 손규칙보다 정교하다.

학습데이터(누수 없음): splits_final.json 을 --train-only 로 재생성한 덕분에
fold1~4 의 val 합집합이 **train 292 케이스를 정확히 커버**한다(73+72+72+75).
즉 모든 train 케이스에 대해 "그 케이스를 학습에 쓰지 않은 모델"의 예측을 얻을 수 있어
TP/FP 라벨이 낙관 편향 없이 만들어진다.

목적: 검출 임계를 낮춰 민감도를 회복하고(천장 0.3259 vs e2e 0.2207, 격차 0.105),
      늘어난 FP를 이 기각기로 걷어낸다.

사용:
  # 1) out-of-fold 예측에서 후보 피처 추출
  python c17_fp_reject.py build --pred-root <oof예측루트> --vessel-dir <참조혈관> \
      --bp-dir <c4출력> --out <cand.json>
  # 2) CV로 기각기 성능 확인
  python c17_fp_reject.py cv --cand <cand.json>
  # 3) 실제 예측 폴더에 적용
  python c17_fp_reject.py apply --cand <cand.json> --aneu-dir <..> --vessel-dir <..> \
      --bp-dir <..> --split val --out <필터된 폴더> [--thresh 0.5]
"""
import argparse, json, collections
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8

ST = np.ones((3, 3, 3), dtype=bool)
LAB720 = L.TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset720_TopAneuBinary417" / "labelsTr"


def candidates_for_case(cid, aneu_path, ves_dir, bp_dir, ves_names, gt_path=None):
    """검출 성분마다 C5 피처 + (GT가 있으면) TP/FP 라벨."""
    ap = Path(aneu_path)
    vp = Path(ves_dir) / f"{cid}.nii.gz"
    if not ap.exists() or not vp.exists():
        return []
    ai = nib.load(ap)
    aneu = np.asanyarray(ai.dataobj)
    if not (aneu > 0).any():
        return []
    ves = np.asanyarray(nib.load(vp).dataobj)
    spacing = np.array(ai.header.get_zooms()[:3], dtype=float)
    nodes = C5.load_bp(bp_dir, cid)
    rows, lesions = C5.extract_case_rows(aneu, ves, spacing, ves_names, nodes, None)

    gt = None
    if gt_path and Path(gt_path).exists():
        gt = np.asanyarray(nib.load(gt_path).dataobj) > 0

    out = []
    for r in rows:
        r["case"] = cid
        if gt is not None:
            m = lesions == r["lesion_mask_idx"]
            r["is_tp"] = int(bool((gt & m).any()))
        out.append(r)
    return out


def cmd_build(args):
    ves_names = L.vessel_dense_names()
    splits = json.load(open(L.TOPANEU_ROOT / "nnunet" / "nnUNet_preprocessed"
                           / "Dataset720_TopAneuBinary417" / "splits_final.json"))
    cands = []
    root = Path(args.pred_root)
    for f in (1, 2, 3, 4):                       # fold k 의 val = 그 fold가 학습에 안 쓴 케이스
        d = root / f"oof_f{f}"
        if not d.exists():
            print(f"  [경고] {d} 없음 — 건너뜀"); continue
        for cid in splits[f]["val"]:
            cands.extend(candidates_for_case(cid, d / f"{cid}.nii.gz", args.vessel_dir,
                                             args.bp_dir, ves_names, LAB720 / f"{cid}.nii.gz"))
        print(f"  fold{f} 처리 후 누적 후보 {len(cands)}", flush=True)
    json.dump(cands, open(args.out, "w"), ensure_ascii=False)
    n_tp = sum(c.get("is_tp", 0) for c in cands)
    print(f"[c17] 후보 {len(cands)} (TP {n_tp}, FP {len(cands)-n_tp}) -> {args.out}")


def vecs(rows, ves_axis):
    C5.USE_POS = True
    return np.array([C5.row_to_vec(r, ves_axis, mirror=False) for r in rows])


def cmd_cv(args):
    C5.USE_POS = True
    rows = json.load(open(args.cand))
    rows = [r for r in rows if "is_tp" in r]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    X = vecs(rows, ves_axis)
    y = np.array([r["is_tp"] for r in rows])
    pm = C8.patient_map()
    g = np.array([pm.get(r["case"], r["case"]) for r in rows])
    print(f"[c17] 후보 {len(rows)}  TP {int(y.sum())}  FP {int((1-y).sum())}\n")

    prob = np.zeros(len(rows))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                     random_state=0, n_jobs=-1).fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])[:, list(clf.classes_).index(1)]

    print(f"{'임계':>6}{'TP유지':>9}{'FP제거':>9}{'민감도손실':>12}{'FP감소율':>10}")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        keep = prob >= t
        tp_keep = int((keep & (y == 1)).sum()); fp_keep = int((keep & (y == 0)).sum())
        tp_tot = int((y == 1).sum()); fp_tot = int((y == 0).sum())
        print(f"{t:>6.1f}{tp_keep:>5}/{tp_tot:<4}{fp_tot-fp_keep:>5}/{fp_tot:<4}"
              f"{1-tp_keep/max(tp_tot,1):>12.3f}{1-fp_keep/max(fp_tot,1):>10.3f}")
    out = Path(args.cand).parent / "c17_cv_report.json"
    json.dump({"n": len(rows), "n_tp": int(y.sum())}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {out}")


def cmd_apply(args):
    C5.USE_POS = True
    train_rows = [r for r in json.load(open(args.cand)) if "is_tp" in r]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                 random_state=0, n_jobs=-1)
    clf.fit(vecs(train_rows, ves_axis), np.array([r["is_tp"] for r in train_rows]))
    tp_idx = list(clf.classes_).index(1)
    ves_names = L.vessel_dense_names()
    ids = L.case_ids_by_split()[{"train": 0, "val": 1, "test": 2}[args.split]]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    n_in, n_out = 0, 0
    for i, cid in enumerate(ids, 1):
        ap = Path(args.aneu_dir) / f"{cid}.nii.gz"
        if not ap.exists():
            continue
        ai = nib.load(ap)
        aneu = np.asanyarray(ai.dataobj)
        rows = candidates_for_case(cid, ap, args.vessel_dir, args.bp_dir, ves_names)
        lab, n = ndi.label(aneu > 0, structure=ST)
        n_in += n
        keep = np.zeros(n + 1, dtype=bool)
        if rows:
            p = clf.predict_proba(vecs(rows, ves_axis))[:, tp_idx]
            for r, pi in zip(rows, p):
                if pi >= args.thresh:
                    keep[r["lesion_mask_idx"]] = True
        res = keep[lab].astype(np.int16)
        n_out += int(ndi.label(res > 0, structure=ST)[1])
        nib.save(nib.Nifti1Image(res, ai.affine, ai.header), out / f"{cid}.nii.gz")
        if i % 20 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}  성분 {n_in} -> {n_out}", flush=True)
    print(f"[c17] 적용 완료 성분 {n_in} -> {n_out} (임계 {args.thresh}) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    s = ap.add_subparsers(dest="cmd", required=True)
    b = s.add_parser("build"); b.add_argument("--pred-root", required=True)
    b.add_argument("--vessel-dir", required=True); b.add_argument("--bp-dir", required=True)
    b.add_argument("--out", required=True); b.set_defaults(fn=cmd_build)
    c = s.add_parser("cv"); c.add_argument("--cand", required=True); c.set_defaults(fn=cmd_cv)
    a = s.add_parser("apply"); a.add_argument("--cand", required=True)
    a.add_argument("--aneu-dir", required=True); a.add_argument("--vessel-dir", required=True)
    a.add_argument("--bp-dir", required=True); a.add_argument("--split", required=True)
    a.add_argument("--out", required=True); a.add_argument("--thresh", type=float, default=0.5)
    a.set_defaults(fn=cmd_apply)
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
