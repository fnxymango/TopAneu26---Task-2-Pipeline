#!/usr/bin/env python
"""단일 케이스 전체 파이프라인 — 원본 영상 1개 → 52클래스 위치 라벨맵 (2026-08-28).

도커(GrandChallenge Task 2)의 진입점. **모달리티를 인자로 받는다** — GC 는 파일명이 임의이고
모달리티가 인터페이스(head-ct-angiography / head-mr-angiography)로 정해지므로, 파일명에서
_ct_/_mr_ 를 읽던 기존 방식(p_build_3cls.modality)은 컨테이너에서 ValueError 로 죽는다.

단계 (verify_e2e.sh 로 재현 검증된 경로 그대로):
  1) robust z  — 모달리티별. CT: fg = x > -300 / MR: fg = x > max(1, 0.02*p99.5)
                 (x - median_fg) / IQR_fg  → Dataset722 는 channel_names=noNorm 이라 그대로 들어간다
  2) 검출  Dataset722 · nnUNetTrainer_250epochs · 5폴드 checkpoint_best 확률평균 · --disable_tta
  3) 라벨 2(aneurysm) 이진화
  4) 혈관  Dataset800 · ...ClassWeightedV2_500ep · fold0 checkpoint_best · **원본 영상 입력**
           ※ vendor/Skeleton-Recall 이 PYTHONPATH 에 있어야 트레이너 클래스를 찾는다
  5) 혈관 후처리 V5 (close→prune→adj→endpoint)
  6) 분기점 그래프 C4
  7) 검출 필터 (min_vox=5, max_dist=1.0mm) — GT 불요판
  8) 분류 RF(seed3) + gC 2등 조각 → 52클래스 라벨맵

사용:
  python pipeline_case.py --image <in.nii.gz> --modality CT|MR --out <loc.nii.gz> \
      --bundle <번들루트> [--device cuda] [--keep-temp <dir>]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import numpy as np
import nibabel as nib


# ---------- 1) robust z (p_build_3cls.robust_z 와 동일 수식) ----------
def robust_z(arr, mod):
    a = arr.astype(np.float32)
    if mod == "CT":
        fg = a > -300.0
    else:
        p995 = np.percentile(a, 99.5)
        fg = a > max(1.0, 0.02 * float(p995))
    if fg.sum() < 100:
        fg = np.ones_like(a, dtype=bool)
    v = a[fg]
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)
    if iqr < 1e-6:
        iqr = float(v.std()) or 1.0
    return (a - med) / iqr


def _predict(model_dir, in_dir, out_dir, folds, chk, device, nproc=2):
    """nnUNetv2 CLI 대신 API 로 부른다(컨테이너에 콘솔스크립트가 없어도 동작)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                        perform_everything_on_device=(device == "cuda"),
                        device=torch.device(device), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=tuple(folds), checkpoint_name=chk)
    p.predict_from_files(in_dir, out_dir, save_probabilities=False, overwrite=True,
                         num_processes_preprocessing=nproc, num_processes_segmentation_export=nproc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--modality", required=True, choices=["CT", "MR"])
    ap.add_argument("--out", required=True)
    # --bundle 미지정 시: 환경변수 → 이 스크립트 위치에서 역산(code/sblee/nnunet/scripts/ 기준 4단계 위)
    _self_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
    ap.add_argument("--bundle", default=os.environ.get("TOPANEU_BUNDLE", _self_root))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keep-temp", default=None)
    ap.add_argument("--min-vox", type=int, default=5)
    ap.add_argument("--max-dist", type=float, default=1.0)
    a = ap.parse_args()

    B = os.path.abspath(a.bundle)
    SC = os.path.join(B, "code", "sblee", "nnunet", "scripts")
    sys.path.insert(0, SC)
    sys.path.insert(0, os.path.join(B, "vendor", "Skeleton-Recall"))
    # d9xx_lib 가 dataset_split.json / Dataset800 dataset.json 을 찾는 루트
    os.environ["TOPANEU_ROOT"] = B

    tmp = a.keep_temp or tempfile.mkdtemp(prefix="topaneu_")
    os.makedirs(tmp, exist_ok=True)
    d_in722, d_inraw = os.path.join(tmp, "in722"), os.path.join(tmp, "inraw")
    d_det, d_ves, d_vespp, d_bp = (os.path.join(tmp, x) for x in ("det", "ves", "vespp", "bp"))
    for d in (d_in722, d_inraw, d_det, d_ves, d_vespp, d_bp):
        os.makedirs(d, exist_ok=True)

    CASE = "case"                      # nnUNet 은 {id}_0000 규약을 요구한다. 원본 파일명은 안 쓴다.
    img = nib.load(a.image)
    arr = np.asanyarray(img.dataobj)
    print(f"[1/8] robust z ({a.modality}) · shape {arr.shape} · spacing "
          f"{tuple(round(float(z),4) for z in img.header.get_zooms()[:3])}", flush=True)
    nib.save(nib.Nifti1Image(robust_z(arr, a.modality).astype(np.float32), img.affine, img.header),
             os.path.join(d_in722, f"{CASE}_0000.nii.gz"))
    shutil.copy(a.image, os.path.join(d_inraw, f"{CASE}_0000.nii.gz"))

    DTR = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
    VTR = "nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
    print("[2/8] 검출 5폴드 확률평균", flush=True)
    _predict(os.path.join(B, "models/detector/Dataset722_TopAneuPjh3cls417", DTR),
             d_in722, d_det, (0, 1, 2, 3, 4), "checkpoint_best.pth", a.device)

    print("[3/8] 라벨2(aneurysm) 이진화", flush=True)
    di = nib.load(os.path.join(d_det, f"{CASE}.nii.gz"))
    aneu_bin = (np.asanyarray(di.dataobj) == 2)

    print("[4/8] 혈관 fold0 (원본 영상 입력)", flush=True)
    _predict(os.path.join(B, "models/vessel/Dataset800_TopAneuVessel417", VTR),
             d_inraw, d_ves, (0,), "checkpoint_best.pth", a.device)

    print("[5/8] 혈관 후처리 V5", flush=True)
    import postprocess_vessel as PV
    PV.apply(d_ves, d_vespp)

    print("[6/8] 분기점 그래프", flush=True)
    import c4_branchpoint_graph as C4
    from pathlib import Path as _P
    C4.process_case(_P(d_vespp) / f"{CASE}.nii.gz", _P(d_bp),
                    C4.adjacency_table(), C4.vessel_names(), C4.SPUR_MM, C4.JUNCTION_R_MM)

    print(f"[7/8] 검출 필터 (min_vox={a.min_vox} · max_dist={a.max_dist}mm)", flush=True)
    from det_filter import filter_case
    vi = nib.load(os.path.join(d_vespp, f"{CASE}.nii.gz"))
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    det_m = filter_case(aneu_bin, ves, spacing, a.min_vox, a.max_dist)

    print("[8/8] 위치 분류 (RF seed3 + gC 2등 조각)", flush=True)
    import pickle
    with open(os.path.join(B, "code/sblee/nnunet/analysis/final_rf_seed3.pkl"), "rb") as f:
        M = pickle.load(f)
    tk = M["topk"]
    os.environ.update({"TOPANEU_TOPK": str(tk["n"]), "TOPANEU_TOPK_VOX": str(tk["vox"]),
                       "TOPANEU_TOPK_ICA": str(tk["ica"]), "TOPANEU_TOPK_MARGIN": str(tk["margin"]),
                       "TOPANEU_TOPK_P2": str(tk["p2"]), "TOPANEU_TOPK_OR": str(tk["or"]),
                       "TOPANEU_TOPK_TAU": str(tk["tau"]), "TOPANEU_TOPK_MAXN": str(tk["maxn"]),
                       "TOPANEU_OUT_DILATE": str(M["out_dilate"])})
    import c5_location_v2 as C5, d9xx_lib as L
    C5.USE_POS = bool(M["use_pos"]); C5.CONF_TAU = float(M["conf_tau"]); C5.CONF_BETA_HI = float(M["conf_beta_hi"])
    mdl, beta = M["model"], float(M["beta"])
    _, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()
    nodes = C5.load_bp(d_bp, CASE)
    rows, lesions = C5.extract_case_rows(det_m, ves, spacing, ves_names, nodes, None)
    for r in rows:
        r["case"] = CASE
    out = np.zeros(det_m.shape, dtype=np.int32)
    for r in rows:
        name = C5.predict_one(mdl, r, beta)
        if name is None:
            continue
        oid = name2id.get(name)
        if oid is None:
            continue
        out[lesions == r["lesion_mask_idx"]] = oid
        if C5.TOPK_N > 1:
            C5._emit_topk(out, lesions, r, mdl, beta, name, name2id)
    if C5.OUT_DILATE > 0 and out.any():
        from scipy import ndimage
        fg = out > 0
        grown = ndimage.binary_dilation(fg, iterations=C5.OUT_DILATE)
        _, nn = ndimage.distance_transform_edt(~fg, return_indices=True)
        newv = grown & ~fg
        out[newv] = out[nn[0][newv], nn[1][newv], nn[2][newv]]

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    o = nib.Nifti1Image(out.astype(np.uint8), img.affine, img.header)
    o.set_data_dtype(np.uint8)
    nib.save(o, a.out)
    print(f"[완료] 병변 {len(rows)} · 라벨 {sorted(int(x) for x in np.unique(out) if x)} → {a.out}", flush=True)
    if not a.keep_temp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
