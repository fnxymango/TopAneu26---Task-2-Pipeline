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
  7) 검출 필터 (min_vox=ENV TOPANEU_MIN_VOX, 기본 5 · 최종 구성 12, max_dist=1.0mm) — GT 불요판
  8) 분류 RF(seed3) + gC 2등 조각 → 52클래스 라벨맵

사용:
  python pipeline_case.py --image <in.nii.gz> --modality CT|MR --out <loc.nii.gz> \
      --bundle <번들루트> [--device cuda] [--keep-temp <dir>]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
import time  # ihson: stage timing for the resource sanity check
import numpy as np
import nibabel as nib


# ---------- 1) robust z (p_build_3cls.robust_z 와 동일 수식) ----------

def _release():
    """파이썬이 해제한 힙을 OS 로 반환한다. 컨테이너는 RSS 로 한도를 재기 때문에,
    free 만 하고 arena 에 남겨두면 한도 초과로 죽는다."""
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def robust_z(arr, mod):
    """전경 기준 robust z. 결과는 종전과 동일하고 메모리 사본만 줄였다 (2026-08-29).

    최대 케이스(1억 3640만 복셀)에서 종전 구현은 사본을 5~6개 만들어 3.5 GB 를 썼다.
    컨테이너 한도가 8 GB 라 이 앞단만으로 절반을 소모하면 안 된다.
      · 분위수는 [25,50,75] 를 한 번에 구해 정렬 사본을 2회 → 1회로
      · 최종 정규화를 제자리 연산으로 (중간 배열 2개 제거)
      · 전경 사본 v 는 쓰고 즉시 해제
    """
    a = np.asarray(arr, dtype=np.float32)
    if a is arr:                      # 입력을 훼손하지 않는다
        a = arr.astype(np.float32)
    if mod == "CT":
        fg = a > -300.0
    else:
        p995 = np.percentile(a, 99.5)
        fg = a > max(1.0, 0.02 * float(p995))
    if fg.sum() < 100:
        fg = np.ones_like(a, dtype=bool)
    v = a[fg]
    del fg
    q1, med, q3 = np.percentile(v, [25, 50, 75])   # 정렬 1회로 통합
    med = float(med)
    iqr = float(q3 - q1)
    if iqr < 1e-6:
        iqr = float(v.std()) or 1.0
    del v
    a -= med                          # 제자리 — 중간 배열 없음
    a /= iqr
    return a


def _predict(model_dir, in_dir, out_dir, folds, chk, device, nproc=1, tile_step=None):
    """nnUNetv2 CLI 대신 API 로 부른다(컨테이너에 콘솔스크립트가 없어도 동작).

    predict_from_files 대신 predict_single_npy_array 를 쓴다 (2026-08-29).
      predict_from_files 는 spawn 워커 풀을 띄우고 로짓 배열을 pickle 로 워커에 복사한다.
      1케이스짜리 컨테이너에서는 이득이 없고 메모리만 배로 든다 (트리 합계 22 GB 관측).
      단일 배열 경로는 전처리·추론·내보내기를 한 프로세스에서 처리해 복사본이 없다.
    영상은 반드시 plans 의 image_reader_writer 로 읽는다 — 직접 transpose 하면 축이 틀어져
    조용히 잘못된 결과가 나온다. nnUNet 자신의 리더를 쓰면 그 위험이 없다.
    """
    import torch, glob as _glob
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    # 검출기/혈관을 따로 조절한다: 시간은 혈관이 55%, FP 증가는 검출기 쪽에서 온다
    _which = "DET" if "detector" in str(model_dir) else "VES"
    _ts = float(os.environ.get(f"TOPANEU_TILE_STEP_{_which}",
                               os.environ.get("TOPANEU_TILE_STEP", 0.5))) if tile_step is None else float(tile_step)
    p = nnUNetPredictor(tile_step_size=_ts, use_gaussian=True, use_mirroring=False,
                        perform_everything_on_device=(device == "cuda"),
                        device=torch.device(device), verbose=False,
                        verbose_preprocessing=False, allow_tqdm=False)
    p.initialize_from_trained_model_folder(model_dir, use_folds=tuple(folds), checkpoint_name=chk)

    srcs = sorted(_glob.glob(os.path.join(in_dir, "*_0000.nii.gz")))
    assert len(srcs) == 1, f"입력이 1개가 아니다: {srcs}"
    case = os.path.basename(srcs[0])[:-len("_0000.nii.gz")]

    rw = p.plans_manager.image_reader_writer_class()
    img, props = rw.read_images([srcs[0]])
    os.makedirs(out_dir, exist_ok=True)
    p.predict_single_npy_array(img, props, None, os.path.join(out_dir, case), False)
    del img
    if device == "cuda":
        torch.cuda.empty_cache()


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
    ap.add_argument("--min-vox", type=int, default=int(os.environ.get("TOPANEU_MIN_VOX", "5")))
    ap.add_argument("--max-dist", type=float, default=1.0)
    a = ap.parse_args()

    _T0 = time.time()
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
    print(f"[{time.time()-_T0:6.1f}s] [1/8] robust z ({a.modality}) · shape {arr.shape} · spacing "
          f"{tuple(round(float(z),4) for z in img.header.get_zooms()[:3])}", flush=True)
    _z = robust_z(arr, a.modality)          # 이미 float32 — astype 재복사 불필요
    del arr
    nib.save(nib.Nifti1Image(_z, img.affine, img.header),
             os.path.join(d_in722, f"{CASE}_0000.nii.gz"))
    del _z
    _release()                              # 해제한 메모리를 OS 로 돌려준다
    shutil.copy(a.image, os.path.join(d_inraw, f"{CASE}_0000.nii.gz"))

    DTR = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
    VTR = "nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
    _DET_FOLDS = tuple(int(x) for x in
                       os.environ.get("TOPANEU_DET_FOLDS", "0,1,2,3,4").split(","))
    print(f"[{time.time()-_T0:6.1f}s] [2/8] 검출 {len(_DET_FOLDS)}폴드 확률평균 folds={_DET_FOLDS}", flush=True)
    _predict(os.path.join(B, "models/detector/Dataset722_TopAneuPjh3cls417", DTR),
             d_in722, d_det, _DET_FOLDS, "checkpoint_best.pth", a.device)

    _release(); print(f"[{time.time()-_T0:6.1f}s] [3/8] 라벨2(aneurysm) 이진화", flush=True)
    di = nib.load(os.path.join(d_det, f"{CASE}.nii.gz"))
    aneu_bin = (np.asanyarray(di.dataobj) == 2)

    _release(); print(f"[{time.time()-_T0:6.1f}s] [4/8] 혈관 fold0 (원본 영상 입력)", flush=True)
    _predict(os.path.join(B, "models/vessel/Dataset800_TopAneuVessel417", VTR),
             d_inraw, d_ves, (0,), "checkpoint_best.pth", a.device)

    _release(); print(f"[{time.time()-_T0:6.1f}s] [5/8] 혈관 후처리 V5", flush=True)
    import postprocess_vessel as PV
    PV.apply(d_ves, d_vespp)

    _release(); print(f"[{time.time()-_T0:6.1f}s] [6/8] 분기점 그래프", flush=True)
    import c4_branchpoint_graph as C4
    from pathlib import Path as _P
    C4.process_case(_P(d_vespp) / f"{CASE}.nii.gz", _P(d_bp),
                    C4.adjacency_table(), C4.vessel_names(), C4.SPUR_MM, C4.JUNCTION_R_MM)

    _release(); print(f"[{time.time()-_T0:6.1f}s] [7/8] 검출 필터 (min_vox={a.min_vox} · max_dist={a.max_dist}mm)", flush=True)
    # ihson: crop-based replacement for the full-volume EDT. The original builds an EDT over the
    # entire volume and then reads only min-distance per blob against a 1.0 mm threshold, so the
    # distance only has to be correct within that radius. Verified output-identical on 11 real
    # cases (6 MR, 5 CT), 6-9x faster. Falls back to the original if the module is unavailable.
    try:
        # /opt/app in the image; TOPANEU_APP_DIR lets a local dry run find it too, so the dry
        # run exercises the same code path the container will.
        sys.path.insert(0, os.environ.get("TOPANEU_APP_DIR", "/opt/app"))
        from fast_stages import filter_case_fast as filter_case
        print("[fast] det_filter: crop-based EDT", flush=True)
    except Exception as _e:
        print(f"[fast] det_filter fast path unavailable ({_e}) -- using the original", flush=True)
        from det_filter import filter_case
    vi = nib.load(os.path.join(d_vespp, f"{CASE}.nii.gz"))
    ves = np.asanyarray(vi.dataobj)
    spacing = np.array(vi.header.get_zooms()[:3], dtype=float)
    det_m = filter_case(aneu_bin, ves, spacing, a.min_vox, a.max_dist)

    _exp = os.environ.get("TOPANEU_EXPORT_VESSEL")
    if _exp:
        nib.save(nib.Nifti1Image(ves.astype(np.uint8), vi.affine, vi.header), _exp)
        print(f"[export] vessel map -> {_exp}", flush=True)

    _release(); print(f"[{time.time()-_T0:6.1f}s] [8/8] 위치 분류 (RF seed3 + gC 2등 조각)", flush=True)
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
    # 출력 부피 보정 (ENV TOPANEU_OUT_GROW) — c5_location_v2.cmd_eval 의 S5 블록과 같은 코드·같은 순서.
    # 컨테이너는 cmd_eval 을 거치지 않으므로 여기에도 있어야 한다.
    if C5.OUT_GROW > 1.0 and out.any():
        from scipy import ndimage
        fg = out > 0
        n0 = int(fg.sum())
        need = int(round(n0 * C5.OUT_GROW)) - n0
        if need > 0:
            dist, nn = ndimage.distance_transform_edt(~fg, return_indices=True)
            shell = np.flatnonzero((dist.ravel() > 0) & (dist.ravel() <= 4.0))
            if shell.size:
                order = shell[np.argsort(dist.ravel()[shell], kind="stable")[:need]]
                ii = np.unravel_index(order, out.shape)
                out[ii] = out[nn[0][ii], nn[1][ii], nn[2][ii]]
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
