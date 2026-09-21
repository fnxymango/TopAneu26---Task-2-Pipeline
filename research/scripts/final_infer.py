#!/usr/bin/env python
"""최종 모델(X5+gC) 단독 추론기 — 도커 제작자용 (2026-08-28).

케이스 하나의 세 입력(검출 마스크 · 후처리 혈관 마스크 · 분기점 json)과 고정된 분류기 피클을 받아
52클래스 위치 라벨맵(.nii.gz)을 쓴다. c5_location_v2.py eval 이 내부에서 하는 것과 **같은 함수**를
호출하므로 experiments/final_pred_seed3_test/ 의 기준 마스크와 복셀 단위로 일치해야 한다(sanity).

사용:
  TOPANEU_ROOT=<번들루트> python final_infer.py \
      --model analysis/final_rf_seed3.pkl \
      --aneu  <case>.nii.gz  (X5 5폴드 확률평균 → 라벨2 추출 → c7 필터 결과, 이진)  \
      --vessel <case>.nii.gz (V4-2 fold0 → postprocess_vessel apply 결과, 36클래스) \
      --bp    <case>.json    (c4_branchpoint_graph 결과)  \
      --out   <case>_location.nii.gz
환경변수는 피클에 저장된 값으로 자동 설정된다(gC 게이트·β·τ·팽창).
"""
import argparse, os, sys, pickle, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, nibabel as nib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--aneu", required=True)
    ap.add_argument("--vessel", required=True); ap.add_argument("--bp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with open(a.model, "rb") as f:
        M = pickle.load(f)
    # c5 는 import 시점에 환경변수를 읽으므로 먼저 세팅한다
    tk = M["topk"]
    os.environ.update({"TOPANEU_TOPK": str(tk["n"]), "TOPANEU_TOPK_VOX": str(tk["vox"]),
                       "TOPANEU_TOPK_ICA": str(tk["ica"]), "TOPANEU_TOPK_MARGIN": str(tk["margin"]),
                       "TOPANEU_TOPK_P2": str(tk["p2"]), "TOPANEU_TOPK_OR": str(tk["or"]),
                       "TOPANEU_TOPK_TAU": str(tk["tau"]), "TOPANEU_TOPK_MAXN": str(tk["maxn"]),
                       "TOPANEU_OUT_DILATE": str(M["out_dilate"])})
    import c5_location_v2 as C5, d9xx_lib as L
    from scipy import ndimage
    C5.USE_POS = bool(M["use_pos"]); C5.CONF_TAU = float(M["conf_tau"]); C5.CONF_BETA_HI = float(M["conf_beta_hi"])
    mdl, ves_axis, beta = M["model"], M["ves_axis"], float(M["beta"])
    id2name, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()

    vi = nib.load(a.vessel); ves = np.asanyarray(vi.dataobj)
    sp = np.array(vi.header.get_zooms()[:3], dtype=float)
    ai = nib.load(a.aneu); pred = np.asanyarray(ai.dataobj)
    nodes = C5.load_bp(os.path.dirname(a.bp), os.path.basename(a.bp).replace(".json", ""))
    rows, lesions = C5.extract_case_rows(pred, ves, sp, ves_names, nodes, None)
    cid = os.path.basename(a.aneu).replace(".nii.gz", "")
    for r in rows: r["case"] = cid
    out = np.zeros(pred.shape, dtype=np.int32)
    for r in rows:
        name = C5.predict_one(mdl, r, beta)
        if name is None: continue
        oid = name2id.get(name)
        if oid is None: continue
        out[lesions == r["lesion_mask_idx"]] = oid
        if C5.TOPK_N > 1:
            C5._emit_topk(out, lesions, r, mdl, beta, name, name2id)
    if C5.OUT_DILATE > 0 and out.any():
        fg = out > 0
        grown = ndimage.binary_dilation(fg, iterations=C5.OUT_DILATE)
        _, nn = ndimage.distance_transform_edt(~fg, return_indices=True)
        newv = grown & ~fg
        out[newv] = out[nn[0][newv], nn[1][newv], nn[2][newv]]
    img = nib.Nifti1Image(out.astype(np.uint8), ai.affine, ai.header); img.set_data_dtype(np.uint8)
    nib.save(img, a.out)
    print(f"[final_infer] {cid}: 병변 {len(rows)} → 라벨 {sorted(int(x) for x in np.unique(out) if x)}  → {a.out}")


if __name__ == "__main__":
    main()
