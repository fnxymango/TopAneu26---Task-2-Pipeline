#!/usr/bin/env python3
"""V4-D 1단계 — train 검출 blob OOF 학습표 구축 (train 만 · test·val 안 봄 · 검출기는 재학습하지 않는다).

왜: 학습표는 GT 병변 · GT 혈관에서 피처를 재는데 추론은 검출 blob · 예측 혈관에서 잰다(POSTMORTEM 3 · PROJECT_RULES.md).
V3-P 는 혈관 쪽만 추론 조건으로 바꿨다. 여기서는 병변 쪽도 **실제 검출 blob** 으로 바꾼 행을 만든다.
기록 grep(2026-09-15): 검출 blob 을 분류기 학습 행으로 쓴 실험 없음(C17 은 FP 기각기 · 미적용, blob 마스크는 소실).

OOF 출처: E9 번들 검출기 fold_6~9 (= 개정판 P5 run 의 split 1~4 · checkpoint init_args.fold 로 확인).
  split: nnunet/preproc_json_backup/Dataset722_TopAneuPjh3cls417_20260904run/splits_final.json
  split 0 은 전체 train(val=공식 val) 이라 쓰지 않는다. split 1~4 의 val 합집합 = train 291 전부.
  각 train 케이스는 자기를 학습에 안 쓴 폴드 1개로만 추론한다 → 누수 없음.
  ⚠ 제출 검출기(b1ff)와는 다른 모델이다. blob 성질은 비슷하나 같지 않다(e2e 에서 확인).
경로: pipeline_case 와 같다 — robust z → 단일 폴드 예측 → 라벨 2 → det_filter(5복셀·1mm, vespp_train)
      → C5.extract_case_rows(blob, vespp_train, vespp_train 그래프) → GT 위치 마스크와 최다 겹침 클래스를 gt_loc 로.
      겹침 없음(환각)은 gt_loc=None 으로 남긴다(학습에서 제외).
출력: v4d_oof/rows/<case>.json · v4d_oof/seg/<case>.nii.gz · 합본 code/sblee/nnunet/analysis/v4d_feat_blob_oof.json

── 이후 단계 관문 (결과 보기 전 고정 · 2026-09-15) ──
 2단계 스크리닝(train OOF · 케이스 묶음 5폴드 × 5시드 · V5 틀 · 평가 행 = 보류 폴드의 **검출 blob TP 행**):
   팔 base = 기준 하이브리드표(e11_feat_hyb_ov_NEW) 학습 · 팔 V4D = 검출 blob TP 행 학습 · 팔 V4D+ = 두 표 행 합침.
   통과 = 평균 Δmacro-recall ≥ +0.02 ∧ 평균 Δtop1 ≥ −0.005 ∧ Δmacro>0 시드 ≥ 4/5.
   두 팔 모두 통과하면 Δmacro 큰 쪽 하나만 e2e. 둘 다 미달 → V4-D 닫음.
 3단계 e2e: K0 장치(α 0.05 · 안전 ① ΔFP ≤ ΔTP · 안전 ② 기준 무오답 클래스 새 오답 ≤ ΔTP) → 통과 시 시드 5~9 복제.
사용: v4d_oof.py run <shard> <nshard> · v4d_oof.py merge
"""
import json, os, sys, glob, tempfile, collections
import numpy as np, nibabel as nib
from scipy import ndimage

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
D = f"{R}/experiments/D1_newdata"; A = f"{R}/code/sblee/nnunet/analysis"
SC = f"{R}/code/sblee/nnunet/scripts"
DET = "/home/sblee/e9_bundle_experiment/models/detector/Dataset722_TopAneuPjh3cls417/nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans722iso04__3d_fullres"
SPLITS = f"{R}/nnunet/preproc_json_backup/Dataset722_TopAneuPjh3cls417_20260904run/splits_final.json"
OUT = f"{D}/v4d_oof"
ST = np.ones((3, 3, 3), bool)


def assignment():
    sp = json.load(open(SPLITS))
    tr = set(json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]["train"])
    fold = {}
    for k in (1, 2, 3, 4):
        for c in sp[k]["val"]:
            assert c in tr and c not in fold, c
            fold[c] = k + 5
    assert set(fold) == tr, (len(fold), len(tr))
    return fold


def run(shard, nshard):
    sys.path.insert(0, SC); os.environ.setdefault("TOPANEU_ROOT", R)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from pipeline_case import robust_z
    from det_filter import filter_case
    import c5_location_v2 as C5, d9xx_lib as L
    id2name, _ = L.official_location_names()
    ves_names = L.vessel_dense_names()
    fold = assignment()
    cs = sorted(fold)[shard::nshard]
    os.makedirs(f"{OUT}/rows", exist_ok=True); os.makedirs(f"{OUT}/seg", exist_ok=True)
    preds = {}
    tmp = tempfile.mkdtemp(prefix=f"v4d{shard}_", dir="/tmp")
    for i, cid in enumerate(cs, 1):
        fo = f"{OUT}/rows/{cid}.json"
        if os.path.exists(fo):
            continue
        k = fold[cid]
        if k not in preds:
            p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                                device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
            p.initialize_from_trained_model_folder(DET, use_folds=(k,), checkpoint_name="checkpoint_best.pth")
            preds[k] = p
        p = preds[k]
        img = nib.load(f"{R}/dataset/TopAneu/images/{cid}_0000.nii.gz")
        mod = "CT" if "_ct_" in cid else "MR"
        z = robust_z(np.asanyarray(img.dataobj), mod)
        zin = f"{tmp}/{cid}_0000.nii.gz"
        nib.save(nib.Nifti1Image(z, img.affine, img.header), zin); del z
        rw = p.plans_manager.image_reader_writer_class()
        im, props = rw.read_images([zin])
        segp = f"{OUT}/seg/{cid}"
        p.predict_single_npy_array(im, props, None, segp, False)
        os.remove(zin); del im
        torch.cuda.empty_cache()
        seg = np.asanyarray(nib.load(segp + ".nii.gz").dataobj)
        vi = nib.load(f"{R}/experiments/_c1_realpred/vespp_train/{cid}.nii.gz")
        ves = np.asanyarray(vi.dataobj); sp = np.array(vi.header.get_zooms()[:3], float)
        assert seg.shape == ves.shape, (cid, seg.shape, ves.shape)
        det_m = filter_case(seg == 2, ves, sp, 5, 1.0)
        nodes = C5.load_bp(f"{R}/experiments/_c4_bpgraph/vespp_train", cid)
        rows, lesions = C5.extract_case_rows(det_m, ves, sp, ves_names, nodes, None)
        loc = np.asanyarray(nib.load(f"{R}/dataset/TopAneu/location_masks/{cid}.nii.gz").dataobj)
        glab, _ = ndimage.label(loc > 0, structure=ST)
        for r in rows:
            m = lesions == r["lesion_mask_idx"]
            v = loc[m]; v = v[v > 0]
            r["case"] = cid; r["det_fold"] = k
            if v.size:
                gid = int(np.bincount(v).argmax())
                r["gt_loc"] = id2name[gid]
                g = glab[m]; g = g[g > 0]
                r["gt_comp"] = int(np.bincount(g).argmax())
                r["gt_ov_frac"] = float(v.size / m.sum())
            else:
                r["gt_loc"] = None
        json.dump(rows, open(fo, "w"))
        print(f"[{shard}] {i}/{len(cs)} {cid} f{k} · blob {len(rows)} · TP {sum(r['gt_loc'] is not None for r in rows)}", flush=True)


def merge():
    fold = assignment()
    got = [c for c in sorted(fold) if os.path.exists(f"{OUT}/rows/{c}.json")]
    assert len(got) == len(fold), (len(got), len(fold))
    rows = [r for c in got for r in json.load(open(f"{OUT}/rows/{c}.json"))]
    json.dump(rows, open(f"{A}/v4d_feat_blob_oof.json", "w"))
    base = [r for r in json.load(open(f"{A}/e11_feat_hyb_ov_NEW.json")) if r.get("gt_loc")]
    tp = [r for r in rows if r["gt_loc"]]
    comps = {(r["case"], r["gt_comp"]) for r in tp}
    print(f"blob {len(rows)} · TP {len(tp)} · 환각 {len(rows) - len(tp)} · 검출된 GT 병변 {len(comps)} / 기준표 GT 병변 {len(base)}")
    cb = collections.Counter(r["gt_loc"] for r in base); ct = collections.Counter(r["gt_loc"] for r in tp)
    print("클래스 수 기준표", len(cb), "· blob TP", len(ct), "· blob 에 없는 클래스", sorted(set(cb) - set(ct)))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]))
    else:
        merge()
