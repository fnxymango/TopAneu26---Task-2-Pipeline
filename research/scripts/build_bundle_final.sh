#!/usr/bin/env bash
# 최종모델(X5+gC · 시드3) 도커 제작용 번들 (2026-08-28). build_bundles.sh(08-21, B) 를 새 검출기로 갱신.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; OUT="${OUT_DIR:-/home/sblee}"; D="${BUNDLE_NAME:-$OUT/sblee_final_model}"
S="$R/code/sblee/nnunet/scripts"; AN="$R/code/sblee/nnunet/analysis"; E="$R/experiments"
P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
log(){ echo "[bundle $(TZ=Asia/Seoul date +'%m-%d %H:%M')] $*"; }
rm -rf "$D"; mkdir -p "$D"/{models/detector,models/vessel,code/sblee/nnunet/scripts,code/sblee/nnunet/analysis,code/TopAneu-26/eval/task2,experiments/V5_vessel_classweighted_postproc_417,vendor,nnunet_raw_meta/{Dataset722_TopAneuPjh3cls417,Dataset800_TopAneuVessel417},dataset,sanity/{input_raw,expected/{stage1_det_P55ff,stage2_vessel_pp,stage2_bp,stage3_location}},env}

log "1) 검출기 X5 — Dataset722 pjh 3-class, 5폴드 checkpoint_best"
DTR="nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"
DSRC="$E/P1_pjh3cls_stock250_iso04_f0/results/Dataset722_TopAneuPjh3cls417/$DTR"
DD="$D/models/detector/Dataset722_TopAneuPjh3cls417/$DTR"; mkdir -p "$DD"
cp "$DSRC"/{plans.json,dataset.json} "$DD/"
for i in 0 1 2 3 4; do mkdir -p "$DD/fold_$i"; cp "$DSRC/fold_$i/checkpoint_best.pth" "$DD/fold_$i/"; done
cp "$R/nnunet/nnUNet_raw/Dataset722_TopAneuPjh3cls417/dataset.json" "$D/nnunet_raw_meta/Dataset722_TopAneuPjh3cls417/"

log "2) 혈관 V4-2 fold0 (custom trainer → vendor 필요)"
VTR="nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
VSRC="$E/V4-2_vessel_classweighted_417_500ep/results/Dataset800_TopAneuVessel417/$VTR"
VD="$D/models/vessel/Dataset800_TopAneuVessel417/$VTR"; mkdir -p "$VD/fold_0"
cp "$VSRC"/{plans.json,dataset.json} "$VD/"; cp "$VSRC/fold_0/checkpoint_best.pth" "$VD/fold_0/"   # ★ vespp_* 는 best 로 생성됨(final 아님)
cp "$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json" "$D/nnunet_raw_meta/Dataset800_TopAneuVessel417/"
VEN="$R/code/sblee/nnunet/vendor/Skeleton-Recall"
[ -d "$VEN" ] && cp -r "$VEN" "$D/vendor/" || cp -r /home/sblee/TOPANEU_BUNDLE_B_inference/vendor/Skeleton-Recall "$D/vendor/"
rm -rf "$D/vendor/Skeleton-Recall/build" "$D/vendor/Skeleton-Recall/nnunetv2.egg-info"

log "3) 코드"
cp "$S"/{p_build_3cls.py,c7_detect_postproc.py,postprocess_vessel.py,postproc_params.json,c4_branchpoint_graph.py,c5_location_v2.py,d9xx_lib.py,c8_classifier_cv.py,c10_landmark_coords.py,final_infer.py,pipeline_case.py,det_filter.py,mha_nifti.py} "$D/code/sblee/nnunet/scripts/" 2>/dev/null
cp "$AN"/{e11_feat_hyb_ov.json,final_rf_seed3.pkl,c10_feat_train.json} "$D/code/sblee/nnunet/analysis/"
cp "$R"/code/TopAneu-26/eval/task2/*.py "$D/code/TopAneu-26/eval/task2/" 2>/dev/null
mkdir -p "$D/dataset/TopAneu"; cp "$R/dataset/TopAneu/dataset_split.json" "$D/dataset/TopAneu/"

log "4) sanity 4케이스 (CT1·MR3, 병변/gC조각/무병변 포함): 원본 + 단계별 기대출력"
for c in topaneu_center2_ct_192 topaneu_center1_mr_148 topaneu_center2_mr_038 topaneu_center1_mr_001; do
  cp -L "$P/in_test/${c}_0000.nii.gz" "$D/sanity/input_raw/"
  cp "$P/aneu_test_P55ff/$c.nii.gz" "$D/sanity/expected/stage1_det_P55ff/"
  cp "$P/vespp_test/$c.nii.gz"      "$D/sanity/expected/stage2_vessel_pp/"
  cp "$BP/vespp_test/$c.json"       "$D/sanity/expected/stage2_bp/"
  cp "$E/final_pred_seed3_test/$c.nii.gz" "$D/sanity/expected/stage3_location/"
  cp "$R/dataset/TopAneu/location_masks/$c.nii.gz" "$D/sanity/expected/" 2>/dev/null && mv "$D/sanity/expected/$c.nii.gz" "$D/sanity/expected/${c}_GT_location.nii.gz"
done

cp "$R/experiments/V5_vessel_classweighted_postproc_417/postproc_params.json" "$D/experiments/V5_vessel_classweighted_postproc_417/"

log "4b) d9xx_lib 이 기대하는 경로 레이아웃"
mkdir -p "$D/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417" "$D/nnunet/nnUNet_raw/Dataset722_TopAneuPjh3cls417"
cp "$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json" "$D/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/"
cp "$R/nnunet/nnUNet_raw/Dataset722_TopAneuPjh3cls417/dataset.json" "$D/nnunet/nnUNet_raw/Dataset722_TopAneuPjh3cls417/"

log "5) 환경"
/home/sblee/miniconda3/envs/sblee_topaneu/bin/pip freeze > "$D/env/requirements_nnunet_env.txt" 2>/dev/null
/home/sblee/miniconda3/envs/sbaneu2/bin/pip freeze     > "$D/env/requirements_analysis_env.txt" 2>/dev/null
/home/sblee/miniconda3/envs/sblee_topaneu/bin/python -c "import torch,sys;print('python',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda)" > "$D/env/versions.txt" 2>/dev/null
/home/sblee/miniconda3/envs/sbaneu2/bin/python -c "import sys,sklearn,numpy,scipy,nibabel;print('analysis python',sys.version.split()[0],'sklearn',sklearn.__version__,'numpy',numpy.__version__,'scipy',scipy.__version__,'nibabel',nibabel.__version__)" >> "$D/env/versions.txt" 2>/dev/null
cp "$S/README_BUNDLE.md" "$D/README.md"
( cd "$D" && find . -type f -not -path './vendor/*' | sort > MANIFEST.txt && md5sum models/*/*/*/fold_*/*.pth code/sblee/nnunet/analysis/final_rf_seed3.pkl >> MANIFEST.txt )
log "완료 $(du -sh "$D" | cut -f1) → $D"
