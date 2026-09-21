#!/usr/bin/env bash
# 이식용 번들 2종 생성 (2026-08-21).
#   A_eval       55MB   test83 점수 재현만. 검출/혈관 결과가 이미 들어있어 nnUNet·GPU 불필요.
#   B_inference  ~5GB   새 영상 추론. 체크포인트 6개 + vendor 트레이너 포함.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; OUT="${OUT_DIR:-/home/sblee}"
A_DIR="$OUT/TOPANEU_BUNDLE_A_eval"; B_DIR="$OUT/TOPANEU_BUNDLE_B_inference"
S="$R/code/sblee/nnunet/scripts"; AN="$R/code/sblee/nnunet/analysis"
P="$R/experiments/_c1_realpred"; BP="$R/experiments/_c4_bpgraph"
log(){ echo "[bundle $(date -u +'%m-%d %H:%M:%S')] $*"; }

log "=== A: 평가 재현 번들 ==="
rm -rf "$A_DIR"; mkdir -p "$A_DIR"/{code/sblee/nnunet/{scripts,analysis},code/TopAneu-26/eval/task2,experiments/_c1_realpred,experiments/_c4_bpgraph,dataset/TopAneu,nnunet/nnUNet_raw/Dataset800_TopAneuVessel417}
cp "$S"/{c5_location_v2.py,d9xx_lib.py,c8_classifier_cv.py} "$A_DIR/code/sblee/nnunet/scripts/"
cp "$R"/code/TopAneu-26/eval/task2/*.py "$A_DIR/code/TopAneu-26/eval/task2/"
cp "$AN"/{e11_feat_hyb_ov.json,c10_feat_train.json} "$A_DIR/code/sblee/nnunet/analysis/"
cp -r "$P/vespp_test" "$P/aneu_test_probavgf" "$A_DIR/experiments/_c1_realpred/"
cp -r "$BP/vespp_test" "$A_DIR/experiments/_c4_bpgraph/"
cp "$R/dataset/TopAneu/dataset_split.json" "$A_DIR/dataset/TopAneu/"
cp -r "$R/dataset/TopAneu/location_masks" "$A_DIR/dataset/TopAneu/"
cp "$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/dataset.json" "$A_DIR/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/"
log "  A 완료 $(du -sh "$A_DIR"|cut -f1)"

log "=== B: 추론 번들 (A 위에 얹는다) ==="
rm -rf "$B_DIR"; cp -r "$A_DIR" "$B_DIR"
mkdir -p "$B_DIR"/{models/{detector,vessel},vendor,nnunet/nnUNet_raw/Dataset720_TopAneuBinary417}
# 검출 A6-2 5폴드 — nnUNet 모델폴더 구조 그대로(plans/dataset + fold_N/checkpoint)
DTR="nnUNetTrainerTverskyTopkCE__nnUNetResEncUNetLPlansAdaptive__3d_fullres"
DD="$B_DIR/models/detector/Dataset720_TopAneuBinary417/$DTR"; mkdir -p "$DD"
for i in 0 1 2 3 4; do
  SRC="$R/experiments/A6-2_resencl_adaptivenorm_topk_417_f$i/results/Dataset720_TopAneuBinary417/$DTR"
  [ -d "$SRC" ] || { log "  ★ 검출 f$i 없음"; continue; }
  [ -f "$DD/plans.json" ] || cp "$SRC"/{plans.json,dataset.json} "$DD/" 2>/dev/null
  mkdir -p "$DD/fold_$i"; cp "$SRC/fold_$i/checkpoint_best.pth" "$DD/fold_$i/"
  log "  검출 fold$i 복사"
done
# 혈관 V4-2 fold0
VTR="nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres"
VSRC="$R/experiments/V4-2_vessel_classweighted_417_500ep/results/Dataset800_TopAneuVessel417/$VTR"
VD="$B_DIR/models/vessel/Dataset800_TopAneuVessel417/$VTR"; mkdir -p "$VD/fold_0"
cp "$VSRC"/{plans.json,dataset.json} "$VD/" 2>/dev/null
cp "$VSRC/fold_0/checkpoint_final.pth" "$VD/fold_0/"
log "  혈관 fold0 복사"
# 추론에 필요한 나머지 스크립트
cp "$S"/{c4_branchpoint_graph.py,c7_detect_postproc.py,c10_landmark_coords.py,postprocess_vessel.py,postproc_params.json} \
   "$B_DIR/code/sblee/nnunet/scripts/" 2>/dev/null
cp "$R/nnunet/nnUNet_raw/Dataset720_TopAneuBinary417/dataset.json" \
   "$B_DIR/nnunet/nnUNet_raw/Dataset720_TopAneuBinary417/" 2>/dev/null
# 혈관 트레이너가 사는 vendor — stock nnUNet 으로는 못 부른다
VSRC_DIR="$R/code/sblee/nnunet/vendor/Skeleton-Recall"
[ -d "$VSRC_DIR" ] && { cp -r "$VSRC_DIR" "$B_DIR/vendor/"; log "  vendor/Skeleton-Recall 복사"; } || log "  ★ vendor/Skeleton-Recall 없음"
log "  B 완료 $(du -sh "$B_DIR"|cut -f1)"
log "=== 완료 ==="
