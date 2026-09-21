#!/usr/bin/env bash
# C16 — 5-fold softmax 확률맵 평균 앙상블 (2026-08-16)
#
# 왜: 현행 vote2는 **이진 마스크 다수결**이라 여러 모델의 합집합 성분이 되어 경계가 뭉개진다.
#     실측에서 앙상블/합성/beta 등 "희귀클래스를 공격적으로 예측하게 만드는" 개입이 전부
#     MCC는 올리면서 HD95를 악화시켰다 (0.5887 -> 0.6490 -> 0.6713). 공식 랭킹이 6지표 평균이라
#     이 손실이 MCC 이득을 상쇄한다.
#     nnU-Net의 -f 0 1 2 3 4 는 **softmax를 평균**한 뒤 argmax 하므로 경계가 매끄럽고 보정도 낫다.
#
# 새 학습 없음 — 기존 5개 fold 체크포인트를 심볼릭으로 묶어 재추론만 한다.
# val 은 GPU0, test 는 GPU1 로 병렬.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; M="$E/_c16_a62_5fold/results"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
LOG="$E/c16_probavg.log"
log(){ echo "[c16 $(date -u +'%m-%d %H:%M:%S')] $*" >> "$LOG"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

predict(){ # $1=split $2=gpu
  local SP=$1 G=$2 OUT="$P/aneu_${SP}_probavg"
  [ -d "$OUT" ] && [ "$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)" -gt 0 ] && { log "$SP 이미 있음"; return; }
  log "$SP 5-fold 확률평균 추론 시작 (GPU$G)"
  nnUNet_results="$M" CUDA_VISIBLE_DEVICES="$G" "$ENVBIN/nnUNetv2_predict" \
    -i "$P/in_$SP" -o "$OUT" -d 720 -c 3d_fullres -f 0 1 2 3 4 \
    -tr nnUNetTrainerTverskyTopkCE -p nnUNetResEncUNetLPlansAdaptive \
    -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 >> "$LOG" 2>&1
  log "$SP 종료 status=$?"
}

predict val 0 & PV=$!
predict test 1 & PT=$!
wait $PV; wait $PT
log "추론 완료"

# c7 후처리 (A6-2 최적: min_vox 5, dist 3mm) + 검출지표
for SP in val test; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  $PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_${SP}_probavg" --vessel-dir "$P/$VES" \
      --split "$SP" --tag "probavg_$SP" --save-best "$P/aneu_${SP}_probavgf" \
      --force-cfg 5,3 >> "$LOG" 2>&1
  log "$SP c7 필터 완료"
done

# 위치분류 평가 (현재 최고 설정: C10 좌표 + beta=1.0, C11 합성은 test에서 악화라 제외)
FEAT="$R/code/sblee/nnunet/analysis/c10_feat_train.json"
BEST="--model rf --beta 1.0 --use-pos"
for SP in val test; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  VB=$([ "$SP" = val ] && echo val_pred || echo vespp_test)
  log "C5 평가 $SP"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$SP" \
    --vessel-dir "$P/$VES" --bp-dir "$E/_c4_bpgraph/$VB" \
    --aneurysm-pred-dir "$P/aneu_${SP}_probavgf" $BEST --tag "probavg" >> "$LOG" 2>&1
  log "  $SP 종료 status=$?"
done
log "=== C16 완료 ==="
$PY -u c18_composite_reselect.py >> "$LOG" 2>&1
