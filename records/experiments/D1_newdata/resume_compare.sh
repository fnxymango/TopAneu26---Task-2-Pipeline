#!/usr/bin/env bash
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; ST=$D/STATUS.log
L=$R/nnunet/nnUNet_raw/Dataset720_TopAneuBinary417/labelsTr
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][resume] $*" | tee -a "$ST"; }
log "720 재생성 대기"
for i in $(seq 1 60); do
  n=$(ls "$L"/*.nii.gz 2>/dev/null | wc -l)
  [ "$n" -ge 415 ] && break
  sleep 15
done
n=$(ls "$L"/*.nii.gz 2>/dev/null | wc -l)
if [ "$n" -lt 415 ]; then log "★720 라벨 $n/415 — 재개 취소"; exit 1; fi
log "720 라벨 $n 확인 — c7 산출물 초기화 후 대조 재개"
rm -f "$D/.failed_compare" "$D/cmp_c7.log"
setsid nohup "$D/COMPARE.sh" > "$D/COMPARE.out.2" 2>&1 </dev/null & disown
log "COMPARE 재기동"
