#!/usr/bin/env bash
# 감시자 (사용자 지시 2026-08-15: "30분 이상 변하는 거 없으면 멈췄는지 의심하고 해결해.
# 계속 안 끊기고 마무리되게 신경써").
#
# 5분마다 세 작업의 **진행 지표**를 보고, 30분간 변화가 없으면 멈춘 것으로 판단해 되살린다.
#   1) A6-2 fold3/4 학습   지표=epoch 카운트   복구=--c 로 재개
#   2) CPU 실험 큐(C11/C15/C14)  지표=로그 mtime  복구=큐 재기동(완료분은 캐시로 건너뜀)
#   3) 앙상블 파이프라인    지표=로그 mtime      복구=재기동(멱등: 산출물 있으면 건너뜀)
# 디스크가 모자라면 안전한 임시물만 정리한다.
#
# 중지: kill $(cat experiments/watchdog_0815.pid)
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
LOG="$E/watchdog_0815.log"; STALL=1800; INTERVAL=300; MIN_FREE_GB=15
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
echo $$ > "$E/watchdog_0815.pid"
exec >> "$LOG" 2>&1
log(){ echo "[wd $(date -u +'%m-%d %H:%M:%S')] $*"; }

epochs(){ # $1=fold
  local d="$E/A6-2_resencl_adaptivenorm_topk_417_f$1"
  [ -d "$d" ] || { echo 0; return; }
  grep -ch "Epoch time" "$d"/results/*/*/fold_$1/training_log_*.txt 2>/dev/null | tail -1 || echo 0
}
mtime(){ [ -f "$1" ] && stat -c %Y "$1" || echo 0; }
now(){ date +%s; }

log "=========================================================="
log "감시 시작 (주기 ${INTERVAL}s, 정체판정 ${STALL}s)"

declare -A last_val last_chg
for k in f3 f4 queue ens; do last_val[$k]=""; last_chg[$k]=$(now); done

while :; do
  # ---- 디스크 ----
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "${FREE:-999}" -lt "$MIN_FREE_GB" ]; then
    log "디스크 여유 ${FREE}GB — 임시물 정리"
    rm -rf "$E"/_discarded_* "$E"/__t__* 2>/dev/null
    log "  정리 후 $(df --output=avail -BG / | tail -1 | tr -dc '0-9')GB"
  fi

  # ---- 1) 학습 fold3/4 ----
  train_alive=$(pgrep -c -f 'nnUNetv2_train 720' || echo 0)
  all_done=1
  for f in 3 4; do
    [ -f "$E/A6-2_resencl_adaptivenorm_topk_417_f$f"/results/*/*/fold_$f/checkpoint_final.pth ] 2>/dev/null || all_done=0
  done
  if [ "$all_done" -eq 0 ]; then
    for f in 3 4; do
      v=$(epochs "$f"); k="f$f"
      if [ "$v" != "${last_val[$k]}" ]; then last_val[$k]="$v"; last_chg[$k]=$(now); fi
      age=$(( $(now) - ${last_chg[$k]} ))
      if [ "$age" -gt "$STALL" ]; then
        log "⚠️ fold$f 정체 ${age}s (epoch $v) — checkpoint에서 재개"
        GPU=$(( f == 3 ? 0 : 1 )) NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          nohup bash "$S/run_experiment.sh" 720 3d_fullres "$f" \
          "A6-2_resencl_adaptivenorm_topk_417_f$f" \
          -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE --c \
          >> "$E/a62_5fold.log" 2>&1 &
        disown; last_chg[$k]=$(now)
        log "   fold$f 재개 기동"
      fi
    done
  fi

  # ---- 2) CPU 실험 큐 ----
  qlog="$E/c11_c15_chain.log"
  if ! grep -q "큐 완료" "$qlog" 2>/dev/null; then
    v=$(mtime "$qlog"); k=queue
    if [ "$v" != "${last_val[$k]}" ]; then last_val[$k]="$v"; last_chg[$k]=$(now); fi
    age=$(( $(now) - ${last_chg[$k]} ))
    if [ "$age" -gt "$STALL" ] && ! pgrep -f 'chain_c11_c1[5]\.sh' >/dev/null; then
      log "⚠️ CPU 큐 정체/사망 ${age}s — 재기동 (완료분은 캐시로 건너뜀)"
      TOPANEU_ROOT="$R" nohup bash "$S/chain_c11_c15.sh" >> "$qlog" 2>&1 & disown
      last_chg[$k]=$(now)
    fi
  fi

  # ---- 3) 앙상블 파이프라인 ----
  elog="$E/ensemble_a62.log"
  if ! grep -q "앙상블 파이프라인 완료" "$elog" 2>/dev/null; then
    if ! pgrep -f 'chain_ensemble_a6[2]\.sh' >/dev/null; then
      log "앙상블 파이프라인 미기동 — 시작"
      TOPANEU_ROOT="$R" nohup bash "$S/chain_ensemble_a62.sh" >/dev/null 2>&1 & disown
      last_chg[ens]=$(now)
    else
      v=$(mtime "$elog"); k=ens
      if [ "$v" != "${last_val[$k]}" ]; then last_val[$k]="$v"; last_chg[$k]=$(now); fi
      age=$(( $(now) - ${last_chg[$k]} ))
      # 학습 대기 중에는 로그가 안 변하는 게 정상이므로 학습이 끝난 뒤에만 정체로 본다
      if [ "$age" -gt "$STALL" ] && [ "$all_done" -eq 1 ]; then
        log "⚠️ 앙상블 정체 ${age}s — 재기동"
        pkill -f 'chain_ensemble_a6[2]\.sh'
        TOPANEU_ROOT="$R" nohup bash "$S/chain_ensemble_a62.sh" >/dev/null 2>&1 & disown
        last_chg[$k]=$(now)
      fi
    fi
  fi

  # ---- 전부 끝났으면 종료 ----
  if [ "$all_done" -eq 1 ] && grep -q "큐 완료" "$qlog" 2>/dev/null \
     && grep -q "앙상블 파이프라인 완료" "$elog" 2>/dev/null; then
    log "✅ 모든 작업 완료 — 감시 종료"
    exit 0
  fi

  # 2시간마다 생존 신호
  if [ $(( $(date +%s) % 7200 )) -lt "$INTERVAL" ]; then
    log "(heartbeat) 학습 프로세스 ${train_alive}개 | fold3 $(epochs 3)/250 | fold4 $(epochs 4)/250"
  fi
  sleep "$INTERVAL"
done
