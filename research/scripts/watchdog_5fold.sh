#!/usr/bin/env bash
# fold 1~4 학습 감시. 세션 종료/OOM 등으로 죽으면 --c(체크포인트 이어받기)로 재시작.
# fold 1~3은 작업 세션 SID에 묶인 채 시작됐고(실행 중이라 SID 변경 불가) nohup만 걸려 있어서,
# 세션이 SIGTERM으로 정리되는 경우를 대비한 보험이다.
# 사용: setsid nohup bash scripts/watchdog_5fold.sh > experiments/watchdog.log 2>&1 &
set -uo pipefail

BASE=/home/user/TopAneu/seg/sblee/nnunet
RUN="$BASE/scripts/run_experiment.sh"
ENVBIN=/home/user/anaconda3/envs/sbaneu2/bin
declare -A GPU=([1]=0 [2]=1 [3]=2 [4]=3)
declare -A TRIES=([1]=0 [2]=0 [3]=0 [4]=0)
MAX_TRIES=2
INTERVAL=300

done_fold() {  # 최종 체크포인트가 있으면 완료
  compgen -G "$BASE/experiments/D600_vessel_skelrec_resencm_250ep_bd0_f$1/results/*/*/fold_$1/checkpoint_final.pth" > /dev/null
}
alive_fold() {
  pgrep -f "nnUNetv2_train 600 3d_fullres $1 " > /dev/null
}

echo "[watchdog] 시작 $(date -Iseconds) | 간격 ${INTERVAL}s, fold당 최대 ${MAX_TRIES}회 재시작"
while true; do
  remaining=0
  for f in 1 2 3 4; do
    if done_fold "$f"; then continue; fi
    remaining=1
    if alive_fold "$f"; then continue; fi
    if [ "${TRIES[$f]}" -ge "$MAX_TRIES" ]; then
      echo "[watchdog] fold$f 재시작 한도(${MAX_TRIES}) 초과 — 포기. 로그 확인 필요. $(date -Iseconds)"
      continue
    fi
    TRIES[$f]=$(( TRIES[$f] + 1 ))
    echo "[watchdog] fold$f 죽음 감지 → --c 로 재시작 (${TRIES[$f]}회차) $(date -Iseconds)"
    GPU="${GPU[$f]}" NPROC=1 ENVBIN="$ENVBIN" nohup "$RUN" 600 3d_fullres "$f" \
      "D600_vessel_skelrec_resencm_250ep_bd0_f$f" \
      -p nnUNetResEncUNetMPlansBD0 -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep --c \
      >> "$BASE/experiments/restart_f$f.out" 2>&1 &
    sleep 30
  done
  [ "$remaining" -eq 0 ] && { echo "[watchdog] 4개 fold 모두 완료 $(date -Iseconds)"; break; }
  sleep "$INTERVAL"
done
