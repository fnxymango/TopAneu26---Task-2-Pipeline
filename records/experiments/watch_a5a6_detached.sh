#!/usr/bin/env bash
# A5-2(plain z) vs A6-2(adaptive norm) 감시 — 세션과 무관하게 도는 detached 버전.
# 이벤트는 아래 EVLOG에 append 된다. 중지: kill <이 스크립트 PID>  (PID는 .pid 파일에 기록)
E=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments
EVLOG=$E/watch_a5a6_events.log
A5=$E/A5-2_resencl_plainz_topk_417_f0/results/Dataset720_TopAneuBinary417/nnUNetTrainerTverskyTopkCE__nnUNetResEncUNetLPlans__3d_fullres/fold_0/training_log_2026_8_13_16_40_18.txt
A6=$E/A6-2_resencl_adaptivenorm_topk_417_f0/results/Dataset720_TopAneuBinary417/nnUNetTrainerTverskyTopkCE__nnUNetResEncUNetLPlansAdaptive__3d_fullres/fold_0/training_log_2026_8_13_16_40_18.txt

exec >> "$EVLOG" 2>&1
echo "$$" > "$E/watch_a5a6.pid"
now() { echo "$(date -u +'%m-%d %H:%M')UTC / $(TZ=Asia/Seoul date +'%H:%M')KST"; }

echo "=========================================================="
echo "[$(now)] 감시 시작 (detached, PID $$)"
echo "  A5-2 plainZ  : $(grep -c 'Epoch time' "$A5") epoch 완료"
echo "  A6-2 adaptive: $(grep -c 'Epoch time' "$A6") epoch 완료"

TRAIN_PAT='Mean Validation Dice|Training done|Traceback|out of memory|CUDA error|RuntimeError|Killed|OOM'
CHAIN_PAT='^\[gpu0-chain|^\[gpu1-c1|Traceback|out of memory|CUDA error|ABORT'

tail -n0 -F "$A5" 2>/dev/null | grep -E --line-buffered "$TRAIN_PAT" | sed -u 's/^/[A5-2 plainZ  ] /' &
tail -n0 -F "$A6" 2>/dev/null | grep -E --line-buffered "$TRAIN_PAT" | sed -u 's/^/[A6-2 adaptive] /' &
tail -n0 -F "$E/gpu1_chain_c1.log" 2>/dev/null | grep -E --line-buffered "$CHAIN_PAT" | sed -u 's/^/[GPU1 체인] /' &
tail -n0 -F "$E/gpu0_chain_a6.log" 2>/dev/null | grep -E --line-buffered "$CHAIN_PAT" | sed -u 's/^/[GPU0 체인] /' &

# 안전망: 로그 없이 죽는 경우(강제 kill 등)
watch_pid() {
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
  echo "[$(now)] [프로세스종료] $2 (PID $1) 종료 — 직전에 Dice 줄이 없었다면 비정상 종료"
}
watch_pid 3073497 "A6-2 adaptive / GPU0" &
watch_pid 3073499 "A5-2 plainZ / GPU1" &

# 2시간마다 생존 신호 (감시가 살아있는지 확인용)
while true; do
  sleep 7200
  echo "[$(now)] (heartbeat) 감시 동작 중 — GPU0/1 학습 프로세스: $(pgrep -fc 'nnUNetv2_train 720' || echo 0)개"
done
