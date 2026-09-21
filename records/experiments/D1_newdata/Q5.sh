#!/usr/bin/env bash
# Q5 — 유형 마스크 활용 트랙 (2026-09-15 사용자 지시 "진행 중 큐와 안 겹치게")
#  대기: K1·K2 e2e 가 둘 다 끝날 때까지(.done_k1 · .done_k2) — CPU 를 겹치지 않는다. 영상 재수신은 네트워크 작업이라 겹쳐도 됨.
#  순서(직렬): ① T1 라벨 규약 측정 → ② T3 비낭형 가중 스크리닝 → ③ T2 분절 걸침 스크리닝 → ④ T3 통과 시 T3.sh e2e
#  T1 규칙 · T2 는 통과해도 추론 구현·설계가 필요해 자동 e2e 하지 않고 보고만 한다. 각 관문은 스크립트 머리말에 고정.
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata; V=$D/../V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
exec 9>"$D/q5.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][q5] $*" | tee -a "$D/STATUS.log"; }
until [ -e "$D/.done_k1" ] && [ -e "$D/.done_k2" ]; do sleep 60; done
log "K1·K2 종료 확인 → 유형 트랙 시작"
filt(){ grep -v "UserWarning\|n_jobs = min"; }
log "① T1 걸친 병변 라벨 규약 측정"
"$PY" -u "$D/t1_convention.py" 2>&1 | filt > "$V/RESULTS_T1.md" || log "★T1 실패"
log "  T1 → $(grep -o '판정:[^*]*' $V/RESULTS_T1.md)"
log "② T3 비낭형 표본 가중 스크리닝"
"$PY" -u "$D/t3_screen.py" 2>&1 | filt > "$V/RESULTS_T3_SCREEN.md" || log "★T3 스크리닝 실패"
log "  T3 → $(grep -o '관문 →[^*]*' $V/RESULTS_T3_SCREEN.md)"
log "③ T2 분절 걸침 피처 스크리닝"
"$PY" -u "$D/t2_span.py" 2>&1 | filt > "$V/RESULTS_T2_SCREEN.md" || log "★T2 실패"
log "  T2 → $(grep -o '관문 →[^*]*' $V/RESULTS_T2_SCREEN.md)"
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$D/t3_gate.json'))['ok'] else 1)" 2>/dev/null; then
  log "④ T3 스크리닝 통과 → T3.sh e2e (K0 장치)"
  bash "$D/T3.sh" > "$D/T3.out" 2>&1
else
  log "④ T3 스크리닝 미달 → e2e 생략"
fi
touch "$D/.done_q5"; log "Q5 완료"
