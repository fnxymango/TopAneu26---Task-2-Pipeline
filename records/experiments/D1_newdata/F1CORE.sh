#!/usr/bin/env bash
# F1CORE — 앞으로 쓸 세 팔만 동결 eval(660da7a · 7지표)로 채점.
# 168판을 다 매기지 않는 이유: 128판이 검출기 실험(H1 e9off · H4 폴드 · H6 교체)인데
# 검출기는 제출본 유지로 확정됐고, H5 의 구 학습표 팔도 폐기됐다(PROJECT_RULES.md 0-2장).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; H=$R/experiments/H1_patchfilter
PY=$HOME/miniconda3/envs/sbaneu2/bin/python; ST=$D/STATUS.log
exec 3>"$D/f1core.lock"; flock -n 3 || exit 0
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][f1core] $*" | tee -a "$ST"; }
alive(){ local n=0 p c; for p in /proc/[0-9]*; do [ -r "$p/cmdline" ] || continue
  c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null); case "$c" in *neweval2.py*) n=$((n+1));; esac; done; echo $n; }
log "이전 채점 잔여 대기 ($(alive)개)"
for i in $(seq 1 180); do [ "$(alive)" -eq 0 ] && break; sleep 20; done
log "잔여 0 — 시작"
n=0
for t in b1on_pf b1Non_pf b1Noff_pf; do for sp in test val; do for sd in 0 1 2 3 4; do
  b="${t}_${sp}_s${sd}"
  grep -q '"F1"' "$H/scores/$b.json" 2>/dev/null && continue
  [ -d "$H/pred/$b" ] || { log "★예측 없음 $b"; continue; }
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PY" "$D/neweval2.py" "$sp" "$H/pred/$b" >> "$D/f1core.log" 2>&1 &
  n=$((n+1)); [ $((n%15)) -eq 0 ] && { wait; log "  진행 $n/30"; }
done; done; done
wait
ok=0; for t in b1on_pf b1Non_pf b1Noff_pf; do for sp in test val; do for sd in 0 1 2 3 4; do
  grep -q '"F1"' "$H/scores/${t}_${sp}_s${sd}.json" 2>/dev/null && ok=$((ok+1)); done; done; done
log "완료 · F1 있음 $ok/30"; touch "$D/.done_f1core"
