#!/usr/bin/env bash
# F1FILL — 저장된 모든 채점본을 **동결 eval(660da7a · F1 포함)** 로 다시 매긴다.
#
# 왜: 2026-09-11 10:27 KST, 채점이 도는 중에 ~/TopAneu-26 이 갱신됐다(F1 추가).
#     이미 떠 있던 프로세스는 옛 모듈을 들고 있어 F1 없이 기록했다 —
#     한 실험 안에서 지표 집합이 갈렸다. eval 은 고정본이어야 한다.
# 무엇이 바뀌었나: **F1 추가뿐**. 기존 6지표 계산식은 한 글자도 안 바뀌었다(diff 확인).
#     따라서 기존 판정은 전부 유효하고, F1 칸만 채우면 된다.
# 앞으로: 전부 neweval2.py 를 쓴다 — 커밋 해시를 결과에 남기고 per-class TP/FP/FN/TN
#     원자료를 저장하므로, 공식이 지표를 또 추가해도 재채점이 필요 없다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; H=$R/experiments/H1_patchfilter
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
exec 3>"$D/f1fill.lock"; flock -n 3 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][f1] $*" | tee -a "$ST"; }

todo=(); skip=0
for f in "$H"/scores/*.json; do
  b=$(basename "$f" .json)
  grep -q '"F1"' "$f" 2>/dev/null && { skip=$((skip+1)); continue; }
  [ -d "$H/pred/$b" ] || { log "★예측 없음 $b — 건너뜀"; continue; }
  todo+=("$b")
done
log "대상 ${#todo[@]}판 (이미 F1 있음 $skip)"

n=0
for b in "${todo[@]}"; do
  sp=test; case "$b" in *_val_s*) sp=val;; esac
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    "$PY" "$D/neweval2.py" "$sp" "$H/pred/$b" >> "$D/f1fill.log" 2>&1 &
  n=$((n+1))
  [ $((n%16)) -eq 0 ] && { wait; log "  진행 $n/${#todo[@]}"; }
done
wait

ok=0; bad=0
for f in "$H"/scores/*.json; do
  if grep -q '"F1"' "$f" 2>/dev/null; then ok=$((ok+1)); else bad=$((bad+1)); fi
done
log "완료 · F1 있음 $ok · 없음 $bad"
touch "$D/.done_f1fill"
