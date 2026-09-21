#!/usr/bin/env bash
# RESCORE — H6 가 끝나면 **검출기 결정에 쓰이는 팔만** 동결 eval(660da7a · F1 포함)로 다시 매긴다.
#
# 왜 전부가 아닌가: 60765a5 → 660da7a 변경은 **F1 추가뿐**이고 기존 6지표 계산식은 안 바뀌었다.
#   따라서 이미 끝난 판정(패치필터 채택 5/6 · 개정피처 미채택 0/6 · gC OFF 무효)은 여유가 커서
#   F1 을 넣어도 안 뒤집힌다. 다시 매길 값어치가 있는 건 **아슬아슬한 판정**뿐이다 —
#   H4 의 e9f3P3 가 val 4/6 으로 통과했는데 7지표 기준에서 갈릴 수 있다.
#
# 나머지 110 판은 6지표 버전으로 남긴다. 앞으로는 전부 neweval2.py 를 쓴다
# (동결 eval · 커밋해시 기록 · per-class 원자료 저장 → 지표가 또 늘어도 재채점 불필요).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; H=$R/experiments/H1_patchfilter
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
exec 5>"$D/rescore.lock"; flock -n 5 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][rescore] $*" | tee -a "$ST"; }

log "H6 완료 대기"
for i in $(seq 1 720); do [ -f "$D/.done_h6" ] && break; sleep 20; done
[ -f "$D/.done_h6" ] || { log "★H6 미완 — 중단"; exit 1; }
log "H6 완료 확인 · 재채점 시작"

TAGS="b1on_pf e9on_pf e9f3P3on_pf e9f3P3_pf e9off_pf"
# split 별로 묶어 한 프로세스가 여러 태그를 처리한다 — GT 로드를 공유해 시간을 줄인다.
n=0
for sp in val test; do
  for t in $TAGS; do
    d="$H/pred/${t}_${sp}_s"
    for sd in 0 1 2 3 4; do
      [ -d "${d}${sd}" ] || { log "★예측 없음 ${t}_${sp}_s${sd}"; continue; }
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        "$PY" "$D/neweval2.py" "$sp" "${d}${sd}" >> "$D/rescore.log" 2>&1 &
      n=$((n+1)); [ $((n%12)) -eq 0 ] && wait
    done
  done
done
wait
log "재채점 완료 $n 판"

ok=0; bad=0
for t in $TAGS; do for sp in test val; do for sd in 0 1 2 3 4; do
  f="$H/scores/${t}_${sp}_s${sd}.json"
  if grep -q '"F1"' "$f" 2>/dev/null && grep -q '660da7a' "$f" 2>/dev/null; then ok=$((ok+1)); else bad=$((bad+1)); fi
done; done; done
log "검증: F1+해시 있음 $ok · 없음 $bad"
touch "$D/.done_rescore"
