#!/usr/bin/env bash
# CV 예측·채점 드라이버 v2 — 동시 실행 수를 CV_MAXPROC로 조절(RAM 사고 방지).
# 남은 물량에 비례해 fold별 파트 수를 배분한다. 완료된 예측은 건너뛰므로 재실행 안전.
set -uo pipefail
BASE=/home/user/TopAneu/seg/sblee/nnunet
PY=/home/user/anaconda3/envs/sbaneu2/bin/python
N=$BASE/experiments/D600_vessel_skelrec_resencm_250ep_bd0
LOG=$N/cv.log
CV=$N/cv
MAXP=${CV_MAXPROC:-8}
export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
export nnUNet_results=$N/results

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
rem() { echo $(( $(ls "$CV/fold$1/in" 2>/dev/null | wc -l) - $(ls "$CV/fold$1/out"/*.nii.gz 2>/dev/null | wc -l) )); }

say "=== 드라이버 v2 (동시 $MAXP) ==="

for round in 1 2 3; do
  TOTAL=0; for f in 1 2 3 4; do TOTAL=$((TOTAL + $(rem $f))); done
  [ "$TOTAL" -le 0 ] && { say "예측 전부 완료"; break; }
  say "라운드 $round — 남은 $TOTAL 케이스"

  i=0
  for f in 1 2 3 4; do
    R=$(rem $f); [ "$R" -le 0 ] && continue
    P=$(( (MAXP * R + TOTAL - 1) / TOTAL )); [ "$P" -lt 1 ] && P=1; [ "$P" -gt "$R" ] && P=$R
    for ((p=0; p<P; p++)); do
      CUDA_VISIBLE_DEVICES=$((i%4)) "$PY" "$BASE/scripts/predict_seq.py" "$f" "$p" "$P" \
        > "$CV/fold$f/seq_r${round}_p$p.log" 2>&1 &
      i=$((i+1))
    done
  done
  say "프로세스 $i개 기동 — 대기"
  wait
  for f in 1 2 3 4; do
    say "  fold$f: $(ls "$CV/fold$f/out"/*.nii.gz 2>/dev/null | wc -l)/$(ls "$CV/fold$f/in" | wc -l)"
  done
done

TOTAL=0; for f in 1 2 3 4; do TOTAL=$((TOTAL + $(rem $f))); done
if [ "$TOTAL" -gt 0 ]; then
  say "⚠ 3라운드 후에도 $TOTAL 케이스 미완 — 부분 데이터로 채점하지 않고 중단"; exit 1
fi
say "전 fold 예측 완료"

say "후처리 (fold1~4)"
for f in 1 2 3 4; do
  "$PY" -u "$BASE/scripts/postprocess_vessel.py" apply "$CV/fold$f/out" "$CV/fold$f/pp" >> "$LOG" 2>&1
done
say "집계: raw"
"$PY" -u "$BASE/scripts/eval_cv_5fold.py" raw \
  "$N/val_predict_out" "$CV/fold1/out" "$CV/fold2/out" "$CV/fold3/out" "$CV/fold4/out" 2>&1 | tee -a "$LOG"
say "집계: pp"
"$PY" -u "$BASE/scripts/eval_cv_5fold.py" pp \
  "$N/val_predict_out_pp" "$CV/fold1/pp" "$CV/fold2/pp" "$CV/fold3/pp" "$CV/fold4/pp" 2>&1 | tee -a "$LOG"
say "완료"
