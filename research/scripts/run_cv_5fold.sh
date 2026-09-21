#!/usr/bin/env bash
# 5-fold CV 채점 — 각 fold를 '자기 val'로만 채점(유출 없음). held-out test 14는 미사용.
#
# 예측은 predict_seq.py(멀티프로세싱 없음)로 수행한다.
#   nnUNetv2_predict CLI는 이 데이터셋에서 두 가지로 실패했다:
#     (1) 4 fold 동시 실행 → RAM 압박으로 워커 사망(KeyError / workers no longer alive)
#     (2) -num_parts 분할 → 케이스 종료 후 export 대기에서 행(hang), 56분 무진행
#   predict_seq.py는 전 과정을 메인 프로세스에서 처리해 두 실패 모드가 없다.
# 이미 있는 예측은 건너뛴다(재실행 안전).
set -uo pipefail
BASE=/home/user/TopAneu/seg/sblee/nnunet
PY=/home/user/anaconda3/envs/sbaneu2/bin/python
N=$BASE/experiments/D600_vessel_skelrec_resencm_250ep_bd0
LOG=$N/cv.log
CV=$N/cv
export nnUNet_raw=$BASE/nnUNet_raw
export nnUNet_preprocessed=$BASE/nnUNet_preprocessed
export nnUNet_results=$N/results

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mkdir -p "$CV"
say "=== CV 재개 (predict_seq, 12프로세스 / GPU 4장) ==="

# ---- 입력 폴더 구성 ----
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import json, os
B='/home/user/TopAneu/seg/sblee/nnunet'
s=json.load(open(f'{B}/nnUNet_preprocessed/Dataset600_TopAneuVessel/splits_final.json'))
for f in range(1,5):
    d=f'{B}/experiments/D600_vessel_skelrec_resencm_250ep_bd0/cv/fold{f}/in'
    os.makedirs(d, exist_ok=True)
    for cid in s[f]['val']:
        src=f'{B}/nnUNet_raw/Dataset600_TopAneuVessel/imagesTr/{cid}_0000.nii.gz'
        dst=f'{d}/{cid}_0000.nii.gz'
        if not os.path.lexists(dst): os.symlink(os.path.realpath(src), dst)
PY

# ---- 예측: (fold, part) 12개를 GPU 4장에 3개씩 배치, 전부 동시 실행 ----
# 남은 물량 비례 배분: fold2 3파트 · fold3 1파트 · fold4 8파트
launch() {  # fold, part, nparts, gpu
  CUDA_VISIBLE_DEVICES=$4 "$PY" "$BASE/scripts/predict_seq.py" "$1" "$2" "$3" \
    > "$CV/fold$1/seq_p$2.log" 2>&1 &
}
mkdir -p "$CV"/fold{1,2,3,4}
i=0
for p in 0 1 2; do launch 2 $p 3 $((i%4)); i=$((i+1)); done
launch 3 0 1 $((i%4)); i=$((i+1))
for p in 0 1 2 3 4 5 6 7; do launch 4 $p 8 $((i%4)); i=$((i+1)); done
say "예측 프로세스 12개 기동 — 완료 대기"
wait
say "예측 라운드 종료"

# ---- 완료 검증 + 미완 시 1회 재시도 ----
for attempt in 1 2; do
  MISS=0
  for f in 1 2 3 4; do
    TOT=$(ls "$CV/fold$f/in" | wc -l); HAVE=$(ls "$CV/fold$f/out"/*.nii.gz 2>/dev/null | wc -l)
    say "  fold$f: $HAVE/$TOT"
    [ "$HAVE" -lt "$TOT" ] && MISS=1
  done
  [ "$MISS" -eq 0 ] && break
  [ "$attempt" -eq 2 ] && { say "예측이 불완전해 채점을 중단합니다(부분 데이터로 수치를 만들지 않음)"; exit 1; }
  say "미완 감지 — 재시도"
  i=0
  for f in 1 2 3 4; do for p in 0 1 2; do launch $f $p 3 $((i%4)); i=$((i+1)); done; done
  wait
done
say "전 fold 예측 완료 확인"

# ---- 후처리 (병렬) ----
say "후처리 (fold1~4)"
for f in 1 2 3 4; do
  "$PY" -u "$BASE/scripts/postprocess_vessel.py" apply "$CV/fold$f/out" "$CV/fold$f/pp" >> "$LOG" 2>&1
done

# ---- 집계 (병렬) ----
say "집계: 후처리 없음 (raw)"
"$PY" -u "$BASE/scripts/eval_cv_5fold.py" raw \
  "$N/val_predict_out" "$CV/fold1/out" "$CV/fold2/out" "$CV/fold3/out" "$CV/fold4/out" 2>&1 | tee -a "$LOG"
say "집계: 후처리 적용 (pp)"
"$PY" -u "$BASE/scripts/eval_cv_5fold.py" pp \
  "$N/val_predict_out_pp" "$CV/fold1/pp" "$CV/fold2/pp" "$CV/fold3/pp" "$CV/fold4/pp" 2>&1 | tee -a "$LOG"
say "완료"
