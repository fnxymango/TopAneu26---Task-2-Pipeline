#!/usr/bin/env bash
# Q2 — ICA C6/C7/terminus 분할 큐 (사용자 지시 2026-09-14 "추천순서에 있는거 다 걸어 큐에")
#
#   1단계 V2-0b   분할 기준 Dice 비교           ← 이미 끝남 (RESULTS_V20B.md · 규칙상 판정 보류)
#   2단계 V2-1    병변 단위 일관성 (GT 혈관)     v21_lesion.py gt
#   3단계 V2-2    폴백 적합 + 최종 규칙 선택      v22_select.py select → v2_rule.json
#   4단계 V2-3    예측 혈관에서 규칙이 버티나     v21_lesion.py pred + v22_select.py pred
#   5단계 V2-A    상한 실험 (GT 혈관 36 vs 40)    gen40 → c4 → 학습표 → c5 20런 → 패치필터 → 채점 → 판정
#
# 판정규칙은 각 스크립트 머리말에 결과 보기 전에 고정했다.
# 단계마다 .q2_sN 마커를 남겨 재실행 시 끝난 단계는 건너뛴다.
# fine GT 는 train·val 만 연다. test 10케이스는 어느 단계에서도 읽지 않는다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/q2.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][q2] $*" | tee -a "$D/STATUS.log"; }
stage(){ [ -f "$D/.q2_$1" ]; }
mark(){ touch "$D/.q2_$1"; }

# ── 2단계 ─────────────────────────────────────────────────────────────────
if ! stage s2; then
  log "2단계 · 병변 단위 일관성 측정 (GT 혈관 · fine GT train·val 30케이스)"
  "$PY" -u "$D/v21_lesion.py" gt > "$D/q2_s2.log" 2>&1 || { log "★2단계 실패 — q2_s2.log"; exit 1; }
  log "  $(tail -1 "$D/q2_s2.log")"; mark s2
fi

# ── 3단계 ─────────────────────────────────────────────────────────────────
if ! stage s3; then
  log "3단계 · 폴백 적합 + 분할 규칙 선택"
  "$PY" "$D/v22_select.py" select > "$V/RESULTS_V21V22.md" 2> "$D/q2_s3.err" || { log "★3단계 실패 — q2_s3.err"; exit 1; }
  [ -s "$D/v2_rule.json" ] || { log "★v2_rule.json 없음"; exit 1; }
  log "  채택 규칙 $("$PY" -c "import json;print(json.load(open('$D/v2_rule.json'))['name'])")"; mark s3
fi

# ── 4단계 ─────────────────────────────────────────────────────────────────
if ! stage s4; then
  log "4단계 · 예측 혈관(vespp)에서 같은 규칙 점검"
  "$PY" -u "$D/v21_lesion.py" pred > "$D/q2_s4.log" 2>&1 || { log "★4단계 측정 실패 — q2_s4.log"; exit 1; }
  "$PY" "$D/v22_select.py" pred > "$V/RESULTS_V23.md" 2>> "$D/q2_s4.log" || { log "★4단계 보고 실패"; exit 1; }
  log "  $(grep '판정 →' "$V/RESULTS_V23.md" | tail -1)"; mark s4
fi

# ── 5단계 V2-A ────────────────────────────────────────────────────────────
if ! stage s5a; then
  log "5단계 · V2-A (a) GT 혈관 415케이스 40클래스 분할"
  "$PY" -u "$D/v2a_gen40.py" > "$D/q2_s5a.log" 2>&1 || { log "★gen40 실패 — q2_s5a.log"; exit 1; }
  log "  $(tail -1 "$D/q2_s5a.log")"; mark s5a
fi

if ! stage s5b; then
  log "5단계 · V2-A (b) 분기점 그래프 40클래스 (c4_bp40 · 10분할 병렬)"
  mkdir -p "$BP/all_ref40"
  ls "$E/_v2_ves40/gt" | grep '\.nii\.gz$' | sed 's/\.nii\.gz$//' > "$D/q2_cases.txt"
  split -n l/10 -d "$D/q2_cases.txt" "$D/q2_shard_"
  for f in "$D"/q2_shard_*; do
    "$PY" -u "$S/c4_bp40.py" --vessel-dir "$E/_v2_ves40/gt" --out "$BP/all_ref40" \
      --cases "$(paste -sd, "$f")" > "$f.log" 2>&1 &
  done; wait
  n=$(ls "$BP/all_ref40" | grep -c '\.json$'); m=$(wc -l < "$D/q2_cases.txt")
  [ "$n" -ge "$m" ] || { log "★c4 결과 $n/$m"; exit 1; }
  rm -f "$D"/q2_shard_*; log "  분기점 그래프 $n/$m"; mark s5b
fi

if ! stage s5c; then
  log "5단계 · V2-A (c) 학습표 2종 (GT 혈관만 · 36 / 40)"
  "$PY" -u "$S/c5_location_v2.py" build --split train --vessel-dir "$R/dataset/TopAneu/vessel_masks" \
    --bp-dir "$BP/all_ref" --out "$A/v2a_feat_gt36.json" > "$D/q2_s5c_36.log" 2>&1 &
  "$PY" -u "$S/c5_v40.py" build --split train --vessel-dir "$E/_v2_ves40/gt" \
    --bp-dir "$BP/all_ref40" --out "$A/v2a_feat_gt40.json" > "$D/q2_s5c_40.log" 2>&1 &
  wait
  for t in 36 40; do
    n=$("$PY" -c "import json;print(len(json.load(open('$A/v2a_feat_gt$t.json'))))" 2>/dev/null || echo 0)
    [ "$n" -ge 250 ] || { log "★학습표 gt$t $n행"; exit 1; }
    log "  학습표 gt$t ${n}행"
  done; mark s5c
fi

ARMS="v2a36:c5_location_v2.py:$A/v2a_feat_gt36.json:$R/dataset/TopAneu/vessel_masks:$BP/all_ref v2a40:c5_v40.py:$A/v2a_feat_gt40.json:$E/_v2_ves40/gt:$BP/all_ref40"

if ! stage s5d; then
  log "5단계 · V2-A (d) c5 20런 (팔2 × split2 × 시드5) · gC ON · 검출기 b1ff"
  clf(){ local TAG=$1 SCR=$2 FEAT=$3 VD=$4 BPD=$5 sp=$6 sd=$7 exp n
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/$SCR" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$VD" --bp-dir "$BPD" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "v2a_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  n=0
  for spec in $ARMS; do IFS=: read -r TAG SCR FEAT VD BPD <<<"$spec"
    for sp in test val; do for sd in 0 1 2 3 4; do
      clf "$TAG" "$SCR" "$FEAT" "$VD" "$BPD" "$sp" "$sd" & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
    done; done
  done; wait
  miss=0
  for spec in $ARMS; do IFS=: read -r TAG _ <<<"$spec"
    for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      for sd in 0 1 2 3 4; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l)
        [ "$c" = "$exp" ] || { log "★c5 누락 ${TAG}_${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
  done
  [ $miss = 0 ] || { log "★c5 단계 실패 — 중단"; exit 1; }
  mark s5d
fi

if ! stage s5e; then
  log "5단계 · V2-A (e) 패치필터 20런"
  pf(){ local TAG=$1 sp=$2 sd=$3 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
      --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
      --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
  }
  n=0
  for TAG in v2a36 v2a40; do for sp in test val; do for sd in 0 1 2 3 4; do
    pf "$TAG" "$sp" "$sd" & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait
  done; done; done; wait
  mark s5e
fi

if ! stage s5f; then
  log "5단계 · V2-A (f) 채점 20런 (동결 eval 660da7a · 7지표)"
  n=0
  for TAG in v2a36 v2a40; do for sp in val test; do for sd in 0 1 2 3 4; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/q2_s5f.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; done; wait
  mark s5f
fi

log "5단계 · V2-A (g) 판정"
"$PY" "$D/v2a_report.py" > "$V/RESULTS_V2A.md" 2>&1
log "Q2 완료 → V1_vessel_axis/RESULTS_V21V22.md · RESULTS_V23.md · RESULTS_V2A.md"
touch "$D/.done_q2"
