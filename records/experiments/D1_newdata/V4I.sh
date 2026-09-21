#!/usr/bin/env bash
# V4-I — ICA 원위 기하 룰 (설계·τ 선택 근거는 v4i_screen.py · v4i_make.py 머리말)
#
# 기준선: b1Non_pf = 제출본 검출기 + 개정판 학습표 + gC ON + 패치필터 (PROJECT_RULES.md 0-2)
# 후보:   b1v4i_pf = 기준선과 **완전히 같은 설정** + c5_v4i.py(TOPANEU_V4I=1 · M1 τ3.0)
#         룰이 발동하지 않은 병변은 기준선과 출력이 같아야 한다 → 보고서에서 검증.
#
# ── 판정규칙 (결과 보기 전에 고정 · 2026-09-15) ─────────────────────────────
#  1차(주):  병변 단위 TP (tpcount.py) — b1Non_pf 대비 test+val 합계가 늘어야 한다.
#  안전:     룰로 **새로 틀린** 병변(원래 GT 와 맞던 것을 바꿈) ≤ **새로 맞힌** 병변 — test·val 각각.
#  2차:      신 eval 7지표 평균 ΔMCC ≥ −0.005 — test·val 각각.
#  셋 다 만족 → 채택. 1차가 음수거나 안전 위반이면 기각.
#  ⚠ 이 룰은 RF 1등이 ICA 원위인 blob 만 건드린다. 7지표 '개선 수'는 대부분 0 변화라 기준으로 쓰지 않는다
#    (V1-D 와 같은 틀). 스크리닝(train OOF +7.9%p · 고침49/망침19)은 GT 병변 기준이라 검출 blob 에서 줄 수 있다.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/v4i.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v4i] $*" | tee -a "$D/STATUS.log"; }
TAG=b1v4i
FEAT=$A/e11_feat_hyb_ov_NEW.json
rm -f "$D"/v4i_dump_*.jsonl

log "1단계 · c5 10런 · c5_v4i (TOPANEU_V4I=1) · 개정판 학습표 · gC ON · 검출기 b1ff"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_V4I=1 TOPANEU_V4I_DUMP="$D/v4i_dump_${sp}_s${sd}.jsonl" \
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_v4i.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "v4i_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
miss=0
for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  for sd in 0 1 2 3 4; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
    [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
[ $miss = 0 ] || { log "★1단계 실패 — 중단"; exit 1; }
log "  룰 발동 $(cat "$D"/v4i_dump_*.jsonl 2>/dev/null | wc -l)건 (10런 합)"

log "2단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait

log "3단계 · 채점 10런 (동결 eval 660da7a · 7지표)"
n=0; for sp in val test; do for sd in 0 1 2 3 4; do
  d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
  grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/v4i_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; wait

log "4단계 · 보고서"
"$PY" "$D/v4i_report.py" > "$V/RESULTS_V4I.md" 2>&1
log "V4I 완료 → V1_vessel_axis/RESULTS_V4I.md"; touch "$D/.done_v4i"
