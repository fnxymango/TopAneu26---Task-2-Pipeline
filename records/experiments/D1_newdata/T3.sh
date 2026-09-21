#!/usr/bin/env bash
# T3 — 비낭형(방추·박리) 학습 행 표본 가중 e2e (설계는 t3_make.py 머리말 · 스크리닝 t3_screen.py 통과 시에만 Q5.sh 가 실행)
#
# 무엇이 바뀌나: c5_t3.py + TOPANEU_T3=1 — 학습 때 비낭형 행에 N낭형/N비낭형 가중. 추론·학습표·검출기·gC·패치필터는 기준선과 같다.
# 기준선: b1Non_pf (시드 0~4).
#
# ── 판정규칙 (결과 보기 전에 고정 · 2026-09-15 · K0 장치) ──────────────────────
#  주판정: k0_judge.py (α 0.05) — 130 단위 단측 부호검정 p < 0.05 ∧ 오른 단위 > 내린 단위.
#  안전:   ΔFP ≤ ΔTP (5시드 합).   둘 다 → 채택 후보 · 주판정 미충족 → 미채택 · 안전 위반 → 기각.
#  보조 기록: tpcount · 7지표. 채택 시 제출 반영은 모델 파일 교체만(추론 코드 불변).
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/t3.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][t3] $*" | tee -a "$D/STATUS.log"; }
TAG=b1t3
FEAT=$A/e11_feat_hyb_ov_NEW.json
log "1단계 · c5 10런 · c5_t3 (비낭형 표본 가중) · gC ON · 검출기 b1ff"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_T3=1 TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_t3.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "t3_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
miss=0
for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  for sd in 0 1 2 3 4; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
    [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
[ $miss = 0 ] || { log "★1단계 실패"; exit 1; }
log "2단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
log "3단계 · 채점 10런"
n=0; for sp in val test; do for sd in 0 1 2 3 4; do
  d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
  grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/t3_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; wait
log "4단계 · 보고서"
until [ -e "$D/k0_null.json" ]; do sleep 60; done
{ echo "# T3 — 비낭형 표본 가중 e2e"; echo
  echo "판정규칙은 T3.sh 머리말에 결과 보기 전에 고정 (K0 장치)."; echo
  "$PY" "$D/k0_judge.py" b1Non_pf "${TAG}_pf"
  echo; echo "## 보조 · 병변 단위 TP (tpcount)"; echo; "$PY" "$D/tpcount.py" b1Non_pf "${TAG}_pf"
  echo; echo "## 보조 · 7지표 (${TAG}_pf − b1Non_pf)"; echo; "$PY" "$D/g2_seeds.py" "$H/scores" "${TAG}_pf" b1Non_pf "비낭형가중 − 기준선"
} > "$V/RESULTS_T3.md" 2>&1
log "T3 완료 → V1_vessel_axis/RESULTS_T3.md · $(grep -o 'K0 판정 → [^*]*' $V/RESULTS_T3.md)"; touch "$D/.done_t3"
