#!/usr/bin/env bash
# G3 — 세 후보(B1 구번들 · B2 재학습 · E9 10폴드) 모두 gC OFF(topk1) 로 맞비교. c5 정식(구 eval 내부 채점 포함) 5시드 × test/val
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; G=$E/G3_gcoff_candidates; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/g3.lock"; flock -n 9 || exit 0; echo $$ > "$D/G3.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g3] $*" | tee -a "$ST"; }
mkdir -p "$G"/{logs,raw_results,scores}; cd "$S"
clf(){ local det=$1 feat=$2 tag=$3 sp=$4 sd=$5 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  [ -s "$A/c5_eval_${sp}_${tag}_${sp}_s${sd}.json" ] && return 0
  TOPANEU_TOPK=1 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u c5_location_v2.py eval --train-feat "$A/$feat" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${det}ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "${tag}_${sp}_s${sd}" > "$G/logs/${tag}_${sp}_s${sd}.log" 2>&1
}
log "gC OFF 3후보 × 5시드 × test/val = 30런 (15 병렬 × 2)"
n=0
for spec in "b1 e11_feat_hyb_ov.json b1off" "b2 e11_feat_hyb_ov_NEW.json b2off" "e9 e11_feat_hyb_ov_NEW.json e9off"; do set -- $spec
  for sp in test val; do for sd in 0 1 2 3 4; do clf $1 $2 $3 $sp $sd & n=$((n+1)); [ $((n%15)) -eq 0 ] && wait; done; done
done; wait
miss=0; for t in b1off b2off e9off; do for sp in test val; do for sd in 0 1 2 3 4; do [ -s "$A/c5_eval_${sp}_${t}_${sp}_s${sd}.json" ] || { log "★누락 ${t}_${sp}_s${sd}"; miss=1; }; done; done; done
[ $miss = 0 ] || exit 1
cp "$A"/c5_eval_*_{b1off,b2off,e9off}_*_s*.json "$A"/c5_percase_*_{b1off,b2off,e9off}_*_s*.json "$G/raw_results/" 2>/dev/null
{
  echo "# G3 — 세 후보 모두 gC OFF(topk n=1) · 5시드 · test/val"; echo
  echo "후보: B1 = 구 번들 가중치(구피처) · B2 = 번들 레시피 개정판 재학습(검출기+분류기, 신피처) · E9 = ResEncL 10폴드(P3구+P5신, 신피처)"; echo
  echo "## 새 공식 eval (60765a5) · 5시드 평균 절대값"; echo
  "$PY" "$D/g3_abs.py"
  echo; echo "## 새 eval · 쌍별 시드짝 Δ (후보 − 기준) · 규칙: ≥4/6 개선 ∧ 평균ΔMCC ≥ 0 · 두 집합"; echo
  for pair in "b1off b2off B2재학습 B1구번들" "b1off e9off E9_10폴드 B1구번들" "b2off e9off E9_10폴드 B2재학습"; do set -- $pair
    "$PY" "$D/g2_seeds.py" "$G/scores" $2 $1 "$3 − $4" | sed "s/gC on(topk2) − off(topk1)/후보 − 기준/; s/gC 유지(topk2)/후보 채택/; s/gC OFF(topk1) 권고/후보 미채택(기준 유지)/"; echo
  done
  echo "## 구 eval (c5 내부 · official÷52 / covered_gt · 케이스 부트스트랩 2000회)"; echo
  for pair in "b1off b2off B1구번들 B2재학습" "b1off e9off B1구번들 E9_10폴드" "b2off e9off B2재학습 E9_10폴드"; do set -- $pair
    "$PY" "$D/cmp_tags.py" $1 $2 "$3" "$4" | tail -n +2; echo
  done
} > "$G/RESULTS.md" 2>&1
log "G3 완료 → $G/RESULTS.md"; touch "$D/.done_g3"
