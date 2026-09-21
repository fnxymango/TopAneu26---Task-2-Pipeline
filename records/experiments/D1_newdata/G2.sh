#!/usr/bin/env bash
# G2 — 개정판 빌드(B2 단독, E9 10폴드) 위에서 gC on(topk2)/off(topk1) 둘 다 5시드 생성 → 새 eval 채점 → 판정
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; G=$E/G2_gc_onoff_newbuild; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/g2.lock"; flock -n 9 || exit 0; echo $$ > "$D/G2.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g2] $*" | tee -a "$ST"; }
mkdir -p "$G"/{pred,scores,logs,raw_results}; cd "$S"
FEAT=$A/e11_feat_hyb_ov_NEW.json   # 프로덕션 후보 = 개정판 피처
gen(){ local det=$1 sp=$2 sd=$3 topk=$4 tag=$5 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  [ "$(ls "$G/pred/${tag}_${sp}_s${sd}"/*.nii.gz 2>/dev/null | wc -l)" -ge 40 ] && return 0
  TOPANEU_TOPK=$topk TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" --fast \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${det}ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$G/pred/${tag}_${sp}_s${sd}" --tag "${tag}_${sp}_s${sd}" > "$G/logs/${tag}_${sp}_s${sd}.log" 2>&1
}
score(){ local tag=$1 sp=$2 sd=$3
  [ -s "$G/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$G/scores/${tag}_${sp}_s${sd}.json" && return 0
  "$PY" "$D/neweval.py" "$G/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" > "$G/scores/${tag}_${sp}_s${sd}.json" 2> "$G/logs/score_${tag}_${sp}_s${sd}.err"
}
stage(){ local det=$1 name=$2 mark=$3   # det: b2 | e9
  [ -f "$D/$mark" ] && return 0
  log "$name: gC on/off × 5시드 × test/val 생성 (20런)"
  for sp in test val; do for sd in 0 1 2 3 4; do gen $det $sp $sd 2 ${det}gcon & gen $det $sp $sd 1 ${det}gcoff & done; done; wait
  for t in ${det}gcon ${det}gcoff; do for sp in test val; do for sd in 0 1 2 3 4; do
    [ "$(ls "$G/pred/${t}_${sp}_s${sd}"/*.nii.gz 2>/dev/null | wc -l)" -ge 40 ] || { log "★$name 생성 실패 ${t}_${sp}_s${sd}"; return 1; }
  done; done; done
  log "$name: 새 eval 채점 20건 (8 병렬)"
  n=0; for t in ${det}gcon ${det}gcoff; do for sp in test val; do for sd in 0 1 2 3 4; do
    score $t $sp $sd & n=$((n+1)); [ $((n%8)) -eq 0 ] && wait
  done; done; done; wait
  cp "$A"/c5_eval_*_${det}gc*_s*.json "$A"/c5_percase_*_${det}gc*_s*.json "$G/raw_results/" 2>/dev/null
  { "$PY" "$D/g2_seeds.py" "$G/scores" ${det}gcon ${det}gcoff "$name"
    echo; echo "## 구 eval (c5 내부 · covered_gt / ÷52 · 부트스트랩)"; echo '```'
    "$PY" "$D/cmp_tags.py" ${det}gcoff ${det}gcon "gC off" "gC on" | tail -n +2; echo '```'
  } > "$G/RESULTS_${det}.md" 2>&1
  log "$name 판정: $(grep -o '→ gC [^*]*' "$G/RESULTS_${det}.md" | tail -1)"; touch "$D/$mark"
}
log "대기: .done_b2"; for i in $(seq 1 4320); do [ -f "$D/.done_b2" ] && break; sleep 60; done
[ -f "$D/.done_b2" ] || { log "★B2 미완(72h) — 종료"; exit 1; }
stage b2 "B2 단독(개정판 재학습 · 신피처)" .done_g2b || exit 1
log "대기: .done_ens"; for i in $(seq 1 4320); do [ -f "$D/.done_ens" ] && break; sleep 60; done
[ -f "$D/.done_ens" ] || { log "★ENS 미완(72h) — 종료"; exit 1; }
stage e9 "E9 10폴드 앙상블(신피처)" .done_g2e || exit 1
{ echo "# G2 — 개정판 빌드 gC on/off 종합"; echo; cat "$G/RESULTS_b2.md"; echo; echo "---"; echo; cat "$G/RESULTS_e9.md"; } > "$G/RESULTS.md"
log "G2 완료 → $G/RESULTS.md"; touch "$D/.done_g2"
