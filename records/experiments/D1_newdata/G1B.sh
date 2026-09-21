#!/usr/bin/env bash
# G1B — G1 의 병렬판. B단계(번들 검출기 B1 위) 예측맵 4건 동시 생성 → 4건 동시 채점 → A단계 결과와 합쳐 보고.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; G=$E/G1_gc_neweval; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/g1b.lock"; flock -n 9 || exit 0
echo $$ > "$D/G1B.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g1] $*" | tee -a "$ST"; }
cd "$S"; mkdir -p "$G"/{pred,logs,scores_par}
gen(){ local sp=$1 topk=$2 tag=$3 det=$4 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  TOPANEU_TOPK=$topk TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=3 OMP_NUM_THREADS=2 \
  "$PY" -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$det" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$G/pred/${tag}_${sp}" --tag "g1_${tag}_${sp}_s3" > "$G/logs/${tag}_${sp}.log" 2>&1
}
score(){ "$PY" "$D/neweval.py" "$G/pred/$1" "$2" "$1" > "$G/scores_par/$1.json" 2> "$G/logs/score_par_$1.err"; }
ok(){ "$PY" - "$1" <<'PYEOF'
import runpy,sys
ns=runpy.run_path("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata/g1_report.py",run_name="x")
sys.exit(0 if ns["L"](sys.argv[1]) else 1)
PYEOF
}
log "B단계 예측맵 4건 병렬 생성 (번들 검출기 B1 위)"
gen test 2 b1gcON aneu_test_b1ff & gen test 1 b1gcOFF aneu_test_b1ff &
gen val  2 b1gcON aneu_val_b1ff  & gen val  1 b1gcOFF aneu_val_b1ff  &
wait
log "B단계 채점 4건 병렬"
score b1gcON_test test & score b1gcOFF_test test & score b1gcON_val val & score b1gcOFF_val val &
wait
log "B단계 채점 완료"
# A단계 파일 완성 대기 (gcON_test/gcOFF_test 는 scores/, val 은 scores_par/)
for i in $(seq 1 120); do ok gcON_test && ok gcOFF_test && ok gcON_val && ok gcOFF_val && break; sleep 30; done
"$PY" "$D/g1_report.py" > "$G/RESULTS.md" 2>&1
log "G1 완료 · $G/RESULTS.md"
touch "$D/.done_g1"
