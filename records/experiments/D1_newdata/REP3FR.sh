#!/usr/bin/env bash
# REP3F-R — 3폴드 min_vox 12 재현 **복제** (시드 5~9) · 2026-09-18 사용자 승인
#  원판(시드 0~4)은 사전 규칙 3개를 전부 충족했다: ΔPRECISION test +0.0507 / val +0.0207 ·
#  ΔMCC +0.0368 / +0.0259 · ΔRECALL 0.0000 / 0.0000 · TP 완전 불변 · FP −6.0 / −1.0.
#  판정규칙은 REP3F.sh 머리말과 **동일**하다(결과 보기 전 고정 · 바꾸지 않는다):
#    ① test·val 양쪽 ΔPRECISION > 0  ② 양쪽 ΔMCC ≥ 0  ③ 양쪽 ΔRECALL ≥ −0.01
#  복제도 셋 다 충족해야 "채택 권고"(반영은 사용자 결정). A단계 산출물은 원판 것을 그대로 쓴다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/rep3fr.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][rep3fr] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov.json
SEEDS="5 6 7 8 9"

run_arm(){ local TAG=$1 SUF=$2
  for sp in test val; do [ -d "$P/aneu_${sp}_${SUF}" ] || { log "★검출 디렉터리 없음: $P/aneu_${sp}_${SUF}"; return 1; }; done
  log "복제 c5 · $TAG (검출 aneu_{split}_$SUF) · 시드 $SEEDS"
  clf(){ local sp=$1 sd=$2 bp exp n
    bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_TOPK_FRAC=0.35 TOPANEU_OUT_GROW=1.32 TOPANEU_OUT_DILATE=0 \
    CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${SUF}" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "rep3fr_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
  for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
      [ "$c" = "$exp" ] || { log "★c5 누락 ${TAG} ${sp}_s${sd} ($c/$exp)"; return 1; }; done; done
  log "복제 패치필터 · $TAG"
  pf(){ local sp=$1 sd=$2 exp n; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
      --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
      --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
  }
  n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
  log "복제 채점 · $TAG"
  n=0; for sp in val test; do for sd in $SEEDS; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/rep3fr_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; wait
  return 0
}

run_arm b3fg    b3ff      || { log "★복제 기준팔 실패"; exit 1; }
run_arm b3fg12  b3ff12    || { log "★복제 mv12 팔 실패"; exit 1; }

{ echo "# REP3F-R — 3폴드 min_vox 12 **복제** (시드 5~9) · 규칙은 REP3F.sh 머리말과 동일"; echo
  BASE=b3fg_pf TAG=b3fg12_pf SEEDS=5,6,7,8,9 LABEL_BASE="3폴드 기준" LABEL_TAG="min_vox 12" \
    "$PY" "$D/metrics_diff.py"
  echo; echo "## 원판+복제 통합 (시드 0~9)"; echo
  BASE=b3fg_pf TAG=b3fg12_pf SEEDS=0,1,2,3,4,5,6,7,8,9 LABEL_BASE="3폴드 기준" LABEL_TAG="min_vox 12" \
    "$PY" "$D/metrics_diff.py"
} > "$V/RESULTS_REP3F_R.md" 2>&1
touch "$D/.done_rep3fr"
log "REP3F 복제 끝 → $V/RESULTS_REP3F_R.md"
