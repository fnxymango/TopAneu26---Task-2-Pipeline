#!/usr/bin/env bash
# H7 — 제출본에서 **검출기만** 개정판 데이터 5폴드로 바꾸면 오르는가
#
# 목표: 제출본 대비 검출기 성능이 실제로 올라가는지 본다. 바꾸는 축은 **하나뿐**이다.
#
#   b1on_pf   제출본      PlainConv 30.8M · 3폴드 · (학습 데이터 시점 미확인)
#   e9P5on_pf 후보        ResEncL 101.9M · **개정판 데이터 fold 5~9 단독**
#
# 나머지는 제출본과 **전부 동일**: 구피처 e11_feat_hyb_ov · gC ON(topk2) · 혈관 · 분기점 ·
#   c7 · 패치필터. 검출 마스크만 다르다.
#
# 왜 P5 단독인가: E9 10폴드는 보통의 10-fold 가 아니라 **구 데이터 5폴드 + 개정판 5폴드**를
#   모아 확률평균한 것이다. 앞으로 구 데이터는 쓰지 않으므로 개정판 폴드만 남긴다.
#   (10폴드·3폴드는 H6 에서 이미 측정 — 참고용 대조로만 쓴다.)
# 왜 구피처인가: 신피처는 H2 에서 −0.0240/−0.0144 로 더 나빴다. 차이는 병변 6개에서 온
#   잡음 수준이지만 굳이 나쁜 쪽을 택할 이유가 없고, 무엇보다 **축을 하나만 움직여야** 한다.
#
# ── 판정규칙 (결과 보기 전에 고정) ─────────────────────────────────────────
#   채택: e9P5on_pf − b1on_pf 가 신 eval **7지표**(F1 포함) 중 ≥5 개선 ∧ 평균 ΔMCC ≥ 0 을
#         **test·val 둘 다** 만족.
#   참고(판정 아님): e9on_pf(10폴드) − e9P5on_pf → 구 데이터를 섞는 게 이득이었나.
#   시드 산포가 평균보다 크면 결론에 명시한다.
#   런타임: 5폴드 = 케이스당 약 43s(A5000 실측 8.6s/폴드). T4 환산 1.5~1.7배.
#           한도 12분·32GB 로 완화됐으므로 여유.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; G=$E/H4_folds
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 4>"$D/h7.lock"; flock -n 4 || { echo "이미 실행 중"; exit 0; }
echo $$ > "$D/H7.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h7] $*" | tee -a "$ST"; }
mkdir -p "$G"
TAG=e9P5on
FOLDS=5,6,7,8,9

log "1단계 · 검출 추론 — test/val 을 GPU 한 장씩에 (H6 는 CPU 채점 중이라 GPU 가 논다)"
run_one(){ local sp=$1 gpu=$2 exp c
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  c=$(ls -1 "$G/pred_${TAG}_${sp}"/*.nii.gz 2>/dev/null | wc -l)
  if [ "$c" -ge "$exp" ]; then log "  건너뜀 $sp ($c/$exp)"; return 0; fi
  log "  $sp 폴드 $FOLDS → GPU$gpu (기존 $c/$exp 재사용)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$D/h4_predict.py" --folds "$FOLDS" --tag "$TAG" \
    --split "$sp" --nproc 2 > "$G/predict_${TAG}_${sp}.log" 2>&1
  tail -1 "$G/predict_${TAG}_${sp}.log" | tee -a "$ST"
}
run_one test 0 &
run_one val  1 &
wait
for sp in test val; do
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  c=$(ls -1 "$G/pred_${TAG}_${sp}"/*.nii.gz 2>/dev/null | wc -l)
  [ "$c" -ge "$exp" ] || { log "★검출 누락 $sp ($c/$exp) — 중단"; exit 1; }
done
log "  검출 완료"

log "2단계 · 재채점 완료 대기 (CPU 경합 회피)"
for i in $(seq 1 540); do [ -f "$D/.done_rescore" ] && break; sleep 20; done
[ -f "$D/.done_rescore" ] || log "★재채점 미완 — 그래도 진행"

log "3단계 · 라벨2 이진화 + c7 필터"
for sp in test val; do "$PY" "$D/h4_post.py" --tag "$TAG" --split "$sp" >> "$G/post.log" 2>&1; done
for sp in test val; do
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  c=$(ls -1 "$P/aneu_${sp}_${TAG}ff" 2>/dev/null | wc -l)
  [ "$c" = "$exp" ] || { log "★c7 누락 $sp ($c/$exp) — 중단"; exit 1; }
done

log "4단계 · c5 10런 · 구피처 · gC ON (제출본과 동일)"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${TAG}ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "h7_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait

log "5단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait

log "6단계 · 채점 20런 (동결 eval 660da7a · 7지표)"
n=0
for sp in val test; do for t in ${TAG}_pf ${TAG}; do for sd in 0 1 2 3 4; do
  d="$H/pred/${t}_${sp}_s${sd}"; [ -d "$d" ] || continue
  grep -q '"F1"' "$H/scores/${t}_${sp}_s${sd}.json" 2>/dev/null && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/h7_score.log" 2>&1 &
  n=$((n+1)); [ $((n%12)) -eq 0 ] && wait
done; done; done; wait
log "  채점 $n 판"

log "7단계 · 보고서"
{
  echo "# H7 — 제출본에서 검출기만 개정판 5폴드로"; echo
  echo '판정규칙은 H7.sh 머리말에 결과 보기 전에 고정. 신 eval 7지표(F1 포함) · 동결 eval 660da7a.'; echo
  echo "## 채택 판정 · e9P5on_pf − b1on_pf (제출본)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" "${TAG}_pf" b1on_pf "개정판5폴드 − 제출본" 2>&1
  echo; echo "## 참고 · e9on_pf(10폴드=구5+신5) − e9P5on_pf(개정판5 단독)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" e9on_pf "${TAG}_pf" "10폴드 − 개정판5폴드" 2>&1
  echo; echo "## 런타임 실측"; echo '```'
  for f in "$G"/predict_${TAG}_*.log; do echo "$(basename "$f"): $(tail -1 "$f")"; done; echo '```'
  echo; echo "## 절대값 (5시드 평균)"; echo
  H3_TAGS="b1on_pf e9f3P3on_pf ${TAG}_pf e9on_pf" "$PY" "$D/h3_abs.py" "$H/scores" 2>&1
} > "$G/RESULTS_H7.md" 2>&1
log "H7 완료 → $G/RESULTS_H7.md"; touch "$D/.done_h7"
