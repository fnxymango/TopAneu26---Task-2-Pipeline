#!/usr/bin/env bash
# H4 — E9(ResEncL) 검출기를 폴드 수를 줄여 돌리면 이득이 얼마나 남는가
#
# 왜: H1/H2 에서 E9 검출기가 제출본 검출기보다 크게 낫다는 게 나왔다(피처를 맞춘 뒤 test
#     +0.0442 6/6 · val +0.0326 5/6). 그런데 E9 는 10폴드 ResEncL 이라 GC 420초 한도를 넘긴다.
#     제출본과 **같은 3폴드 예산**으로 줄였을 때 이득이 남는지가 채택 여부를 가른다.
#
# 두 검출기의 차이는 셋뿐이다 — 아키텍처(PlainConv 30.8M → ResEncL 101.9M) · 폴드 수 ·
# 학습데이터(E9 는 folds5-9 가 2026-09-01 개정판). 여기서는 **폴드 수만** 줄여
# 아키텍처 효과와 폴드 효과를 가른다. 해상도·패치·batch·epoch 는 두 검출기가 이미 동일하다.
#
#   e9f3P5  folds 5,6,7   개정판 데이터로 학습한 폴드 3개
#   e9f3P3  folds 0,1,2   구 데이터로 학습한 폴드 3개
#   (참고) e9off = 10폴드 전부, H1 에서 측정 완료
#
# 혈관·분기점·c7·분류기 피처·gC·패치필터는 전부 고정. 바뀌는 것은 검출 마스크뿐이다.
#
# ── 판정규칙 (결과 보기 전에 고정) ──────────────────────────────────────────
#   주 질문: {tag}_pf − b1on_pf(제출본) 가 신 eval 6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을
#            test·val 둘 다 만족하는가. 만족하면 그 폴드 수로 채택 가능.
#   부 질문: {tag}_pf − e9off_pf(10폴드) 로 10폴드 대비 얼마를 잃는지 본다(채택 판정 아님).
#   시드 산포가 평균보다 크면 결론에 명시한다.
#   런타임: h4_predict.py 가 케이스당·폴드당 초를 찍는다. 제출본 3폴드 PlainConv 대비
#           몇 배인지 함께 적는다. 이 값이 채택의 실질 관문이다.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; G=$E/H4_folds
ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/h4.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
echo $$ > "$D/H4.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h4] $*" | tee -a "$ST"; }
mkdir -p "$G" "$H"/{pred,scores,logs}

RAW=$R/nnunet/nnUNet_raw/Dataset722_TopAneuPjh3cls417/imagesTr
log "0단계 · Dataset722 정규화 영상 재생성 대기 (p_build_3cls.py)"
for i in $(seq 1 240); do
  n=$(ls -1 "$RAW" 2>/dev/null | wc -l)
  [ "$n" -ge 124 ] && break
  [ $((i % 10)) -eq 0 ] && log "  $n/124"
  sleep 30
done
n=$(ls -1 "$RAW" 2>/dev/null | wc -l)
[ "$n" -ge 124 ] || { log "★영상 재생성 미완 ($n/124) — 중단"; exit 1; }
log "  영상 $n/124 준비됨"

log "1단계 · 검출 추론 (GPU 2장 병렬 · 팔 하나씩)"
# A5000 이 2장이라 두 팔을 각각 한 장에 붙인다. 한 장 안에서는 직렬(test → val).
# 스모크 실측: 1케이스 3폴드 45.7s(모델 로딩 포함).
run_arm(){ local tag=$1 folds=$2 gpu=$3
  for sp in test val; do
    local exp c
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    c=$(ls -1 "$G/pred_${tag}_${sp}" 2>/dev/null | grep -c '\.nii\.gz$')
    if [ "$c" -ge "$exp" ]; then log "  건너뜀 $tag $sp ($c/$exp)"; continue; fi
    log "  $tag $sp 폴드 $folds → GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" -u "$D/h4_predict.py" --folds "$folds" --tag "$tag" \
      --split "$sp" > "$G/predict_${tag}_${sp}.log" 2>&1
    tail -1 "$G/predict_${tag}_${sp}.log" | tee -a "$ST"
  done
}
run_arm e9f3P5 5,6,7 0 &
run_arm e9f3P3 0,1,2 1 &
wait

log "2단계 · 라벨2 이진화 + c7 필터"
for t in e9f3P5 e9f3P3; do for sp in test val; do
  "$PY" "$D/h4_post.py" --tag "$t" --split "$sp" >> "$G/post.log" 2>&1
done; done
miss=0
for t in e9f3P5 e9f3P3; do for sp in test val; do
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  c=$(ls -1 "$P/aneu_${sp}_${t}ff" 2>/dev/null | wc -l)
  [ "$c" = "$exp" ] || { log "★c7 누락 ${t} ${sp} ($c/$exp)"; miss=1; }
done; done
[ $miss = 0 ] || { log "★2단계 실패 — 중단"; exit 1; }

log "3단계 · c5 20런 (10 병렬) · 개정피처 · gC OFF"
clf(){ local tag=$1 sp=$2 sd=$3 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${tag}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=1 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$A/e11_feat_hyb_ov_NEW.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${tag}ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${tag}_${sp}_s${sd}" --tag "h4_${tag}_${sp}_s${sd}" \
    > "$H/logs/${tag}_${sp}_s${sd}.log" 2>&1
}
n=0
for t in e9f3P5 e9f3P3; do for sp in test val; do for sd in 0 1 2 3 4; do
  clf $t $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; done; wait

log "4단계 · 패치필터 20런"
pf(){ local tag=$1 sp=$2 sd=$3 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${tag}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${tag}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${tag}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${tag}_${sp}_s${sd}.json" > "$H/logs/pf_${tag}_${sp}_s${sd}.log" 2>&1
}
n=0
for t in e9f3P5 e9f3P3; do for sp in test val; do for sd in 0 1 2 3 4; do
  pf $t $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait
done; done; done; wait

log "5단계 · neweval 40런 (14 병렬)"
score(){ local tag=$1 sp=$2 sd=$3 t
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  t="$H/scores/.tmp_${tag}_${sp}_s${sd}.$$"
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$t" 2> "$H/logs/score_${tag}_${sp}_s${sd}.err"
  if grep -q '"label"' "$t"; then mv -f "$t" "$H/scores/${tag}_${sp}_s${sd}.json"
  else rm -f "$t"; log "★채점 실패 ${tag}_${sp}_s${sd}"; fi
}
n=0
for t in e9f3P5_pf e9f3P3_pf e9f3P5 e9f3P3; do for sp in val test; do for sd in 0 1 2 3 4; do
  score $t $sp $sd & n=$((n+1)); [ $((n%14)) -eq 0 ] && wait
done; done; done; wait

log "6단계 · 보고서"
{
  echo "# H4 — E9 검출기 폴드 축소 · 이득이 남는가"; echo
  echo '판정규칙은 H4.sh 머리말에 결과 보기 전에 고정.'; echo
  echo "## 런타임 실측"; echo
  echo '```'; for f in "$G"/predict_*.log; do echo "$(basename "$f"): $(tail -1 "$f")"; done; echo '```'
  for t in e9f3P5 e9f3P3; do
    echo; echo "## ${t}_pf − b1on_pf (제출본) · 채택 판정"; echo
    "$PY" "$D/g2_seeds.py" "$H/scores" "${t}_pf" b1on_pf "$t − 제출본" \
      | sed 's/gC on(topk2) − off(topk1)/후보 − 제출본/; s/gC 유지(topk2)/**채택 가능**/; s/gC OFF(topk1) 권고/**미충족**/'
    echo; echo "### ${t}_pf − e9off_pf (10폴드) · 10폴드 대비 손실"; echo
    "$PY" "$D/g2_seeds.py" "$H/scores" "${t}_pf" e9off_pf "$t − 10폴드" \
      | sed 's/gC on(topk2) − off(topk1)/후보 − 10폴드/; s/gC 유지(topk2)/10폴드와 대등하거나 우세/; s/gC OFF(topk1) 권고/10폴드보다 열세/'
  done
  echo; echo "## 절대값 (5시드 평균)"; echo
  H3_TAGS="b1on_pf e9f3P3_pf e9f3P5_pf e9off_pf" "$PY" "$D/h3_abs.py" "$H/scores"
} > "$G/RESULTS.md" 2>&1
log "H4 완료 → $G/RESULTS.md"; touch "$D/.done_h4"
