#!/usr/bin/env bash
# H6 — 제출본에서 **검출기만** 바꾼다 (구피처 유지 · gC ON 유지)
#
# 왜: 지금까지 E9 를 잰 모든 팔(e9off_pf 등)은 검출기 말고도 개정피처와 gC OFF 가 함께 들어가
#     있었다. 그런데 H2 는 개정피처가 −0.0240/−0.0144 로 해롭다고, H5 는 gC OFF 가 구피처에서
#     ±0.001 로 무효라고 말한다. 즉 그 사슬은 스스로 낸 손해를 스스로 메우고 있었다.
#     실제로 제출할 구성은 **제출본에서 검출기만 교체한 것**이고, 그 팔이 아직 없다.
#
# 제출본 대비 바뀌는 것: aneurysm-pred-dir 하나뿐.
#   피처(구 e11_feat_hyb_ov) · gC ON(topk2) · 혈관 · 분기점 · c7 · 패치필터 전부 제출본과 동일.
#   검출 추론은 이미 있는 결과를 재사용하므로 GPU 를 쓰지 않는다.
#
#   e9on    = E9 ResEncL 10폴드 검출기
#   e9f3P3on = E9 ResEncL 3폴드(0,1,2) 검출기   ← 폴드 수 질문도 같은 피처 조건에서 재확인
#   (대조) b1on_pf = 제출본, H1 에서 측정 완료
#
# ── 판정규칙 (결과 보기 전에 고정) ─────────────────────────────────────────
#   채택: {tag}_pf − b1on_pf 가 신 eval 6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을 test·val 둘 다 만족.
#   부 질문: e9f3P3on_pf − e9on_pf 로 구피처 조건에서의 폴드 손실을 본다(채택 판정 아님).
#   cov MCC 는 외부보고용으로 병기하되 채택 판정에 쓰지 않는다(공식 eval 이 결정).
#   시드 산포가 평균보다 크면 결론에 명시한다.
#   런타임 한도는 12분·메인메모리 32GB 로 완화됨 — 10폴드(T4 환산 380~430s)도 들어간다.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 6>"$D/h6.lock"; flock -n 6 || { echo "이미 실행 중"; exit 0; }
echo $$ > "$D/H6.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h6] $*" | tee -a "$ST"; }

# tag → 검출 결과 디렉터리
det_of(){ case "$1" in e9on) echo e9ff;; e9f3P3on) echo e9f3P3ff;; esac; }

log "1단계 · c5 20런 (10 병렬) · 구피처 · gC ON · 검출기만 교체"
clf(){ local tag=$1 sp=$2 sd=$3 bp exp n det
  det=$(det_of "$tag")
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${tag}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${det}" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${tag}_${sp}_s${sd}" --tag "h6_${tag}_${sp}_s${sd}" \
    > "$H/logs/${tag}_${sp}_s${sd}.log" 2>&1
}
n=0; for t in e9on e9f3P3on; do for sp in test val; do for sd in 0 1 2 3 4; do
  clf $t $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; done; wait
miss=0
for t in e9on e9f3P3on; do for sp in test val; do for sd in 0 1 2 3 4; do
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  c=$(ls -1 "$H/pred/${t}_${sp}_s${sd}" 2>/dev/null | wc -l)
  [ "$c" = "$exp" ] || { log "★c5 누락 ${t}_${sp}_s${sd} ($c/$exp)"; miss=1; }
done; done; done
[ $miss = 0 ] || { log "★1단계 실패 — 중단"; exit 1; }

log "2단계 · 패치필터 20런 (3 병렬)"
pf(){ local tag=$1 sp=$2 sd=$3 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${tag}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${tag}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${tag}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${tag}_${sp}_s${sd}.json" > "$H/logs/pf_${tag}_${sp}_s${sd}.log" 2>&1
}
n=0; for t in e9on e9f3P3on; do for sp in test val; do for sd in 0 1 2 3 4; do
  pf $t $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait
done; done; done; wait

log "3단계 · neweval 40런 (14 병렬 · BLAS 1스레드)"
score(){ local tag=$1 sp=$2 sd=$3 t
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  t="$H/scores/.tmp_${tag}_${sp}_s${sd}.$$"
  # BLAS 스레드 폭주 방지 — 이게 없으면 채점 하나가 20~36 스레드를 띄워 20코어를 과구독한다(2026-09-10 실측)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$t" 2> "$H/logs/score_${tag}_${sp}_s${sd}.err"
  if grep -q '"label"' "$t"; then mv -f "$t" "$H/scores/${tag}_${sp}_s${sd}.json"
  else rm -f "$t"; log "★채점 실패 ${tag}_${sp}_s${sd}"; fi
}
n=0; for t in e9on_pf e9f3P3on_pf e9on e9f3P3on; do for sp in val test; do for sd in 0 1 2 3 4; do
  score $t $sp $sd & n=$((n+1)); [ $((n%14)) -eq 0 ] && wait
done; done; done; wait

log "4단계 · 보고서"
{
  echo "# H6 — 제출본에서 검출기만 교체 (구피처 · gC ON 유지)"; echo
  echo '판정규칙은 H6.sh 머리말에 결과 보기 전에 고정.'; echo
  for t in e9on e9f3P3on; do
    echo "## ${t}_pf − b1on_pf (제출본) · 채택 판정"; echo
    "$PY" "$D/g2_seeds.py" "$H/scores" "${t}_pf" b1on_pf "$t − 제출본" \
      | sed 's/gC on(topk2) − off(topk1)/후보 − 제출본/; s/gC 유지(topk2)/**채택**/; s/gC OFF(topk1) 권고/**미채택**/'
    echo
  done
  echo "## 폴드 손실 · e9f3P3on_pf − e9on_pf (구피처 조건)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" e9f3P3on_pf e9on_pf "3폴드 − 10폴드" \
    | sed 's/gC on(topk2) − off(topk1)/3폴드 − 10폴드/; s/gC 유지(topk2)/3폴드로 충분/; s/gC OFF(topk1) 권고/10폴드가 우세/'
  echo; echo "## 절대값 (5시드 평균) · 전체 후보"; echo
  H3_TAGS="b1on_pf b1off_pf b1Noff_pf e9f3P3on_pf e9on_pf e9f3P3_pf e9off_pf" "$PY" "$D/h3_abs.py" "$H/scores"
} > "$H/report/RESULTS_H6.md" 2>&1
log "H6 완료 → $H/report/RESULTS_H6.md"; touch "$D/.done_h6"
