#!/usr/bin/env bash
# H5 — 제출본에 gC 만 끈다 (구피처 유지). 사용자가 gC OFF 로 확정했으므로 그 구성의 실측치를 채운다.
#
# 왜 필요한가: H2 ② 의 gC 효과(+0.0271/+0.0296)는 **개정피처 위에서** 잰 값인데, 개정피처는
#   ① 에서 탈락했다. 실제로 나갈 구성은 구피처 + gC OFF 이고, 그 팔은 H2 에 없다.
#   제출본 기준 변경 명세(APPLY_CHANGES.md)에 적을 숫자는 이 팔에서 나와야 한다.
#
# 제출본 대비 바뀌는 것: TOPANEU_TOPK 2 → 1 하나뿐. 검출 마스크(b1ff)·피처(구)·혈관·분기점·
#   c7·패치필터 전부 제출본과 동일. 검출 추론을 재사용하므로 GPU 를 쓰지 않는다.
#
# ── 판정규칙 (결과 보기 전에 고정) ─────────────────────────────────────────
#   b1off_pf − b1on_pf 가 신 eval 6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을 test·val 둘 다 만족하면
#   "제출본에서도 gC OFF 가 이득" 으로 확인. 불만족이면 ② 의 이득이 개정피처에 딸린 것이므로
#   그 사실을 APPLY_CHANGES.md 에 명시한다(확정을 뒤집자는 게 아니라 근거를 정확히 적기 위함).
#   cov MCC 는 방향이 반대로 나올 것으로 예상됨 — 그대로 병기한다.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 7>"$D/h5.lock"; flock -n 7 || { echo "이미 실행 중"; exit 0; }
echo $$ > "$D/H5.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h5] $*" | tee -a "$ST"; }

log "1단계 · c5 10런 (구피처 · gC OFF · 검출기 b1ff)"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/b1off_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=1 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/b1off_${sp}_s${sd}" --tag "h5_b1off_${sp}_s${sd}" \
    > "$H/logs/b1off_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait

log "2단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/b1off_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/b1off_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/b1off_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_b1off_${sp}_s${sd}.json" > "$H/logs/pf_b1off_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait

log "3단계 · neweval 20런 (10 병렬)"
score(){ local tag=$1 sp=$2 sd=$3 t
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  t="$H/scores/.tmp_${tag}_${sp}_s${sd}.$$"
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$t" 2> "$H/logs/score_${tag}_${sp}_s${sd}.err"
  if grep -q '"label"' "$t"; then mv -f "$t" "$H/scores/${tag}_${sp}_s${sd}.json"
  else rm -f "$t"; log "★채점 실패 ${tag}_${sp}_s${sd}"; fi
}
n=0; for t in b1off_pf b1off; do for sp in val test; do for sd in 0 1 2 3 4; do
  score $t $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; done; wait

log "4단계 · 보고서"
{
  echo "# H5 — 제출본에 gC 만 끄기 (구피처 유지)"; echo
  echo '판정규칙은 H5.sh 머리말에 결과 보기 전에 고정.'; echo
  echo "## b1off_pf − b1on_pf · 제출본에서 gC 만 끈 효과"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1off_pf b1on_pf "gC OFF − 제출본" \
    | sed 's/gC on(topk2) − off(topk1)/gC OFF − 제출본(gC ON)/; s/gC 유지(topk2)/**제출본에서도 gC OFF 이득 확인**/; s/gC OFF(topk1) 권고/**제출본에서는 미확인**/'
  echo; echo "## 참고 · 필터 없이 b1off − b1on"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1off b1on "gC OFF − 제출본 (필터 off)" \
    | sed 's/gC on(topk2) − off(topk1)/gC OFF − 제출본/; s/gC 유지(topk2)/이득 확인/; s/gC OFF(topk1) 권고/미확인/'
  echo; echo "## 절대값 (5시드 평균)"; echo
  H3_TAGS="b1on b1on_pf b1off b1off_pf b1Noff_pf e9off_pf" "$PY" "$D/h3_abs.py" "$H/scores"
} > "$H/report/RESULTS_H5.md" 2>&1
log "H5 완료 → $H/report/RESULTS_H5.md"; touch "$D/.done_h5"
