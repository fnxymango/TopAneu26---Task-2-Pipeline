#!/usr/bin/env bash
# H2 — 검출기는 제출본 것으로 freeze 하고 분류기 피처만 개정판으로 바꾼다 (+ gC 레버 분리)
#
# 왜: E9 는 검출기·피처·gC 세 가지가 한꺼번에 바뀌어 있어 기여도를 귀속할 수 없다.
#     검출기를 고정하면 런타임을 전혀 안 건드리면서 얻을 수 있는 몫이 얼마인지 나오고,
#     동시에 ResEncL 의 순수 기여도(③)가 분리된다.
#
#   b1on   구검출기 + 구피처   + gC ON   ← 제출본 (H1 에서 이미 측정)
#     │ ① 피처
#   b1Non  구검출기 + 개정피처 + gC ON   ← 이번에 추가
#     │ ② gC
#   b1Noff 구검출기 + 개정피처 + gC OFF  ← 이번에 추가
#     │ ③ 검출기(ResEncL)
#   e9off  E9검출기 + 개정피처 + gC OFF  ← H1 에서 이미 측정
#
# 피처 json 은 GT 병변에서 뽑은 학습셋이라 검출기와 무관하다. 그래서 구 검출기에 개정판
# 피처를 그대로 물릴 수 있다. 검출 마스크는 양쪽 다 aneu_{sp}_b1ff 로 동일하다.
#
# ── 판정규칙 (결과 보기 전에 고정) ──────────────────────────────────────────
#   채택 판정은 ① 피처 효과에만 적용한다: b1Non_pf − b1on_pf 가
#   신 eval 6지표 중 **≥4 개선 ∧ 평균 ΔMCC ≥ 0** 을 **test·val 둘 다** 만족하면 채택.
#   (제출본이 이미 패치필터를 갖고 있으므로 _pf 끼리 비교하는 것이 실제 조건이다.)
#   ②·③ 은 기여도 분해용이며 채택 판정이 아니다.
#   시드별 Δ 산포가 평균보다 크면 그 사실을 결론에 명시한다(PROJECT_RULES.md 6-1b).
#   보고는 신 eval 6지표 + 구 eval ÷52 + covered_gt(test ÷36 · val ÷33 환산, HD95 제외).
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/h2.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
echo $$ > "$D/H2.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h2] $*" | tee -a "$ST"; }
mkdir -p "$H"/{pred,scores,logs,report}

clf(){ local tag=$1 tk=$2 sp=$3 sd=$4 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  local n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${tag}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=$tk TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$A/e11_feat_hyb_ov_NEW.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${tag}_${sp}_s${sd}" --tag "h2_${tag}_${sp}_s${sd}" \
    > "$H/logs/${tag}_${sp}_s${sd}.log" 2>&1
}
log "1단계 · c5 20런 (10 병렬) — 검출기 b1ff 고정 · 개정판 피처"
n=0
for spec in "b1Non 2" "b1Noff 1"; do set -- $spec
  for sp in test val; do for sd in 0 1 2 3 4; do
    clf $1 $2 $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done
done; wait
miss=0
for t in b1Non b1Noff; do for sp in test val; do for sd in 0 1 2 3 4; do
  c=$(ls -1 "$H/pred/${t}_${sp}_s${sd}" 2>/dev/null | wc -l); exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  [ "$c" = "$exp" ] || { log "★c5 누락 ${t}_${sp}_s${sd} ($c/$exp)"; miss=1; }
done; done; done
[ $miss = 0 ] || { log "★1단계 실패 — 중단"; exit 1; }

log "2단계 · 패치필터 20런"
pf(){ local tag=$1 sp=$2 sd=$3 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${tag}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${tag}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${tag}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${tag}_${sp}_s${sd}.json" > "$H/logs/pf_${tag}_${sp}_s${sd}.log" 2>&1
}
n=0
for t in b1Non b1Noff; do for sp in test val; do for sd in 0 1 2 3 4; do
  pf $t $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait
done; done; done; wait

log "3단계 · neweval 40런 (14 병렬)"
score(){ local tag=$1 sp=$2 sd=$3 t
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  t="$H/scores/.tmp_${tag}_${sp}_s${sd}.$$"
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$t" 2> "$H/logs/score_${tag}_${sp}_s${sd}.err"
  if grep -q '"label"' "$t"; then mv -f "$t" "$H/scores/${tag}_${sp}_s${sd}.json"; else rm -f "$t"; log "★채점 실패 ${tag}_${sp}_s${sd}"; fi
}
n=0
for t in b1Non_pf b1Non b1Noff_pf b1Noff; do for sp in val test; do for sd in 0 1 2 3 4; do
  score $t $sp $sd & n=$((n+1)); [ $((n%14)) -eq 0 ] && wait
done; done; done; wait

log "4단계 · 보고서"
{
  echo "# H2 — 검출기 freeze · 분류기 피처만 개정판으로"; echo
  echo '판정규칙은 H2.sh 머리말에 결과 보기 전에 고정. 채택 판정은 ① 피처 효과에만 적용한다.'; echo
  echo "## ① 피처 효과 (채택 판정) · b1Non_pf − b1on_pf · 검출기·gC·필터 모두 동일"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1Non_pf b1on_pf "개정피처 − 구피처 (검출기 freeze · gC ON · 필터 on)" \
    | sed 's/gC on(topk2) − off(topk1)/개정피처 − 구피처/; s/gC 유지(topk2)/**개정피처 채택**/; s/gC OFF(topk1) 권고/**개정피처 미채택**/'
  echo; echo "## ①′ 같은 비교, 필터 없이 · b1Non − b1on"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1Non b1on "개정피처 − 구피처 (필터 off)" \
    | sed 's/gC on(topk2) − off(topk1)/개정피처 − 구피처/; s/gC 유지(topk2)/충족/; s/gC OFF(topk1) 권고/미충족/'
  echo; echo "## ② gC 효과 (분해용) · b1Noff_pf − b1Non_pf · 개정피처 고정"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1Noff_pf b1Non_pf "gC OFF − gC ON (개정피처 · 검출기 freeze)" \
    | sed 's/gC on(topk2) − off(topk1)/gC OFF − ON/; s/gC 유지(topk2)/gC OFF 우세/; s/gC OFF(topk1) 권고/gC ON 우세/'
  echo; echo "## ③ 검출기 효과 (분해용) · e9off_pf − b1Noff_pf · 피처·gC·필터 모두 동일, 검출기만 다름"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" e9off_pf b1Noff_pf "ResEncL 10폴드 − stock 5폴드 (다른 조건 동일)" \
    | sed 's/gC on(topk2) − off(topk1)/ResEncL − stock/; s/gC 유지(topk2)/ResEncL 우세/; s/gC OFF(topk1) 권고/ResEncL 미충족/'
  echo; echo "## 절대값 (5시드 평균) · 사슬 전체"; echo
  "$PY" "$D/h2_abs.py" "$H/scores"
} > "$H/report/RESULTS_H2.md" 2>&1
log "H2 완료 → $H/report/RESULTS_H2.md"; touch "$D/.done_h2"
