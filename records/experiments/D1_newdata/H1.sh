#!/usr/bin/env bash
# H1 — 통합본의 패치 CNN 환각필터를 우리 최신 기여(E9 10폴드 + 개정판 RF)에 그대로 입혀 재측정
#
# 질문: 통합본이 안 쓴 우리 최신 front end 에 jslee 패치필터를 붙이면 더 좋아지는가.
#
# 검출·혈관 추론은 다시 돌리지 않는다. 남겨둔 산출물에서 분류 단계부터 재개한다:
#   검출 aneu_{sp}_{e9ff|b1ff} · 혈관 vespp_{sp} · 분기점 _c4_bpgraph · 원본 in_{sp}
# 패치필터 코드(src/)와 가중치(patchclf/)는 통합 컨테이너·모델 tar 에서 바이트 그대로 가져왔다.
#
# 팔 4개 (시드 0~4 × test/val)
#   e9off      E9 10폴드 + 개정 RF(NEW 피처) + gC OFF(topk1)      ← 우리 최종
#   e9off_pf   e9off + 패치필터                                    ← 이번 질문
#   b1on       구 번들 검출기 + 구 RF(구 피처) + gC ON(topk2)      ← 통합본 근사(단, 5폴드)
#   b1on_pf    b1on + 패치필터                                     ← 그들이 보고한 +0.0148 재현 확인
#
# ── 판정규칙 (결과를 보기 전에 고정한다) ─────────────────────────────────────
#   1차(채택 판정) e9off_pf − e9off :
#       시드짝 Δ 로 신 eval 6지표 중 **≥4 개선 ∧ 평균 ΔMCC ≥ 0** 을
#       **test 와 val 둘 다** 만족할 때만 채택. (g2_seeds.py 와 동일한 규칙 + 두 집합 규칙)
#   2차(재현 확인, 채택 판정 아님) b1on_pf − b1on :
#       그들이 보고한 방향(+)과 일치하는지만 본다.
#   보고는 신 eval 6지표 + 구 eval official_div52 + covered_gt(÷52 를 ÷36[test]·÷33[val] 로 환산,
#   HD95 는 비례하지 않으므로 환산하지 않고 ÷52 값만 적는다).
#   패치필터 임계는 meta.json 값(2.0064e-4)을 그대로 쓴다 — 여기서 다시 고르지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/h1.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
echo $$ > "$D/H1.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h1] $*" | tee -a "$ST"; }
mkdir -p "$H"/{pred,scores,logs,report}

# ── 1단계 · c5 분류 (팔 2개 × 시드 5 × split 2 = 20런) ──────────────────────
clf(){ local tag=$1 det=$2 feat=$3 tk=$4 sp=$5 sd=$6 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  [ -d "$H/pred/${tag}_${sp}_s${sd}" ] && [ "$(ls -1 "$H/pred/${tag}_${sp}_s${sd}" 2>/dev/null | wc -l)" -gt 0 ] && return 0
  TOPANEU_TOPK=$tk TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 TOPANEU_TOPK_P2=0.0 \
  TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$A/$feat" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${det}ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${tag}_${sp}_s${sd}" --tag "h1_${tag}_${sp}_s${sd}" \
    > "$H/logs/${tag}_${sp}_s${sd}.log" 2>&1
}
log "1단계 · c5 20런 (8 병렬)"
n=0
for spec in "e9off e9 e11_feat_hyb_ov_NEW.json 1" "b1on b1 e11_feat_hyb_ov.json 2"; do set -- $spec
  for sp in test val; do for sd in 0 1 2 3 4; do
    clf $1 $2 $3 $4 $sp $sd & n=$((n+1)); [ $((n%8)) -eq 0 ] && wait
  done; done
done; wait
miss=0
for t in e9off b1on; do for sp in test val; do for sd in 0 1 2 3 4; do
  c=$(ls -1 "$H/pred/${t}_${sp}_s${sd}" 2>/dev/null | wc -l)
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  [ "$c" = "$exp" ] || { log "★c5 누락 ${t}_${sp}_s${sd} ($c/$exp)"; miss=1; }
done; done; done
[ $miss = 0 ] || { log "★1단계 실패 — 중단"; exit 1; }

# ── 2단계 · 패치필터 (GPU 직렬 2병렬) ───────────────────────────────────────
log "2단계 · 패치필터 20런"
pf(){ local tag=$1 sp=$2 sd=$3
  [ -d "$H/pred/${tag}_pf_${sp}_s${sd}" ] && [ "$(ls -1 "$H/pred/${tag}_pf_${sp}_s${sd}" 2>/dev/null | wc -l)" -gt 0 ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${tag}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${tag}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${tag}_${sp}_s${sd}.json" > "$H/logs/pf_${tag}_${sp}_s${sd}.log" 2>&1
}
n=0
for t in e9off b1on; do for sp in test val; do for sd in 0 1 2 3 4; do
  pf $t $sp $sd & n=$((n+1)); [ $((n%2)) -eq 0 ] && wait
done; done; done; wait

# ── 3단계 · 채점 (40런) ─────────────────────────────────────────────────────
log "3단계 · neweval 40런 (6 병렬)"
score(){ local tag=$1 sp=$2 sd=$3
  [ -s "$H/scores/${tag}_${sp}_s${sd}.json" ] && grep -q '"label"' "$H/scores/${tag}_${sp}_s${sd}.json" && return 0
  "$PY" "$D/neweval.py" "$H/pred/${tag}_${sp}_s${sd}" "$sp" "${tag}_${sp}_s${sd}" \
    > "$H/scores/${tag}_${sp}_s${sd}.json" 2> "$H/logs/score_${tag}_${sp}_s${sd}.err"
}
n=0
for t in e9off e9off_pf b1on b1on_pf; do for sp in test val; do for sd in 0 1 2 3 4; do
  score $t $sp $sd & n=$((n+1)); [ $((n%6)) -eq 0 ] && wait
done; done; done; wait

# ── 4단계 · 보고 ────────────────────────────────────────────────────────────
log "4단계 · 보고서"
{
  echo "# H1 — 패치 CNN 환각필터를 우리 최신 기여(E9 10폴드 + 개정 RF)에 적용"; echo
  echo '판정규칙은 H1.sh 머리말에 결과 보기 전에 고정해 두었다: 신 eval 6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을 test·val 둘 다 만족할 때만 채택.'; echo
  echo "## 1차 · e9off_pf − e9off (채택 판정)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" e9off_pf e9off "E9 + 패치필터 − E9" \
    | sed 's/gC on(topk2) − off(topk1)/패치필터 on − off/; s/gC 유지(topk2)/**패치필터 채택**/; s/gC OFF(topk1) 권고/**패치필터 미채택**/'
  echo; echo "## 2차 · b1on_pf − b1on (그들이 보고한 +0.0148 재현 확인 · 채택 판정 아님)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1on_pf b1on "구 front end + 패치필터 − 구 front end" \
    | sed 's/gC on(topk2) − off(topk1)/패치필터 on − off/; s/gC 유지(topk2)/방향 일치/; s/gC OFF(topk1) 권고/방향 불일치/'
  echo; echo "## 절대값 (5시드 평균) · 신 eval 6지표 + 구 eval ÷52 + covered_gt"; echo
  "$PY" "$D/h1_abs.py" "$H/scores"
  echo; echo "## 필터가 실제로 지운 blob"; echo
  "$PY" "$D/h1_drop.py" "$H/logs"
} > "$H/report/RESULTS.md" 2>&1
log "H1 완료 → $H/report/RESULTS.md"; touch "$D/.done_h1"
