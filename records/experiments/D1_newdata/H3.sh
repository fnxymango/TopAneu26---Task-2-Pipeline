#!/usr/bin/env bash
# H3 / R1 — 곁가지 신뢰도(vesconf) 피처 e2e (2026-09-17 · 사용자 승인)
#   설계·근거는 `experiments/H3_vesconf/NOTES.md` (2026-09-10 작성, 그동안 미실행).
#
#  왜 지금: R0(ICA 천장) 측정에서 **ICA 원위 오답의 61%(127/209)가 같은 분절 안**이었다.
#      3.5↔3.4 · 3.3↔3.2 · 3.4↔3.6 — 전부 곁가지(AChA·Pcom·OA) 구분 문제다.
#      ICA 를 토막내는 축은 상한이 시드당 5.6개뿐이라 접었고, 이 축이 그 61% 를 정면으로 겨냥한다.
#      H3 관찰: 병변 5mm 안에 **가짜 곁가지**가 있으면 분류 정답률 71.3% → 54.7%.
#      가짜 곁가지 내역 L-Pcom 7 · L-AChA 5 · R-AChA 4 · R-Pcom 3 · R-OA 1.
#      현 혈관모델 Dice — R-AChA 0.52 · L-AChA 0.58 · L-Pcom 0.62 (허깨비가 나올 만하다).
#
#  무엇: conf[c] = clip(최대연결성분_복셀[c] / 학습중앙값[c], 0, 2). 굵고 이어진 관이면 1 근처,
#      작고 끊긴 조각이면 0 근처. 기준 중앙값은 **학습 케이스에서만** 잡아 누수가 없다.
#      block = conf 36차원을 피처 뒤에 붙인다(RF 가 쓸지 말지 스스로 정함)
#      gate  = 거리/중첩 피처에 conf 를 곱한다(허깨비 신호를 직접 누른다)
#
#  0단계 검사 완료(2026-09-17): 표 3종 존재 · 학습 214/val 41/test 83 **커버리지 100%** ·
#      vesconf(08-25) 가 vespp(08-14) 보다 나중 생성. 1단계 후 로그에 `[vesconf]` 가 찍혔는지 본다 —
#      조용히 off 로 돌면 기준팔과 같은 결과가 나와 "효과 없음" 으로 오독된다(NOTES 경고).
#
#  기준선: **b1frac035_pf** (= 제출본 구성 + 채택 확정된 FRAC 0.35 · 시드 0~9 보유).
#      두 팔 모두 FRAC 0.35 를 켜고 돈다 — 채택된 구성 위에서의 한계 이득을 묻는다.
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-17 10:25 KST) ─────────────────────
#  NOTES 의 "6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을 test·val 둘 다" 를 현행 7지표 기준(PROJECT_RULES.md 6-0)
#  으로 옮긴다: **7지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 · test·val 둘 다**.
#  더불어 K0 장치(130단위)도 같이 찍되 **주판정은 위 지표 규칙**으로 한다(NOTES 가 그렇게 고정했다).
#  두 모드가 다 통과하면 **val 의 ΔMCC 가 큰 쪽**을 고른다(test 로 고르지 않는다).
#  통과 시 시드 5~9 복제에서도 같은 조건을 만족해야 "채택 권고"(반영은 사용자 결정).
#  검출력 한계: val 41 은 +0.018 수준의 진짜 효과도 t=2.3 밖에 못 낸다(NOTES). 결론에 명시한다.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
export TOPANEU_VESCONF_DIR=$A
exec 9>"$D/h3.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][h3] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov.json
BASE=b1frac035_pf
MODES="block gate"
tag_of(){ echo "b1vc$1"; }

run_seeds(){
  local SEEDS="$1"
  for M in $MODES; do
    local TAG; TAG=$(tag_of "$M")
    log "c5 · vesconf=$M · 태그 $TAG · 시드 $SEEDS · FRAC 0.35"
    clf(){ local sp=$1 sd=$2 bp exp n
      bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
      exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
      TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
      TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
      TOPANEU_TOPK_FRAC=0.35 TOPANEU_OUT_DILATE=0 TOPANEU_VESCONF=$M \
      CLF_SEED=$sd OMP_NUM_THREADS=1 \
      "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
        --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
        --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
        --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "h3_${TAG}_${sp}_s${sd}" \
        > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
    }
    local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
    # ★ NOTES 경고 — vesconf 가 조용히 off 로 돌면 기준팔과 같은 결과가 나와 오독된다
    if ! grep -qi "vesconf" "$H/logs/${TAG}_test_s$(echo $SEEDS | cut -d' ' -f1).log" 2>/dev/null; then
      log "★[vesconf] 로그가 안 찍힘 — 모드 $M 이 적용되지 않았을 수 있다. 중단."
      return 1
    fi
    for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
        [ "$c" = "$exp" ] || { log "★c5 누락 ${TAG} ${sp}_s${sd} ($c/$exp)"; return 1; }; done; done
    log "패치필터 · $M · 시드 $SEEDS"
    pf(){ local sp=$1 sd=$2 exp n; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
      "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
        --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
        --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
    }
    n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
    log "채점 · $M · 시드 $SEEDS"
    n=0; for sp in val test; do for sd in $SEEDS; do
      d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
      grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/h3_score.log" 2>&1 &
      n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
    done; done; wait
  done
  return 0
}

report(){  # $1 시드(콤마) · $2 출력 · $3 제목
  { echo "# $3"; echo
    for M in $MODES; do
      T=$(tag_of "$M")
      echo "## 모드 \`$M\` — K0 단위 판정 (참고 · 주판정 아님)"; echo
      K0_SEEDS=$1 "$PY" "$D/k0_judge.py" $BASE ${T}_pf
      echo
      BASE=$BASE TAG=${T}_pf SEEDS=$1 LABEL_BASE="기준(FRAC 0.35)" LABEL_TAG="vesconf $M" \
        "$PY" "$D/metrics_diff.py"
      echo; echo "---"; echo
    done
  } > "$2" 2>&1
}

run_seeds "0 1 2 3 4" || { log "★원판 실패"; exit 1; }
report "0,1,2,3,4" "$V/RESULTS_H3.md" "H3/R1 — 곁가지 신뢰도 vesconf (시드 0~4) · 규칙은 H3.sh 머리말 고정"
touch "$D/.done_h3"
log "H3 원판 끝 → $V/RESULTS_H3.md"
