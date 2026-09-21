#!/usr/bin/env bash
# FRAC — gC 2등 조각의 복셀 지분(`TOPANEU_TOPK_FRAC`) 복구 실험 (2026-09-16 · 사용자 승인)
#
#  왜: C60_metric_structure/summary.md §22·§24 가 `TOPK_FRAC=0.50` 을 **채택 ②** 로 판정했는데
#      (당시 6지표 test +0.0042 · val +0.0025), 제출본에도 우리 실험 스크립트에도 들어가 있지 않다.
#      `final_rf_seed3.pkl` 의 topk 딕셔너리에 frac 키가 없고 pipeline_case.py 도 설정하지 않는다.
#      즉 지금 2등 조각은 blob 281복셀 중 3복셀(1%)만 받는다. 이건 새 축이 아니라 **누락 복구**다.
#      되돌린 기록은 문서 어디에도 없다(PROJECT_RULES.md · NOTES.md · APPLY_CHANGES.md · chain_status.md 검색).
#
#  왜 기대하나: 이번 주 태운 축은 전부 **결정 경계를 옮기는** 개입이었고 전부 죽었다
#      (B1 · K7 · R1 · R3 · DuoRF · 개정판 학습표). 이건 경계를 안 옮긴다 —
#      TP 성립은 1복셀이면 되므로 존재기반 4지표는 **정의상 불변**이고, 복셀 지분에 비례하는
#      DICE·VOLSIM 만 움직인다. 과적합할 자유도가 구조적으로 없다(C60 §20).
#      단 C60 의 "무손실" 주장은 현행 7지표에서 그대로 성립하지 않는다 — HD95 는 예측 영역
#      모양이 바뀌면 같이 바뀐다. 그래서 4지표 불변 + 2지표 개선 + HD95 안전장치로 판정한다.
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-16 16:35 KST) ──────────────────────
#  대상 지분: 0.35 · 0.50 · 0.65   (기준 = 현행 FRAC 0 = 3복셀 고정 · 태그 b1on_pf 시드 0~9 보유)
#  주판정(지분값마다):
#    ① DICE 와 VOLSIM 이 **test·val 양쪽에서** 시드평균 개선
#    ② PRECISION·RECALL·F1·MCC 가 **전 시드 불변**(절대편차 < 1e-9) — 깨지면 코드 가정이
#       틀린 것이므로 그 지분값은 즉시 미달 처리하고 원인부터 본다
#    ③ 안전: HD95 악화가 test·val 어느 쪽에서도 시드평균 +3 을 넘지 않을 것
#  값 선택: 통과한 지분 중 **val 의 DICE+VOLSIM 합이 가장 큰 것**. test 로 고르지 않는다.
#           train OOF 는 쓰지 않는다 — 학습표는 병변 단위라 e2e 복셀 지분을 잴 경로가 없다.
#  통과 시 시드 5~9 복제에서도 ①②③ 을 만족해야 "채택 권고"(반영은 사용자 결정).
#  전부 미달이면 FRAC 축을 닫는다.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/frac.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][frac] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov.json          # 기준선 = 구 학습표 (PROJECT_RULES.md 0-2 · 2026-09-16 개정)
FRACS="0.35 0.50 0.65"
tag_of(){ echo "b1frac$(echo "$1" | tr -d '.')"; }      # 0.35 -> b1frac035

run_seeds(){
  local SEEDS="$1"
  for FR in $FRACS; do
    local TAG; TAG=$(tag_of "$FR")
    log "c5 · 지분 $FR · 태그 $TAG · 시드 $SEEDS"
    clf(){ local sp=$1 sd=$2 bp exp n
      bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
      exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
      TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
      TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
      TOPANEU_TOPK_FRAC=$FR TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
      "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
        --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
        --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
        --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "frac_${TAG}_${sp}_s${sd}" \
        > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
    }
    local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
    for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
        [ "$c" = "$exp" ] || { log "★c5 누락 ${TAG} ${sp}_s${sd} ($c/$exp)"; return 1; }; done; done

    log "패치필터 · 지분 $FR · 시드 $SEEDS"
    pf(){ local sp=$1 sd=$2 exp n; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
      n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
      "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
        --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
        --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
    }
    n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait

    log "채점 · 지분 $FR · 시드 $SEEDS"
    n=0; for sp in val test; do for sd in $SEEDS; do
      d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
      grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/frac_score.log" 2>&1 &
      n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
    done; done; wait
  done
  return 0
}

TAGS=""; LABELS=""
for FR in $FRACS; do TAGS="$TAGS,$(tag_of "$FR")_pf"; LABELS="$LABELS,$FR"; done
TAGS=${TAGS#,}; LABELS=${LABELS#,}

run_seeds "0 1 2 3 4" || { log "★원판 실패"; exit 1; }
SEEDS=0,1,2,3,4 TAGS="$TAGS" LABELS="$LABELS" BASE=b1on_pf \
  "$PY" "$D/frac_report.py" > "$V/RESULTS_FRAC.md" 2>&1
touch "$D/.done_frac"
log "FRAC 원판 끝 → $(grep -o '\*\*판정 →.*' $V/RESULTS_FRAC.md | tail -1)"

if grep -q '\*\*판정 → 통과' "$V/RESULTS_FRAC.md"; then
  log "복제 시작 · 시드 5~9"
  run_seeds "5 6 7 8 9" || { log "★복제 실패"; exit 1; }
  SEEDS=5,6,7,8,9 TAGS="$TAGS" LABELS="$LABELS" BASE=b1on_pf \
    "$PY" "$D/frac_report.py" > "$V/RESULTS_FRAC_REP.md" 2>&1
  touch "$D/.done_fracrep"
  log "FRAC 복제 끝 → $(grep -o '\*\*판정 →.*' $V/RESULTS_FRAC_REP.md | tail -1) · 반영은 사용자 결정"
else
  log "원판 미달 → 복제 생략 · FRAC 축 닫음"
fi
