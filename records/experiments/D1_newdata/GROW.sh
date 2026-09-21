#!/usr/bin/env bash
# GROW — 출력 부피 보정 `TOPANEU_OUT_GROW` (S5) e2e (2026-09-17 · 사용자 승인)
#
#  왜: repo c5 에 `OUT_GROW`(S5 목표 부피 배수)가 구현돼 있는데 **실험 기록이 전무하다**
#      (`S5` grep 0 건). `crop_to_vessel` 처럼 짜놓고 안 쓴 코드다.
#      동작은 예측 마스크를 목표 배수까지 **최근접 껍질 복셀(4mm 이내)로 키우되 라벨은 최근접 상속** —
#      새 클래스가 생기지 않으므로 **클래스 존재가 불변**이다. FRAC 과 같은 계열(지표 구조 개입)이다.
#
#  값 근거 (C 진단 · 2026-09-17 09:38 KST · 규칙은 결과 보기 전 고정했다):
#      적중 병변의 (예측 부피 / GT 부피) 중앙값 — val **0.756** · test 0.816.
#      val 에서 병변의 81% 를 GT 보다 작게 낸다. 사전 규칙 "val 중앙 ≤ 0.85 면 진행,
#      OUT_GROW 는 val 중앙의 역수" 에 따라 **OUT_GROW = 1.32** 하나만 돌린다(스윕 없음 = 선택 자유도 0).
#
#  주의 — S4 의 교훈: 맹목적 +1복셀 팽창은 실패했다(복합 −0.0010). 이건 목표 부피를 맞추는
#      보정이라 다르지만, 키우는 방향이라는 점은 같다. HD95 가 나빠질 수 있어 안전장치를 둔다.
#
#  기준선: **b1frac035_pf** (= 제출본 구성 + 채택 확정된 FRAC 0.35 · 시드 0~9 보유).
#      FRAC 이 이미 채택됐으므로 B 는 그 위에서의 **한계 이득**을 물어야 한다.
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-17 09:40 KST) ─────────────────────
#  ① DICE 와 VOLSIM 이 **test·val 양쪽에서** 시드평균 개선
#  ② PRECISION·RECALL·F1·MCC 가 **전 시드 불변**(절대편차 < 1e-9) — 깨지면 "존재 불변" 가정이
#     틀린 것이므로 즉시 미달 처리하고 원인부터 본다
#  ③ 안전: HD95 악화가 test·val 어느 쪽에서도 시드평균 **+3** 을 넘지 않을 것
#  셋 다 만족하면 시드 5~9 복제로 진행, 복제도 만족해야 "채택 권고"(반영은 사용자 결정).
#  값은 이미 val 로 정해졌다 — 결과를 보고 다른 배수를 시도하지 않는다.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/grow.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][grow] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov.json
TAG=b1fg           # FRAC 0.35 + OUT_GROW 1.32
FRAC=0.35
GROW=1.32
BASE=b1frac035_pf

run_seeds(){
  local SEEDS="$1"
  log "c5 · 시드 $SEEDS · FRAC $FRAC + OUT_GROW $GROW · 기준 $BASE"
  clf(){ local sp=$1 sd=$2 bp exp n
    bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_TOPK_FRAC=$FRAC TOPANEU_OUT_GROW=$GROW TOPANEU_OUT_DILATE=0 \
    CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "grow_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
  for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
      [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; return 1; }; done; done
  log "패치필터 · 시드 $SEEDS"
  pf(){ local sp=$1 sd=$2 exp n; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
      --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
      --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
  }
  n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
  log "채점 · 시드 $SEEDS"
  n=0; for sp in val test; do for sd in $SEEDS; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/grow_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; wait
  return 0
}

run_seeds "0 1 2 3 4" || { log "★원판 실패"; exit 1; }
SEEDS=0,1,2,3,4 TAGS=${TAG}_pf LABELS=$GROW BASE=$BASE "$PY" "$D/frac_report.py" > "$V/RESULTS_GROW.md" 2>&1
sed -i "1s/.*/# GROW — 출력 부피 보정 OUT_GROW=$GROW (FRAC 0.35 위 · 기준 $BASE) · 규칙은 GROW.sh 머리말 고정/" "$V/RESULTS_GROW.md"
touch "$D/.done_grow"
log "GROW 원판 끝 → $(grep -o '\*\*판정 →.*' $V/RESULTS_GROW.md | tail -1)"

if grep -q '\*\*판정 → 통과' "$V/RESULTS_GROW.md"; then
  log "복제 시작 · 시드 5~9"
  run_seeds "5 6 7 8 9" || { log "★복제 실패"; exit 1; }
  SEEDS=5,6,7,8,9 TAGS=${TAG}_pf LABELS=$GROW BASE=$BASE "$PY" "$D/frac_report.py" > "$V/RESULTS_GROW_REP.md" 2>&1
  touch "$D/.done_growrep"
  log "GROW 복제 끝 → $(grep -o '\*\*판정 →.*' $V/RESULTS_GROW_REP.md | tail -1) · 반영은 사용자 결정"
else
  log "원판 미달 → 복제 생략 · GROW 닫음"
fi
