#!/usr/bin/env bash
# G-1 — 혈관 연결관계 블록 e2e (2026-09-17 · 사용자 승인 · GNN 제작 전 단계)
#
#  왜 GNN 이 아니라 이걸 먼저: GNN 은 같은 그래프 정보를 파라미터만 더 써서 읽는다.
#      블록이 e2e 에서 죽으면 GNN 도 죽는다. 4~5시간 대 며칠이다.
#
#  G-0 관문 결과 (LOCO 214폴드 · 시드 0,1,2 · 규칙 3개 전부 충족 → 통과):
#      표적 21행 top-2  6.0 → **7.0**   전체 top-1 69.3% → 69.7%
#      **비관계형만 팔은 6.0** — 움직인 것은 연결관계다(제안서가 요구한 간선 제거 대조군).
#      단, +1.0 은 전부 L-3.5 한 클래스이고 **표적 top-1 은 2.0 → 1.3 으로 내렸다**. 실질 병변 1개다.
#
#  블록 10차원 (근거: 예측혈관 그래프 노드의 23% 가 JUNCTION_PAIRS 34쌍 밖 — 2혈관 34쌍밖 18% + 3혈관 이상 5%)
#      [관계형 5] 지역차수 · 주혈관 인접차수 · 주혈관 노드수 · 최근접노드가 주혈관 포함 · 3혈관 노드수
#      [비관계형 5] 최근접 노드 · 최근접 3혈관 노드 · 최근접 34쌍밖 노드 · 5mm 내 개수 · 10mm 내 개수
#      학습표 `e11_feat_hyb_ov_g.json` = 기준표 + `gfeat` 키만 추가(행 268 동일 · 기존 값 불일치 0건 실측).
#      좌우 미러 불변 실측 확인. GRAPH=0 이면 112차원 그대로(기본동작 불변 실측).
#
#  ★ 알고 도는 차이 — G-0 과 e2e 는 **같은 피처가 아니다.** G-0 은 원시 10차원을 RF 에 그냥 붙였고,
#      c5 는 기존 블록 규약대로 **블록 정규화 + 가중치 0.5 + 전체 재정규화**를 한다. RF 는 축정렬이라
#      단조 변환에는 불변이지만 전체 재정규화는 표본별 스케일이라 불변이 아니다. 그래서 G-0 이
#      e2e 를 보장하지 않는다 — 이 판정이 진짜 판정이다.
#
#  기준선: **b1fg_pf** (= 제출본 구성 + FRAC 0.35 + OUT_GROW 1.32, 둘 다 채택 확정 · 시드 0~9 보유)
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-17 16:05 KST) ─────────────────────
#  ① 적용 확인: 로그에 `[graph] 연결관계 블록 ON` 이 찍혀야 한다. 없으면 즉시 중단
#     (H3 교훈 — 조용히 off 로 돌면 기준팔과 같은 결과가 나와 "효과 없음" 으로 오독된다).
#  ② **주판정 = K0 장치** (PROJECT_RULES.md 분류기 실험 표준): test+val 130단위 · 시드 0~4 ·
#     단위별 적중 시드수 순증의 단측 부호검정 **p < 0.05 ∧ 오른 > 내린**
#     · 안전① ΔFP ≤ ΔTP · 안전② 기준 무오답 클래스의 새 오답 ≤ ΔTP.
#  ③ **두 집합 규칙**(사용자 2026-09-16): 7지표에서 **test 평균 ΔMCC ≥ 0**. 음수면 미달.
#  ②∧③ 을 만족해야 복제(시드 5~9)로 간다. 복제도 같은 조건을 만족해야 "채택 권고"(반영은 사용자 결정).
#  미달이면 **블록을 닫고, GNN 도 같이 닫는다** — 같은 정보를 읽는 축이기 때문이다.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/g1graph.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g1graph] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov_g.json
TAG=b1gph
BASE=b1fg_pf

run_seeds(){
  local SEEDS="$1"
  log "c5 · 시드 $SEEDS · FRAC 0.35 + OUT_GROW 1.32 + GRAPH 10차원 · 기준 $BASE"
  clf(){ local sp=$1 sd=$2 bp exp n
    bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_TOPK_FRAC=0.35 TOPANEU_OUT_GROW=1.32 TOPANEU_OUT_DILATE=0 TOPANEU_GRAPH=1 \
    CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "g1graph_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
  # ★ 규칙 ① — 블록이 실제로 걸렸는지
  local first; first=$(echo $SEEDS | cut -d' ' -f1)
  if ! grep -q "\[graph\] 연결관계 블록 ON" "$H/logs/${TAG}_test_s${first}.log" 2>/dev/null; then
    log "★[graph] 로그 없음 — 블록이 적용되지 않았다. 중단."; return 1
  fi
  log "적용 확인: $(grep -m1 '\[graph\]' $H/logs/${TAG}_test_s${first}.log)"
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
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/g1graph_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; wait
  return 0
}

report(){  # $1 시드(콤마) · $2 출력 · $3 제목
  { echo "# $3"; echo
    echo "## K0 병변 짝지은 판정 (주판정 · 규칙 ②)"; echo
    K0_SEEDS=$1 "$PY" "$D/k0_judge.py" $BASE ${TAG}_pf
    echo; echo "## 7지표 (규칙 ③ — test 평균 ΔMCC ≥ 0)"; echo
    BASE=$BASE TAG=${TAG}_pf SEEDS=$1 LABEL_BASE="기준(FRAC+GROW)" LABEL_TAG="연결관계 블록" \
      "$PY" "$D/metrics_diff.py"
  } > "$2" 2>&1
}

run_seeds "0 1 2 3 4" || { log "★원판 실패"; exit 1; }
report "0,1,2,3,4" "$V/RESULTS_G1GRAPH.md" "G-1 — 혈관 연결관계 블록 e2e (시드 0~4) · 규칙은 G1GRAPH.sh 머리말 고정"
touch "$D/.done_g1graph"
log "G-1 원판 끝 → $V/RESULTS_G1GRAPH.md"
