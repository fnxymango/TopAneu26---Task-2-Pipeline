#!/usr/bin/env bash
# DET9 — 검출기만 E9(ResEncL 10폴드)로 갈아끼운 맞비교 (2026-09-16 · 사용자 승인)
#
#  왜 지금인가: E9 를 뺀 이유는 **오직 런타임**이었다. A5000 최악 253초 → T4 환산 380~430초,
#      구 GC 한도 420초 경계선(~/e9_bundle_experiment/NOTES.md · APPLY_CHANGES.md §4).
#      2026-09-16 한도가 **720초 · RAM 31GB** 로 완화됐다 → 약 290초 여유. 보류 사유가 사라졌다.
#      E9 는 신 eval 로 test·val 둘 다 우세(ΔMCC test +0.0203 · val +0.0182)한,
#      우리의 몇 안 되는 두 집합 동시 개선이다.
#
#  왜 다시 재나: 기존 E9 측정은 번들 통째 비교였다 — 검출기 말고도 **개정판 학습표**와 **gC OFF**
#      가 같이 섞여 있었다. 그 둘은 이후 각각 미채택(TBL10 K0 p=0.828) · gC ON 유지(H5)로 정리됐다.
#      따라서 **검출기 하나만** 바꿔 다시 잰다. 나머지는 전부 제출본 구성 그대로다.
#      기준 b1on_pf = b1ff + 구 학습표 + gC ON + 패치필터 (= 제출본 구성 · 시드 0~9 보유)
#      후보 b1e9_pf = **e9ff** + 구 학습표 + gC ON + 패치필터
#      NOTES 경고: "val 시드별 Δ 가 −0.055~+0.072 로 산포가 크다" → 통과해도 복제 필수.
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-16 16:45 KST) ─────────────────────
#  주판정: K0 장치(test+val 130단위) 시드 0~4 · 단측 부호검정 p<0.05 ∧ 오른>내린
#  안전 ①: ΔFP ≤ ΔTP   안전 ②: 기준에서 오답 0 이던 클래스의 새 오답 ≤ ΔTP
#  두 집합 규칙: 7지표에서 **test·val 양쪽** 개선 지표 수가 4/7 이상일 것
#                (test 가 안 오르면 채택하지 않는다 — 2026-09-16 사용자 지시)
#  통과 시 시드 5~9 복제에서도 같은 조건을 만족해야 "채택 권고"(반영은 사용자 결정)
#  런타임은 이 실험의 판정 대상이 아니다 — 별도로 실측한다.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/det9.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][det9] $*" | tee -a "$D/STATUS.log"; }
TAG=b1e9
FEAT=$A/e11_feat_hyb_ov.json          # 구 학습표 (PROJECT_RULES.md 0-2 · 2026-09-16 개정)
PAR=5                                  # FRAC 과 병행 중이라 절반만 쓴다 (20코어)

run_seeds(){
  local SEEDS="$1"
  log "c5 · 시드 $SEEDS · 검출기 e9ff · 구 학습표 · gC ON"
  clf(){ local sp=$1 sd=$2 bp exp n
    bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_e9ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "det9_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%PAR)) -eq 0 ] && wait; done; done; wait
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
  n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%2)) -eq 0 ] && wait; done; done; wait
  log "채점 · 시드 $SEEDS"
  n=0; for sp in val test; do for sd in $SEEDS; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/det9_score.log" 2>&1 &
    n=$((n+1)); [ $((n%PAR)) -eq 0 ] && wait
  done; done; wait
  return 0
}

report(){   # $1 = 시드(콤마) · $2 = 출력파일 · $3 = 제목
  { echo "# $3"; echo
    K0_SEEDS=$1 "$PY" "$D/k0_judge.py" b1on_pf ${TAG}_pf
    echo
    BASE=b1on_pf TAG=${TAG}_pf SEEDS=$1 LABEL_BASE="제출본 검출기(b1ff)" LABEL_TAG="E9 검출기(e9ff)" \
      "$PY" "$D/metrics_diff.py"
  } > "$2" 2>&1
}

run_seeds "0 1 2 3 4" || { log "★원판 실패"; exit 1; }
report "0,1,2,3,4" "$V/RESULTS_DET9.md" "DET9 — 검출기만 E9 로 교체 (시드 0~4) · 규칙은 DET9.sh 머리말 고정"
touch "$D/.done_det9"
log "DET9 원판 끝 → $(grep -o 'K0 판정 →.*' $V/RESULTS_DET9.md | tail -1)"

if grep -q '\*\*K0 판정 → 채택 후보\*\*' "$V/RESULTS_DET9.md"; then
  log "복제 시작 · 시드 5~9"
  run_seeds "5 6 7 8 9" || { log "★복제 실패"; exit 1; }
  report "5,6,7,8,9" "$V/RESULTS_DET9_REP.md" "DET9 복제 — 시드 5~9"
  touch "$D/.done_det9rep"
  log "DET9 복제 끝 → $(grep -o 'K0 판정 →.*' $V/RESULTS_DET9_REP.md | tail -1) · 반영은 사용자 결정"
else
  log "원판 미채택 → 복제 생략"
fi
