#!/usr/bin/env bash
# K1OLD — K1(두 표 확률평균)을 **구 학습표**(제출본이 쓰는 라벨판) 위에서 재현 (2026-09-16 · 사용자 지시)
#  왜: K1 은 개정판 학습표 위에서만 측정됐다. 그런데 개정판 학습표 자체는 H2 관문에서 미채택이라
#      제출본에 안 들어가 있다. 구 학습표 위에서도 K1 이 같은 이득을 내면 **제출본 대비 변경이 하나로 끝난다.**
#  표: ① e11_feat_hyb_ov.json(구) ② c10_feat_train_predves.json(구 · 같은 라벨판의 예측혈관표) · 268행
#  기준: b1on_pf (구검출기 + 구 학습표 + gC ON + 패치필터 = 제출본과 같은 구성) · 시드 0~4
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-16) ───────────────────────────────
#  K0 장치 α 0.05 · 안전 ① ΔFP ≤ ΔTP · 안전 ② 기준 무오답 클래스 새 오답 ≤ ΔTP
#  보조 기록: 7지표 test·val (공식 개정 eval · PROJECT_RULES.md 6-0) · 개정판 위 K1 과 같은 방향인지
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-16) ─────────────────────────────
#  주판정: K0 장치 시드 0~4 · α 0.05 · 부호검정 p<0.05 ∧ 오른>내린
#  안전 ①: ΔFP ≤ ΔTP · 안전 ②: 기준에서 오답 0 이던 클래스의 새 오답 ≤ ΔTP
#  통과 시 시드 5~9 복제에서도 같은 조건을 만족해야 "채택 권고"(반영은 사용자 결정)
#  보조 기록(판정 미사용): 7지표 test·val 평균 Δ 와 시드별 개선 지표 수
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/k1orep.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][k1o] $*" | tee -a "$D/STATUS.log"; }
TAG=b1k1o
FEAT=$A/e11_feat_hyb_ov.json
export TOPANEU_K7_FEATS="$A/c10_feat_train_predves.json"

run_seeds(){
  local SEEDS="$1"
  log "c5 · 시드 $SEEDS · 구 학습표 2표 확률평균 · gC ON · 검출기 b1ff"
  clf(){ local sp=$1 sd=$2 bp exp n
    bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_k7.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "k1o_${TAG}_${sp}_s${sd}" \
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
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/k1o_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; wait
  return 0
}
report(){   # $1 = 시드목록(콤마) · $2 = 출력파일 · $3 = 제목
  { echo "# $3"; echo
    K0_SEEDS=$1 "$PY" "$D/k0_judge.py" b1on_pf ${TAG}_pf
    echo
    SEEDS=$1 TAG=$TAG "$PY" - <<'PYEOF'
import json, os, numpy as np
H = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/H1_patchfilter/scores"
M = ["PRECISION","RECALL","F1","MCC","DICE","VOLSIM","HD95"]
seeds = [int(x) for x in os.environ["SEEDS"].split(",")]; TAG = os.environ["TAG"]
g = lambda t, sp, s: json.load(open(f"{H}/{t}_{sp}_s{s}.json"))["new"]
print("## 참고 · 7지표 (판정 미사용 · 기준 b1on_pf 대비 평균 Δ · HD95 는 낮을수록 개선)\n")
print("| split | " + " | ".join(M) + " | 개선 |\n|" + "---|" * (len(M) + 2))
for sp in ("test", "val"):
    d = {m: np.mean([g(f"{TAG}_pf", sp, s)[m] - g("b1on_pf", sp, s)[m] for s in seeds]) for m in M}
    imp = sum((d[m] > 0) if m != "HD95" else (d[m] < 0) for m in M)
    print(f"| {sp} | " + " | ".join(f"{d[m]:+.4f}" for m in M) + f" | {imp}/7 |")
PYEOF
  } > "$2" 2>&1
}
run_seeds "5 6 7 8 9" || { log "★복제 실패"; exit 1; }
report "5,6,7,8,9" "$V/RESULTS_K1OLD_REP.md" "K1OLD 복제 — 구 학습표 위 K1 시드 5~9 (기술 통계 · 원판 p=0.0898 미채택은 뒤집히지 않는다)"
touch "$D/.done_k1orep"
log "K1OLD 복제 끝 → $(grep -o 'K0 판정 →.*' $V/RESULTS_K1OLD_REP.md | tail -1)"
