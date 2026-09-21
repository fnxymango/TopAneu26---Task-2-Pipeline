#!/usr/bin/env bash
# GROW1 / G1 — 밝기순 출력 부피 보정 (2026-09-17 · 사용자 승인)
#
#  왜: GROW(거리순, OUT_GROW 1.32)는 원판을 통과했지만 **DICE 가 VOLSIM 의 1/8** 이었다
#      (test DICE +0.0015 vs VOLSIM +0.0128 · val +0.0027 vs +0.0233).
#      부피만 맞추고 겹침은 못 늘렸다는 뜻이라 D1 진단으로 원인을 특정했다(시드 3):
#
#      | | val | test |
#      |---|---|---|
#      | 기준 예측 정밀도(예측 복셀 중 GT 안) | 87.8% | 81.9% |
#      | **GROW 추가 복셀 정밀도** | **46.4%** | **16.1%** |
#      | 추가 복셀 강도분위 중앙 | 0.306 | 0.529 |
#      | 놓친 GT 강도분위 중앙 | **0.444** | **0.814** |
#      | 놓친 GT 의 예측까지 거리 중앙 | 0.95mm | 0.85mm |
#
#      **채워야 할 곳은 밝은데 거리순으로 채우니 어두운 배경을 집어왔다.** test 는 추가분의 84% 가 GT 밖이다.
#
#  무엇: 껍질 복셀(≤4mm)을 **거리순이 아니라 밝기순**으로 채운다. 반경도 need 도 그대로,
#      **순서만** 바꾼다(자유도 0 · 임계나 반경을 스윕하지 않는다).
#      구현 `TOPANEU_OUT_GROW_BY=int` + `--image-dir`. 기본값 dist 는 기존 동작과 동일하다.
#      원본 영상은 추론 시점에 있다(우리 `in_{test,val}` · 컨테이너는 입력으로 받는다).
#
#  G2(혈관 제외)는 D1 결과로 접었다 — 추가 복셀 중 예측혈관 안은 11.6~11.9% 뿐이라 과녁이 작고,
#      **놓친 GT 의 54.9%(test)가 오히려 예측 혈관 안**이라 혈관을 빼면 살릴 곳을 막는다.
#
#  기준선: b1frac035_pf (제출본 구성 + 채택 확정 FRAC 0.35)
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-17 12:00 KST) ─────────────────────
#  ① DICE·VOLSIM 이 **test·val 양쪽** 시드평균 개선
#  ② PRECISION·RECALL·F1·MCC 전 시드 불변(절대편차 < 1e-9)
#  ③ HD95 악화가 어느 쪽에서도 시드평균 +3 이하
#  ④ **DICE Δ 가 현행 GROW(test +0.0015 · val +0.0027)보다 클 것** — 이 변형의 존재 이유다
#  넷 다 충족해야 통과. 미달이면 G1 을 닫고 GROW(거리순) 쪽으로 복제 여부를 다시 묻는다.
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/grow1.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][grow1] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov.json
TAG=b1fgi
BASE=b1frac035_pf
SEEDS="0 1 2 3 4"

log "c5 · 시드 $SEEDS · FRAC 0.35 + OUT_GROW 1.32 **밝기순** · 기준 $BASE"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_TOPK_FRAC=0.35 TOPANEU_OUT_GROW=1.32 TOPANEU_OUT_GROW_BY=int \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --image-dir "$P/in_${sp}" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "grow1_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
# ★ 밝기순이 실제로 걸렸는지 — 낙하 경고가 찍혔으면 거리순으로 돈 것이라 GROW 와 같아진다
if grep -q "밝기순 실패" "$H/logs/${TAG}_test_s0.log" 2>/dev/null; then
  log "★밝기순 낙하 발생 — 영상 경로/shape 불일치. 중단."; exit 1
fi
for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
    [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; exit 1; }; done; done

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
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/grow1_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; wait

SEEDS=0,1,2,3,4 TAGS=${TAG}_pf LABELS=밝기순 BASE=$BASE "$PY" "$D/frac_report.py" > "$V/RESULTS_GROW1.md" 2>&1
sed -i "1s|.*|# GROW1/G1 — 밝기순 출력 부피 보정 (OUT_GROW 1.32 · FRAC 0.35 위 · 기준 $BASE) · 규칙은 GROW1.sh 머리말 고정|" "$V/RESULTS_GROW1.md"
{ echo; echo "## 현행 GROW(거리순)와의 비교 — 규칙 ④"; echo
  echo "| split | 거리순 DICE Δ | 밝기순 DICE Δ | 거리순 VOLSIM Δ | 밝기순 VOLSIM Δ |"
  echo "|---|---|---|---|---|"
  BASE=$BASE TAG=b1fg_pf SEEDS=0,1,2,3,4 "$PY" - <<'PYEOF'
import json, os
import numpy as np
H="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/H1_patchfilter/scores"
g=lambda t,sp,s: json.load(open(f"{H}/{t}_{sp}_s{s}.json"))["new"]
for sp in ("test","val"):
    row=[]
    for m in ("DICE","VOLSIM"):
        for t in ("b1fg_pf","b1fgi_pf"):
            row.append(np.mean([g(t,sp,s)[m]-g("b1frac035_pf",sp,s)[m] for s in range(5)]))
    print(f"| {sp} | {row[0]:+.4f} | {row[1]:+.4f} | {row[2]:+.4f} | {row[3]:+.4f} |")
PYEOF
} >> "$V/RESULTS_GROW1.md" 2>&1
touch "$D/.done_grow1"
log "GROW1 끝 → $(grep -o '\*\*판정 →.*' $V/RESULTS_GROW1.md | tail -1)"
