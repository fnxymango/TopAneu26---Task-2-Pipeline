#!/usr/bin/env bash
# K2 — 좌우 대칭 앵커 보간 e2e (설계는 k2_sym.py · 재료는 k2_graphs.py 머리말)
#
# 왜: 한쪽 ICA 곁가지 분기점(OA·Pcom·AChA)이 빠지면 3.2/3.4/3.5 를 가를 입력이 사라진다. 반대쪽 분기점을 정중면 거울상으로 옮겨 채운다.
# 무엇이 바뀌나: 학습표 bp_mm (k2_feat_hyb_sym.json · 참조 그래프 보간 · 다른 필드 기준표와 동일 확인)
#               + 추론 그래프 --bp-dir vespp_test_sym · val_pred_sym (예측 혈관 그래프에 같은 보간). 분류기 코드·검출기·gC·패치필터 동일.
# 기준선: b1Non_pf (시드 0~4).
#
# ── 판정규칙 (결과 보기 전에 고정 · 2026-09-15 · K0 장치) ────────────────────────
#  ⚠ 사전 규칙에서 벗어난 진행임을 먼저 적는다: train 스크리닝 관문(Δmacro-recall ≥ +0.02 · 4/5)을 **미달**했다
#    (Δtop1 +0.013 · Δmacro +0.006 · 3/5 · 표적 ICA 3.2~3.6 top1 0.445→0.484). 사용자 승인(2026-09-15)으로 e2e 에 올린다.
#    스크리닝은 train 만 봤으므로 test·val 을 엿본 선택은 아니다. 탈락 후보를 올리는 몫만큼 유의수준을 낮춘다.
#  주판정: k0_judge.py (K0_ALPHA=0.01) — 130 단위 단측 부호검정 p < 0.01 ∧ 오른 단위 > 내린 단위.
#  안전:   ΔFP ≤ ΔTP (5시드 합).   둘 다 → 채택 후보 · 주판정 미충족 → 미채택 · 안전 위반 → 기각.
#  전제: K0 영가설 보정 통과. 보조 기록: tpcount · 7지표.
#  ⚠ 채택되면 추론 파이프라인에 그래프 후처리 단계 추가(이미지 변경) → APPLY_CHANGES.md · 런타임 확인.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/k2.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][k2] $*" | tee -a "$D/STATUS.log"; }
TAG=b1sym
FEAT=$A/k2_feat_hyb_sym.json
until [ -e "$D/.done_k0c" ]; do sleep 60; done   # 영가설 보정과 CPU 를 겹치지 않게
log "1단계 · c5 10런 · 대칭 앵커 보간(학습표 bp · 추론 그래프 *_sym) · gC ON · 검출기 b1ff"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test_sym || echo val_pred_sym )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "k2_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
miss=0
for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  for sd in 0 1 2 3 4; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
    [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
[ $miss = 0 ] || { log "★1단계 실패"; exit 1; }
log "2단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
log "3단계 · 채점 10런"
n=0; for sp in val test; do for sd in 0 1 2 3 4; do
  d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
  grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/k2_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; wait
log "4단계 · 보고서"
until [ -e "$D/k0_null.json" ]; do sleep 60; done
{ echo "# K2 — 좌우 대칭 앵커 보간 e2e"; echo
  echo "판정규칙은 K2.sh 머리말에 결과 보기 전에 고정 (K0 장치 · 유의수준 0.01)."; echo
  K0_ALPHA=0.01 "$PY" "$D/k0_judge.py" b1Non_pf "${TAG}_pf"
  echo; echo "## 보조 · 병변 단위 TP (tpcount)"; echo; "$PY" "$D/tpcount.py" b1Non_pf "${TAG}_pf"
  echo; echo "## 보조 · 7지표 (${TAG}_pf − b1Non_pf)"; echo; "$PY" "$D/g2_seeds.py" "$H/scores" "${TAG}_pf" b1Non_pf "대칭앵커 − 기준선"
} > "$V/RESULTS_K2.md" 2>&1
log "K2 완료 → V1_vessel_axis/RESULTS_K2.md · $(grep -o 'K0 판정 → [^*]*' $V/RESULTS_K2.md)"; touch "$D/.done_k2"
