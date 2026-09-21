#!/usr/bin/env bash
# V1-D — 병변 위치 기준을 **낭 중심 → 목(낭∩모혈관)** 으로 바꾼다
#
# 왜: 학습 병변의 n_vox 가 중앙 748, 최대 702,062 로 **940배** 차이다(2026-09-11 실측).
#   거대 낭은 여러 혈관을 동시에 물어 dist_mm 이 전부 0.25~0.45mm 로 붙고, 중심점이 목에서
#   10mm 넘게 벗어난다. 실제 학습표에서:
#     'L-1.1 VA trunk'(702k) 가 **반대쪽** R-PICA 에 가장 가깝다
#     '1.4 BA trunk'(92k) 는 어떤 혈관도 안 물고 L-VA/L-PICA 가 가장 가깝다
#     'L-1.3 VA-PICA junction'(42k) 은 BA 만 물고 PICA 분기에서 13.9mm 떨어져 있다
#   임상에서 동맥류 위치는 **목**으로 정의된다. 작은 낭은 목≈중심이라 값이 거의 안 변하고,
#   거대 낭에서만 달라진다 — 그래서 크기 문턱(분기)을 두지 않는다.
#
# 무엇이 바뀌나: dist_mm · bp_mm · pos(중심) 의 **기준 점집합**. overlap 과 n_vox 는 낭 전체 유지
#   (크기 정보는 남겨야 하고 겹침은 본래 낭의 성질). **차원은 안 늘어난다.**
#
# 구현: c5_location_v2.py / d9xx_lib.py 를 건드리지 않고 복사본 c5_neck.py / d9xx_lib_neck.py.
#   목 = 낭∩혈관의 **최대 연결성분**(스친 조각 배제). 비면 낭 전체로 되돌림.
#   거리는 기준집합과 겹치면 0 (원본의 `& (~mc)` 방식은 목이 작을 때 MAX_R 에 걸려 무너진다).
#
# 기준선: 제출본 검출기 + **개정판 학습표** + gC ON + 패치필터 = b1Non_pf (PROJECT_RULES.md 0-2장)
#
# ── 판정규칙 (결과 보기 전에 고정) ─────────────────────────────────────────
#  1차(주): **병변 단위 TP 수** — b1Non_pf 대비 test+val 합계가 늘어야 한다.
#           오늘 실측한 노이즈 바닥(학습표 6줄 차이 → CV ±1.31%p · e2e MCC ±0.02~0.03)
#           때문에 공식 지표만으로는 병변 몇 개짜리 변화를 판정할 수 없다.
#  2차(안전): 신 eval 7지표에서 **악화가 없어야** 한다 — test·val 각각 평균 ΔMCC ≥ −0.005.
#  둘 다 만족하면 채택. 1차가 음수면 기각.
#  시드 산포가 평균보다 크면 결론에 명시한다.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet TOPANEU_NECK=1
exec 9>"$D/v1d.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v1d] $*" | tee -a "$ST"; }
TAG=b1neck
FEAT=$A/e11_feat_neck.json

log "1단계 · 목 기준 학습표 생성 (train 291케이스)"
if [ ! -s "$FEAT" ]; then
  "$PY" -u "$S/c5_neck.py" build --split train \
    --vessel-dir "$P/vespp_train" --bp-dir "$BP/vespp_train" --out "$FEAT" \
    > "$D/v1d_build.log" 2>&1
fi
n=$("$PY" -c "import json;print(len(json.load(open('$FEAT'))))" 2>/dev/null || echo 0)
[ "$n" -ge 250 ] || { log "★학습표 생성 실패 ($n행) — v1d_build.log 확인"; exit 1; }
log "  학습표 $n행 (대조: 개정판 낭기준 271행)"

log "2단계 · c5 10런 · 목기준 학습표 · gC ON · 검출기 b1ff"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_NECK=1 TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_neck.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "v1d_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
miss=0
for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  for sd in 0 1 2 3 4; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
    [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
[ $miss = 0 ] || { log "★2단계 실패 — 중단"; exit 1; }

log "3단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 0 1 2 3 4; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait

log "4단계 · 채점 10런 (동결 eval 660da7a · 7지표)"
n=0; for sp in val test; do for sd in 0 1 2 3 4; do
  d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
  grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/v1d_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; wait

log "5단계 · 보고서"
{
  echo "# V1-D — 병변 위치 기준을 목(낭∩모혈관)으로"; echo
  echo '판정규칙은 V1D.sh 머리말에 결과 보기 전에 고정. 1차=병변 단위 TP, 2차=7지표 악화 없음.'; echo
  echo "## 1차 · 병변 단위 TP (b1Non_pf → ${TAG}_pf)"; echo
  "$PY" "$D/tpcount.py" b1Non_pf "${TAG}_pf" 2>&1
  echo; echo "## 2차 · 신 eval 7지표 (${TAG}_pf − b1Non_pf)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" "${TAG}_pf" b1Non_pf "목기준 − 낭기준" 2>&1
  echo; echo "## 참고 · 제출본(b1on_pf) 대비"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" "${TAG}_pf" b1on_pf "목기준 − 제출본" 2>&1
} > "$E/V1_vessel_axis/RESULTS_V1D.md" 2>&1
log "V1D 완료 → $E/V1_vessel_axis/RESULTS_V1D.md"; touch "$D/.done_v1d"
