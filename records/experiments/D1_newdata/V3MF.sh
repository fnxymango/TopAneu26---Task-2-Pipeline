#!/usr/bin/env bash
# V3-MF — 병합표 + 파편 제거 (V3-M·V3-F 둘 다 채택 후보일 때만 · Q4.sh 가 조건 검사)
#
# 왜: 두 축(학습표 혈관 출처 · 파편 위생)은 서로 다른 행을 건드려 독립이다. 둘 다 단독으로 채택 후보면 합친 것을 잰다.
# 무엇이 바뀌나: --train-feat → analysis/v3mf_feat_merged_nofrag.json (v3m 542행에서 n_vox<20 10행 제거 · 532행).
# 기준선: b1Non_pf. ⚠ 채택되면 PROJECT_RULES.md 0-2(분류기 기준 학습표)를 바꾸는 사안이라 사용자 확인 후 반영.
#
# ── 판정규칙 (결과 보기 전에 고정 · 2026-09-14 · V1-D 와 같은 틀) ─────────────
#  1차(주): 병변 단위 TP — b1Non_pf 대비 test+val 합계가 늘어야 한다.
#  2차:     신 eval 7지표 — test·val 각각 평균 ΔMCC ≥ −0.005.
#  참고:    표준 채택선(7지표 ≥5 ∧ test·val)도 함께 적는다. 둘 다 만족 → 채택 후보. 1차 음수 → 기각.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/v3mf.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v3mf] $*" | tee -a "$D/STATUS.log"; }
TAG=b1mf
FEAT=$A/v3mf_feat_merged_nofrag.json
log "1단계 · c5 10런 · 병합+파편제거 학습표 532행 · gC ON · 검출기 b1ff"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "v3mf_${TAG}_${sp}_s${sd}" \
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
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/v3mf_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
done; done; wait
log "4단계 · 보고서"
{ echo "# V3-MF — 병합표 + 파편 제거"; echo
  echo "판정규칙은 V3MF.sh 머리말에 결과 보기 전에 고정. 1차=병변 TP · 2차=ΔMCC ≥ −0.005"; echo
  echo "## 1차 · 병변 단위 TP (b1Non_pf → ${TAG}_pf)"; echo; "$PY" "$D/tpcount.py" b1Non_pf "${TAG}_pf"
  echo; echo "## 2차 · 7지표 (${TAG}_pf − b1Non_pf)"; echo; "$PY" "$D/g2_seeds.py" "$H/scores" "${TAG}_pf" b1Non_pf "병합파편제거표 − 하이브리드표"
} > "$V/RESULTS_V3MF.md" 2>&1
log "V3MF 완료 → V1_vessel_axis/RESULTS_V3MF.md"; touch "$D/.done_v3mf"
