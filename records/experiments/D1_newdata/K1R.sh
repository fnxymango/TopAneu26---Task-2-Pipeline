#!/usr/bin/env bash
# K1R — K1(두 학습표 모델 확률 평균) 복제 검증 · 시드 5~9 (2026-09-15 사용자 "다 걸어")
#
# 왜: K1 은 K0 장치에서 채택 후보(시드 0~4 · 오른 7 · 내린 0 · p 0.0078 · ΔTP +14 · ΔFP −2)였지만
#     보조 7지표 test MCC 가 5시드 모두 음수(−0.014)였다 — 원인은 깨끗하던 클래스(L-5.1 0→5)에 새 오답이 생긴 것.
#     같은 설정을 **새 시드**로 다시 돌려 우연인지 가린다. 기준선 b1Non_pf 시드 5~9 는 K0NULL 이 이미 만들었다(채점은 여기서).
# 무엇이 바뀌나: K1.sh 와 완전히 같고 시드만 5~9.
#
# ── 판정규칙 (결과 보기 전에 고정 · 2026-09-15) ────────────────────────────────
#  복제 통과 = 시드 5~9 에서도 K0 장치 통과: 부호검정 p < 0.05 ∧ 오른 > 내린 · 안전 ΔFP ≤ ΔTP.
#  K1 최종 권고 = 원판(0~4) ∧ 복제(5~9) 모두 통과 → "채택 권고"(실제 반영은 사용자 결정 · 모델 2개 → APPLY_CHANGES.md)
#                 복제 실패 → "보류".
#  참고 기록(판정 미사용): 깨끗하던 클래스 새 오답 수 · 신 eval MCC 평균 Δ (test·val).
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/k1r.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][k1r] $*" | tee -a "$D/STATUS.log"; }
TAG=b1k1
FEAT=$A/e11_feat_hyb_ov_NEW.json
FEAT2=$A/c10_feat_train_predves_NEW.json
log "1단계 · c5 10런 · c5_k1 복제(시드 5~9) · gC ON · 검출기 b1ff"
clf(){ local sp=$1 sd=$2 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_K1_FEAT2="$FEAT2" TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/c5_k1.py" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "k1r_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 5 6 7 8 9; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
miss=0
for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  for sd in 5 6 7 8 9; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
    [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
[ $miss = 0 ] || { log "★1단계 실패"; exit 1; }
log "2단계 · 패치필터 10런"
pf(){ local sp=$1 sd=$2 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0; for sp in test val; do for sd in 5 6 7 8 9; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
log "3단계 · 채점 10런"
n=0; for sp in val test; do for sd in 5 6 7 8 9; do
  for t in "$TAG" b1Non; do d="$H/pred/${t}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
  grep -q '"F1"' "$H/scores/${t}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/k1r_score.log" 2>&1 &
  n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done
done; done; wait
log "4단계 · 보고서"
{ echo "# K1R — K1 복제 검증 (시드 5~9)"; echo
  echo "판정규칙은 K1R.sh 머리말에 결과 보기 전에 고정."; echo
  K0_SEEDS=5,6,7,8,9 "$PY" "$D/k0_judge.py" b1Non_pf "${TAG}_pf"
  echo; echo "## 참고 · 신 eval MCC (시드 5~9 평균)"; echo
  "$PY" - <<PYEOF
import json
H="$H"
for sp in ("test","val"):
    b=[json.load(open(f"{H}/scores/b1Non_pf_{sp}_s{s}.json"))["new"]["MCC"] for s in range(5,10)]
    c=[json.load(open(f"{H}/scores/b1k1_pf_{sp}_s{s}.json"))["new"]["MCC"] for s in range(5,10)]
    d=[y-x for x,y in zip(b,c)]
    print(f"- {sp}: 기준 {sum(b)/5:.4f} · K1 {sum(c)/5:.4f} · 시드별 Δ {' '.join(f'{v:+.4f}' for v in d)} · 평균 {sum(d)/5:+.4f}")
PYEOF
  o=$(grep -o 'K0 판정 → [^*]*' "$V/RESULTS_K1.md"); r=""
} > "$V/RESULTS_K1R.md" 2>&1
orig=$(grep -c 'K0 판정 → 채택 후보' "$V/RESULTS_K1.md"); rep=$(grep -c 'K0 판정 → 채택 후보' "$V/RESULTS_K1R.md")
if [ "$orig" = 1 ] && [ "$rep" = 1 ]; then fin="채택 권고(원판·복제 모두 통과 · 반영은 사용자 결정)"; else fin="보류(복제 실패)"; fi
echo -e "\n**K1 최종 권고 → $fin**" >> "$V/RESULTS_K1R.md"
log "K1R 완료 → V1_vessel_axis/RESULTS_K1R.md · $fin"; touch "$D/.done_k1r"
