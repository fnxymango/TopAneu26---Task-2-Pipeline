#!/usr/bin/env bash
# B1E — 관문 E (e2e). 관문 V 통과(b1_gate_v.json.pick != null) 일 때만 돈다. 규칙은 PLAN.md 에 결과 보기 전 고정.
#  1 혈관 재예측(test 83 · train 291 · val 은 관문 V 산출 재사용) → 2 V5 후처리 + C4 그래프
#  3 학습표 재구축(overlap 만 새 예측혈관 · dist/bp/pos 는 GT 그대로) → 4 c5 시드 0~4(gC ON · 검출기 b1ff)
#  5 패치필터(새 혈관) → 6 채점 → 7 K0 판정(안전 ①② · α 0.05) → 8 통과 시 시드 5~9 복제
# 기준선 교체·제출 반영은 하지 않는다(사용자 확인 필요).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis; B=$E/B1_vessel_contact_ft
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
export nnUNet_raw=/tmp nnUNet_preprocessed=/tmp nnUNet_results=/tmp
exec 9>"$B/b1e.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b1e] $*" | tee -a "$D/STATUS.log"; }
TAG=b1ves

log "대기 · .done_b1v"
while [ ! -f "$B/.done_b1v" ]; do sleep 120; done
ARM=$("$PY" -c "import json;print(json.load(open('$B/b1_gate_v.json'))['pick'] or '')")
[ -n "$ARM" ] || { log "관문 V 미달 → e2e 취소 · B1 닫음"; touch "$B/.skip_b1e"; exit 0; }
log "관문 V 통과 · 선택 $ARM · e2e 시작"

# ── 1 혈관 재예측 (GPU 2장)
for sp in test train; do
  exp=$([ "$sp" = test ] && echo 83 || echo 291)
  n=$(ls -1 "$B/e2e/ves_$sp" 2>/dev/null | wc -l)
  if [ "$n" != "$exp" ]; then
    log "1단계 · 혈관 재예측 $sp ($n/$exp)"
    CUDA_VISIBLE_DEVICES=0 "$PY" -u "$B/b1e_pred.py" "$sp" 0 2 > "$B/e2e_${sp}_g0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 "$PY" -u "$B/b1e_pred.py" "$sp" 1 2 > "$B/e2e_${sp}_g1.log" 2>&1 &
    wait
    n=$(ls -1 "$B/e2e/ves_$sp" 2>/dev/null | wc -l)
    [ "$n" = "$exp" ] || { log "★1단계 실패 $sp ($n/$exp)"; exit 1; }
  fi
done
mkdir -p "$B/e2e/vespp_val" "$B/e2e/bp_val"
cp -n "$B/val/vespp_$ARM"/*.nii.gz "$B/e2e/vespp_val/" 2>/dev/null
cp -n "$B/val/graph_$ARM"/*.json "$B/e2e/bp_val/" 2>/dev/null

# ── 2 후처리 + 분기점 그래프
for sp in test train; do
  exp=$([ "$sp" = test ] && echo 83 || echo 291)
  if [ "$(ls -1 "$B/e2e/bp_$sp" 2>/dev/null | wc -l)" != "$exp" ]; then
    log "2단계 · V5 후처리 + C4 그래프 $sp"
    "$PY" "$S/postprocess_vessel.py" apply "$B/e2e/ves_$sp" "$B/e2e/vespp_$sp" > "$B/e2e_pp_$sp.log" 2>&1 \
      || { log "★2단계 후처리 실패 $sp"; exit 1; }
    "$PY" "$S/c4_branchpoint_graph.py" --vessel-dir "$B/e2e/vespp_$sp" --out "$B/e2e/bp_$sp" >> "$B/e2e_pp_$sp.log" 2>&1 \
      || { log "★2단계 그래프 실패 $sp"; exit 1; }
  fi
done

# ── 3 학습표 재구축 (기준 규약과 동일 · overlap 만 새 예측혈관)
FEAT=$A/b1_feat_hyb_ov_${ARM}.json
if [ ! -f "$FEAT" ]; then
  log "3단계 · 학습표 재구축 (overlap 만 교체)"
  "$PY" -u "$S/c5_location_v2.py" build --split train --vessel-dir "$B/e2e/vespp_train" \
    --bp-dir "$B/e2e/bp_train" --out "$A/b1_feat_train_predves_${ARM}.json" > "$B/e2e_build.log" 2>&1 \
    || { log "★3단계 피처 실패"; exit 1; }
  "$PY" - "$A" "$ARM" <<'PYEOF' >> "$B/e2e_build.log" 2>&1 || { log "★3단계 병합 실패"; exit 1; }
import json, sys
A, arm = sys.argv[1], sys.argv[2]
ref = json.load(open(f"{A}/c10_feat_train_NEW.json"))          # GT 혈관 피처(기준과 동일)
prd = json.load(open(f"{A}/b1_feat_train_predves_{arm}.json"))  # 새 예측 혈관 피처
key = lambda r: (r["case"], r.get("lesion_mask_idx"))
pm = {key(r): r for r in prd}
n = 0
for r in ref:
    q = pm.get(key(r))
    if q is not None and "overlap" in q:
        r["overlap"] = q["overlap"]; n += 1
json.dump(ref, open(f"{A}/b1_feat_hyb_ov_{arm}.json", "w"))
print(f"  overlap 교체 {n}/{len(ref)}")
PYEOF
fi

run_seeds(){   # $1 = "0 1 2 3 4" 또는 "5 6 7 8 9"
  local SEEDS="$1"
  log "c5 · 시드 $SEEDS · 새 혈관 · gC ON"
  clf(){ local sp=$1 sd=$2 exp n
    exp=$([ "$sp" = test ] && echo 83 || echo 41)
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$B/e2e/vespp_${sp}" --bp-dir "$B/e2e/bp_${sp}" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "b1e_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
  for sp in test val; do exp=$([ "$sp" = test ] && echo 83 || echo 41)
    for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l)
      [ "$c" = "$exp" ] || { log "★c5 누락 ${sp}_s${sd} ($c/$exp)"; return 1; }; done; done
  log "패치필터 · 시드 $SEEDS · 새 혈관"
  pf(){ local sp=$1 sd=$2 exp n; exp=$([ "$sp" = test ] && echo 83 || echo 41)
    n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$B/e2e/vespp_${sp}" \
      --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
      --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
  }
  n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
  log "채점 · 시드 $SEEDS"
  n=0; for sp in val test; do for sd in $SEEDS; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$B/e2e_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; wait
  return 0
}

# ── 4~7 원판 시드 0~4
run_seeds "0 1 2 3 4" || { log "★원판 시드 실패"; exit 1; }
log "K0 판정 · 시드 0~4"
{
  echo "# B1 관문 E — 새 혈관($ARM) e2e · 판정규칙은 PLAN.md 에 결과 보기 전 고정"
  echo
  K0_SEEDS=0,1,2,3,4 "$PY" "$D/k0_judge.py" b1Non_pf ${TAG}_pf
  echo
  "$PY" - "$H" "$TAG" <<'PYEOF'
import json, sys, numpy as np
H, TAG = sys.argv[1], sys.argv[2]
print("## 참고 · 신 eval MCC (시드 0~4 평균)\n")
for sp in ("test", "val"):
    b = [json.load(open(f"{H}/scores/b1Non_pf_{sp}_s{s}.json"))["MCC"] for s in range(5)]
    c = [json.load(open(f"{H}/scores/{TAG}_pf_{sp}_s{s}.json"))["MCC"] for s in range(5)]
    d = np.array(c) - np.array(b)
    print(f"- {sp}: 기준 {np.mean(b):.4f} · B1 {np.mean(c):.4f} · 시드별 Δ " +
          " ".join(f"{x:+.4f}" for x in d) + f" · 평균 {d.mean():+.4f}")
PYEOF
} > "$V/RESULTS_B1_E.md" 2>&1
touch "$B/.done_b1e"
log "관문 E(시드 0~4) 끝 → $(grep -o 'K0 판정 →.*' $V/RESULTS_B1_E.md | tail -1)"

# ── 8 통과 시 복제(시드 5~9)
if grep -q '\*\*K0 판정 → 채택 후보\*\*' "$V/RESULTS_B1_E.md"; then
  log "복제 검증 시작 · 시드 5~9"
  run_seeds "5 6 7 8 9" || { log "★복제 시드 실패"; exit 1; }
  {
    echo "# B1 관문 E 복제 — 시드 5~9"
    echo
    K0_SEEDS=5,6,7,8,9 "$PY" "$D/k0_judge.py" b1Non_pf ${TAG}_pf
  } > "$V/RESULTS_B1_E_REP.md" 2>&1
  touch "$B/.done_b1erep"
  log "복제 끝 → $(grep -o 'K0 판정 →.*' $V/RESULTS_B1_E_REP.md | tail -1) · 반영은 사용자 결정"
else
  log "원판 미채택 → 복제 생략"
fi
