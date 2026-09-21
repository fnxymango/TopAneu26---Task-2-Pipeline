#!/usr/bin/env bash
# V1-E — 목(낭∩모혈관) 기준 위치를 **어떻게** 쓸지 두 팔로 가른다
#
# ── 왜 다시 도나 (V1-D 의 교란) ────────────────────────────────────────────
# 기준 학습표 e11_feat_hyb_ov_NEW.json 은 **하이브리드**다 (C1C2.sh:44-64):
#     dist_mm · bp_mm · pos  ← GT 혈관마스크(dataset/TopAneu/vessel_masks + bp/all_ref)
#     overlap                ← 예측 혈관(vespp_train)
# 그런데 V1D.sh 는 학습표를 **예측 혈관만으로** 만들었다. 즉 V1-D 는
#   (a) 낭 중심 → 목   (b) GT 혈관 → 예측 혈관
# 두 가지를 동시에 바꿨다. test 7지표 −0.0208 을 (a) 탓으로 돌릴 근거가 없다.
# 여기서는 두 팔 다 **기준표와 같은 하이브리드 방식**으로 만들어 (b) 를 제거한다.
#
# ── 두 팔 ──────────────────────────────────────────────────────────────────
#  D2팔 (b1neckD2) = V1-D 재실험 · **갈아치우기**
#      c5_neck.py — dist_mm·bp_mm·pos 의 기준점집합을 목으로 교체. 112차원 유지.
#  E팔 (b1neck2) = 신규 · **나란히 주기**
#      c5_neck2.py — 기존 112차원은 그대로 두고 목 기준 혈관근접 36차원을 덧붙임(148).
#      RF 가 병변/클래스마다 어느 기준을 볼지 스스로 고른다.
#      근거: V1-D 오예측 집계상 늘어난 FP 가 분산이 아니라 **이득 본 클래스에 집중**
#      (1.5 VA-BA: TP +4 · 오예측 +11 / R-3.5 AChA: TP +8 · 오예측 +7). 전역 교체가
#      그 클래스들의 결정영역을 통째로 넓힌 것이므로, 선택권을 RF 에 넘긴다.
#
# 기준선: b1Non_pf (제출본 검출기 + 개정판 학습표 + gC ON + 패치필터) — PROJECT_RULES.md 0-2장
#
# ── 판정규칙 (결과 보기 전에 고정) ─────────────────────────────────────────
#  1차(주):   병변 단위 TP — b1Non_pf 대비 test+val 합계가 늘어야 한다 (tpcount.py).
#             노이즈 바닥(학습표 6줄 → e2e MCC ±0.02~0.03) 때문에 공식 지표만으로는
#             병변 몇 개짜리 변화를 판정할 수 없다.
#  2차(안전): 신 eval 7지표에서 악화가 없어야 한다 — test·val 각각 평균 ΔMCC ≥ −0.005.
#  정식채택:  7지표 중 ≥5 개선 ∧ test·val 동시 (PROJECT_RULES.md 2장 표준).
#  1·2차 통과 = 축 유지하고 다음 단계로. 1차가 음수면 그 팔은 기각.
#  두 팔 다 통과하면 병변 TP 가 큰 쪽을 남긴다. 시드 산포가 평균보다 크면 명시.
# ───────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter
ST=$D/STATUS.log; PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/v1e.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][v1e] $*" | tee -a "$ST"; }

# 팔 정의: 태그 · 스크립트 · 학습표 · 추가 env
ARMS="b1neckD2:c5_neck.py:$A/e11_feat_neckD2.json b1neck2:c5_neck2.py:$A/e11_feat_neck2.json"

log "1단계 · 학습표 2종 생성 (기준표와 같은 하이브리드: 위치=GT혈관 · overlap=예측혈관)"
for spec in $ARMS; do
  IFS=: read -r TAG SCR FEAT <<<"$spec"
  [ -s "$FEAT" ] && { log "  $TAG 학습표 있음 — 건너뜀"; continue; }
  TOPANEU_NECK=1 TOPANEU_NECKBLOCK=1 "$PY" -u "$S/$SCR" build --split train \
    --vessel-dir "$R/dataset/TopAneu/vessel_masks" --bp-dir "$BP/all_ref" \
    --out "${FEAT}.ref" > "$D/v1e_build_$TAG.log" 2>&1 \
    || { log "★$TAG 위치피처 실패"; exit 1; }
  # overlap 만 예측혈관 것으로 교체 — C1C2.sh 와 같은 규칙. 예측혈관 표는 이미 있다.
  "$PY" - "$A" "${FEAT}.ref" "$FEAT" <<'PYEOF' >> "$D/v1e_build_$TAG.log" 2>&1 \
    || { log "★$TAG 병합 실패"; exit 1; }
import json, sys
A, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
ref = json.load(open(src))
prd = json.load(open(f"{A}/c10_feat_train_predves_NEW.json"))
key = lambda r: (r["case"], r.get("lesion_mask_idx"))
pm = {key(r): r for r in prd}
n = 0
for r in ref:
    r.pop("_neck_idx", None)
    q = pm.get(key(r))
    if q is not None and "overlap" in q:
        r["overlap"] = q["overlap"]; n += 1
json.dump(ref, open(dst, "w"), ensure_ascii=False)
print(f"  overlap 교체 {n}/{len(ref)}")
PYEOF
  rm -f "${FEAT}.ref"
  n=$("$PY" -c "import json;print(len(json.load(open('$FEAT'))))" 2>/dev/null || echo 0)
  [ "$n" -ge 250 ] || { log "★$TAG 학습표 $n행 — 중단"; exit 1; }
  log "  $TAG 학습표 $n행 (대조: 기준표 271행)"
done

log "2단계 · c5 20런 (팔2 × split2 × 시드5) · gC ON · 검출기 b1ff"
clf(){ local TAG=$1 SCR=$2 FEAT=$3 sp=$4 sd=$5 bp exp n
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  TOPANEU_NECK=1 TOPANEU_NECKBLOCK=1 \
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=$sd OMP_NUM_THREADS=1 \
  "$PY" -u "$S/$SCR" eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_b1ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "v1e_${TAG}_${sp}_s${sd}" \
    > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0
for spec in $ARMS; do IFS=: read -r TAG SCR FEAT <<<"$spec"
  for sp in test val; do for sd in 0 1 2 3 4; do
    clf "$TAG" "$SCR" "$FEAT" "$sp" "$sd" & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done
done; wait
miss=0
for spec in $ARMS; do IFS=: read -r TAG SCR FEAT <<<"$spec"
  for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    for sd in 0 1 2 3 4; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
      [ "$c" = "$exp" ] || { log "★c5 누락 ${TAG}_${sp}_s${sd} ($c/$exp)"; miss=1; }; done; done
done
[ $miss = 0 ] || { log "★2단계 실패 — 중단"; exit 1; }

log "3단계 · 패치필터 20런"
pf(){ local TAG=$1 sp=$2 sd=$3 n exp; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
  n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
  "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
    --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
    --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
}
n=0
for spec in $ARMS; do IFS=: read -r TAG SCR FEAT <<<"$spec"
  for sp in test val; do for sd in 0 1 2 3 4; do
    pf "$TAG" "$sp" "$sd" & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait
  done; done
done; wait

log "4단계 · 채점 20런 (동결 eval 660da7a · 7지표)"
n=0
for spec in $ARMS; do IFS=: read -r TAG SCR FEAT <<<"$spec"
  for sp in val test; do for sd in 0 1 2 3 4; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/v1e_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done
done; wait

log "5단계 · 보고서"
{
  echo "# V1-E — 목 기준을 '갈아치우기' vs '나란히 주기'"; echo
  echo '판정규칙은 V1E.sh 머리말에 결과 보기 전에 고정. 1차=병변 단위 TP, 2차=7지표 악화 없음.'
  echo; echo '두 팔 다 학습표를 **기준표와 같은 하이브리드**(위치=GT혈관 · overlap=예측혈관)로'
  echo '만들었다. V1-D 는 예측혈관만 썼기 때문에 목 기준의 효과와 혈관 출처의 효과가 섞여 있었다.'
  echo
  for arm in b1neckD2 b1neck2; do
    case $arm in
      b1neckD2) t="D2팔 · 갈아치우기 (V1-D 재실험 · 112차원)";;
      b1neck2) t="E팔 · 나란히 주기 (목 기준 36차원 추가 · 148차원)";;
    esac
    echo "---"; echo; echo "# $t"; echo
    echo "## 1차 · 병변 단위 TP (b1Non_pf → ${arm}_pf)"; echo
    "$PY" "$D/tpcount.py" b1Non_pf "${arm}_pf" 2>&1
    echo; echo "## 2차 · 신 eval 7지표 (${arm}_pf − b1Non_pf)"; echo
    "$PY" "$D/g2_seeds.py" "$H/scores" "${arm}_pf" b1Non_pf "$t" 2>&1
    echo
  done
  echo "---"; echo; echo "# 참고 · 두 팔 직접 비교 (E − R)"; echo
  "$PY" "$D/g2_seeds.py" "$H/scores" b1neck2_pf b1neckD2_pf "나란히 − 갈아치우기" 2>&1
} > "$E/V1_vessel_axis/RESULTS_V1E.md" 2>&1
log "V1E 완료 → $E/V1_vessel_axis/RESULTS_V1E.md"; touch "$D/.done_v1e"
