#!/usr/bin/env bash
# C3 — 검출기 교체 대조 평가 (사용자 지시 2026-08-14 "평가 한번 더 걸어줘").
#
# GPU1 체인(run_gpu1_chain_c1.sh) STEP5는 A5-2(plain z) 검출 결과로만 8칸을 채운다:
#     A5-2 × {vespp, ves} × {C1, C2} × {val, test}
# 이 스크립트는 같은 8칸을 A6-2(adaptive norm) 검출 결과로 채워 2x2x2x2 = 16칸을 완성한다.
# 레버는 동맥류 검출기 하나뿐 — 혈관(V4-2)·후처리(V5)·C방법(C1 lookup / C2 kNN k=5 p=1)·split 동일.
#
# 검출기 동작점 (val 42, 병변 단위, checkpoint_best + TTA off — 파이프라인이 실제 쓰는 설정):
#     A5-2  민감도 0.721 (31/43)  FP성분/case 1.31   총 55
#     A6-2  민감도 0.814 (35/43)  FP성분/case 3.62   총 152
#   -> 진양성 +4개를 위양성 +97개로 산 셈. C단계 MCC가 어느 쪽을 택하는지가 이 평가의 질문.
#
# 의존: vespp_test (GPU1 체인 STEP4가 생성) + aneu_test_a62 (run_a62_det_compare.sh가 생성).
#       둘 다 기다렸다 시작한다. GPU 미사용(CPU 평가).
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ANALYSIS="$TOPANEU_ROOT/code/sblee/nnunet/analysis"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
E="$TOPANEU_ROOT/experiments"
WORK="$E/_c1_realpred"
LOG="$E/c3_eval_a62.log"
export TOPANEU_ROOT
log() { echo "[c3-eval $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------------------------------------------------------------- 선행 조건 대기
log "선행 대기: GPU1 체인 종료 + aneu_test_a62 83개"
for i in $(seq 1 180); do
  OK=1
  grep -q "GPU1 체인 완료" "$E/gpu1_chain_c1.log" 2>/dev/null || OK=0
  [ "$(ls "$WORK/aneu_test_a62"/*.nii.gz 2>/dev/null | wc -l)" -eq 83 ] || OK=0
  [ "$OK" -eq 1 ] && { log "선행 조건 충족 (${i}분 대기)"; break; }
  [ $((i % 10)) -eq 0 ] && log "  대기중 ${i}/180분 — gpu1완료=$(grep -c 'GPU1 체인 완료' "$E/gpu1_chain_c1.log" 2>/dev/null) aneu_test_a62=$(ls "$WORK/aneu_test_a62"/*.nii.gz 2>/dev/null | wc -l)/83"
  sleep 60
done
if ! grep -q "GPU1 체인 완료" "$E/gpu1_chain_c1.log" 2>/dev/null; then
  log "🚨 ABORT: 180분 내 GPU1 체인이 끝나지 않음"; exit 1
fi

# ---------------------------------------------------------------- 8칸 평가
cd "$SCRIPTS"
for SP in val test; do
  for PAIR in "vespp_${SP}:pp" "ves_${SP}:raw"; do
    D="${PAIR%%:*}"
    T="${PAIR##*:}"
    if [ ! -d "$WORK/$D" ]; then log "  건너뜀: $D 없음"; continue; fi

    log "C1(signature lookup)  split=$SP  vessel=$T  det=A6-2"
    "$ENVBIN/python" d900_infer_eval.py --split "$SP" \
      --aneurysm-pred-dir "$WORK/aneu_${SP}_a62" --vessel-pred-dir "$WORK/$D" \
      --lookup "$ANALYSIS/d9xx_lookup_V2_nearest_r2.json" \
      --tag "realpred_a62_${T}" >> "$LOG" 2>&1
    log "  status=$?"

    log "C2(weighted-kNN)      split=$SP  vessel=$T  det=A6-2"
    "$ENVBIN/python" d910_infer_eval.py --split "$SP" \
      --aneurysm-pred-dir "$WORK/aneu_${SP}_a62" --vessel-pred-dir "$WORK/$D" \
      --index "$ANALYSIS/d910_index_k5_p1.npz" \
      --tag "realpred_a62_${T}" >> "$LOG" 2>&1
    log "  status=$?"
  done
done

# ---------------------------------------------------------------- 16칸 요약표
log "16칸 요약표 생성"
"$ENVBIN/python" "$SCRIPTS/summarize_c3.py" >> "$LOG" 2>&1
log "완료 — 요약: analysis/c3_comparison.md"
