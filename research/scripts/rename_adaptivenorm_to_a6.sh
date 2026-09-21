#!/usr/bin/env bash
# adaptive-norm 실험 2종을 학습 완료 후 A6으로 rename (사용자 지시 2026-08-13:
# "다른점 있으면 이름 무조건 다르게" — plain-z=A5 / adaptive-norm=A6).
# 학습중 rename하면 nnU-Net이 시작시 확정한 절대경로가 깨지므로 완료까지 대기.
set -uo pipefail
TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
E="$TOPANEU_ROOT/experiments"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
LOG="$TOPANEU_ROOT/experiments/rename_a6.log"
export TOPANEU_ROOT
log() { echo "[rename-a6 $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

declare -A MAP=(
  ["D720_resencl_adaptivenorm_plain_417_f0"]="A6_resencl_adaptivenorm_plain_417_f0"
  ["D720_resencl_adaptivenorm_topk_417_f0"]="A6-2_resencl_adaptivenorm_topk_417_f0"
)

log "두 학습 완료 대기"
for OLD in "${!MAP[@]}"; do
  while ! grep -q "finished with status" "$E/$OLD/train.log" 2>/dev/null; do sleep 60; done
  log "  $OLD 완료 감지"
done
# nnUNetv2_train 프로세스가 완전히 빠질 때까지 여유
while pgrep -f "nnUNetv2_train 720 3d_fullres 0 -p nnUNetResEncUNetLPlansAdaptive" >/dev/null; do sleep 20; done
sleep 10

for OLD in "${!MAP[@]}"; do
  NEW="${MAP[$OLD]}"
  if [ -d "$E/$OLD" ] && [ ! -d "$E/$NEW" ]; then
    mv "$E/$OLD" "$E/$NEW"
    "$ENVBIN/python" - "$E/$NEW" "$NEW" <<'PY'
import json, sys
p, new = sys.argv[1], sys.argv[2]
c = json.load(open(f"{p}/config.json")); c["exp_name"] = new
json.dump(c, open(f"{p}/config.json", "w"), indent=2, ensure_ascii=False)
PY
    "$ENVBIN/python" "$TOPANEU_ROOT/code/sblee/nnunet/scripts/make_summary.py" "$E/$NEW" >> "$LOG" 2>&1
    log "  $OLD -> $NEW (config/summary 갱신)"
  else
    log "  스킵: $OLD (없거나 대상 이미 존재)"
  fi
done
"$ENVBIN/python" "$TOPANEU_ROOT/code/sblee/nnunet/scripts/make_leaderboard.py" >> "$LOG" 2>&1
log "완료"
