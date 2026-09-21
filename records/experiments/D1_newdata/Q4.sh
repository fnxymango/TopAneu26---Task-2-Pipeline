#!/usr/bin/env bash
# Q4 — Q3(V3-M→V3-F) 뒤에 이어 붙는 조건부 후속 (2026-09-14 사용자 "다 걸어놔")
#  V3-MF: V3-M·V3-F 가 **둘 다** 사전 규칙 통과(1차 병변 TP Δ>0 ∧ 2차 test·val 평균ΔMCC ≥ −0.005)일 때만 실행.
#         ⚠ g2_seeds 의 "충족/→ 채택" 표시는 표준 채택선(개선≥NEED ∧ ΔMCC≥0)이라 V3 사전 규칙과 다르다 → 쓰지 않는다.
#         하나만 통과하면 그 하나가 후보이므로 합칠 필요가 없다 → 건너뛰고 기록.
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
V=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/V1_vessel_axis
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][q4] $*" | tee -a "$D/STATUS.log"; }
while [ ! -e "$D/.done_q3" ]; do sleep 60; done
pass(){ local f=$1 d
  d=$(grep -oE '\*\*Δ [+-]?[0-9]+\*\*' "$f" | head -1 | grep -oE '[+-]?[0-9]+')
  [ -n "$d" ] && [ "$d" -gt 0 ] || return 1
  grep -oE '평균ΔMCC [+-][0-9.]+' "$f" | awk '{v=$2+0; n++; if (v < -0.005) bad=1} END{exit !(n==2 && !bad)}'; }
m=0; f=0; pass "$V/RESULTS_V3M.md" && m=1; pass "$V/RESULTS_V3F.md" && f=1
log "조건 검사 · V3-M 통과=$m · V3-F 통과=$f"
if [ $m = 1 ] && [ $f = 1 ]; then
  bash "$D/V3MF.sh" > "$D/V3MF.out" 2>&1
else
  log "V3-MF 건너뜀 (둘 다 통과해야 실행)"
fi
touch "$D/.done_q4"
