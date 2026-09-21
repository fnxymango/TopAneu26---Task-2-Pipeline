#!/usr/bin/env bash
# Q3 — V3-M(병합표) → V3-F(파편 제거) 순차 실행 (2026-09-14 사용자 승인)
D=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata
bash "$D/V3M.sh" > "$D/V3M.out" 2>&1
bash "$D/V3F.sh" > "$D/V3F.out" 2>&1
touch "$D/.done_q3"
