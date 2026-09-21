#!/usr/bin/env bash
# Q-GPU — 검출기 축. 2026-08-19.
#
# 왜 여기가 최우선인가: E8(T13 §6) 시뮬레이션상 검출이 cov.MCC 약 0.10 을 먹고 있다.
#   병변민감도 0.733 -> 0.86 이면 +0.0504, FP 가 2배로 늘어도 +0.0321 이 남는다.
#   분류기 축에서 오늘 하루 얻은 최대치가 +0.031 이므로 자릿수가 다르다.
#
# 게이트: A7(Tversky a0.15/b0.85, FN 벌점 강화) fold0 은 이미 학습돼 있고 val 추론물도 있다.
#   **먼저 A6-2 fold0 과 fold0 대 fold0 으로 공정 비교**한다 (현행은 5폴드 평균이라 직접 비교는 불공정).
#   c7 필터 뒤 민감도가 오르고 FP 가 2배 이내면 -> folds 1~4 를 GPU 2장으로 학습해 5폴드로 간다.
#   아니면 학습에 며칠 쓰지 않고 여기서 접는다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN2="$HOME/miniconda3/envs/sbaneu2/bin"       # 벤더 트레이너 포함 env
AEXP="A7_tversky_a15b85_417_f0"
log(){ echo "[qgpu $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== G1: A7 fold0 vs A6-2 fold0 (val, c7 필터 뒤 병변 민감도/FP) ==="
for D in aneu_val_a7 aneu_val_a62f; do
  n=$(ls "$P/$D"/*.nii.gz 2>/dev/null | wc -l)
  log "  $D : $n 케이스"
  [ "$n" -lt 40 ] && { log "  ★ $D 추론물 부족 — 중단"; exit 1; }
done
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_val_a7"   --vessel-dir "$P/vespp_val" \
    --split val --tag qgpu_a7   > "$E/q_g1_a7.txt"  2>&1 || log "  A7 후처리 실패"
$PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_val_a62f" --vessel-dir "$P/vespp_val" \
    --split val --tag qgpu_a62f > "$E/q_g1_a62f.txt" 2>&1 || log "  A6-2f0 후처리 실패"
log "  --- A7 fold0 ---";    tail -14 "$E/q_g1_a7.txt"
log "  --- A6-2 fold0 ---";  tail -14 "$E/q_g1_a62f.txt"

GATE=$($PY - "$E/q_g1_a7.txt" "$E/q_g1_a62f.txt" <<'PYEOF'
import re,sys
def best(p):
    """c7 출력에서 (민감도, FP) 중 민감도 최대 행을 뽑는다."""
    rows=[]
    for ln in open(p, errors="ignore"):
        nums=re.findall(r"\d+\.\d+|\d+", ln)
        if len(nums)<3: continue
        f=[float(x) for x in nums]
        cand=[x for x in f if 0.0<=x<=1.0]
        if not cand: continue
        rows.append((max(cand), f))
    if not rows: return None
    return max(rows, key=lambda r:r[0])
a,b=best(sys.argv[1]),best(sys.argv[2])
if not a or not b: print("UNKNOWN"); raise SystemExit
print(f"A7={a[0]:.4f} A62f0={b[0]:.4f}")
PYEOF
)
log "  게이트 원자료: $GATE"
log "  ※ 자동 파싱은 참고용이다. 위 두 표를 사람이 직접 보고 folds 1~4 학습 여부를 정한다."
log "  기준: 민감도가 오르고 FP 가 2배 이내면 학습 진행 (T13 §6)."

if [ "${QGPU_TRAIN_FOLDS:-0}" != "1" ]; then
  log "=== folds 1~4 학습은 QGPU_TRAIN_FOLDS=1 일 때만 돈다. 여기서 정지. ==="
  exit 0
fi

log "=== G2: A7 folds 1~4 학습 (GPU 2장, 2개씩) ==="
ARES="$E/$AEXP/results/Dataset720_TopAneuBinary417/nnUNetTrainerTverskyTopkCE_a15b85__nnUNetResEncUNetLPlansAdaptive__3d_fullres"
for PAIR in "1 2" "3 4"; do
  set -- $PAIR; F1=$1; F2=$2
  for FG in "$F1 0" "$F2 1"; do
    set -- $FG; FOLD=$1; GPUID=$2
    [ -f "$ARES/fold_${FOLD}/checkpoint_final.pth" ] && { log "  fold$FOLD 이미 완료"; continue; }
    log "  fold$FOLD -> GPU$GPUID 학습 시작"
    GPU=$GPUID NPROC=2 ENVBIN="$ENVBIN2" TOPANEU_ROOT="$R" nohup bash "$S/run_experiment.sh" \
      720 3d_fullres "$FOLD" "$AEXP" \
      -p nnUNetResEncUNetLPlansAdaptive -tr nnUNetTrainerTverskyTopkCE_a15b85 \
      > "$E/q_g2_train_f${FOLD}.log" 2>&1 &
    disown
  done
  log "  fold $F1,$F2 완료 대기 (30분마다 확인)"
  for i in $(seq 1 96); do
    ok=0
    for FOLD in $F1 $F2; do
      [ -f "$ARES/fold_${FOLD}/checkpoint_final.pth" ] && ok=$((ok+1))
    done
    [ "$ok" -ge 2 ] && { log "  fold $F1,$F2 완료"; break; }
    log "  (대기 $((i*30))분) 완료 $ok/2"
    sleep 1800
  done
done
log "=== G2 종료. 5폴드 확률평균/e2e 평가는 별도 확인 후 진행. ==="
