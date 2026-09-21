#!/usr/bin/env bash
# Per-experiment nnU-Net training runner.
#   Each experiment -> its own folder with: train.log, loss_curve.png, config.json, results/
#
# Usage:
#   run_experiment.sh <DATASET_ID> <CONFIG> <FOLD> <EXP_NAME> [extra nnUNetv2_train args...]
# Example:
#   run_experiment.sh 501 3d_fullres 0 D501_region_3dfr_f0
#   run_experiment.sh 501 3d_fullres 0 D501_region_3dfr_f0_ep200 -tr nnUNetTrainer_50epochs
set -uo pipefail

BASE="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$BASE/nnunet"
SCRIPTS="$BASE/code/sblee/nnunet/scripts"
# conda env bin. 기본 sblee_topaneu(구 sbaneu, 2026-07-31 개명).
# 다른 env로 실행하려면: ENVBIN=$HOME/miniconda3/envs/sbaneu2/bin GPU=n run_experiment.sh ...
ENVBIN="${ENVBIN:-$HOME/miniconda3/envs/sblee_topaneu/bin}"
PY="$ENVBIN/python"
TRAIN="$ENVBIN/nnUNetv2_train"

# setup_env.sh가 이미 export했으면 그 값을 존중, 없으면 새 레이아웃 기본값 사용
export nnUNet_raw="${nnUNet_raw:-$NNUNET_BASE/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$NNUNET_BASE/nnUNet_preprocessed}"

# GPU 선택 (공유 서버): `GPU=1 run_experiment.sh ...` 형태. 기본 0.
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

# 학습 후 내장 validation의 export 워커 수(nnUNet configuration.py의 default_num_processes, 기본 8).
# 36클래스 확률맵을 원본 spacing으로 리샘플하느라 워커당 ~20GB를 먹어서, 기본값이면 공유 서버
# RAM(125GB)이 고갈되고 OOM killer가 워커를 죽인다 → "Some background workers are no longer alive"로
# 학습은 끝났는데 validation만 터짐(D600 런 2건: 07-27 22:06, 07-30 03:42).
# 학습 속도와는 무관(DA 워커는 nnUNet_n_proc_DA, unpack은 get_allowed_n_proc_DA로 별도).
# RAM 여유가 확실하면 `NPROC=2 run_experiment.sh ...`로 올릴 수 있다.
export nnUNet_def_n_proc="${NPROC:-1}"

if [ "$#" -lt 4 ]; then
  echo "Usage: [GPU=n] $0 <DATASET_ID> <CONFIG> <FOLD> <EXP_NAME> [extra args...]"; exit 1
fi
DATASET_ID="$1"; CONFIG="$2"; FOLD="$3"; EXP_NAME="$4"; shift 4
EXTRA=("$@")

EXP_DIR="$BASE/experiments/$EXP_NAME"
mkdir -p "$EXP_DIR"
export nnUNet_results="$EXP_DIR/results"
mkdir -p "$nnUNet_results"
LOG="$EXP_DIR/train.log"

cat > "$EXP_DIR/config.json" <<EOF
{
  "exp_name": "$EXP_NAME",
  "dataset_id": "$DATASET_ID",
  "config": "$CONFIG",
  "fold": "$FOLD",
  "extra_args": "${EXTRA[*]:-}",
  "gpu": "$GPU",
  "nnUNet_def_n_proc": "$nnUNet_def_n_proc",
  "started": "$(date -Iseconds)",
  "cmd": "CUDA_VISIBLE_DEVICES=$GPU nnUNet_def_n_proc=$nnUNet_def_n_proc nnUNetv2_train $DATASET_ID $CONFIG $FOLD ${EXTRA[*]:-}"
}
EOF

# guarantee loss curve + summary.md + master leaderboard CSV on ANY exit (success, crash, or Ctrl-C)
trap '"$PY" "$SCRIPTS/plot_loss.py" "$EXP_DIR" || true; "$PY" "$SCRIPTS/make_summary.py" "$EXP_DIR" || true; "$PY" "$SCRIPTS/make_leaderboard.py" || true' EXIT

echo "[run_experiment] $EXP_NAME | Dataset$DATASET_ID $CONFIG fold=$FOLD GPU=$GPU n_proc=$nnUNet_def_n_proc extra=${EXTRA[*]:-} | $(date -Iseconds)" | tee "$LOG"
"$TRAIN" "$DATASET_ID" "$CONFIG" "$FOLD" "${EXTRA[@]}" 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
echo "[run_experiment] finished with status $STATUS | $(date -Iseconds)" | tee -a "$LOG"
exit $STATUS
