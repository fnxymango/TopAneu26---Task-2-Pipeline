#!/usr/bin/env bash
# D600 vessel (SkelRecall + ResEncM + batch_dice=False) fold 1~4 학습.
#   fold 0은 이미 D600_vessel_skelrec_resencm_250ep_bd0 에 있음 (정식 split, 재학습 불필요).
#
# 왜 stagger 하나:
#   학습 후 내장 validation의 export 워커가 36클래스 확률맵 리샘플로 ~20GB를 쓴다.
#   4개 런이 동시에 validation에 진입하면 125GB RAM이 다시 터진다(07-27, 07-30 OOM 전례).
#   시작을 25분씩 밀어서 종료(=export) 시점을 분산시킨다. NPROC=1과 병행.
#
# 사용: nohup bash scripts/run_5fold_bd0.sh > experiments/nohup_5fold_bd0.out 2>&1 &
set -uo pipefail

BASE=/home/user/TopAneu/seg/sblee/nnunet
RUN="$BASE/scripts/run_experiment.sh"
ENVBIN=/home/user/anaconda3/envs/sbaneu2/bin
SPLIT="$BASE/nnUNet_preprocessed/Dataset600_TopAneuVessel/splits_final.json"
STAGGER=1500   # 25분

# 안전장치: 5-fold split이 아니면 즉시 중단 (fold 1~4가 존재하지 않으면 nnUNet이 실패)
NSPLIT=$("$ENVBIN/python" -c "import json;print(len(json.load(open('$SPLIT'))))")
if [ "$NSPLIT" -lt 5 ]; then
  echo "[abort] splits_final.json에 fold가 $NSPLIT개뿐. scripts/make_5fold_split.py --write 먼저 실행." >&2
  exit 1
fi

launch() {  # launch <fold> <gpu> <delay_sec>
  local FOLD=$1 GPU=$2 DELAY=$3
  local NAME="D600_vessel_skelrec_resencm_250ep_bd0_f${FOLD}"
  (
    sleep "$DELAY"
    echo "[launch] fold=$FOLD gpu=$GPU $NAME | $(date -Iseconds)"
    GPU="$GPU" NPROC=1 ENVBIN="$ENVBIN" "$RUN" 600 3d_fullres "$FOLD" "$NAME" \
      -p nnUNetResEncUNetMPlansBD0 -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep
    echo "[done] fold=$FOLD status=$? | $(date -Iseconds)"
  ) &
}

echo "[run_5fold_bd0] 시작 $(date -Iseconds) | stagger=${STAGGER}s"
launch 1 0 0
launch 2 1 $((STAGGER))
launch 3 2 $((STAGGER * 2))
launch 4 3 $((STAGGER * 3))
wait
echo "[run_5fold_bd0] 전체 종료 $(date -Iseconds)"
