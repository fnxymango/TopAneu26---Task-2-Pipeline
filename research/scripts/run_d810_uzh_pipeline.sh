#!/usr/bin/env bash
# D810 = D800 혈관 데이터(417케이스) + UZH(TopBrain2025 우승) 아키텍처 + UZH 가중치 파인튜닝
#
# 417케이스 이전 서버용으로 이식한 run_d610_uzh_pipeline.sh (98케이스 버전) 포크.
# 원본과의 차이:
#   - 경로: /home/user/... 하드코딩 -> $TOPANEU_ROOT 기반
#   - 데이터셋: 610(98케이스) -> 810(417케이스), split은 D800과 동일 것을 그대로 복사
#   - GPU: 원본은 fold 0/1/2를 GPU 3개에 병렬. 여기는 GPU 1개(GPU1)뿐이라 기본 fold 0만.
#
# 사용: source rebuild_417/setup_env.sh 먼저 하지 않아도 됨(이 스크립트가 자체 export).
#   bash code/sblee/nnunet/scripts/run_d810_uzh_pipeline.sh
set -uo pipefail

TOPANEU_ROOT="${TOPANEU_ROOT:-$HOME/topaneu_sblee}"
NNUNET_BASE="$TOPANEU_ROOT/nnunet"
EXP_BASE="$TOPANEU_ROOT/experiments"
SCRIPTS="$TOPANEU_ROOT/code/sblee/nnunet/scripts"
ENVBIN="${ENVBIN:-$HOME/miniconda3/envs/sbaneu2/bin}"
PY="$ENVBIN/python"
PLANS=nnUNetPlansUZH
SRC_DS=Dataset525_topbrain_AM_case_01_27_wReg
UZH_CKPT_DIR="$TOPANEU_ROOT/pretrained/uzh_topbrain/opt/app/trained_models/$SRC_DS/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres"
PRE="$NNUNET_BASE/nnUNet_preprocessed/Dataset810_TopAneuVesselUZH417"
RAW810="$NNUNET_BASE/nnUNet_raw/Dataset810_TopAneuVesselUZH417"
FOLDS=(${FOLDS_OVERRIDE:-0})     # 기본 fold0만. FOLDS_OVERRIDE="0 1 2"로 여러개 가능(GPU 여유 있을 때).
GPUS=(${GPUS_OVERRIDE:-1})       # 기본 GPU1.

export nnUNet_raw="$NNUNET_BASE/nnUNet_raw"
export nnUNet_preprocessed="$NNUNET_BASE/nnUNet_preprocessed"
export nnUNet_results="$EXP_BASE"
export nnUNet_def_n_proc=4
export TOPANEU_DATA="${TOPANEU_DATA:-$TOPANEU_ROOT/dataset/TopAneu}"

log() { echo "[d810 $(date +%H:%M:%S)] $*"; }
die() { echo "[d810 ABORT] $*" >&2; exit 1; }

log "시작. fold=${FOLDS[*]} gpu=${GPUS[*]}"
[ -d "$RAW810" ] || die "raw 데이터셋 없음: $RAW810 (먼저 810 raw를 만들어야 함)"

# ---------------------------------------------------------------- 1. fingerprint
if [ ! -f "$PRE/dataset_fingerprint.json" ]; then
  log "fingerprint 추출"
  "$ENVBIN/nnUNetv2_extract_fingerprint" -d 810 -np 4 || die "fingerprint 실패"
fi
log "fingerprint OK"

# ---------------------------------------------------------------- 2. plans 이식
if [ ! -f "$PRE/$PLANS.json" ]; then
  log "UZH plans 이식: $SRC_DS -> 810 ($PLANS)"
  "$ENVBIN/nnUNetv2_move_plans_between_datasets" -s "$SRC_DS" -t 810 -sp nnUNetPlans -tp "$PLANS" \
    || die "move_plans 실패"
fi
log "plans OK"

# ---------------------------------------------------------------- 3. split 생성
# make_splits_417.py는 고정 시드(20260807)라 D800에 썼던 것과 항상 동일한 fold 구성이 나온다.
# (D800 preprocessed 폴더는 디스크 확보를 위해 삭제됐으므로 복사 대신 직접 생성)
if [ ! -f "$PRE/splits_final.json" ]; then
  "$PY" "$TOPANEU_ROOT/rebuild_417/make_splits_417.py" 810 || die "splits 생성 실패"
fi
cp -f "$RAW810/dataset.json" "$PRE/dataset.json" || die "dataset.json 복사 실패"
log "splits 동기화 OK ($("$PY" -c "import json;print(len(json.load(open('$PRE/splits_final.json'))))")개 fold)"

# ---------------------------------------------------------------- 4. VRAM 실측 -> batch/patch 확정
log "VRAM 실측 시작"
PLANS_FILE="$PRE/$PLANS.json" "$PY" - <<'PYEOF' || die "VRAM 프로브 실패"
import json, os, torch, sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

PLANS_FILE = os.environ['PLANS_FILE']
plans = json.load(open(PLANS_FILE))
cfg = plans['configurations']['3d_fullres']
arch = cfg['architecture']
NUM_OUT = 37   # background + 36 vessel classes

def fits(patch, bs):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        net = nnUNetTrainer.build_network_architecture(
            arch['network_class_name'], arch['arch_kwargs'],
            arch.get('_kw_requires_import', []), 1, NUM_OUT, True).cuda()
        opt = torch.optim.SGD(net.parameters(), lr=1e-2)
        scaler = torch.cuda.amp.GradScaler()
        x = torch.randn((bs, 1, *patch), device='cuda')
        with torch.cuda.amp.autocast():
            out = net(x)
            loss = sum(o.mean() for o in out) if isinstance(out, (list, tuple)) else out.mean()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        peak = torch.cuda.max_memory_reserved() / 2**30
        del net, opt, x, out, loss
        return True, peak
    except torch.cuda.OutOfMemoryError:
        return False, None
    except RuntimeError as e:
        if 'out of memory' not in str(e).lower():
            raise
        return False, None
    finally:
        torch.cuda.empty_cache()

CANDIDATES = [([128,256,256],2), ([128,256,256],1), ([96,192,192],1), ([64,160,160],1)]
chosen = None
for patch, bs in CANDIDATES:
    ok, peak = fits(patch, bs)
    print(f"  patch={patch} batch={bs} -> {'OK %.1fGB' % peak if ok else 'OOM'}", flush=True)
    if ok and peak < 19.0:
        chosen = (patch, bs, peak); break
    if ok:
        print(f"    (여유 부족 {peak:.1f}GB, 다음 후보로)", flush=True)

if chosen is None:
    sys.exit("모든 후보가 OOM")
patch, bs, peak = chosen
cfg['patch_size'] = patch
cfg['batch_size'] = bs
json.dump(plans, open(PLANS_FILE, 'w'), indent=1)
print(f"확정: patch={patch} batch={bs} (peak {peak:.1f}GB) -> plans 반영", flush=True)
PYEOF
log "VRAM 실측 OK"

# ---------------------------------------------------------------- 5. 전처리
if [ ! -d "$PRE/${PLANS}_3d_fullres" ]; then
  log "전처리 시작 (UZH spacing)"
  "$ENVBIN/nnUNetv2_preprocess" -d 810 -plans_name "$PLANS" -c 3d_fullres -np 4 || die "전처리 실패"
fi
N=$(ls "$PRE/${PLANS}_3d_fullres"/*.b2nd "$PRE/${PLANS}_3d_fullres"/*.npz 2>/dev/null | wc -l)
[ "$N" -ge 200 ] || die "전처리 결과가 부족하다 ($N개 파일)"
log "전처리 OK ($N 파일)"

# ---------------------------------------------------------------- 6. fold 학습
for i in "${!FOLDS[@]}"; do
  F=${FOLDS[$i]}; G=${GPUS[$((i % ${#GPUS[@]}))]}
  CKPT="$UZH_CKPT_DIR/fold_$F/checkpoint_final.pth"
  [ -f "$CKPT" ] || die "UZH 체크포인트 없음: $CKPT"
  NAME="D810_vessel_uzh_ft_f${F}"
  (
    log "fold=$F gpu=$G 학습 시작 (init: UZH $SRC_DS fold_$F)"
    GPU="$G" NPROC=1 ENVBIN="$ENVBIN" TOPANEU_ROOT="$TOPANEU_ROOT" \
      bash "$SCRIPTS/run_experiment.sh" 810 3d_fullres "$F" "$NAME" \
      -p "$PLANS" -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -pretrained_weights "$CKPT"
    log "fold=$F 학습 종료 status=$?"

    # ------------------------------------------------------------ 7. 예측 + skeleton
    EXP="$EXP_BASE/$NAME"
    IN="$EXP/val_predict_in"; OUT="$EXP/val_predict_out"
    mkdir -p "$IN" "$OUT"
    RAW810_DIR="$RAW810/imagesTr" PRE_SPLITS="$PRE/splits_final.json" FOLD_IDX="$F" INDIR="$IN" "$PY" <<'PYEOF'
import json, os
raw = os.environ['RAW810_DIR']
fold = int(os.environ['FOLD_IDX'])
indir = os.environ['INDIR']
sp = json.load(open(os.environ['PRE_SPLITS']))
for c in sp[fold]['val']:
    src, dst = f"{raw}/{c}_0000.nii.gz", f"{indir}/{c}_0000.nii.gz"
    if not os.path.exists(dst): os.symlink(os.path.realpath(src), dst)
print(f"fold {fold}: val {len(sp[fold]['val'])} 케이스 준비")
PYEOF
    log "fold=$F 예측 시작 (-nps 1)"
    CUDA_VISIBLE_DEVICES="$G" nnUNet_results="$EXP/results" "$ENVBIN/nnUNetv2_predict" -i "$IN" -o "$OUT" \
      -d 810 -c 3d_fullres -f "$F" -p "$PLANS" -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep \
      -nps 1 -npp 1 --disable_tta >> "$EXP/predict.log" 2>&1
    log "fold=$F 예측 종료 status=$?"

    log "fold=$F skeleton 생성"
    "$PY" "$SCRIPTS/skeleton_from_predictions.py" "$OUT" "$EXP/val_skeletons" \
      >> "$EXP/skeleton.log" 2>&1
    log "fold=$F skeleton 종료 status=$?"
  ) &
done
wait
log "D810 전체 완료"
