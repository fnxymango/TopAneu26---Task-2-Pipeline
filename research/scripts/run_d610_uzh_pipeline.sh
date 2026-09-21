#!/usr/bin/env bash
# D610 = D600 혈관 데이터 + UZH(TopBrain2025 우승) 아키텍처 + UZH 가중치 파인튜닝, fold 0/1/2
#
#   왜 이 구조인가:
#     - UZH 가중치를 얹으려면 아키텍처가 정확히 같아야 한다(PlainConvUNet, 6 stages,
#       features [32,64,128,256,320,320], strides 고정). 그래서 우리 ResEncM plans 대신
#       nnUNetv2_move_plans_between_datasets 로 UZH plans를 610에 이식한다.
#     - 클래스 수 불일치(UZH 13 vs 우리 36)는 문제 없음 — load_pretrained_weights 가
#       '.seg_layers.' 를 건너뛰고 나머지 shape 일치만 검증한다(확인함).
#     - fold k 는 UZH Dataset525(AM, CT/MR 공용) 의 fold k 체크포인트에서 출발한다.
#
#   OOM 대비 (2중):
#     VRAM — UZH 원본은 patch 128x256x256 / batch 2 = 16.8M voxel. D600(4.2M voxel)이
#            10.7GB를 썼으므로 24GB 3090에 batch 2는 불가. 학습 전에 실제로 forward+backward를
#            돌려 batch 2 -> 1 -> patch 축소 순으로 들어가는 설정을 고른다.
#     RAM  — run_experiment.sh 가 nnUNet_def_n_proc=1 을 걸어 export 워커 폭주를 막는다.
#            추가로 fold 시작을 25분씩 밀어 validation(=export) 시점을 분산시킨다.
#
# 사용: setsid nohup bash scripts/run_d610_uzh_pipeline.sh > experiments/nohup_d610.out 2>&1 &
set -uo pipefail

BASE=/home/user/TopAneu/seg/sblee/nnunet
ENVBIN=/home/user/anaconda3/envs/sbaneu2/bin
PY="$ENVBIN/python"
PLANS=nnUNetPlansUZH
SRC_DS=Dataset525_topbrain_AM_case_01_27_wReg
UZH_CKPT_DIR="$BASE/pretrained/uzh_topbrain/opt/app/trained_models/$SRC_DS/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres"
PRE="$BASE/nnUNet_preprocessed/Dataset610_TopAneuVesselUZH"
STAGGER=1500
FOLDS=(0 1 2)
GPUS=(0 1 2)          # GPU 3은 다른 사람 몫으로 비워둔다

export nnUNet_raw="$BASE/nnUNet_raw"
export nnUNet_preprocessed="$BASE/nnUNet_preprocessed"
export nnUNet_results="$BASE/experiments"
export nnUNet_def_n_proc=4      # 전처리용. 학습은 run_experiment.sh 가 1로 덮는다.

log() { echo "[d610 $(date +%H:%M:%S)] $*"; }
die() { echo "[d610 ABORT] $*" >&2; exit 1; }

log "시작. fold=${FOLDS[*]} gpu=${GPUS[*]}"

# ---------------------------------------------------------------- 1. fingerprint
if [ ! -f "$PRE/dataset_fingerprint.json" ]; then
  log "fingerprint 추출"
  "$ENVBIN/nnUNetv2_extract_fingerprint" -d 610 -np 4 || die "fingerprint 실패"
fi
log "fingerprint OK"

# ---------------------------------------------------------------- 2. plans 이식
if [ ! -f "$PRE/$PLANS.json" ]; then
  log "UZH plans 이식: $SRC_DS -> 610 ($PLANS)"
  "$ENVBIN/nnUNetv2_move_plans_between_datasets" -s "$SRC_DS" -t 610 -sp nnUNetPlans -tp "$PLANS" \
    || die "move_plans 실패"
fi
log "plans OK"

# ---------------------------------------------------------------- 3. split 동기화
# D600과 완전히 같은 fold 구성을 써야 비교가 성립한다.
cp -f "$BASE/nnUNet_preprocessed/Dataset600_TopAneuVessel/splits_final.json" "$PRE/splits_final.json" \
  || die "splits 복사 실패"
# move_plans 경로는 nnUNetv2_plan_experiment 를 안 거치므로 dataset.json 이 복사되지 않는다. 전처리가 이걸 읽는다.
cp -f "$nnUNet_raw/Dataset610_TopAneuVesselUZH/dataset.json" "$PRE/dataset.json" || die "dataset.json 복사 실패"
log "splits 동기화 OK ($("$PY" -c "import json;print(len(json.load(open('$PRE/splits_final.json'))))")개 fold)"

# ---------------------------------------------------------------- 4. VRAM 실측 -> batch/patch 확정
log "VRAM 실측 시작"
"$PY" - <<'PYEOF' || die "VRAM 프로브 실패"
import json, torch, sys
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

PLANS_FILE = "/home/user/TopAneu/seg/sblee/nnunet/nnUNet_preprocessed/Dataset610_TopAneuVesselUZH/nnUNetPlansUZH.json"
plans = json.load(open(PLANS_FILE))
cfg = plans['configurations']['3d_fullres']
arch = cfg['architecture']
NUM_OUT = 37   # background + 36

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
        # allocated 가 아니라 reserved 로 잰다. 카드를 실제로 점유하는 건 캐싱 할당자의 reserved 이고,
        # 단편화 때문에 allocated 보다 항상 크다. OOM 판정은 이쪽이 진짜 기준.
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

# 원본 -> batch 축소 -> patch 축소 순. patch 축소해도 strides/stages가 그대로라 가중치는 계속 호환된다.
CANDIDATES = [([128,256,256],2), ([128,256,256],1), ([96,192,192],1), ([64,160,160],1)]
chosen = None
for patch, bs in CANDIDATES:
    ok, peak = fits(patch, bs)
    print(f"  patch={patch} batch={bs} -> {'OK %.1fGB' % peak if ok else 'OOM'}", flush=True)
    if ok and peak < 19.0:          # 24GB 중 5GB는 학습 중 변동/종료 후 sliding-window 추론 여유
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
  log "전처리 시작 (UZH spacing 0.6/0.375/0.375)"
  "$ENVBIN/nnUNetv2_preprocess" -d 610 -plans_name "$PLANS" -c 3d_fullres -np 4 || die "전처리 실패"
fi
N=$(ls "$PRE/${PLANS}_3d_fullres"/*.npz 2>/dev/null | wc -l)
[ "$N" -ge 90 ] || die "전처리 결과가 부족하다 ($N개)"
log "전처리 OK ($N 케이스)"

# ---------------------------------------------------------------- 6. fold 0/1/2 학습
for i in "${!FOLDS[@]}"; do
  F=${FOLDS[$i]}; G=${GPUS[$i]}; D=$((i * STAGGER))
  CKPT="$UZH_CKPT_DIR/fold_$F/checkpoint_final.pth"
  [ -f "$CKPT" ] || die "UZH 체크포인트 없음: $CKPT"
  NAME="D610_vessel_uzh_ft_f${F}"
  (
    sleep "$D"
    log "fold=$F gpu=$G 학습 시작 (init: UZH $SRC_DS fold_$F)"
    GPU="$G" NPROC=1 ENVBIN="$ENVBIN" "$BASE/scripts/run_experiment.sh" 610 3d_fullres "$F" "$NAME" \
      -p "$PLANS" -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -pretrained_weights "$CKPT"
    log "fold=$F 학습 종료 status=$?"

    # ------------------------------------------------------------ 7. 예측 + skeleton
    EXP="$BASE/experiments/$NAME"
    IN="$EXP/val_predict_in"; OUT="$EXP/val_predict_out"
    mkdir -p "$IN" "$OUT"
    "$PY" - "$F" "$IN" <<'PYEOF'
import json, os, sys
fold, indir = int(sys.argv[1]), sys.argv[2]
raw = "/home/user/TopAneu/seg/sblee/nnunet/nnUNet_raw/Dataset610_TopAneuVesselUZH/imagesTr"
sp = json.load(open("/home/user/TopAneu/seg/sblee/nnunet/nnUNet_preprocessed/Dataset610_TopAneuVesselUZH/splits_final.json"))
for c in sp[fold]['val']:
    src, dst = f"{raw}/{c}_0000.nii.gz", f"{indir}/{c}_0000.nii.gz"
    if not os.path.exists(dst): os.symlink(os.path.realpath(src), dst)
print(f"fold {fold}: val {len(sp[fold]['val'])} 케이스 준비")
PYEOF
    log "fold=$F 예측 시작 (-nps 1)"
    CUDA_VISIBLE_DEVICES="$G" "$ENVBIN/nnUNetv2_predict" -i "$IN" -o "$OUT" \
      -d 610 -c 3d_fullres -f "$F" -p "$PLANS" -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep \
      -nps 1 -npp 1 --disable_tta >> "$EXP/predict.log" 2>&1
    log "fold=$F 예측 종료 status=$?"

    log "fold=$F skeleton 생성"
    "$PY" "$BASE/scripts/skeleton_from_predictions.py" "$OUT" "$EXP/val_skeletons" \
      >> "$EXP/skeleton.log" 2>&1
    log "fold=$F skeleton 종료 status=$?"
  ) &
done
wait
log "D610 전체 완료"
