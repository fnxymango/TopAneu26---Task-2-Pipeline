#!/usr/bin/env bash
# C16 추론 완료 후 이어지는 전체 큐 (2026-08-16)
#   C16-post : c7 필터 -> C5 평가 -> 복합지표 재집계
#   C17      : out-of-fold 추론 -> FP 기각기 학습/CV -> 적용/평가
#   C19      : A4 경계 정밀화
#   C14-2    : 혈관망 내부표현 재도전
# 각 단계 독립 실행 — 앞이 실패해도 뒤가 계속 돈다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
NN="$HOME/miniconda3/envs/sblee_topaneu/bin/nnUNetv2_predict"
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
FEAT="$A/c10_feat_train.json"; BEST="--model rf --beta 1.0 --use-pos"
M5="$E/_c16_a62_5fold/results"
log(){ echo "[q $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ---------- 0) C16 추론 완료 대기 ----------
# 주의: 프로세스 존재(pgrep)로 대기하면 **내 명령줄 wrapper까지 매칭**돼 영원히 안 풀린다.
# chain_best.sh 에서 2.5시간, 여기서 3.4시간을 이렇게 날렸다. 반드시 **산출물 개수**로 판정한다.
log "C16 추론 완료 대기 (val 42 / test 83)"
for _ in $(seq 1 120); do
  nv=$(ls "$P"/aneu_val_probavg/*.nii.gz 2>/dev/null | wc -l)
  nt=$(ls "$P"/aneu_test_probavg/*.nii.gz 2>/dev/null | wc -l)
  [ "$nv" -ge 42 ] && [ "$nt" -ge 83 ] && break
  sleep 60
done
log "val $(ls $P/aneu_val_probavg/*.nii.gz 2>/dev/null|wc -l)/42, test $(ls $P/aneu_test_probavg/*.nii.gz 2>/dev/null|wc -l)/83"

# ---------- 1) C16 후처리 + 평가 ----------
for SP in val test; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  VB=$([ "$SP" = val ] && echo val_pred || echo vespp_test)
  n=$(ls $P/aneu_${SP}_probavg/*.nii.gz 2>/dev/null|wc -l)
  [ "$n" -eq 0 ] && { log "C16 $SP 예측 없음 — 건너뜀"; continue; }
  log "C16 $SP: c7 필터"
  $PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_${SP}_probavg" --vessel-dir "$P/$VES" \
      --split "$SP" --tag "probavg_$SP" --save-best "$P/aneu_${SP}_probavgf" --force-cfg 5,3 \
      || log "  c7 실패"
  log "C16 $SP: C5 평가"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$SP" \
    --vessel-dir "$P/$VES" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/aneu_${SP}_probavgf" \
    $BEST --tag "c16_probavg" 2>&1 | tail -18 || log "  C5 실패"
done
log "=== C16 완료 ==="; $PY -u c18_composite_reselect.py 2>&1 | tail -25

# ---------- 2) C17 FP 기각기 ----------
log "=== C17: out-of-fold 추론 (train 292 커버) ==="
mkdir -p "$E/_c17_oof"
for f in 1 2 3 4; do
  OUT="$E/_c17_oof/oof_f$f"
  [ -d "$OUT" ] && [ "$(ls $OUT/*.nii.gz 2>/dev/null|wc -l)" -gt 50 ] && { log "  fold$f 이미 있음"; continue; }
  mkdir -p "$E/_c17_oof/in_f$f"
  $PY - "$f" "$E/_c17_oof/in_f$f" <<'PYEOF'
import json,os,sys,glob
from pathlib import Path
f=int(sys.argv[1]); dst=Path(sys.argv[2])
R=Path(os.environ["TOPANEU_ROOT"])
sp=json.load(open(R/"nnunet"/"nnUNet_preprocessed"/"Dataset720_TopAneuBinary417"/"splits_final.json"))
src=R/"nnunet"/"nnUNet_raw"/"Dataset720_TopAneuBinary417"/"imagesTr"
for c in sp[f]["val"]:
    for g in glob.glob(str(src/f"{c}_*.nii.gz")):
        t=dst/Path(g).name
        if not t.exists(): os.symlink(g,t)
print(f"fold{f} 입력 {len(list(dst.glob('*.nii.gz')))}")
PYEOF
  log "  fold$f 추론"
  nnUNet_results="$M5" CUDA_VISIBLE_DEVICES=0 "$NN" -i "$E/_c17_oof/in_f$f" -o "$OUT" \
    -d 720 -c 3d_fullres -f "$f" -tr nnUNetTrainerTverskyTopkCE \
    -p nnUNetResEncUNetLPlansAdaptive -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
    > "$E/c17_pred_f$f.log" 2>&1 || log "  fold$f 추론 실패"
done
log "=== C17: 후보 피처 추출 ==="
$PY -u c17_fp_reject.py build --pred-root "$E/_c17_oof" --vessel-dir "$REFVES" \
    --bp-dir "$BP/all_ref" --out "$A/c17_candidates.json" || log "C17 build 실패"
log "=== C17: CV ==="
$PY -u c17_fp_reject.py cv --cand "$A/c17_candidates.json" || log "C17 cv 실패"

# ---------- 3) C19 A4 경계 정밀화 ----------
log "=== C19: A4 전체볼륨 추론 ==="
A4="$E/A4_binary_lesionscale_417/results"
for SP in val test; do
  OUT="$P/a4_$SP"
  [ -d "$OUT" ] && [ "$(ls $OUT/*.nii.gz 2>/dev/null|wc -l)" -gt 10 ] && continue
  nnUNet_results="$A4" CUDA_VISIBLE_DEVICES=1 "$NN" -i "$P/in_$SP" -o "$OUT" \
    -d 740 -c 3d_fullres -f 0 -tr nnUNetTrainerTverskyTopkCE -p nnUNetPlans \
    -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 > "$E/c19_pred_$SP.log" 2>&1 \
    || log "  A4 $SP 추론 실패"
  log "  A4 $SP 완료"
done

# ---------- 4) C14-2 ----------
log "=== C14-2: 혈관망 내부표현 재도전 (전량) ==="
$PY -u c14_vessel_encoder_feat.py --feat "$FEAT" --folds 5 --max-cases 250 \
    --cache "$A/c14v2_enc_feats.json" || log "C14-2 실패"

log "=== 전체 큐 완료 ==="
