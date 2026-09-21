#!/usr/bin/env bash
# E1 — e2e 를 292 학습코호트 OOF 로 평가하는 하니스 (2026-08-19). 점수용이 아니라 **판정 장치**다.
#
# 왜 만드는가 (2026-08-19 측정):
#   test 는 병변 87개 / 분모클래스 36개다. 여기서 n=1 희소 클래스 하나를 맞히면
#   그 클래스 MCC 가 0 -> 1.0 이라 cov.MCC 가 **한 번에 +0.0278** 뛴다.
#   우리가 지금까지 쌓은 개선 전체(C26 0.3188 -> C36 0.3609 = +0.0421)가 병변 1.5개분이다.
#   즉 실험 채택 여부를 병변 1~2개짜리 노이즈 위에서 결정해 왔다.
#   C39 가 val +0.006 -> test -0.074 로 뒤집힌 것도 이 해상도에서는 이상한 일이 아니다.
#
#   같은 계산을 292 코호트(병변 268 / 클래스 43)에 하면 병변 1개의 값이 +0.0029 다.
#   **분해능이 3.6배** 올라간다. +0.01 짜리 차이가 실재하는지 비로소 판정할 수 있다.
#
# 구성 — 사용자 지시(2026-08-19): 검출기는 5폴드, 혈관은 1폴드 그대로, 전체를 CV 로.
#   검출  Dataset720 splits 의 fold1~4 는 train 292 를 정확히 4분할한다(73/72/72/75).
#         각 폴드 모델이 자기 held-out 만 추론 -> 292 전체가 정직한 OOF 가 된다. 누수 0.
#   혈관  V4-2 fold0 를 그대로 쓴다. **이 폴드는 292 를 학습했으므로 여기 예측은 train-fit 이다.**
#         따라서 E1 은 혈관 품질에 기대는 피처(측지·호위치·분기점/앵커=C44 계열)에 대해서는
#         낙관적이다. 그 계열은 E1 을 통과해도 test 로 재확인해야 한다.
#         반대로 결정규칙 축(β·τ·분류기 종류)은 혈관을 안 건드리므로 E1 판정이 유효하다.
#   분류  학습 피처는 프로덕션과 동일하게 c10_feat_train.json(GT병변·참조혈관)을 쓰되,
#         평가 폴드의 케이스를 학습에서 빼고 적합한다. 적용 대상은 위 OOF 검출 병변.
#
# 순서: 검출 OOF -> 혈관 -> c7 필터 -> 분기점 그래프. GPU0 하나만 쓴다(GPU1 은 vLLM 점유).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
# 혈관 트레이너(SkeletonRecall...ClassWeightedV2_500ep)는 vendor/Skeleton-Recall 에만 있고
# 그 nnunetv2 를 쓰는 건 sbaneu2 환경이다. sblee_topaneu 로 부르면 트레이너를 못 찾는다.
ENVBIN2="$HOME/miniconda3/envs/sbaneu2/bin"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
LOG="$E/e1_chain.log"
log(){ echo "[e1 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── STEP0: 폴드별 입력 스테이징 ────────────────────────────────────────
log "=== STEP0: fold1~4 held-out 입력 링크 ==="
$PY - "$SP720" "$P" <<'PYEOF'
import json,os,sys
from pathlib import Path
sp=json.load(open(sys.argv[1])); P=Path(sys.argv[2])
img=Path(os.environ["TOPANEU_ROOT"])/"nnunet"/"nnUNet_raw"/"Dataset720_TopAneuBinary417"/"imagesTr"
tot=0
for i in (1,2,3,4):
    out=P/f"in_train_f{i}"; out.mkdir(parents=True,exist_ok=True); n=0
    for cid in sp[i]["val"]:
        s=img/f"{cid}_0000.nii.gz"; d=out/f"{cid}_0000.nii.gz"
        if s.exists() and not d.exists(): os.symlink(s,d); n+=1
    have=len(list(out.glob("*.nii.gz"))); tot+=have
    print(f"  fold{i}: {have}건 (신규 {n})")
out=P/"in_train"; out.mkdir(parents=True,exist_ok=True)
for cid in sp[0]["train"]:
    s=img/f"{cid}_0000.nii.gz"; d=out/f"{cid}_0000.nii.gz"
    if s.exists() and not d.exists(): os.symlink(s,d)
print(f"  합계 {tot} · in_train {len(list(out.glob('*.nii.gz')))}")
PYEOF

# ── STEP1: 검출 OOF (fold1~4, 각자 자기 held-out) ──────────────────────
log "=== STEP1: 검출 OOF 추론 (A6-2 f1~f4) ==="
OOF="$P/aneu_train_oof"; mkdir -p "$OOF"
for i in 1 2 3 4; do
  NEED=$(ls "$P/in_train_f$i"/*.nii.gz 2>/dev/null | wc -l)
  HAVE=$($PY - "$SP720" "$OOF" "$i" <<'PYEOF'
import json,sys
from pathlib import Path
sp=json.load(open(sys.argv[1])); O=Path(sys.argv[2]); i=int(sys.argv[3])
print(sum(1 for c in sp[i]["val"] if (O/f"{c}.nii.gz").exists()))
PYEOF
)
  if [ "$HAVE" -ge "$NEED" ] && [ "$NEED" -gt 0 ]; then log "  fold$i 이미 완료 ($HAVE/$NEED)"; continue; fi
  log "  fold$i 추론 ($HAVE/$NEED)"
  nnUNet_results="$E/A6-2_resencl_adaptivenorm_topk_417_f$i/results" CUDA_VISIBLE_DEVICES=0 \
    "$ENVBIN/nnUNetv2_predict" -i "$P/in_train_f$i" -o "$OOF" \
    -d 720 -c 3d_fullres -f "$i" -tr nnUNetTrainerTverskyTopkCE \
    -p nnUNetResEncUNetLPlansAdaptive -chk checkpoint_best.pth \
    --disable_tta --continue_prediction -npp 2 -nps 2 >> "$LOG" 2>&1 || log "    fold$i 실패 status=$?"
done
log "  OOF 검출 $(ls "$OOF"/*.nii.gz 2>/dev/null | wc -l)/292"

# ── STEP2: 혈관 (V4-2 fold0 — train-fit 임을 알고 쓴다) ────────────────
log "=== STEP2: 혈관 추론 (V4-2 fold0) ==="
if [ "$(ls "$P/ves_train"/*.nii.gz 2>/dev/null | wc -l)" -lt 292 ]; then
  nnUNet_results="$E/V4-2_vessel_classweighted_417_500ep/results" CUDA_VISIBLE_DEVICES=0 \
    "$ENVBIN2/nnUNetv2_predict" -i "$P/in_train" -o "$P/ves_train" \
    -d 800 -c 3d_fullres -f 0 -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep \
    -p nnUNetResEncUNetMPlans -chk checkpoint_best.pth \
    --disable_tta --continue_prediction -npp 2 -nps 2 >> "$LOG" 2>&1 || log "  실패 status=$?"
fi
log "  ves_train $(ls "$P/ves_train"/*.nii.gz 2>/dev/null | wc -l)/292"
if [ "$(ls "$P/vespp_train"/*.nii.gz 2>/dev/null | wc -l)" -lt 292 ]; then
  log "  혈관 후처리 (V5 레시피)"
  "$ENVBIN/python" "$S/postprocess_vessel.py" apply "$P/ves_train" "$P/vespp_train" >> "$LOG" 2>&1
fi
log "  vespp_train $(ls "$P/vespp_train"/*.nii.gz 2>/dev/null | wc -l)/292"

# ── STEP3: c7 필터 (프로덕션과 동일: min_vox 5, dist 3mm) ──────────────
NV=$(ls "$P/vespp_train"/*.nii.gz 2>/dev/null | wc -l)
[ "$NV" -lt 250 ] && { log "혈관 $NV/292 — 부족해서 중단 (빈 입력으로 뒤 단계가 죽는다)"; exit 1; }

log "=== STEP3: c7 후처리 필터 ==="
if [ "$(ls "$P/aneu_train_ooff"/*.nii.gz 2>/dev/null | wc -l)" -lt 292 ]; then
  $PY -u c7_detect_postproc.py --aneu-dir "$OOF" --vessel-dir "$P/vespp_train" \
      --split train --tag "e1_oof" --save-best "$P/aneu_train_ooff" --force-cfg 5,3 \
      >> "$LOG" 2>&1 || log "  실패"
fi
log "  aneu_train_ooff $(ls "$P/aneu_train_ooff"/*.nii.gz 2>/dev/null | wc -l)/292"

# ── STEP4: 분기점 그래프 ──────────────────────────────────────────────
log "=== STEP4: 분기점 그래프 (vespp_train) ==="
if [ "$(ls "$BP/vespp_train"/*.json 2>/dev/null | wc -l)" -lt 292 ]; then
  $PY -u c4_branchpoint_graph.py --vessel-dir "$P/vespp_train" --out "$BP/vespp_train" \
      >> "$LOG" 2>&1 || log "  실패"
fi
log "  bpgraph $(ls "$BP/vespp_train"/*.json 2>/dev/null | wc -l)/292"

log "=== E1 데이터 준비 완료 — 다음은 c5 CV 평가 ==="
