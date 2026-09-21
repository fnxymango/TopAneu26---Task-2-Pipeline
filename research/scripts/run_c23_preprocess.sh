#!/usr/bin/env bash
# C23 — Dataset900(43클래스 위치라벨) 전처리 + splits 재생성 (2026-08-17).
# 전처리 산출물은 디스크 여유(/ 68GB)가 부족해 /mnt/hdd 로 빼고 심볼릭 링크했다.
# PROJECT_RULES.md §1: 전처리 직후 **반드시** make_splits_417.py 로 자동생성 split 을 덮어쓴다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; ENV="$HOME/miniconda3/envs/sblee_topaneu/bin"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
export nnUNet_raw="$R/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$R/nnunet/nnUNet_preprocessed"
export nnUNet_results="$R/nnunet/nnUNet_results"
log(){ echo "[c23 $(date -u +'%m-%d %H:%M:%S')] $*"; }

log "STEP1: plan_and_preprocess (Dataset900, 43클래스)"
"$ENV/nnUNetv2_plan_and_preprocess" -d 900 -c 3d_fullres --verify_dataset_integrity -np 4
log "STEP1 종료 status=$?  ($(du -sh /mnt/hdd/sblee/nnUNet_preprocessed/Dataset900_TopAneuLoc417 2>/dev/null | cut -f1))"

log "STEP2: splits 재생성 (--train-only, 자동생성분 덮어쓰기)"
cd "$R/rebuild_417" || exit 1
nnUNet_preprocessed="$nnUNet_preprocessed" TOPANEU_DATA="$R/dataset/TopAneu" \
  $PY make_splits_417.py 900 --train-only
log "STEP2 종료 status=$?"

log "STEP3: 누출 검증"
$PY - <<'PYEOF'
import json, os
R = os.environ["TOPANEU_ROOT"]
s = json.load(open(f"{R}/nnunet/nnUNet_preprocessed/Dataset900_TopAneuLoc417/splits_final.json"))
d = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
val, test = set(d["val"]), set(d["test"])
ok = True
for i, f in enumerate(s):
    tr, va = set(f["train"]), set(f["val"])
    lv, lt = len(tr & val), len((tr | va) & test)
    print(f"  fold{i}: train {len(tr):3d} val {len(va):3d} | val42누출 {lv} | test83누출 {lt}")
    ok &= (lt == 0)
print("검증", "통과 — test 83 전 fold 미포함" if ok else "실패!")
PYEOF
log "=== C23 전처리 완료 ==="
