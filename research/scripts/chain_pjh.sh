#!/usr/bin/env bash
# P — pjh Stage1 검출기 재현 (2026-08-25). REIMPLEMENT_jslee_pjh.md §3.2~3.3.
#
# 왜: 팀 문서 대조로 우리 검출기가 test recall 73.3% vs pjh 81.8% 로 밀린다.
#   TTA 로는 안 좁혀졌다(D1: 0.721, 오히려 -1병변). A7(Tversky FN벌점 2배)도 민감도 불변.
#   남은 차이는 **3-class · 스톡손실 · 모달리티별 robust z · 0.4mm iso** 넷이다.
#   spacing 은 원본 z 중앙 0.5mm 라 84% 가 보간 업샘플이지만, 레시피 재현이 목적이므로 포함한다.
#
# fold0 만 먼저 한다 — 우리 fold0 의 val 이 곧 공식 val 42 라 A6-2 fold0(recall 0.814 / FP 53)
# 과 같은 자로 직접 비교된다. 이기면 5폴드로 확대, 지면 접는다.
#
# 각 스텝 시작/종료를 STATUS 파일에 한 줄씩 찍는다(감시자가 읽는다).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
DS=722; DSN="Dataset722_TopAneuPjh3cls417"
EXP="P1_pjh3cls_stock250_iso04_f0"
TR=nnUNetTrainer_250epochs
STATUS="$E/pjh_status.log"
export nnUNet_raw="$R/nnunet/nnUNet_raw" nnUNet_preprocessed="$R/nnunet/nnUNet_preprocessed"
export TOPANEU_ROOT="$R"
cd "$S" || exit 1
st(){ echo "[STEP] $*" | tee -a "$STATUS"; }
er(){ echo "[FAIL] $*" | tee -a "$STATUS"; exit 1; }
log(){ echo "[pjh $(date -u +'%m-%d %H:%M:%S')] $*"; }

st "P1 시작 · 3-class 라벨 + 모달리티별 robust z 빌드 (417케이스, CPU ~30분)"
if [ "$(ls "$nnUNet_raw/$DSN/labelsTr"/*.nii.gz 2>/dev/null | wc -l)" -lt 417 ]; then
  $PY -u p_build_3cls.py > "$E/pjh_p1_build.log" 2>&1 || er "P1 빌드 실패 — $E/pjh_p1_build.log"
fi
n=$(ls "$nnUNet_raw/$DSN/labelsTr"/*.nii.gz 2>/dev/null | wc -l)
[ "$n" -lt 417 ] && er "P1 라벨 $n/417 — 부족"
st "P1 완료 · 라벨 $n / 이미지 $(ls "$nnUNet_raw/$DSN/imagesTr"/*.nii.gz | wc -l)"
tail -8 "$E/pjh_p1_build.log" 2>/dev/null

st "P2 시작 · fingerprint + plan (0.4mm iso, patch 112x160x128 로 고정)"
if [ ! -f "$nnUNet_preprocessed/$DSN/dataset_fingerprint.json" ]; then
  "$ENVBIN/nnUNetv2_extract_fingerprint" -d $DS -np 8 > "$E/pjh_p2_fp.log" 2>&1 || er "fingerprint 실패"
fi
if [ ! -f "$nnUNet_preprocessed/$DSN/nnUNetPlans.json" ]; then
  "$ENVBIN/nnUNetv2_plan_experiment" -d $DS > "$E/pjh_p2_plan.log" 2>&1 || er "plan 실패"
fi
$PY - "$nnUNet_preprocessed/$DSN/nnUNetPlans.json" <<'PYEOF' || er "plan 수정 실패"
import json,sys
p=sys.argv[1]; d=json.load(open(p)); c=d["configurations"]["3d_fullres"]
before=(list(c["spacing"]), list(c["patch_size"]), c["batch_size"])
c["spacing"]=[0.4,0.4,0.4]; c["patch_size"]=[112,160,128]; c["batch_size"]=2
json.dump(d,open(p,"w"),indent=2)
print(f"  plan 수정  spacing {before[0]} -> {c['spacing']}")
print(f"             patch   {before[1]} -> {c['patch_size']}")
print(f"             batch   {before[2]} -> {c['batch_size']}")
print(f"             정규화  {c.get('normalization_schemes')}")
PYEOF
st "P2 완료 · plan 확정"

st "P3 시작 · 전처리 (0.4mm iso, ~50GB, -np 6, 2~3시간)"
DID=$($PY -c "import json;print(json.load(open('$nnUNet_preprocessed/$DSN/nnUNetPlans.json'))['configurations']['3d_fullres']['data_identifier'])")
if [ "$(ls "$nnUNet_preprocessed/$DSN/$DID"/*.b2nd 2>/dev/null | wc -l)" -lt 417 ]; then
  "$ENVBIN/nnUNetv2_preprocess" -d $DS -c 3d_fullres -np 6 > "$E/pjh_p3_pp.log" 2>&1 || er "전처리 실패 — $E/pjh_p3_pp.log"
fi
np=$(ls "$nnUNet_preprocessed/$DSN/$DID"/*.b2nd 2>/dev/null | wc -l)
[ "$np" -lt 417 ] && er "전처리 $np/417 — 부족"
st "P3 완료 · 전처리 $np 건 · $(du -shx "$nnUNet_preprocessed/$DSN/$DID" | cut -f1) · SSD 여유 $(df -h / | tail -1 | awk '{print $4}')"

st "P4 시작 · split 주입 (fold0 val = 공식 val 42)"
$PY - "$R" "$nnUNet_preprocessed/$DSN/splits_final.json" <<'PYEOF' || er "split 주입 실패"
import json,sys,os
sys.path.insert(0,os.path.join(sys.argv[1],"code/sblee/nnunet/scripts"))
os.environ.setdefault("TOPANEU_ROOT",sys.argv[1])
import d9xx_lib as L
tr,va,te=L.case_ids_by_split()
# fold0 = 공식 val 42 를 held-out 으로. test 83 은 어느 폴드에도 안 넣는다.
sp=[{"train":sorted(tr),"val":sorted(va)}]
json.dump(sp,open(sys.argv[2],"w"),indent=2)
print(f"  fold0  train {len(tr)} / val {len(va)}  · test {len(te)} 는 제외")
print(f"  겹침 검사  train∩val {len(set(tr)&set(va))} · train∩test {len(set(tr)&set(te))}")
PYEOF
st "P4 완료"

st "P5 시작 · fold0 학습 ($TR, 250epoch, GPU0, 12~16시간)"
RES="$E/$EXP/results"
CK="$RES/$DSN/${TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
if [ ! -f "$CK" ]; then
  GPU=0 NPROC=6 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
    bash "$S/run_experiment.sh" $DS 3d_fullres 0 "$EXP" -tr "$TR" \
    > "$E/pjh_p5_train.log" 2>&1
fi
[ -f "$CK" ] || er "학습 미완 — $E/pjh_p5_train.log"
st "P5 완료 · 체크포인트 생성"

st "P6 시작 · val 42 추론 + c7 필터 + A6-2 fold0 대비 비교"
OUT="$P/aneu_val_pjh3cls"
if [ "$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)" -lt 42 ]; then
  nnUNet_results="$RES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
    -i "$P/in_val" -o "$OUT" -d $DS -c 3d_fullres -f 0 -tr "$TR" \
    -chk checkpoint_best.pth --continue_prediction -npp 2 -nps 2 \
    > "$E/pjh_p6_pred.log" 2>&1 || er "추론 실패"
fi
# 3-class 출력에서 동맥류(라벨2)만 뽑아 이진화 — c7 은 이진 마스크를 받는다
BIN="$P/aneu_val_pjh3cls_bin"
$PY - "$OUT" "$BIN" <<'PYEOF' || er "라벨2 추출 실패"
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2(동맥류) 추출 {n}건 -> {dst}")
PYEOF
$PY -u c7_detect_postproc.py --aneu-dir "$BIN" --vessel-dir "$P/vespp_val" \
    --split val --tag pjh3cls_val > "$E/pjh_p6_c7.txt" 2>&1 || er "c7 실패"
st "P6 완료"
echo "=== pjh 재현 fold0 (val 42) ==="   | tee -a "$STATUS"
sed -n '/min_vox/,$p' "$E/pjh_p6_c7.txt" | head -12 | tee -a "$STATUS"
echo "=== A6-2 fold0 (대조) ==="          | tee -a "$STATUS"
sed -n '/min_vox/,$p' "$E/q_g1b_aneu_val_a62.txt" 2>/dev/null | head -12 | tee -a "$STATUS"
st "ALL DONE · 게이트: recall 상승 + FP 2배 이내면 5폴드 확대"
