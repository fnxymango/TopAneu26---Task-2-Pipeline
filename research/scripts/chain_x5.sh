#!/usr/bin/env bash
# X5 — 검출기 5폴드 확대 + e2e 재측정 (2026-08-25). 사용자 승인: "비용 커도 진행해".
#
# 근거: test 83 GT 병변 86개 중 23개 미검출. 부피구간별로 5-15mm^3 에서 10개, 15mm^3 이하에서 13개.
#   pjh 프로파일을 얻으면 11개 회수 -> 오라클 병변당 +0.0092 기준 상한 +0.1008,
#   라벨정확도 0.72 반영 현실치 +0.0726. 분류기 축(Maha +0.0183)보다 한 자릿수 크다.
#
# 함정(이미 한 번 밟았다): Dataset722 는 dataset.json 이 channel_names=noNorm 이다.
#   정규화가 빌드시 들어가 있어 nnUNet 이 추론 입력을 그대로 통과시킨다. 원래 chain_pjh/chain_abl 은
#   -i in_val (원시 강도 mean 71) 을 썼다 -> 학습분포(mean -0.23)와 딴판. 반드시 in_val_722 를 쓴다.
#   이 버그 때문에 chain_pjh 의 P6 와 chain_abl 의 추론 단계를 죽이고 여기서 다시 한다.
#
# 게이트: fold0 val 42 에서 민감도 > A6-2 이고 FP/case <= 2배. 둘 다 미달이면 확대하지 않는다.
#   민감도로 판정하는 이유: 오라클에서 미검출 회수(+0.0092/병변)가 환각 제거(+0.0022/병변)보다
#   4배 값어치가 있다. A7 은 FP 만 절반으로 줄이고 민감도가 그대로여서 버렸다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; A="$R/code/sblee/nnunet/analysis"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
DS=722; DSN="Dataset722_TopAneuPjh3cls417"
PEXP="P1_pjh3cls_stock250_iso04_f0"; PTR=nnUNetTrainer_250epochs
AEXP="P2_pjh3cls_ourloss_iso04_f0";  ATR=nnUNetTrainerTverskyTopkCE
STATUS="$E/x5_status.log"
export nnUNet_raw="$R/nnunet/nnUNet_raw" nnUNet_preprocessed="$R/nnunet/nnUNet_preprocessed"
export TOPANEU_ROOT="$R"
cd "$S" || exit 1
st(){ echo "[STEP] $(TZ=Asia/Seoul date +%H:%M) $*" | tee -a "$STATUS"; }
er(){ echo "[FAIL] $(TZ=Asia/Seoul date +%H:%M) $*" | tee -a "$STATUS"; exit 1; }
ck(){ echo "$E/$1/results/$DSN/$2__nnUNetPlans__3d_fullres/fold_$3/checkpoint_final.pth"; }

val_eval(){ # $1=EXP $2=TR $3=tag $4=gpu   fold0 val 추론 -> 라벨2 -> c7
  local EXP=$1 TR=$2 TAG=$3 G=$4
  local RES="$E/$EXP/results" OUT="$P/aneu_val_$TAG" BIN="$P/aneu_val_${TAG}_bin"
  if [ "$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)" -lt 42 ]; then
    nnUNet_results="$RES" CUDA_VISIBLE_DEVICES=$G "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$OUT" -d $DS -c 3d_fullres -f 0 -tr "$TR" \
      -chk checkpoint_best.pth --continue_prediction -npp 2 -nps 2 \
      > "$E/x5_${TAG}_pred.log" 2>&1 || return 1
  fi
  $PY - "$OUT" "$BIN" <<'PYEOF' || return 1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  $PY -u c7_detect_postproc.py --aneu-dir "$BIN" --vessel-dir "$P/vespp_val" \
      --split val --tag "x5_$TAG" > "$E/x5_${TAG}_c7.txt" 2>&1 || return 1
}

gate(){ # c7 표에서 (민감도 최대, 동률시 FP 최소) 행 -> "sens fp min_vox dist"
  $PY - "$1" <<'PYEOF'
import re,sys
best=None
for L in open(sys.argv[1],errors="replace"):
    m=re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+\((\d+)/(\d+)\)\s+([\d.]+)\s+(\d+)",L)
    if not m: continue
    s=float(m.group(3)); fp=float(m.group(6))
    if best is None or (s,-fp)>(best[0],-best[1]): best=(s,fp,int(m.group(1)),float(m.group(2)))
print(f"{best[0]:.4f} {best[1]:.4f} {best[2]} {best[3]}" if best else "0 0 0 0")
PYEOF
}

# ---------- 1) pjh fold0 재평가 (정규화 입력 수정본) --------------------------
st "X5-1 · P6' pjh fold0 val 재평가 (in_val_722, GPU0)"
val_eval "$PEXP" "$PTR" pjh3cls 0 || er "P6' 실패 — $E/x5_pjh3cls_pred.log"
st "X5-1 완료"

# ---------- 2) ABL 학습 대기 -> fold0 평가 ------------------------------------
st "X5-2 · ABL 학습 완료 대기"
ACK=$(ck "$AEXP" "$ATR" 0)
while [ ! -f "$ACK" ]; do sleep 120; done
st "X5-2 · ABL 학습 완료 · val 재평가 (in_val_722, GPU1)"
val_eval "$AEXP" "$ATR" ablourloss 1 || er "ABL 평가 실패"
st "X5-2 완료"

# ---------- 3) 3자 게이트 -----------------------------------------------------
read BS BF BV BD < <(gate "$E/q_g1b_aneu_val_a62.txt")
read PS PF PV PD < <(gate "$E/x5_pjh3cls_c7.txt")
read AS AF AV AD < <(gate "$E/x5_ablourloss_c7.txt")
{
echo ""
echo "=== 검출 게이트 (val 42 · fold0) ==="
printf "  %-30s %8s %10s  %s\n" 설정 민감도 FP/case "최적 min_vox,dist"
printf "  %-30s %8s %10s  %s\n" "A6-2 현행 (Dataset720)"     "$BS" "$BF" "$BV,$BD"
printf "  %-30s %8s %10s  %s\n" "P5 pjh 스톡손실 (D722)"     "$PS" "$PF" "$PV,$PD"
printf "  %-30s %8s %10s  %s\n" "ABL 우리 손실 (D722)"       "$AS" "$AF" "$AV,$AD"
} | tee -a "$STATUS"

read WIN WEXP WTR < <($PY -c "
b,bf=$BS,$BF; p,pf=$PS,$PF; a,af=$AS,$AF
c=[]
if p>b and pf<=2*bf: c.append((p,-pf,'P5','$PEXP','$PTR'))
if a>b and af<=2*bf: c.append((a,-af,'ABL','$AEXP','$ATR'))
print(' '.join(max(c)[2:]) if c else 'NONE - -')")

if [ "$WIN" = "NONE" ]; then
  st "게이트 미달 — 어느 설정도 A6-2 민감도를 못 넘겼다. 5폴드 확대 중단."
  st "X5 종료 (확대 없음)"
  exit 0
fi
st "게이트 통과 · 승자 = $WIN ($WTR) · 5폴드 확대 진행"

# ---------- 4) split 주입 + folds 1~4 학습 ------------------------------------
cp "$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json" \
   "$R/nnunet/nnUNet_preprocessed/$DSN/splits_final.json" || er "split 주입 실패"
$PY -c "
import json;d=json.load(open('$R/nnunet/nnUNet_preprocessed/$DSN/splits_final.json'))
print('  split 폴드 %d 개 · fold0 val %d (공식 val 42 와 일치해야 함)'%(len(d),len(d[0]['val'])))" | tee -a "$STATUS"

train_fold(){ # $1=fold $2=gpu
  local F=$1 G=$2
  [ -f "$(ck "$WEXP" "$WTR" $F)" ] && { st "fold$F 이미 있음 — 건너뜀"; return 0; }
  st "fold$F 학습 시작 (GPU$G · 250ep · 약 5.5시간)"
  GPU=$G NPROC=4 ENVBIN="$ENVBIN" TOPANEU_ROOT="$R" \
    bash "$S/run_experiment.sh" $DS 3d_fullres $F "$WEXP" -tr "$WTR" \
    > "$E/x5_${WIN}_f${F}.log" 2>&1
  [ -f "$(ck "$WEXP" "$WTR" $F)" ] || { st "[FAIL] fold$F 미완"; return 1; }
  st "fold$F 완료"
}
train_fold 1 0 & T1=$!
train_fold 2 1 & T2=$!
wait $T1; wait $T2
train_fold 3 0 & T3=$!
train_fold 4 1 & T4=$!
wait $T3; wait $T4
st "X5-4 · 5폴드 학습 전부 완료"

# ---------- 5) test 83 e2e 재측정 ---------------------------------------------
st "X5-5 · test 83 5폴드 확률평균 추론"
RES="$E/$WEXP/results"; OUT="$P/aneu_test_${WIN}5f"
if [ "$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)" -lt 83 ]; then
  nnUNet_results="$RES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
    -i "$P/in_test_722" -o "$OUT" -d $DS -c 3d_fullres -f 0 1 2 3 4 -tr "$WTR" \
    -chk checkpoint_best.pth --disable_tta -npp 2 -nps 2 \
    > "$E/x5_test_pred.log" 2>&1 || er "test 추론 실패"
fi
BIN="$P/aneu_test_${WIN}5f_bin"
$PY - "$OUT" "$BIN" <<'PYEOF' || er "라벨2 추출 실패"
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True)
n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
st "X5-5 · c7 필터 (fold0 최적 cfg 를 그대로 적용)"
$PY -u c7_detect_postproc.py --aneu-dir "$BIN" --vessel-dir "$P/vespp_test" \
    --split test --tag "x5_${WIN}_test" --save-best "$P/aneu_test_${WIN}5ff" \
    --force-cfg "$([ "$WIN" = P5 ] && echo "$PV,$PD" || echo "$AV,$AD")" \
    > "$E/x5_test_c7.txt" 2>&1 || er "c7 실패"

st "X5-5 · 분류기 5시드 e2e (T16 설정 · 혈관/그래프/학습피처는 그대로)"
for SD in 0 1 2 3 4; do
  CLF_SEED=$SD OMP_NUM_THREADS=4 $PY -u c5_location_v2.py eval \
    --train-feat "$A/e11_feat_hyb_ov.json" --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
    --aneurysm-pred-dir "$P/aneu_test_${WIN}5ff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "x5_${WIN}_s${SD}" > "$E/x5_e2e_s${SD}.log" 2>&1
done
st "X5-5 완료 · 최종 비교"
$PY - "$A" "x5_${WIN}" <<'PYEOF' 2>&1 | tee -a "$STATUS"
import json,glob,os,sys,re
import numpy as np
A,tag=sys.argv[1],sys.argv[2]
def load(t):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_test_{t}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,of=d.get("adjusted_div_present"),d.get("official_div52")
        if c and of: o[sd]=(c["MCC"],of["MCC"],c.get("Precision"),c.get("Recall"))
    return o
b=load("e11_hyb_ov"); g=load(tag); s=sorted(set(b)&set(g))
print("\n=== e2e 최종 (test 83 · 5시드) ===")
if len(s)<2: print("  데이터 부족"); raise SystemExit
ga=np.array([g[x][0] for x in s]); ba=np.array([b[x][0] for x in s]); d=ga-ba
t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
print(f"  {'설정':<26}{'cov.MCC':>10}{'±':>8}{'off÷52':>9}{'Δ':>10}{'t':>7}{'승':>6}")
print(f"  {'T16 현행 검출기':<26}{ba.mean():>10.4f}{ba.std():>8.4f}{np.mean([b[x][1] for x in s]):>9.4f}")
print(f"  {'X5 새 검출기 5폴드':<26}{ga.mean():>10.4f}{ga.std():>8.4f}{np.mean([g[x][1] for x in s]):>9.4f}"
      f"{d.mean():>+10.4f}{t:>+7.2f}{int((d>0).sum()):>4}/{len(s)}")
print(f"  시드별 Δ {[round(x,4) for x in d]}")
PYEOF
st "X5 전부 끝"
