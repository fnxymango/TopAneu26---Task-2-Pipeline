#!/usr/bin/env bash
# 케이스당 소요시간 측정 (2026-08-21). "가장 늦은 케이스" 를 알아야 제출 시간제한을 판단한다.
# 평균이 아니라 **최악 케이스**가 기준이므로, 볼륨이 큰 것과 작은 것을 섞어 3케이스만 잰다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"; AN="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
ENVBIN2="$HOME/miniconda3/envs/sbaneu2/bin"
W="$E/_timing"; rm -rf "$W"; mkdir -p "$W"/{in,det,ves,vespp,bp,aneu}
log(){ echo "[time $(date -u +'%H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# 1) test 입력 중 볼륨 최대·중앙·최소 3개
log "=== 케이스 선정 (볼륨 기준) ==="
mapfile -t SEL < <($PY - "$P/in_test" <<'PYEOF'
import sys,glob,os,nibabel as nib
fs=sorted(glob.glob(os.path.join(sys.argv[1],"*_0000.nii.gz")))
sz=[]
for f in fs:
    sh=nib.load(f).shape
    sz.append((sh[0]*sh[1]*sh[2], os.path.basename(f)[:-12], sh))
sz.sort()
for lab,i in (("최소",0),("중앙",len(sz)//2),("최대",len(sz)-1)):
    v,n,sh=sz[i]; print(n)
    print(f"  {lab} {n} {sh} {v/1e6:.1f}Mvox", file=sys.stderr)
PYEOF
)
log "  선정: ${SEL[*]}"
for c in "${SEL[@]}"; do cp "$P/in_test/${c}_0000.nii.gz" "$W/in/"; done

T(){ local lab=$1; shift; local t0=$(date +%s.%N); "$@" > "$W/${lab}.log" 2>&1; local rc=$?
     local t1=$(date +%s.%N); printf "%-28s %8.1f 초  %s\n" "$lab" "$(echo "$t1-$t0"|bc)" "$([ $rc -eq 0 ]&&echo ok||echo 실패)" | tee -a "$W/RESULT.txt"; }

DTR=nnUNetTrainerTverskyTopkCE; DPL=nnUNetResEncUNetLPlansAdaptive
log "=== 검출 5폴드 (${#SEL[@]}케이스) ==="
for i in 0 1 2 3 4; do
  RES="$E/A6-2_resencl_adaptivenorm_topk_417_f$i/results"
  T "검출 fold$i" env nnUNet_results="$RES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$W/in" -o "$W/det/f$i" -d 720 -c 3d_fullres -f $i -tr $DTR -p $DPL \
      -chk checkpoint_best.pth --disable_tta --save_probabilities -npp 2 -nps 2
done
log "=== 혈관 1폴드 ==="
VRES="$E/V4-2_vessel_classweighted_417_500ep/results"
T "혈관 V4-2 fold0" env nnUNet_results="$VRES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN2/nnUNetv2_predict" \
    -i "$W/in" -o "$W/ves" -d 800 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep -p nnUNetResEncUNetMPlans \
    -chk checkpoint_final.pth --disable_tta -npp 2 -nps 2
log "=== 후처리·그래프·분류 ==="
T "혈관 V5 후처리" $PY -u postprocess_vessel.py --in-dir "$W/ves" --out-dir "$W/vespp" \
    --params "$S/postproc_params.json"
T "분기점 그래프 c4" $PY -u c4_branchpoint_graph.py --vessel-dir "$W/vespp" --out "$W/bp"
echo "  (검출 확률평균 c16 · c7 필터 · 분류기는 별도 측정)" | tee -a "$W/RESULT.txt"
log "=== 측정표 ==="; cat "$W/RESULT.txt"
