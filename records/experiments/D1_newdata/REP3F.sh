#!/usr/bin/env bash
# REP3F — 동료 보고서(min_vox 12) 를 **제출본과 같은 3폴드 검출기**에서 재현 (2026-09-18 · 사용자 지시)
#
#  왜 재현하나
#      우리 `b1ff` 는 검출기 **5폴드**이고 제출 컨테이너는 ENV `TOPANEU_DET_FOLDS=0,1,2` 로 **3폴드**다.
#      오늘 FPCHAR 로 우리 5폴드 FP 를 해부하니 **blob 크기 중앙 292복셀**(동료 보고 85복셀)이고
#      min_vox 12 로 지워지는 FP 단위가 **test 0.0 / val 1.0** 뿐이었다. 즉 우리 근사에는 지울
#      작은 쓰레기가 없다. 폴드를 적게 평균할수록 작은 허위 blob 이 남으므로, **폴드 수가 원인일 수
#      있다** — 이건 추측이라 재현으로 가른다.
#
#  재료 (전부 로컬에 있다)
#      검출기 = 제출 모델 tar 에서 꺼낸 **그 체크포인트**
#        topaneu-26-task2-integrated-model.tar.gz → Dataset722.../nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres
#        PlainConvUNet · iso 0.4mm · patch [112,160,128] · fold_0~4 (md5 8ce281d3/2d9e1dad/2fc5d472)
#      입력 = `_c1_realpred/in_{test,val}_722` (제출 파이프라인과 같은 robust-z 정규화본)
#      혈관 = `vespp_{test,val}` (fold0) — 동료 조건 "seg model(1-fold)" 과 일치. 팔 사이 불변.
#      분류 = 구 학습표 + FRAC 0.35 + OUT_GROW 1.32 (어제 채택분을 켠 상태)
#
# ── 판정규칙 (결과 보기 전 고정 · 2026-09-18 00:20 KST) ─────────────────────
#  [A단계 관문 · 기전]
#    3폴드 출력에 **12복셀 미만 blob 이 실제로 더 많은가**. k=12 에서 제거되는 blob 이 0 이면
#    재현 자체가 성립하지 않으므로 **B단계를 돌리지 않고** 보고하고 멈춘다.
#
#  [B단계 주판정]  ※ K0 는 이 축의 주판정이 아니다 — 근거를 남긴다:
#    K0 주판정은 "적중 순증 단위가 늘었는가" 다. FP 제거 축은 설계상 적중을 늘리지 않으므로
#    (오른 0 · 내린 n) 이 되어 **항상 기각된다**. 장치를 축에 맞게 고른다. K0 표는 보조로 찍는다.
#
#    ① **test·val 양쪽에서 평균 ΔPRECISION > 0** — FP 제거가 기전이므로 여기가 안 오르면 실패다
#    ② **test·val 양쪽에서 평균 ΔMCC ≥ 0** — 잃은 TP 가 지운 FP 보다 크면 여기서 음수가 된다
#    ③ 안전: 어느 쪽에서도 **ΔRECALL ≥ −0.01** — 진짜 병변을 크게 자르지 않았다는 확인
#    ①∧②∧③ 을 모두 만족해야 시드 5~9 복제로 간다. 복제도 만족해야 "채택 권고"(반영은 사용자 결정).
#    비교 대상은 **같은 3폴드 기준팔**이다(5폴드와 섞어 비교하지 않는다).
#
#  참고 목표치(동료 보고 · test): PRECISION 0.5613→0.6139 · MCC 0.5975→0.6360 · FP 50→44 · TP 58 불변
# ─────────────────────────────────────────────────────────────────────────
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; H=$E/H1_patchfilter; V=$E/V1_vessel_axis
SC=/tmp/scratch
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
ENVBIN=$HOME/miniconda3/envs/sblee_topaneu/bin
DETRES=$SC/det3f/models/detector
export TOPANEU_ROOT=$R PYTHONPATH=$S:$R/code/sblee:$R/code/sblee/nnunet
export nnUNet_raw=$R/nnunet/nnUNet_raw nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
exec 9>"$D/rep3f.lock"; flock -n 9 || { echo "이미 실행 중"; exit 0; }
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][rep3f] $*" | tee -a "$D/STATUS.log"; }
FEAT=$A/e11_feat_hyb_ov.json
SEEDS="0 1 2 3 4"
MV=12

# ============================================================ A단계 · 3폴드 검출
if [ ! -f "$D/.done_rep3f_det" ]; then
  log "A · 3폴드 검출 (제출 체크포인트 · folds 0,1,2)"
  ( nnUNet_results="$DETRES" CUDA_VISIBLE_DEVICES=0 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_test_722" -o "$P/aneu_test_b3" -d 722 -c 3d_fullres -f 0 1 2 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 > "$D/rep3f_pred_test.log" 2>&1 ) & T=$!
  ( nnUNet_results="$DETRES" CUDA_VISIBLE_DEVICES=1 "$ENVBIN/nnUNetv2_predict" \
      -i "$P/in_val_722" -o "$P/aneu_val_b3" -d 722 -c 3d_fullres -f 0 1 2 \
      -p nnUNetPlans -tr nnUNetTrainer_250epochs -chk checkpoint_best.pth \
      --disable_tta -npp 2 -nps 2 > "$D/rep3f_pred_val.log" 2>&1 ) & W=$!
  wait $T || { log "★test 검출 실패"; exit 1; }
  wait $W || { log "★val 검출 실패"; exit 1; }
  touch "$D/.done_rep3f_det"; log "A · 검출 완료"
fi

# 라벨2 이진화 + c7 (제출 설정 5,1.0) — B1.sh 와 같은 절차
cd "$S" || exit 1
for sp in test val; do
  [ -d "$P/aneu_${sp}_b3ff" ] && [ "$(ls -1 $P/aneu_${sp}_b3ff 2>/dev/null|wc -l)" -gt 0 ] && continue
  "$PY" - "$P/aneu_${sp}_b3" "$P/aneu_${sp}_b3_bin" <<'PYEOF' >> "$D/rep3f_c7.log" 2>&1
import sys,os,glob,numpy as np,nibabel as nib
src,dst=sys.argv[1],sys.argv[2]; os.makedirs(dst,exist_ok=True); n=0
for f in sorted(glob.glob(os.path.join(src,"*.nii.gz"))):
    i=nib.load(f); a=np.asanyarray(i.dataobj)
    o=nib.Nifti1Image((a==2).astype(np.uint8),i.affine,i.header); o.set_data_dtype(np.uint8)
    nib.save(o,os.path.join(dst,os.path.basename(f))); n+=1
print(f"  라벨2 추출 {n}건")
PYEOF
  "$PY" -u c7_detect_postproc.py --aneu-dir "$P/aneu_${sp}_b3_bin" --vessel-dir "$P/vespp_${sp}" \
      --split $sp --tag b3_${sp} --save-best "$P/aneu_${sp}_b3ff" --force-cfg "5,1.0" \
      >> "$D/rep3f_c7.log" 2>&1 || { log "★c7 $sp 실패"; exit 1; }
done
log "A · c7(5,1.0) 완료"

# min_vox 12 팔 생성 + ★관문
"$PY" - "$MV" > "$V/RESULTS_REP3F_A.md" 2>&1 <<'PYEOF'
import json, os, sys, collections
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; P=f"{R}/experiments/_c1_realpred"
G=f"{R}/dataset/TopAneu/location_masks"; ST=np.ones((3,3,3),bool); MV=int(sys.argv[1])
sp_ids=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
KS=[5,8,10,12,15,20,25,30]
print("# REP3F-A — 3폴드 검출 blob 해부 (규칙은 REP3F.sh 머리말 고정)\n")
print("| split | 검출기 | blob | 12복셀 미만 | blob 크기 중앙 | GT 적중 |")
print("|---|---|---|---|---|---|")
gate=0
for split in ("test","val"):
    for tag,lab in (("b1ff","5폴드(기존)"),("b3ff","**3폴드(제출조건)**")):
        src=f"{P}/aneu_{split}_{tag}"; tot=0; small=0; sizes=[]; hit=0; nles=0
        for cid in sp_ids[split]:
            sh=cid.replace("topaneu_","")
            p=next((c for c in (f"{src}/{cid}.nii.gz",f"{src}/{sh}.nii.gz",f"{src}/topaneu_{sh}.nii.gz") if os.path.exists(c)),None)
            if p is None: continue
            a=np.asanyarray(nib.load(p).dataobj)>0
            l,n=ndi.label(a,structure=ST); tot+=n
            for i in range(1,n+1):
                s=int((l==i).sum()); sizes.append(s); small += s<MV
            gp=f"{G}/{cid}.nii.gz" if os.path.exists(f"{G}/{cid}.nii.gz") else f"{G}/topaneu_{sh}.nii.gz"
            gl,gn=ndi.label(np.asanyarray(nib.load(gp).dataobj)>0,structure=ST); nles+=gn
            for i in range(1,gn+1): hit+=bool(a[gl==i].any())
        med=int(np.median(sizes)) if sizes else 0
        print(f"| {split} | {lab} | {tot} | **{small}** | {med} | {hit}/{nles} |")
        if tag=="b3ff": gate+=small
print()
print("## 3폴드 min_vox 스윕 (blob 단위 · 참고)\n")
print("| k | test blob | test 적중 | val blob | val 적중 |")
print("|---|---|---|---|---|")
res={}
for split in ("test","val"):
    src=f"{P}/aneu_{split}_b3ff"
    per={k:[0,0] for k in KS}
    for cid in sp_ids[split]:
        sh=cid.replace("topaneu_","")
        p=next((c for c in (f"{src}/{cid}.nii.gz",f"{src}/{sh}.nii.gz",f"{src}/topaneu_{sh}.nii.gz") if os.path.exists(c)),None)
        if p is None: continue
        a=np.asanyarray(nib.load(p).dataobj)>0
        l,n=ndi.label(a,structure=ST)
        sz=np.array([0]+[int((l==i).sum()) for i in range(1,n+1)])
        gp=f"{G}/{cid}.nii.gz" if os.path.exists(f"{G}/{cid}.nii.gz") else f"{G}/topaneu_{sh}.nii.gz"
        gl,gn=ndi.label(np.asanyarray(nib.load(gp).dataobj)>0,structure=ST)
        for k in KS:
            keep=[i for i in range(1,n+1) if sz[i]>=k]
            m=np.isin(l,keep)
            per[k][0]+=len(keep)
            for i in range(1,gn+1): per[k][1]+=bool(m[gl==i].any())
    res[split]=per
for k in KS:
    print(f"| {k} | {res['test'][k][0]} | {res['test'][k][1]} | {res['val'][k][0]} | {res['val'][k][1]} |")
print()
print(f"**관문 → {'통과 — 3폴드에 12복셀 미만 blob 이 %d 개 있다. B단계 진행.'%gate if gate>0 else '미달 — 12복셀 미만 blob 이 0 개다. 재현 불가, B단계 생략.'}**")
open(f"{R}/experiments/D1_newdata/.rep3f_gate","w").write(str(gate))
PYEOF
GATE=$(cat "$D/.rep3f_gate" 2>/dev/null || echo 0)
log "A · 관문 결과 → 3폴드 12복셀 미만 blob $GATE 개"
if [ "$GATE" -le 0 ]; then log "★관문 미달 — B단계 생략"; touch "$D/.done_rep3f"; exit 0; fi

# min_vox 12 적용본 생성
"$PY" - "$MV" >> "$D/rep3f_c7.log" 2>&1 <<'PYEOF'
import json,os,sys,numpy as np,nibabel as nib
from scipy import ndimage as ndi
R="/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"; P=f"{R}/experiments/_c1_realpred"
ST=np.ones((3,3,3),bool); MV=int(sys.argv[1])
ids=json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
for split in ("test","val"):
    src=f"{P}/aneu_{split}_b3ff"; dst=f"{P}/aneu_{split}_b3ff{MV}"; os.makedirs(dst,exist_ok=True)
    for cid in ids[split]:
        sh=cid.replace("topaneu_","")
        p=next((c for c in (f"{src}/{cid}.nii.gz",f"{src}/{sh}.nii.gz",f"{src}/topaneu_{sh}.nii.gz") if os.path.exists(c)),None)
        if p is None: continue
        im=nib.load(p); a=np.asanyarray(im.dataobj)
        l,n=ndi.label(a>0,structure=ST)
        keep=[i for i in range(1,n+1) if (l==i).sum()>=MV]
        nib.save(nib.Nifti1Image(np.isin(l,keep).astype(np.int16),im.affine,im.header),
                 f"{dst}/{os.path.basename(p)}")
    print(f"  {split}: min_vox {MV} 적용본 생성")
PYEOF
log "A · min_vox $MV 적용본 생성 완료"

# ============================================================ B단계 · e2e 두 팔
run_arm(){ local TAG=$1 SUF=$2
  # 경로 규약: aneu_{split}_{suffix} — 2026-09-18 09:54 실패 원인이 이 순서를 뒤집은 것이었다
  for sp in test val; do [ -d "$P/aneu_${sp}_${SUF}" ] || { log "★검출 디렉터리 없음: $P/aneu_${sp}_${SUF}"; return 1; }; done
  log "B · c5 · $TAG (검출 aneu_{split}_$SUF) · 시드 $SEEDS · FRAC 0.35 + GROW 1.32"
  clf(){ local sp=$1 sd=$2 bp exp n
    bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
    exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
    TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
    TOPANEU_TOPK_FRAC=0.35 TOPANEU_OUT_GROW=1.32 TOPANEU_OUT_DILATE=0 \
    CLF_SEED=$sd OMP_NUM_THREADS=1 \
    "$PY" -u "$S/c5_location_v2.py" eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/aneu_${sp}_${SUF}" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --save-pred-dir "$H/pred/${TAG}_${sp}_s${sd}" --tag "rep3f_${TAG}_${sp}_s${sd}" \
      > "$H/logs/${TAG}_${sp}_s${sd}.log" 2>&1
  }
  local n=0; for sp in test val; do for sd in $SEEDS; do clf $sp $sd & n=$((n+1)); [ $((n%10)) -eq 0 ] && wait; done; done; wait
  for sp in test val; do exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    for sd in $SEEDS; do c=$(ls -1 "$H/pred/${TAG}_${sp}_s${sd}" 2>/dev/null|wc -l)
      [ "$c" = "$exp" ] || { log "★c5 누락 ${TAG} ${sp}_s${sd} ($c/$exp)"; return 1; }; done; done
  log "B · 패치필터 · $TAG"
  pf(){ local sp=$1 sd=$2 exp n; exp=$( [ "$sp" = test ] && echo 83 || echo 41 )
    n=$(ls -1 "$H/pred/${TAG}_pf_${sp}_s${sd}" 2>/dev/null | wc -l); [ "$n" = "$exp" ] && return 0
    "$PY" "$H/pf_dir.py" --pred "$H/pred/${TAG}_${sp}_s${sd}" --vessel "$P/vespp_${sp}" \
      --image "$P/in_${sp}" --split "$sp" --out "$H/pred/${TAG}_pf_${sp}_s${sd}" --quiet \
      --report "$H/logs/pf_${TAG}_${sp}_s${sd}.json" > "$H/logs/pf_${TAG}_${sp}_s${sd}.log" 2>&1
  }
  n=0; for sp in test val; do for sd in $SEEDS; do pf $sp $sd & n=$((n+1)); [ $((n%3)) -eq 0 ] && wait; done; done; wait
  log "B · 채점 · $TAG"
  n=0; for sp in val test; do for sd in $SEEDS; do
    d="$H/pred/${TAG}_pf_${sp}_s${sd}"; [ -d "$d" ] || continue
    grep -q '"F1"' "$H/scores/${TAG}_pf_${sp}_s${sd}.json" 2>/dev/null && continue
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" "$D/neweval2.py" "$sp" "$d" >> "$D/rep3f_score.log" 2>&1 &
    n=$((n+1)); [ $((n%10)) -eq 0 ] && wait
  done; done; wait
  return 0
}

run_arm b3fg    b3ff       || { log "★기준팔 실패"; exit 1; }
run_arm b3fg12  b3ff${MV}  || { log "★mv12 팔 실패"; exit 1; }

{ echo "# REP3F-B — 3폴드 조건에서 min_vox 12 재현 (시드 0~4) · 규칙은 REP3F.sh 머리말 고정"; echo
  echo "기준팔 \`b3fg\` = 3폴드 검출 + c7(5,1.0) + FRAC 0.35 + GROW 1.32"
  echo "후보팔 \`b3fg12\` = 위와 동일하되 검출 blob 을 **12복셀 미만 제거**"; echo
  echo "## 7지표 (주판정 — ① ΔPRECISION>0 ② ΔMCC≥0 ③ ΔRECALL≥−0.01, test·val 양쪽)"; echo
  BASE=b3fg_pf TAG=b3fg12_pf SEEDS=0,1,2,3,4 LABEL_BASE="3폴드 기준" LABEL_TAG="min_vox 12" \
    "$PY" "$D/metrics_diff.py"
  echo; echo "## K0 (보조 · 이 축의 주판정 아님 — 머리말 근거 참조)"; echo
  K0_SEEDS=0,1,2,3,4 "$PY" "$D/k0_judge.py" b3fg_pf b3fg12_pf
} > "$V/RESULTS_REP3F_B.md" 2>&1
touch "$D/.done_rep3f"
log "REP3F 끝 → $V/RESULTS_REP3F_B.md"
