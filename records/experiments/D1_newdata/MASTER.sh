#!/usr/bin/env bash
# 새 데이터셋(2026-09-01 릴리스) 수신 → 대조 → 재빌드 → 전처리 → 검출기·혈관 병렬학습.
# 세션과 무관하게 끝까지 돈다. 각 단계는 멱등(이미 끝난 단계는 건너뜀).
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts
OLD=$R/dataset/TopAneu; NEW=$R/dataset/TopAneu_new; ARCH=$R/dataset/TopAneu_old_20260801
D=$E/D1_newdata; ST=$D/STATUS.log
PY2=$HOME/miniconda3/envs/sbaneu2/bin/python
BIN1=$HOME/miniconda3/envs/sblee_topaneu/bin      # 스톡 nnunetv2 — 검출기 722
BIN2=$HOME/miniconda3/envs/sbaneu2/bin            # vendored Skeleton-Recall — 혈관 800
export TOPANEU_ROOT=$R
export nnUNet_raw=$R/nnunet/nnUNet_raw
export nnUNet_preprocessed=$R/nnunet/nnUNet_preprocessed
export TOPANEU_DATA=$R/dataset/TopAneu
export PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet:${PYTHONPATH:-}
DSN722=Dataset722_TopAneuPjh3cls417
DSN800=Dataset800_TopAneuVessel417
PLANS722=nnUNetResEncUNetLPlans722iso04
TR722=nnUNetTrainer_250epochs
TR800=nnUNetTrainerSkeletonRecallNoMirroringClassWeighted_500ep
BK=$R/nnunet/preproc_json_backup

log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST] $*" | tee -a "$ST"; }
die(){ log "★중단: $*"; exit 1; }
mark(){ touch "$D/.done_$1"; }
have(){ [ -f "$D/.done_$1" ]; }

mkdir -p "$nnUNet_preprocessed" "$nnUNet_raw" "$E"
# 중복 실행 구조적 차단 — 감시견이 오판해도 두 번째 인스턴스는 즉시 빠진다
exec 9>"$D/master.lock"
flock -n 9 || { echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST] MASTER 이미 실행중 — 중복 기동 취소" >> "$ST"; exit 0; }
echo $$ > "$D/MASTER.pid"
log "================ MASTER 시작 (PID $$) ================"

# ---------------------------------------------------------------- P0 다운로드 완료 대기
if ! have p0; then
  log "P0 · 다운로드 완료 대기"
  for round in 1 2 3 4; do
    while ! grep -q "=== 완료" "$NEW/.listing/fetch.log" 2>/dev/null; do sleep 60; done
    n=$(find "$NEW"/images "$NEW"/location_masks "$NEW"/vessel_masks \
             "$NEW"/type_masks "$NEW"/location_jsons -type f 2>/dev/null | wc -l)
    log "  라운드 $round 종료 · $n / 2075 파일"
    [ "$n" -ge 2075 ] && break
    log "  누락 재시도"
    rm -f "$NEW/.listing/failed.txt"
    : > "$NEW/.listing/fetch.log"
    bash "$NEW/.listing/fetch.sh" "$NEW" >> "$NEW/.listing/fetch.log" 2>&1
  done
  n=$(find "$NEW"/images "$NEW"/location_masks "$NEW"/vessel_masks \
           "$NEW"/type_masks "$NEW"/location_jsons -type f 2>/dev/null | wc -l)
  [ "$n" -ge 2075 ] || die "다운로드 $n/2075 — 4라운드 후에도 미완"
  # 크기 전수 검증
  bad=$($PY2 - "$NEW" <<'PY'
import sys,os
NEW=sys.argv[1]; bad=0
for l in open(f"{NEW}/.listing/manifest.tsv"):
    p,sz=l.rstrip("\n").split("\t")
    f=f"{NEW}/{p}"
    if not os.path.exists(f) or os.path.getsize(f)!=int(sz): bad+=1; print("크기불일치",p)
print("BAD",bad)
PY
)
  echo "$bad" | tail -3 | tee -a "$ST"
  echo "$bad" | grep -q "^BAD 0$" || die "파일 크기 검증 실패"
  log "P0 완료 · 2075 파일 · $(du -sh "$NEW" | cut -f1)"
  mark p0
fi

# ---------------------------------------------------------------- P1 구/신 전수 대조
if ! have p1; then
  log "P1 · md5 + 복셀 수준 전수 대조 (20~40분)"
  DIFF_OUT=$D/diff.json $PY2 -u "$D/diff2.py" "$OLD" "$NEW" > "$D/DIFF_REPORT.txt" 2>&1 \
    || die "대조 실패 — $D/DIFF_REPORT.txt"
  tail -14 "$D/DIFF_REPORT.txt" | tee -a "$ST"
  mark p1
fi

# ---------------------------------------------------------------- P2 split 갱신 + 교체
if ! have p2; then
  log "P2 · dataset_split.json 갱신 + 디렉터리 교체"
  $PY2 - "$OLD" "$NEW" <<'PY' | tee -a "$ST" || exit 1
import json,sys,glob,os
import numpy as np, nibabel as nib
OLD,NEW=sys.argv[1],sys.argv[2]
sp=json.load(open(f"{OLD}/dataset_split.json"))
present={os.path.basename(p).replace(".nii.gz","") for p in glob.glob(f"{NEW}/location_masks/*.nii.gz")}
drop=[]
for k in ["train","val","test"]:
    keep=[c for c in sp["splits"][k] if c in present]
    drop+= [c for c in sp["splits"][k] if c not in present]
    sp["splits"][k]=keep
    sp["cases"][k]=[c for c in sp["cases"][k] if c["case_id"] in present]
    print(f"  {k}: {len(keep)}건")
print("  제외:",drop)
sp["n_cases"]=sum(len(sp["splits"][k]) for k in ["train","val","test"])
sp["n_patients"]=len({c["patient_id"] for k in ["train","val","test"] for c in sp["cases"][k]})
# 존재 클래스 재계산 (라벨 수정으로 바뀔 수 있다 — 추론하지 않고 스캔한다)
seen=set()
for f in sorted(glob.glob(f"{NEW}/location_masks/*.nii.gz")):
    seen|={int(v) for v in np.unique(np.asanyarray(nib.load(f).dataobj)) if v}
old_pres=set(sp.get("location_classes_present_in_data",[]))
sp["location_classes_present_in_data"]=sorted(seen)
sp["location_classes_absent_from_data"]=sorted(set(range(1,53))-seen)
print(f"  존재 클래스 {len(old_pres)} -> {len(seen)}   추가 {sorted(seen-old_pres)}  소멸 {sorted(old_pres-seen)}")
sp["notes"]=sp.get("notes",[])+[f"2026-09-01 릴리스 반영 · 제외 {drop}"]
json.dump(sp,open(f"{NEW}/dataset_split.json","w"),indent=1,ensure_ascii=False)
print(f"  n_cases {sp['n_cases']} · n_patients {sp['n_patients']}")
PY
  [ -f "$NEW/dataset_split.json" ] || die "split 생성 실패"
  [ -d "$ARCH" ] || mv "$OLD" "$ARCH" || die "구데이터 보관 실패"
  [ -d "$OLD" ] || mv "$NEW" "$OLD" || die "신데이터 배치 실패"
  log "P2 완료 · 구=$ARCH · 신=$OLD ($(ls "$OLD/images" | wc -l) images)"
  mark p2
fi

# ---------------------------------------------------------------- P3 nnUNet_raw 재생성
if ! have p3; then
  log "P3 · nnUNet_raw 재생성 (722 검출기 · 800 혈관)"
  # 720 은 격리하면 안 된다 — c7/c5/c17/c20 이 labelsTr 을 GT 병변마스크로 읽는다. 재생성 대상이다.
  for ds in Dataset700_TopAneuRegion417 Dataset710_TopAneuLoc417 \
            Dataset810_TopAneuVesselUZH417 Dataset900_TopAneuLoc417; do
    [ -d "$nnUNet_raw/$ds" ] && [ ! -d "$nnUNet_raw/${ds}_STALE20260801" ] && \
      mv "$nnUNet_raw/$ds" "$nnUNet_raw/${ds}_STALE20260801" && log "  구버전 격리: $ds"
  done
  rm -rf "$nnUNet_raw/$DSN722" "$nnUNet_raw/$DSN800" "$nnUNet_raw/Dataset720_TopAneuBinary417"
  cd "$S" || die "cd 실패"
  $PY2 -u p_build_3cls.py > "$D/build722.log" 2>&1 || die "722 빌드 실패 — $D/build722.log"
  n=$(ls "$nnUNet_raw/$DSN722/labelsTr"/*.nii.gz 2>/dev/null | wc -l)
  log "  722 라벨 $n / 이미지 $(ls "$nnUNet_raw/$DSN722/imagesTr"/*.nii.gz | wc -l)"
  [ "$n" -ge 415 ] || die "722 라벨 $n — 부족"
  cd "$R" && $PY2 -u rebuild_417/build_datasets_417.py 800 720 > "$D/build800.log" 2>&1 \
    || die "800/720 빌드 실패 — $D/build800.log"
  n8=$(ls "$nnUNet_raw/$DSN800/labelsTr"/*.nii.gz 2>/dev/null | wc -l)
  log "P3 완료 · 800 라벨 $n8"
  [ "$n8" -ge 415 ] || die "800 라벨 $n8 — 부족"
  mark p3
fi

# ---------------------------------------------------------------- P4 검출기 전처리
if ! have p4; then
  log "P4 · 722 fingerprint + 전처리 (0.4mm iso, ~2.5시간)"
  "$BIN1/nnUNetv2_extract_fingerprint" -d 722 -np 8 > "$D/fp722.log" 2>&1 || die "722 fingerprint 실패"
  # plans 는 구버전을 그대로 복원한다 — 아키텍처가 같아야 구/신 비교가 성립한다
  cp "$BK/$DSN722/$PLANS722.json" "$nnUNet_preprocessed/$DSN722/$PLANS722.json" || die "plans 복원 실패"
  log "  plans 복원: $PLANS722 (spacing 0.4iso · patch 112x160x128)"
  # plan_experiment 를 건너뛰었으므로 dataset.json 이 복사되지 않는다 — 전처리가 이걸 읽는다
  cp "$nnUNet_raw/$DSN722/dataset.json" "$nnUNet_preprocessed/$DSN722/dataset.json" || die "dataset.json 복사 실패"
  DID=$($PY2 -c "import json;print(json.load(open('$nnUNet_preprocessed/$DSN722/$PLANS722.json'))['configurations']['3d_fullres']['data_identifier'])")
  "$BIN1/nnUNetv2_preprocess" -d 722 -c 3d_fullres -plans_name "$PLANS722" -np 6 > "$D/pp722.log" 2>&1 \
    || die "722 전처리 실패 — $D/pp722.log"
  np=$(ls "$nnUNet_preprocessed/$DSN722/$DID"/*.b2nd 2>/dev/null | wc -l)
  log "P4 완료 · 전처리 $np 건 · $(du -shx "$nnUNet_preprocessed/$DSN722/$DID" | cut -f1) · 여유 $(df -h / | tail -1 | awk '{print $4}')"
  [ "$np" -ge 415 ] || die "722 전처리 $np — 부족"
  $PY2 "$R/rebuild_417/make_splits_417.py" 722 --train-only >> "$D/pp722.log" 2>&1 || die "722 split 실패"
  $PY2 -c "
import json;s=json.load(open('$nnUNet_preprocessed/$DSN722/splits_final.json'))
print('  splits '+str(len(s))+'폴드  '+' '.join('f%d:tr%d/va%d'%(i,len(f['train']),len(f['val'])) for i,f in enumerate(s)))" | tee -a "$ST"
  mark p4
fi

# ---------------------------------------------------------------- P5 검출기 5폴드 · GPU 2장 분할
# P1 대조 결과 vessel_masks 실질 변경 1/415 → 혈관 재학습 불필요.
# GPU1 을 혈관 대신 검출기 폴드에 붙여 30h -> 18h 로 줄인다.
mk_det(){   # $1=gpu  $2..=folds
  local g=$1; shift
  cat > "$D/det_gpu$g.sh" <<TEOF
#!/usr/bin/env bash
set -uo pipefail
R=$R; D=\$R/experiments/D1_newdata; ST=\$D/STATUS.log
export TOPANEU_ROOT=\$R nnUNet_raw=\$R/nnunet/nnUNet_raw nnUNet_preprocessed=\$R/nnunet/nnUNet_preprocessed
export PYTHONPATH=\$R/code/sblee:\$R/code/sblee/nnunet:\${PYTHONPATH:-}
EXP=P5_newdata_detector
CK=\$R/experiments/\$EXP/results/$DSN722/${TR722}__${PLANS722}__3d_fullres
log(){ echo "[\$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][det-gpu$g] \$*" | tee -a "\$ST"; }
echo \$\$ > "\$D/det_gpu$g.pid"
for f in $*; do
  if [ -f "\$CK/fold_\$f/checkpoint_final.pth" ]; then log "fold\$f 이미 완료"; continue; fi
  CONT=""
  if [ -f "\$CK/fold_\$f/checkpoint_latest.pth" ]; then CONT="--c"; log "fold\$f 중단지점부터 이어받음(--c)"; else log "fold\$f 학습 시작"; fi
  GPU=$g NPROC=1 ENVBIN=\$HOME/miniconda3/envs/sblee_topaneu/bin TOPANEU_ROOT=\$R \\
    bash "\$R/code/sblee/nnunet/scripts/run_experiment.sh" 722 3d_fullres \$f "\$EXP" \\
    -p $PLANS722 -tr $TR722 \$CONT >> "\$D/det_f\$f.log" 2>&1
  if [ -f "\$CK/fold_\$f/checkpoint_final.pth" ]; then
    log "fold\$f 완료 · \$(grep -oE 'Mean Validation Dice: [0-9.]+' "\$D/det_f\$f.log" | tail -1)"
  else
    log "★fold\$f 실패 — \$D/det_f\$f.log"; exit 1
  fi
done
log "GPU$g 담당 폴드($*) 전부 완료"
touch "\$D/.done_det_gpu$g"
TEOF
  chmod +x "$D/det_gpu$g.sh"
}
if ! have launched_p5; then
  log "P5 · 검출기 722 5폴드 · GPU0=fold0,1,2 / GPU1=fold3,4 (약 18시간)"
  mk_det 0 0 1 2
  mk_det 1 3 4
  setsid nohup "$D/det_gpu0.sh" > "$D/det_gpu0.out" 2>&1 </dev/null & disown
  setsid nohup "$D/det_gpu1.sh" > "$D/det_gpu1.out" 2>&1 </dev/null & disown
  mark launched_p5
  log "  검출기 2체인 기동 (GPU0 · GPU1)"
fi

log "================ MASTER 배치 완료 ================"
log "혈관 800: 재학습 생략 (vessel_masks 변경 1/415) · 필요시 run_d800_classweighted_retrain.sh 로 별도 기동"
log "감시: tail -f $ST"
