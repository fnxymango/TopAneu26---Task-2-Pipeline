#!/usr/bin/env bash
# C27 — 검출 동작점(확률 임계)을 MCC 기준으로 고른다 (2026-08-17). **한 스크립트**로.
#
# 근거: C26 에서 official MCC 0.2238 · covered-GT MCC 0.2863 (test). 차이 0.0625 가
# **검출이 놓친 병변의 값**이다(test 커버리지 0.733). 지금 단일 최대 손실이 여기다.
# 확률맵은 이미 저장돼 있으므로 재학습 없이 임계만 낮춰 커버리지를 살 수 있다.
# 임계를 낮추면 FP 가 늘어 precision 이 깎이므로 MCC 최적점이 어디인지는 재봐야 안다.
#
# GPU 는 test 확률맵 추론에, CPU 는 val 평가에 동시에 쓴다 (GPU 유휴 제거).
# 대기 조건에 pgrep 를 절대 쓰지 않는다 — 산출물 개수로만 판정(앞서 두 번 정체시킨 원인).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
NN="$HOME/miniconda3/envs/sblee_topaneu/bin/nnUNetv2_predict"
M5="$E/_c16_a62_5fold/results"
FEAT="$A/c10_feat_train.json"; BETA=0.5
THS="0.15,0.2,0.3,0.4,0.5"
log(){ echo "[c27 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── GPU: test 확률맵 추론을 먼저 띄워 val 평가와 겹친다 ────────────────
# 재기동 시 추론이 이미 돌고 있으면 또 띄우지 않는다 — 로그 mtime 으로 판정(pgrep 금지).
PLOG="$E/c27_pred_test.log"
PRUN=0
NOW=$(date +%s); NEWEST=0
for f in "$PLOG" "$P/aneu_test_prob" "$P"/aneu_test_prob/*; do
  [ -e "$f" ] || continue
  m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
  [ "$m" -gt "$NEWEST" ] && NEWEST=$m
done
[ "$NEWEST" -gt 0 ] && [ $(( NOW - NEWEST )) -lt 900 ] && PRUN=1
if [ "$(ls "$P"/aneu_test_prob/*.npz 2>/dev/null | wc -l)" -lt 83 ] && [ "$PRUN" = 0 ]; then
  log "GPU: test 5-fold 확률맵 추론 시작 (백그라운드, val 평가와 병렬)"
  nnUNet_results="$M5" CUDA_VISIBLE_DEVICES=0 nohup "$NN" -i "$P/in_test" -o "$P/aneu_test_prob" \
    -d 720 -c 3d_fullres -f 0 1 2 3 4 -tr nnUNetTrainerTverskyTopkCE \
    -p nnUNetResEncUNetLPlansAdaptive -chk checkpoint_best.pth --disable_tta --continue_prediction \
    --save_probabilities -npp 2 -nps 2 > "$PLOG" 2>&1 &
  disown
elif [ "$PRUN" = 1 ]; then
  log "test 확률맵 추론이 이미 진행 중 (산출물 갱신 $(( NOW - NEWEST ))s 전) — 재시작 안 함"
else
  log "test 확률맵 이미 있음 ($(ls "$P"/aneu_test_prob/*.npz | wc -l))"
fi

ev(){ # $1=split $2=tag $3=aneu디렉토리
  local sp=$1 tag=$2 ad=$3 VD VB
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; else VD=vespp_test; VB=vespp_test; fi
  [ -d "$ad" ] || { log "    $ad 없음 — 건너뜀"; return; }
  if [ ! -f "$A/c5_eval_${sp}_${tag}.json" ]; then
    log "  official  $sp $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$ad" \
      --model rf --use-pos --beta "$BETA" --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
  fi
  if [ ! -f "$A/c5_eval_${sp}_${tag}_cov.json" ]; then
    log "  coveredGT $sp $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
      --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$ad" --covered-gt \
      --model rf --use-pos --beta "$BETA" --tag "${tag}_cov" 2>&1 \
      | grep -E '"MCC"|"detect_coverage"' | head -2 || log "    실패"
  fi
}

# ── STEP1: val 임계 마스크 생성 ────────────────────────────────────────
log "=== STEP1: val 확률 임계 스윕 (마스크 생성) ==="
if [ "$(ls "$P"/_c27_val/th0.3/*.nii.gz 2>/dev/null | wc -l)" -lt 40 ]; then
  $PY -u c20_prob_threshold.py --prob-dir "$P/aneu_val_prob" --vessel-dir "$P/vespp_val" \
     --split val --ref-dir "$P/aneu_val_probavg" --out-root "$P/_c27_val" \
     --thresholds "$THS" 2>&1 | tail -12 || log "  스윕 실패"
else
  log "  마스크 이미 있음"
fi

# ── STEP2: val 임계별 평가 ────────────────────────────────────────────
log "=== STEP2: val 임계별 평가 (β=$BETA, official + covered-GT) ==="
for T in 0.15 0.2 0.3 0.4 0.5; do ev val "c27_th$T" "$P/_c27_val/th$T"; done
ev val "c27_base" "$P/aneu_val_probavgf"      # 현행 동작점 (비교 기준)

BT=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
best=(None,-1)
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_c27_th*.json")):
    if f.endswith("_cov.json"): continue
    m=re.search(r"c27_th([0-9.]+)\.json$",f)
    if not m: continue
    o=json.load(open(f)).get("official_div52") or {}
    if o.get("MCC",-1)>best[1]: best=(m.group(1),o["MCC"])
b=os.path.join(sys.argv[1],"c5_eval_val_c27_base.json")
bm=(json.load(open(b)).get("official_div52") or {}).get("MCC",0) if os.path.exists(b) else 0
print(best[0] if best[1]>bm else "none")
PYEOF
)
log "=== val 최적 임계: $BT (현행 동작점 대비) ==="

# ── STEP3: test 확률맵 대기 → 마스크 → 평가 ───────────────────────────
if [ "$BT" != "none" ]; then
  log "=== STEP3: test 확률맵 대기 ==="
  for i in $(seq 1 240); do
    n=$(ls "$P"/aneu_test_prob/*.npz 2>/dev/null | wc -l)
    [ "$n" -ge 83 ] && { log "  test 확률맵 $n/83 완료"; break; }
    [ $((i % 10)) -eq 0 ] && log "  대기중 $n/83 (${i}분)"
    sleep 60
  done
  n=$(ls "$P"/aneu_test_prob/*.npz 2>/dev/null | wc -l)
  if [ "$n" -ge 83 ]; then
    log "=== STEP3: test 임계 마스크 + 평가 ==="
    $PY -u c20_prob_threshold.py --prob-dir "$P/aneu_test_prob" --vessel-dir "$P/vespp_test" \
       --split test --ref-dir "$P/aneu_test_probavg" --out-root "$P/_c27_test" \
       --thresholds "$THS" 2>&1 | tail -12 || log "  스윕 실패"
    ev test "c27_th$BT" "$P/_c27_test/th$BT"
  else
    log "  test 확률맵 미완 ($n/83) — test 평가 생략"
  fi
else
  log "임계 조정이 현행보다 낫지 않음 — test 생략"
fi
ev test "c27_base" "$P/aneu_test_probavgf"

# ── STEP4: 요약 ───────────────────────────────────────────────────────
log "=== STEP4: 요약 (official / covered-GT) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
for sp in ("val","test"):
    rows=[]
    for f in sorted(glob.glob(os.path.join(sys.argv[1],f"c5_eval_{sp}_c27_*.json"))):
        if f.endswith("_cov.json"): continue
        tag=os.path.basename(f)[len(f"c5_eval_{sp}_"):-5]
        o=json.load(open(f)).get("official_div52")
        cf=f[:-5]+"_cov.json"; cd=json.load(open(cf)) if os.path.exists(cf) else {}
        c=cd.get("official_div52")
        if not o: continue
        rows.append((tag,o,c,cd.get("top1_accuracy"),cd.get("detect_coverage")))
    if not rows: continue
    print(f"\n===== {sp} =====")
    print(f"{'설정':<14}{'off.MCC':>9}{'off.DICE':>10}{'off.복합':>10}"
          f"{'cov.MCC':>10}{'cov.복합':>10}{'cov.top1':>10}{'커버':>7}")
    for tag,o,c,t1,cov in sorted(rows,key=lambda r:-r[1]['MCC']):
        cs=f"{c['MCC']:>10.4f}{comp(c):>10.4f}" if c else f"{'-':>10}{'-':>10}"
        print(f"{tag:<14}{o['MCC']:>9.4f}{o['DICE']:>10.4f}{comp(o):>10.4f}{cs}"
              f"{(t1 or 0):>10.3f}{(cov or 0):>7.3f}")
PYEOF
log "=== 완료 ==="
