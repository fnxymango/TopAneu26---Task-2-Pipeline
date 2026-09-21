#!/usr/bin/env bash
# C5 체인 — 분기점 기반 위치분류 전체 파이프라인 (2026-08-14, rev2)
#
# rev2 변경점: c7 검출 후처리(혈관거리 게이팅 + 성분크기)로 필터한 검출기 arm 추가.
#   c7 스윕 결과 A5-2는 FP 55->12(민감도 손실 0), A6-2는 152->53(손실 0)이었으므로
#   필터 전/후를 같은 그리드에서 비교한다.
# **test는 val 그리드가 전부 끝난 뒤 최고 조합으로 딱 한 번만 만진다** (선택 편향 방지).
set -uo pipefail
R="${TOPANEU_ROOT:?}"
S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"; A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
BP="$E/_c4_bpgraph"; P="$E/_c1_realpred"
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
FEAT="$A/c5_feat_train.json"; SWEEP="$A/c5_sweep_val.jsonl"

log() { echo "[c5 $(date -u +'%H:%M:%S')] $*"; }
cd "$S" || exit 1
export TOPANEU_ROOT="$R"

# --- 1) train 피처 ---
if [ ! -s "$FEAT" ]; then
  log "STEP1: train 피처 빌드 (참조 혈관)"
  $PY c5_location_v2.py build --split train --vessel-dir "$REFVES" --bp-dir "$BP/all_ref" --out "$FEAT"
fi
log "STEP1 완료 (train 피처 $(stat -c%s "$FEAT") bytes)"

# --- 2) val 평가용 피처 (GT 병변, 혈관 소스별) ---
declare -A VESDIR=( [refves]="$REFVES" [pp]="$P/vespp_val" [raw]="$P/ves_val" )
declare -A BPDIR=(  [refves]="$BP/all_ref" [pp]="$BP/val_pred" [raw]="$BP/ves_val" )
for src in refves pp raw; do
  f="$A/c5_feat_val_${src}.json"
  [ -s "$f" ] || { log "STEP2: val 피처 ($src)"; \
    $PY c5_location_v2.py build --split val --vessel-dir "${VESDIR[$src]}" --bp-dir "${BPDIR[$src]}" --out "$f"; }
done
log "STEP2 완료"

# --- 3) 하이퍼파라미터 스윕 (예측 pp 혈관 기준, top-1만) ---
log "STEP3: 스윕"
: > "$SWEEP"
for k in 1 3 5 10 20; do
  for bal in "" "--no-balance"; do
    for mir in "" "--no-mirror"; do
      $PY c5_location_v2.py eval --train-feat "$FEAT" --split val \
        --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --eval-feat "$A/c5_feat_val_pp.json" \
        --model knn --k "$k" $bal $mir --fast --tag "k${k}" 2>&1 | grep '^{' >> "$SWEEP"
    done
  done
done
$PY c5_location_v2.py eval --train-feat "$FEAT" --split val \
  --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --eval-feat "$A/c5_feat_val_pp.json" \
  --model rf --fast --tag rf 2>&1 | grep '^{' >> "$SWEEP"
log "STEP3 스윕 상위:"
$PY - "$SWEEP" <<'EOF'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows.sort(key=lambda r:-(r["top1_accuracy"] or 0))
print(f"{'model':>6}{'k':>4}{'mirror':>8}{'balance':>9}{'top-1':>9}")
for r in rows[:10]:
    print(f"{r['model']:>6}{r['k']:>4}{str(r['mirror']):>8}{str(r['balance']):>9}{r['top1_accuracy']:>9.3f}")
EOF
BEST=$($PY - "$SWEEP" <<'EOF'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
b=max(rows,key=lambda r:r["top1_accuracy"] or 0)
f=f"--model {b['model']} --k {b['k']}"
if not b["balance"]: f+=" --no-balance"
if not b["mirror"]:  f+=" --no-mirror"
print(f)
EOF
)
log "STEP3 최적 설정: $BEST"

# --- 4) val 천장 (GT 병변) — 혈관 소스별. 오라클 61.0%와 직접 비교 ---
for src in refves pp raw; do
  log "STEP4: val GT-ceiling / 혈관=$src"
  $PY c5_location_v2.py eval --train-feat "$FEAT" --split val \
    --vessel-dir "${VESDIR[$src]}" --bp-dir "${BPDIR[$src]}" $BEST --tag "ceil_$src" 2>&1 | tail -24
done

# --- 5) val 엔드투엔드 그리드 (검출기 x 혈관) ---
#   A52/A62 = 원본, A52f/A62f = c7 필터(거리게이팅+크기) 적용
for arm in "A52:aneu_val:pp" "A62:aneu_val_a62:pp" \
           "A52f:aneu_val_a52f:pp" "A62f:aneu_val_a62f:pp" \
           "A52f:aneu_val_a52f:raw" "A62f:aneu_val_a62f:raw"; do
  an=$(echo "$arm"|cut -d: -f1); ad=$(echo "$arm"|cut -d: -f2); vn=$(echo "$arm"|cut -d: -f3)
  [ -d "$P/$ad" ] || { log "  스킵 $an: $ad 없음"; continue; }
  log "STEP5: val e2e $an x $vn"
  $PY c5_location_v2.py eval --train-feat "$FEAT" --split val \
    --vessel-dir "${VESDIR[$vn]}" --bp-dir "${BPDIR[$vn]}" --aneurysm-pred-dir "$P/$ad" \
    $BEST --tag "e2e_${an}_${vn}" 2>&1 | tail -24
done

log "STEP5 완료 — val 그리드 요약:"
$PY - "$A" <<'EOF'
import json,glob,os,sys
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_*.json")):
    if "SMOKE" in f: continue
    d=json.load(open(f))
    adj=d.get("adjusted_div_present") or {}
    rows.append((os.path.basename(f).replace("c5_eval_val_","").replace(".json",""),
                 d.get("top1_accuracy"), d.get("n_lesions_predicted"),
                 adj.get("MCC"), (d.get("official_div52") or {}).get("MCC")))
rows.sort(key=lambda r:-(r[3] or 0))
print(f"{'arm':<20}{'top-1':>8}{'병변':>6}{'MCC(adj)':>10}{'MCC(공식)':>10}")
for n,t,l,m,o in rows:
    print(f"{n:<20}{(f'{t:.3f}' if t is not None else '-'):>8}{l:>6}"
          f"{(f'{m:.4f}' if m else '-'):>10}{(f'{o:.4f}' if o else '-'):>10}")
EOF

# --- 6) test — val 최고 e2e 조합으로 딱 한 번 ---
BESTCFG=$($PY - "$A" <<'EOF'
import json,glob,os,sys
best=None
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_e2e_*.json")):
    d=json.load(open(f)); m=(d.get("adjusted_div_present") or {}).get("MCC",0)
    if best is None or m>best[0]: best=(m,os.path.basename(f))
print(best[1].replace("c5_eval_val_e2e_","").replace(".json","") if best else "")
EOF
)
AN=${BESTCFG%_*}; VN=${BESTCFG##*_}
case "$AN" in
  A52)  AD=aneu_test ;;      A62)  AD=aneu_test_a62 ;;
  A52f) AD=aneu_test_a52f ;; A62f) AD=aneu_test_a62f ;;  *) AD="" ;;
esac
[ "$VN" = "pp" ] && { VD=vespp_test; VB=vespp_test; } || { VD=ves_test; VB=ves_test; }
if [ -n "$AD" ] && [ -d "$P/$AD" ] && [ "$(ls "$P/$AD" | wc -l)" -ge 83 ]; then
  log "STEP6: test e2e $AN x $VN (aneu=$AD, vessel=$VD) — test 첫 접촉"
  $PY c5_location_v2.py eval --train-feat "$FEAT" --split test \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/$AD" \
    $BEST --tag "e2e_${AN}_${VN}" 2>&1 | tail -24
else
  log "STEP6 보류: $AD 준비 안 됨 (val 최고=$BESTCFG)"
fi
log "C5 체인 완료"
