#!/usr/bin/env bash
# 최고 설정 전체 실험 (2026-08-15)
# 설정: dist+ov+bp+pos / RF / beta=1.0  (C10 CV에서 macro-recall 0.455, 오늘 최고)
#   C9  beta 사전확률 역보정  : macro-recall 0.346 -> 0.410
#   C10 랜드마크 정규화 좌표  : 0.410 -> 0.455
#   기각된 것 — C13 좌우canonical(0.352), C12 그룹전문가(0.360), C9 rel/반경확대
# GT 천장(혈관소스 3종) + 엔드투엔드(A/V 최고 조합)를 val/test 양쪽에서 낸다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"; A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
BP="$E/_c4_bpgraph"; P="$E/_c1_realpred"
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
FEAT="$A/c10_feat_train.json"
BEST="--model rf --beta 1.0 --use-pos"
log(){ echo "[best $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "설정: $BEST | train피처 $FEAT"

# --- GT 천장: 혈관 소스 3종 (val) ---
for a in "refves:$REFVES:$BP/all_ref" "pp:$P/vespp_val:$BP/val_pred" "raw:$P/ves_val:$BP/ves_val"; do
  n=${a%%:*}; vd=$(echo "$a"|cut -d: -f2); vb=$(echo "$a"|cut -d: -f3)
  log "GT천장 val / 혈관=$n"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split val \
    --vessel-dir "$vd" --bp-dir "$vb" $BEST --tag "best_ceil_$n" 2>&1 | tail -20
done

# --- GT 천장 (test) — 분류단계 단독 성능 ---
log "GT천장 test / 혈관=pp"
$PY -u c5_location_v2.py eval --train-feat "$FEAT" --split test \
  --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" $BEST --tag "best_ceil_pp" 2>&1 | tail -20

# --- 엔드투엔드: A 최고(A5-2 + c7필터) x V 최고(V4-2 + V5후처리) ---
for s in val test; do
  [ "$s" = val ] && { AD=aneu_val_a52f; VD=vespp_val; VB=val_pred; } \
                 || { AD=aneu_test_a52f; VD=vespp_test; VB=vespp_test; }
  log "e2e $s / A5-2필터 x V4-2pp"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$s" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/$AD" \
    $BEST --tag "best_e2e_A52f_pp" 2>&1 | tail -20
done

log "=== 최종 요약 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_*_best_*.json"))):
    d=json.load(open(f)); a=d.get("adjusted_div_present") or {}; o=d.get("official_div52") or {}
    rows.append((os.path.basename(f).replace("c5_eval_","").replace(".json",""),
                 d.get("top1_accuracy"), d.get("n_lesions_predicted"), a.get("MCC"), o.get("MCC"), a.get("DICE")))
print(f"{'실험':<34}{'top-1':>8}{'병변':>6}{'MCC(adj)':>10}{'MCC(공식)':>11}{'DICE(adj)':>11}")
for n,t,l,m,oo,dd in rows:
    print(f"{n:<34}{(f'{t:.3f}' if t else '-'):>8}{l:>6}{(f'{m:.4f}' if m else '-'):>10}"
          f"{(f'{oo:.4f}' if oo else '-'):>11}{(f'{dd:.4f}' if dd else '-'):>11}")
PYEOF
log "완료"
