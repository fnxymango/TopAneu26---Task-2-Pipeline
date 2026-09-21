#!/usr/bin/env bash
# 최종 설정 + C11 합성 통합 평가 (2026-08-16)
# 설정: dist+ov+bp+pos / RF / beta=1.0 / C11 합성 x2 / 검출 = 5-fold vote2 + c7필터
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"; A="$R/code/sblee/nnunet/analysis"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
BP="$E/_c4_bpgraph"; P="$E/_c1_realpred"
FEAT="$A/c10_feat_train.json"; SYN="$A/c11_synth_rows.json"
BEST="--model rf --beta 1.0 --use-pos --synth-feat $SYN --synth-repeat 2"
log(){ echo "[fin $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
log "설정: $BEST"

for SP in val test; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  VB=$([ "$SP" = val ] && echo val_pred || echo vespp_test)
  log "GT천장 $SP"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$SP" \
    --vessel-dir "$P/$VES" --bp-dir "$BP/$VB" $BEST --tag "c11_ceil" 2>&1 | tail -18
  log "e2e $SP (vote2 + c7필터)"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$SP" \
    --vessel-dir "$P/$VES" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$P/aneu_${SP}_vote2f" \
    $BEST --tag "c11_e2e_vote2" 2>&1 | tail -18
done

log "=== 최종 비교표 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
rows=[]
for pat in ("c5_eval_*_best_*.json","c5_eval_*_ens_*.json","c5_eval_*_c11_*.json"):
    for f in glob.glob(os.path.join(sys.argv[1],pat)):
        d=json.load(open(f)); a=d.get("adjusted_div_present") or {}; o=d.get("official_div52") or {}
        rows.append((os.path.basename(f).replace("c5_eval_","").replace(".json",""),
                     d.get("top1_accuracy"), d.get("n_lesions_predicted"),
                     a.get("MCC",0), o.get("MCC",0), a.get("DICE",0)))
rows.sort(key=lambda r:(r[0].split('_')[0], -r[3]))
print(f"{'실험':<30}{'top-1':>8}{'병변':>6}{'MCC(adj)':>10}{'MCC(공식)':>11}{'DICE(adj)':>11}")
for n,t,l,m,oo,dd in rows:
    print(f"{n:<30}{(f'{t:.3f}' if t else '-'):>8}{l:>6}{m:>10.4f}{oo:>11.4f}{dd:>11.4f}")
PYEOF
log "완료"
