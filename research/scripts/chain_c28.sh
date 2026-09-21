#!/usr/bin/env bash
# C28 — 검출 필터를 느슨하게 해서 산 병변이 MCC에 이득인지 (2026-08-17).
#
# C27 에서 확인: 확률 임계만 낮춰서는 민감도가 1도 안 오른다(0.15~0.5 전부 33/43).
# 필터를 뗀 원본 확률맵의 천장은 th0.05 에서 35/43=0.814, 다만 FP 11 -> 38.
#   c7 필터(min_vox 5, dist 3mm) 는 th0.5 에서 민감도 손실 0으로 FP 8개를 없앤다 — 그 자체는 옳다.
#   th0.15 에서는 34 -> 33 으로 1병변을 깎는다.
# 남은 질문은 **중간 지점**이다. MCC 는 병변단위(1복셀 겹침=TP)라 새 TP 하나가
# 그 클래스 MCC 를 크게 올리는 반면 FP 는 여러 클래스에 흩어진다. 그래서 재봐야 안다.
#
# test 는 건드리지 않는다 — C27 이 test 경로를 쓰고 있어 충돌한다. 승자는 나중에 한 번만 test.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"; BETA=0.5
log(){ echo "[c28 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=tag $2=aneu디렉토리
  local tag=$1 ad=$2
  [ -d "$ad" ] || { log "  $ad 없음 — 건너뜀"; return; }
  if [ ! -f "$A/c5_eval_val_${tag}.json" ]; then
    log "  official  val $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split val \
      --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --aneurysm-pred-dir "$ad" \
      --model rf --use-pos --beta "$BETA" --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
  fi
  if [ ! -f "$A/c5_eval_val_${tag}_cov.json" ]; then
    log "  coveredGT val $tag"
    $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split val \
      --vessel-dir "$P/vespp_val" --bp-dir "$BP/val_pred" --aneurysm-pred-dir "$ad" --covered-gt \
      --model rf --use-pos --beta "$BETA" --tag "${tag}_cov" 2>&1 \
      | grep -E '"MCC"|"detect_coverage"' | head -2 || log "    실패"
  fi
}

log "=== STEP1: 느슨한 필터 마스크 생성 (min_vox 2, dist 5mm) ==="
if [ "$(ls "$P"/_c28_relax/th0.15/*.nii.gz 2>/dev/null | wc -l)" -lt 40 ]; then
  $PY -u c20_prob_threshold.py --prob-dir "$P/aneu_val_prob" --vessel-dir "$P/vespp_val" \
     --split val --ref-dir "$P/aneu_val_probavg" --out-root "$P/_c28_relax" \
     --thresholds 0.05,0.15,0.3 --min-vox 2 --max-dist 5.0 2>&1 | tail -8 || log "  생성 실패"
else
  log "  이미 있음"
fi

log "=== STEP2: 평가 (β=$BETA, official + covered-GT) ==="
ev c28_nf0.05 "$P/_c27_nofilter/th0.05"     # 필터 완전 제거, 천장 동작점
ev c28_nf0.15 "$P/_c27_nofilter/th0.15"
ev c28_rx0.05 "$P/_c28_relax/th0.05"        # 중간 지점
ev c28_rx0.15 "$P/_c28_relax/th0.15"
ev c28_rx0.3  "$P/_c28_relax/th0.3"

log "=== STEP3: 요약 (C27 결과와 함께, official / covered-GT) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for pat in ("c5_eval_val_c27_*.json","c5_eval_val_c28_*.json","c5_eval_val_c26_b0.5.json"):
    for f in sorted(glob.glob(os.path.join(sys.argv[1],pat))):
        if f.endswith("_cov.json"): continue
        tag=os.path.basename(f)[len("c5_eval_val_"):-5]
        o=json.load(open(f)).get("official_div52")
        cf=f[:-5]+"_cov.json"; cd=json.load(open(cf)) if os.path.exists(cf) else {}
        c=cd.get("official_div52")
        if o: rows.append((tag,o,c,cd.get("top1_accuracy"),cd.get("detect_coverage")))
print(f"\n{'설정':<16}{'off.MCC':>9}{'off.DICE':>10}{'off.복합':>10}"
      f"{'cov.MCC':>10}{'cov.복합':>10}{'cov.top1':>10}{'커버':>7}")
for tag,o,c,t1,cov in sorted(rows,key=lambda r:-r[1]['MCC']):
    cs=f"{c['MCC']:>10.4f}{comp(c):>10.4f}" if c else f"{'-':>10}{'-':>10}"
    print(f"{tag:<16}{o['MCC']:>9.4f}{o['DICE']:>10.4f}{comp(o):>10.4f}{cs}"
          f"{(t1 or 0):>10.3f}{(cov or 0):>7.3f}")
PYEOF
log "=== 완료 ==="
