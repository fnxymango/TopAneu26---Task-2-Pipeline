#!/usr/bin/env bash
# D1 — 검출기 TTA×8 (2026-08-21). 재학습 없음, 추론만 바꾼다.
#
# 왜 지금: 팀 문서(REIMPLEMENT_jslee_pjh.md) 대조로 우리 검출기의 약점이 특정됐다.
#   부피구간별 recall (test83)  <5: 0.000(n3) · 5-15: 0.286(n14) · 15-40: 0.767 · >=40: 0.923
#   pjh Stage1 (전split)        <5: 0.208     · 5-15: 0.672      · 15-40: 0.871 · >=40: 0.969
#   전 구간에서 밀리고 5-15mm3 에서 2.3배 벌어진다. 저쪽 곡선이면 63 -> ~74병변(recall 0.859).
#   E8 표의 0.860 행 = +0.0504(FP동일) / +0.0321(FP2배).
#
#   설정 차이 중 **재학습 없이 옮길 수 있는 것은 TTA 하나**다. 저쪽은 5폴드+TTAx8,
#   우리는 전 추론이 --disable_tta 다. E14 에서 잰 TTA 는 **분류기** TTA 였고
#   검출기 TTA 는 한 번도 안 쟀다. 체크포인트 그대로 쓴다.
#   미러축 확인함: inference_allowed_mirroring_axes=(0,1,2) -> 8회 뒤집기. 학습때 미러링 증강 씀.
#
# 판정선 (결과 보기 전 고정):
#   1단계 검출  recall 이 오르고 FP 가 2배 이내  -> e2e 로 진행 (T13 §6 기준)
#   2단계 e2e   T16 설정 5시드에서 t>2.5 & 5/5   -> 채택
#   어느 하나라도 미달이면 기각하고 현행 유지.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"; A="$R/code/sblee/nnunet/analysis"
M="$E/_c16_a62_5fold/results"
ENVBIN="$HOME/miniconda3/envs/sblee_topaneu/bin"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
log(){ echo "[d1 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

# ── STEP1: 5폴드 확률평균 + TTAx8 ─────────────────────────────────────
predict(){ # $1=split $2=gpu
  local SP=$1 G=$2
  local OUT="$P/aneu_${SP}_tta"
  local n=$(ls "$OUT"/*.nii.gz 2>/dev/null | wc -l)
  local need=$(ls "$P/in_$SP"/*_0000.nii.gz 2>/dev/null | wc -l)
  [ "$n" -ge "$need" ] && { log "  $SP 이미 완료 ($n)"; return; }
  log "  $SP 추론 시작 GPU$G (TTAx8, 5폴드) — 기존 $n/$need"
  nnUNet_results="$M" CUDA_VISIBLE_DEVICES="$G" "$ENVBIN/nnUNetv2_predict" \
    -i "$P/in_$SP" -o "$OUT" -d 720 -c 3d_fullres -f 0 1 2 3 4 \
    -tr nnUNetTrainerTverskyTopkCE -p nnUNetResEncUNetLPlansAdaptive \
    -chk checkpoint_best.pth --continue_prediction -npp 2 -nps 2 \
    > "$E/d1_pred_$SP.log" 2>&1
  log "  $SP 종료 status=$? ($(ls "$OUT"/*.nii.gz 2>/dev/null|wc -l)/$need)"
}
log "=== STEP1: TTAx8 추론 (test=GPU0, val=GPU1 병렬) ==="
predict test 0 & PT=$!
predict val  1 & PV=$!
wait $PT; wait $PV

for SP in test val; do
  need=$(ls "$P/in_$SP"/*_0000.nii.gz 2>/dev/null | wc -l)
  got=$(ls "$P/aneu_${SP}_tta"/*.nii.gz 2>/dev/null | wc -l)
  [ "$got" -lt "$need" ] && { log "★ $SP 추론 미완 $got/$need — 중단"; exit 1; }
done

# ── STEP2: c7 필터 (프로덕션과 동일 5,3) ──────────────────────────────
log "=== STEP2: c7 필터 (min_vox 5 · dist 3mm) ==="
for SP in test val; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  $PY -u c7_detect_postproc.py --aneu-dir "$P/aneu_${SP}_tta" --vessel-dir "$P/$VES" \
      --split "$SP" --tag "d1tta_$SP" --save-best "$P/aneu_${SP}_ttaf" \
      --force-cfg 5,3 > "$E/d1_c7_$SP.txt" 2>&1
  log "  $SP c7 완료 → aneu_${SP}_ttaf"
done

# ── STEP3: 검출 게이트 ────────────────────────────────────────────────
log "=== STEP3: 검출 비교 (TTA 켬 vs 현행) ==="
for SP in test val; do
  VES=$([ "$SP" = val ] && echo vespp_val || echo vespp_test)
  BASE=$([ "$SP" = val ] && echo aneu_val_probavgf || echo aneu_test_probavgf)
  $PY -u c7_detect_postproc.py --aneu-dir "$P/$BASE" --vessel-dir "$P/$VES" \
      --split "$SP" --tag "d1base_$SP" --force-cfg 999,999 > "$E/d1_c7base_$SP.txt" 2>&1
done
log "  --- test: TTA 켬 ---";  sed -n '/min_vox/,$p' "$E/d1_c7_test.txt"  | head -4
log "  --- test: 현행 ---";    sed -n '/min_vox/,$p' "$E/d1_c7base_test.txt" | head -4
log "  --- val:  TTA 켬 ---";  sed -n '/min_vox/,$p' "$E/d1_c7_val.txt"   | head -4
log "  --- val:  현행 ---";    sed -n '/min_vox/,$p' "$E/d1_c7base_val.txt"  | head -4
log "  ※ 게이트: recall 상승 + FP 2배 이내면 STEP4 진행. 사람이 확인할 것."

# ── STEP4: e2e — 검출 게이트를 사람이 통과시킨 뒤에만 돈다 ────────────
if [ "${D1_RUN_E2E:-0}" != "1" ]; then
  log "=== STEP1~3 까지만. e2e 는 D1_RUN_E2E=1 일 때만 돈다. ==="
  exit 0
fi
log "=== STEP4: e2e T16 설정 5시드 (test) ==="
FEAT="$A/e11_feat_hyb_ov.json"
PIDS=()
for SD in 0 1 2 3 4; do
  T="d1_tta_s${SD}"
  [ -f "$A/c5_eval_test_${T}.json" ] && continue
  CLF_SEED=$SD nohup $PY -u c5_location_v2.py eval \
    --train-feat "$FEAT" --split test \
    --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
    --aneurysm-pred-dir "$P/aneu_test_ttaf" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$T" > "$E/d1_e2e_s${SD}.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done

log "=== 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_test_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,ofc=d.get("adjusted_div_present"),d.get("official_div52")
        if c and ofc: o[sd]=(c["MCC"],ofc["MCC"])
    return o
base,g=load("e11_hyb_ov"),load("d1_tta")
a=np.array([base[s][0] for s in sorted(base)])
print(f"\n[test 83] 기준 T16(검출 TTA 끔)  n={len(a)}  cov.MCC {a.mean():.4f} ± {a.std():.4f}")
sds=sorted(set(g)&set(base))
if len(sds)<2:
    print(f"  검출 TTA 켬: 데이터 부족 n={len(sds)}")
else:
    ga=np.array([g[s][0] for s in sds]); ba=np.array([base[s][0] for s in sds]); d=ga-ba
    oa=np.array([g[s][1] for s in sds])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"  검출 TTAx8 켬  n={len(sds)}  cov.MCC {ga.mean():.4f} ± {ga.std():.4f}  off {oa.mean():.4f}")
    print(f"                 Δ {d.mean():+.4f} ± {d.std(ddof=1):.4f}  t={t:+.2f}  {nw}/{len(d)}승  "
          f"{'★채택후보' if ok else '기각'}")
    print(f"                 시드별 Δ {[round(x,4) for x in d]}")
PYEOF
log "=== D1 완료 ==="
