#!/usr/bin/env bash
# E15 — 292 OOF 를 5판으로 확정하고, overlap 단독안도 292 로 잰다 (2026-08-19).
#
# 상태: E1b-2 3판에서 Δ +0.0133 ± 0.0095, t=+2.42, 3/3승.
#   판정선(t>2.5)에 0.08 모자란다. geo 가 3판 t=2.02 -> 5판 t=0.53 으로 무너진 전례가 있어
#   **결과 보기 전에 정한 기준을 지킨다** — 시드 3,4 를 추가해 5판으로 확정한다.
#
# 동시에 E11 블록 분해 결과를 반영한다 (test, 기준선 대비):
#   중첩36만 예측  +0.0314 (t=4.34, 5/5)   <- 이득의 80%
#   거리36만 예측  +0.0073 (t=0.98, 4/5)
#   분기점34만     -0.0070 (t=-0.86)       <- 음수. 268개 전부 값이 바뀌는데 해롭다
#   전부(E10)     +0.0393 (t=4.68, 5/5)
# overlap 단독이면 변경 범위가 훨씬 작고 분기점의 음수를 피한다. 292 에서도 재본다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REF="$A/c10_feat_train.json"; PRD="$A/c10_feat_train_predves.json"
OVF="$A/e11_feat_hyb_ov.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
log(){ echo "[e15 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

ev(){ # $1=tag $2=feat $3=seed
  local tag="e1b2_${1}_s${3}"
  [ -f "$A/c5_eval_train_$tag.json" ] && { log "  $tag 있음"; return; }
  log "  $tag"
  CLF_SEED=$3 $PY -u c5_location_v2.py eval --train-feat "$2" --split train \
    --cv-splits "$SP720" --vessel-dir "$P/vespp_train" --bp-dir "$BP/vespp_train" \
    --aneurysm-pred-dir "$P/aneu_train_ooff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

log "=== 1) 292 시드 3,4 추가 (참조 vs 예측) ==="
for SD in 3 4; do ev ref "$REF" "$SD"; ev prd "$PRD" "$SD"; done

log "=== 2) overlap 단독안 292 (시드 0~4) ==="
for SD in 0 1 2 3 4; do ev ov "$OVF" "$SD"; done

log "=== 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def L(pat):
    o={}
    for f in glob.glob(os.path.join(A,pat)):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present")
        if c: o[sd]=(c["MCC"],d.get("n_lesions_predicted"),d.get("n_present_classes_in_split"))
    return o
r=L("c5_eval_train_e1b2_ref_s?.json")
if r:
    n,ncl=r[sorted(r)[0]][1],r[sorted(r)[0]][2]
    print(f"\n[292 OOF] 예측병변 {n} · 분모 {ncl}클래스 → 병변 1개 = {1.0/ncl:+.4f}  (test {1.0/36:+.4f})")
print(f"\n{'설정':<16}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ vs 참조':>11}{'t':>7}{'승':>6}  판정")
ra=np.array([r[s][0] for s in sorted(r)])
print(f"{'참조혈관(기준)':<16}{len(ra):>3}{ra.mean():>10.4f}{ra.std():>8.4f}")
for tag,lab in (("prd","전부 예측(E10)"),("ov","중첩만 예측")):
    g=L(f"c5_eval_train_e1b2_{tag}_s?.json"); s=sorted(set(g)&set(r))
    if len(s)<2: print(f"{lab:<16}{len(s):>3}  데이터 부족"); continue
    ga=np.array([g[x][0] for x in s]); d=np.array([g[x][0]-r[x][0] for x in s])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0
    v="★ 채택" if d.mean()>0 and t>2.5 else ("열세" if d.mean()<0 and t<-2.5 else "판정 불가")
    print(f"{lab:<16}{len(s):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+11.4f}{t:>+7.2f}"
          f"{int((d>0).sum()):>4}/{len(d)}  {v}")
    print(f"{'  시드별 Δ':<16}   {' '.join(f'{v_:+.4f}' for v_ in d)}")
print(f"\n  [test 참고] 전부 예측 +0.0393 (t=4.68, 5/5) · 중첩만 +0.0314 (t=4.34, 5/5)")
print(f"  [val 참고]  전부 예측 -0.0171 (t=-1.19, 2/5)")
PYEOF
log "=== 완료 ==="
