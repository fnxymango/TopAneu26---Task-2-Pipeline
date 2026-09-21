#!/usr/bin/env bash
# E15p — E15 를 **병렬**로 다시 건다 (2026-08-19). 순차로 짰더니 건당 19분 x 9건 = 3시간인데
# 20코어에 load 1.0 이었다. 평가 하나가 코어 하나만 쓰므로 동시에 던지면 된다.
#
# 확정할 것:
#   1) 292 OOF 를 5판으로 (현재 3판 Δ+0.0133 t=+2.42 3/3 — 판정선 2.5 에 0.08 모자람)
#   2) overlap 단독안 292 5판 (test 블록분해: 중첩만 +0.0314 가 전체 +0.0393 의 80%,
#      분기점 블록은 -0.0070 으로 오히려 해로움)
#
# 동시 실행 8건. RandomForest 적합만 n_jobs=-1 이고 그 구간은 짧다. 나머지는 단일 코어다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REF="$A/c10_feat_train.json"; PRD="$A/c10_feat_train_predves.json"; OVF="$A/e11_feat_hyb_ov.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
log(){ echo "[e15p $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

launch(){ # $1=tag $2=feat $3=seed
  local tag="e1b2_${1}_s${3}"
  [ -f "$A/c5_eval_train_$tag.json" ] && { log "  $tag 있음"; return; }
  CLF_SEED=$3 OMP_NUM_THREADS=2 $PY -u c5_location_v2.py eval --train-feat "$2" --split train \
    --cv-splits "$SP720" --vessel-dir "$P/vespp_train" --bp-dir "$BP/vespp_train" \
    --aneurysm-pred-dir "$P/aneu_train_ooff" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$tag" > "$E/e15p_${tag}.log" 2>&1 &
  log "  띄움 $tag (pid $!)"
}

log "=== 병렬 실행 ==="
for SD in 3 4; do launch ref "$REF" "$SD"; launch prd "$PRD" "$SD"; done
for SD in 0 1 2 3 4; do launch ov "$OVF" "$SD"; done
log "=== 전부 완료 대기 ==="
wait
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
if not r: print("기준선 없음"); raise SystemExit
n,ncl=r[sorted(r)[0]][1],r[sorted(r)[0]][2]
print(f"\n[292 OOF] 예측병변 {n} · 분모 {ncl}클래스 → 병변 1개 = {1.0/ncl:+.4f}  (test {1.0/36:+.4f})")
ra=np.array([r[s][0] for s in sorted(r)])
print(f"\n{'설정':<18}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ vs 참조':>11}{'t':>7}{'승':>6}  판정")
print(f"{'참조혈관(기준)':<18}{len(ra):>3}{ra.mean():>10.4f}{ra.std():>8.4f}")
for tag,lab in (("prd","전부 예측(E10)"),("ov","중첩만 예측")):
    g=L(f"c5_eval_train_e1b2_{tag}_s?.json"); s=sorted(set(g)&set(r))
    if len(s)<2: print(f"{lab:<18}{len(s):>3}  데이터 부족"); continue
    ga=np.array([g[x][0] for x in s]); d=np.array([g[x][0]-r[x][0] for x in s])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0
    v="★ 채택" if d.mean()>0 and t>2.5 else ("열세" if d.mean()<0 and t<-2.5 else "판정 불가")
    print(f"{lab:<18}{len(s):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+11.4f}{t:>+7.2f}"
          f"{int((d>0).sum()):>4}/{len(d)}  {v}")
    print(f"{'  시드별 Δ':<18}   {' '.join(f'{v_:+.4f}' for v_ in d)}")
print(f"\n  [test] 전부 예측 +0.0393 (t=4.68 5/5) · 중첩만 +0.0314 (t=4.34 5/5)")
print(f"  [val ] 전부 예측 -0.0171 (t=-1.19 2/5)")
PYEOF
log "=== 완료 ==="
