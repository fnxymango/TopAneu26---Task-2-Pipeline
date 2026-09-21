#!/usr/bin/env bash
# Q-CPU — T16(중첩만 예측혈관 학습) 기반 분류기 축 전수 스윕. 2026-08-19.
#
# 판정선은 **결과 보기 전에 고정**한다 (PROJECT_RULES.md §6-1b):
#   test 83 · 시드 0~4 · 짝지은 차이 t>2.5 이고 5/5 승  -> 292 OOF 로 승격
#   292 에서도 t>2.5 & 5/5  -> 채택 후보 (사람이 최종 확인)
# 그 외는 전부 기각. 단발 최고값은 채택 근거가 아니다 — C36 ET 로 이미 한 번 속았다.
#
# 기준 = e11_hyb_ov (T16). 모든 설정은 여기서 한 축만 바꾼다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"
SPLITS="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
CAP="${QCAP:-12}"                    # 동시 실행 상한 (GPU 학습과 코어 나눠 쓴다)
log(){ echo "[qcpu $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
[ -f "$FEAT" ] || { log "★ 기준 학습피처 없음: $FEAT"; exit 1; }

# 설정목록:  태그 | 추가인자 | 환경변수(공백구분 KEY=VAL)
CFGS=(
  "q_bpr08|--beta 0.5 --conf-tau 0.5|TOPANEU_BP_MAX_R=8"
  "q_bpr10|--beta 0.5 --conf-tau 0.5|TOPANEU_BP_MAX_R=10"
  "q_bpr20|--beta 0.5 --conf-tau 0.5|TOPANEU_BP_MAX_R=20"
  "q_bpr25|--beta 0.5 --conf-tau 0.5|TOPANEU_BP_MAX_R=25"
  "q_et|--beta 0.5 --conf-tau 0.5 --model et|"
  "q_gb|--beta 0.5 --conf-tau 0.5 --model gb|"
  "q_rfet|--beta 0.5 --conf-tau 0.5 --model rf_et|"
  "q_b03|--beta 0.3 --conf-tau 0.5|"
  "q_b07|--beta 0.7 --conf-tau 0.5|"
  "q_t04|--beta 0.5 --conf-tau 0.4|"
  "q_t06|--beta 0.5 --conf-tau 0.6|"
  "q_pos025|--beta 0.5 --conf-tau 0.5|C10_POS_WEIGHT=0.25"
  "q_pos100|--beta 0.5 --conf-tau 0.5|C10_POS_WEIGHT=1.0"
  "q_arc|--beta 0.5 --conf-tau 0.5 --use-arc|"
  "q_geo|--beta 0.5 --conf-tau 0.5 --use-geo|"
  "q_atlas|--beta 0.5 --conf-tau 0.5 --atlas $A/c42_anchor_atlas.json|"
  "q_nomirror|--beta 0.5 --conf-tau 0.5 --no-mirror|"
  "q_nobal|--beta 0.5 --conf-tau 0.5 --no-balance|"
)

run_batch(){   # $1=split  나머지=태그들.  CAP 만큼씩 끊어 병렬.
  local sp=$1; shift
  local VD VB AD EXTRA=""
  if [ "$sp" = test ]; then VD=vespp_test; VB=vespp_test; AD="$P/aneu_test_probavgf"
  else VD=vespp_train; VB=vespp_train; AD="$P/aneu_train_probavgf"; EXTRA="--cv-splits $SPLITS"; fi
  local PIDS=() n=0
  for spec in "$@"; do
    IFS='|' read -r tag args envs <<< "$spec"
    for SD in 0 1 2 3 4; do
      local T="${tag}_s${SD}"
      [ -f "$A/c5_eval_${sp}_${T}.json" ] && continue
      # shellcheck disable=SC2086
      env $envs TOPANEU_BP_MODE=raw CLF_SEED=$SD nohup $PY -u c5_location_v2.py eval \
        --train-feat "$FEAT" --split "$sp" \
        --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$AD" \
        --model rf --use-pos --conf-beta-hi 0.0 $args $EXTRA \
        --tag "$T" > "$E/q_${sp}_${T}.log" 2>&1 &
      PIDS+=($!); n=$((n+1))
      if [ "${#PIDS[@]}" -ge "$CAP" ]; then
        for p in "${PIDS[@]}"; do wait "$p"; done; PIDS=()
        log "  ($sp) $n건 완료"
      fi
    done
  done
  for p in "${PIDS[@]}"; do wait "$p"; done
  log "  ($sp) 총 $n건 완료"
}

log "=== 1단계: test 83 · ${#CFGS[@]}설정 × 5시드 (상한 $CAP 병렬) ==="
run_batch test "${CFGS[@]}"

log "=== 1단계 판정 ==="
$PY - "$A" "test" "${CFGS[@]}" > "$E/q_verdict_test.txt" 2>&1 <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A,sp=sys.argv[1],sys.argv[2]; specs=sys.argv[3:]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_{sp}_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c,ofc=d.get("adjusted_div_present"),d.get("official_div52")
        if c and ofc: o[sd]=(c["MCC"],ofc["MCC"])
    return o
base=load("e11_hyb_ov")
a=np.array([base[s][0] for s in sorted(base)])
print(f"[{sp}] 기준 T16(중첩만예측) n={len(a)} cov.MCC {a.mean():.4f} ± {a.std():.4f}\n")
print(f"{'설정':<12}{'n':>3}{'cov.MCC':>10}{'±':>8}{'off.MCC':>9}{'Δ':>10}{'t':>7}{'승':>6}  판정")
win=[]
rows=[]
for spec in specs:
    tag=spec.split("|")[0]
    g=load(tag); sds=sorted(set(g)&set(base))
    if len(sds)<2:
        print(f"{tag:<12} 데이터부족 n={len(sds)}"); continue
    ga=np.array([g[s][0] for s in sds]); ba=np.array([base[s][0] for s in sds]); d=ga-ba
    oa=np.array([g[s][1] for s in sds])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok = d.mean()>0 and t>2.5 and nw==len(d)
    if ok: win.append(tag)
    rows.append((d.mean(),tag,len(sds),ga.mean(),ga.std(),oa.mean(),t,nw,ok))
for m,tag,n,gm,gs,om,t,nw,ok in sorted(rows,key=lambda r:-r[0]):
    print(f"{tag:<12}{n:>3}{gm:>10.4f}{gs:>8.4f}{om:>9.4f}{m:>+10.4f}{t:>+7.2f}{nw:>4}/5  {'★통과' if ok else '기각'}")
print(f"\n292 OOF 승격: {win if win else '없음'}")
json.dump(win,open(os.path.join(A,"q_winners_test.json"),"w"))
PYEOF
cat "$E/q_verdict_test.txt"

WIN=$($PY -c "import json;print(' '.join(json.load(open('$A/q_winners_test.json'))))" 2>/dev/null || echo "")
if [ -z "$WIN" ]; then
  log "=== test 통과 없음 — 292 생략. T16 유지. ==="
else
  log "=== 2단계: 292 OOF 중재 — $WIN ==="
  SEL=()
  for spec in "${CFGS[@]}"; do
    for w in $WIN; do [ "${spec%%|*}" = "$w" ] && SEL+=("$spec"); done
  done
  run_batch train "${SEL[@]}"
  log "=== 2단계 판정 ==="
  $PY - "$A" "train" "${SEL[@]}" > "$E/q_verdict_292.txt" 2>&1 <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A,sp=sys.argv[1],sys.argv[2]; specs=sys.argv[3:]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_{sp}_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present")
        if c: o[sd]=c["MCC"]
    return o
base=load("e1b2_ov")
if not base: print("★ 292 기준(e1b2_ov) 없음"); raise SystemExit
a=np.array([base[s] for s in sorted(base)])
print(f"[292 OOF] 기준 T16 n={len(a)} cov.MCC {a.mean():.4f} ± {a.std():.4f}\n")
print(f"{'설정':<12}{'n':>3}{'cov.MCC':>10}{'±':>8}{'Δ':>10}{'t':>7}{'승':>6}  판정")
for spec in specs:
    tag=spec.split("|")[0]; g=load(tag); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"{tag:<12} 데이터부족"); continue
    ga=np.array([g[s] for s in sds]); ba=np.array([base[s] for s in sds]); d=ga-ba
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok=d.mean()>0 and t>2.5 and nw==len(d)
    print(f"{tag:<12}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{d.mean():>+10.4f}{t:>+7.2f}{nw:>4}/5  {'★채택후보' if ok else '기각'}")
PYEOF
  cat "$E/q_verdict_292.txt"
fi
log "=== Q-CPU 완료 ==="
