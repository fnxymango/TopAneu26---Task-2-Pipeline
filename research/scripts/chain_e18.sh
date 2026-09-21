#!/usr/bin/env bash
# E18 — 분기점 블록 재설계 (2026-08-19). E15p 채택안(중첩만 예측) 위에 얹는다.
#
# 왜: E15p 에서 bp 블록이 -0.0070(t=-0.86)로 해로웠고, E16 에서 혼동 상위 10쌍 중
#     6쌍이 junction<->trunk 였다. 268병변 실측 결과 원인이 나왔다 —
#     분기점까지의 거리가 VA-PICA(4.0 vs 1.8) / AChA(3.6 vs 3.1) 에서 **부호가 뒤집힌다**.
#     15mm 게이트도 몸통 병변을 92~100% 켜서 무력하다.
#
#   B  off         블록 제거. **이게 기준선이다** — 이걸 못 넘으면 재설계는 무의미하다.
#   A  branch_mix  거리 대신 "쌍 양쪽 혈관을 다 무는가"의 min. 사전점검 방향 3/5 (raw 2/5).
#   C  sharp       대조군: 축은 그대로 두고 15mm 게이트만 제거 + exp(-d/3).
#                  이게 이기면 문제는 '축'이 아니라 '게이트'였다는 뜻이다.
#
# 판정선(결과 보기 전 고정): test 5시드 t>2.5 & 5/5 통과한 것만 292 OOF 로 보낸다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/e11_feat_hyb_ov.json"          # E15p 채택안: 중첩36만 예측혈관
log(){ echo "[e18 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"
[ -f "$FEAT" ] || { log "★ 학습피처 없음 $FEAT"; exit 1; }

PIDS=()
for MODE in off branch_mix sharp; do
  for SD in 0 1 2 3 4; do
    T="e18_${MODE}_s${SD}"
    [ -f "$A/c5_eval_test_${T}.json" ] && { log "  $T 이미 있음"; continue; }
    TOPANEU_BP_MODE="$MODE" CLF_SEED=$SD nohup $PY -u c5_location_v2.py eval \
      --train-feat "$FEAT" --split test \
      --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" \
      --aneurysm-pred-dir "$P/aneu_test_probavgf" \
      --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
      --tag "$T" > "$E/e18_${T}.log" 2>&1 &
    PIDS+=($!); log "  띄움 $T (pid $!)"
  done
done
log "=== ${#PIDS[@]}건 완료 대기 ==="
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
base=load("e11_hyb_ov")     # E15p 채택안 = 현행 기준
a=np.array([base[s][0] for s in sorted(base)])
print(f"\n[test 83] 기준 = 중첩만예측(E15p 채택안)  n={len(a)}  cov.MCC {a.mean():.4f} ± {a.std():.4f}")
print(f"  {'설정':<26}{'n':>3}{'cov.MCC':>10}{'±':>8}{'off.MCC':>9}{'Δ':>10}{'t':>7}{'승':>6}  판정")
win=[]
for tag,lab in (("e18_off","B  분기점블록 제거"),("e18_branch_mix","A  가지관여도"),
                ("e18_sharp","C  게이트제거+sharp")):
    g=load(tag); sds=sorted(set(g)&set(base))
    if len(sds)<2: print(f"  {lab:<26} 데이터 부족 n={len(sds)}"); continue
    ga=np.array([g[s][0] for s in sds]); ba=np.array([base[s][0] for s in sds]); d=ga-ba
    oa=np.array([g[s][1] for s in sds])
    t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if d.std(ddof=1)>0 else 0.0
    nw=int((d>0).sum()); ok = d.mean()>0 and t>2.5 and nw==len(d)
    if ok: win.append(tag)
    print(f"  {lab:<26}{len(sds):>3}{ga.mean():>10.4f}{ga.std():>8.4f}{oa.mean():>9.4f}"
          f"{d.mean():>+10.4f}{t:>+7.2f}{nw:>4}/{len(d)}  {'★통과' if ok else '기각'}")
    print(f"  {'':<26}   시드별 Δ {[round(x,4) for x in d]}")
print(f"\n  292 OOF 로 보낼 것: {win if win else '없음 — 현행 유지'}")
json.dump(win,open(os.path.join(A,"e18_winners.json"),"w"))
PYEOF
log "=== 완료 ==="
