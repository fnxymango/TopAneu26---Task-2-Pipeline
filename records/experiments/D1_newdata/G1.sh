#!/usr/bin/env bash
# G1 — 새 공식 eval(60765a5) 기준으로 gC(topk=2) 가 여전히 이득인지.
#   판정 규칙(결과 보기 전 고정, 09-04 09:30 KST):
#     리더보드가 지표별 순위평균이므로 스칼라 합산 대신 '6지표 중 몇 개가 개선되는가' 로 본다.
#     gC 유지 = test·val 모두 6지표 중 ≥4 개선 AND MCC 비악화(둘 다).  아니면 '재검토'.
#   A단계: P3 검출기(cmpoldff) 위 — 예비.   B단계: 번들 검출기 B1(b1ff) 위 — 본판정.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; S=$R/code/sblee/nnunet/scripts; A=$R/code/sblee/nnunet/analysis
P=$E/_c1_realpred; BP=$E/_c4_bpgraph; D=$E/D1_newdata; G=$E/G1_gc_neweval; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
export TOPANEU_ROOT=$R PYTHONPATH=$R/code/sblee:$R/code/sblee/nnunet
exec 9>"$D/g1.lock"; flock -n 9 || exit 0
echo $$ > "$D/G1.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][g1] $*" | tee -a "$ST"; }
cd "$S"; mkdir -p "$G"/{pred,logs,scores}
full(){ [ "$(ls "$G/pred/$1"/*.nii.gz 2>/dev/null | wc -l)" -ge "$2" ]; }
gen(){ local sp=$1 topk=$2 tag=$3 det=$4 bp
  bp=$( [ "$sp" = test ] && echo vespp_test || echo val_pred )
  TOPANEU_TOPK=$topk TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=3 OMP_NUM_THREADS=2 \
  "$PY" -u c5_location_v2.py eval --train-feat "$A/e11_feat_hyb_ov.json" --split "$sp" \
    --vessel-dir "$P/vespp_${sp}" --bp-dir "$BP/$bp" --aneurysm-pred-dir "$P/$det" \
    --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --save-pred-dir "$G/pred/${tag}_${sp}" --tag "g1_${tag}_${sp}_s3" > "$G/logs/${tag}_${sp}.log" 2>&1
}
score(){ "$PY" "$D/neweval.py" "$G/pred/$1" "$2" "$1" > "$G/scores/$1.json" 2> "$G/logs/score_$1.err"; }

# ---- A: P3 검출기 위 (이미 기동된 4런 완료 대기)
log "A단계 대기 (P3 검출기 · gC on/off)"
for i in $(seq 1 120); do full gcON_test 83 && full gcOFF_test 83 && full gcON_val 41 && full gcOFF_val 41 && break; sleep 30; done
for t in gcON_test gcOFF_test gcON_val gcOFF_val; do score "$t" "${t#*_}"; done
log "A단계 채점 완료"
# ---- B: 번들 검출기(B1) 위
log "B단계 대기 (B1 c7 산출)"
for i in $(seq 1 240); do [ "$(ls "$P/aneu_test_b1ff"/*.nii.gz 2>/dev/null|wc -l)" -ge 83 ] && [ "$(ls "$P/aneu_val_b1ff"/*.nii.gz 2>/dev/null|wc -l)" -ge 41 ] && break; sleep 30; done
gen test 2 b1gcON  aneu_test_b1ff & gen test 1 b1gcOFF aneu_test_b1ff &
gen val  2 b1gcON  aneu_val_b1ff  & gen val  1 b1gcOFF aneu_val_b1ff  &
wait
for t in b1gcON_test b1gcOFF_test b1gcON_val b1gcOFF_val; do score "$t" "${t#*_}"; done
log "B단계 채점 완료"
# ---- 정리
"$PY" - "$G" <<'PYEOF' > "$G/RESULTS.md"
import json,glob,os,sys
G=sys.argv[1]
def L(t):
    p=f"{G}/scores/{t}.json"
    return json.load(open(p)) if os.path.exists(p) else None
M=["PRECISION","RECALL","MCC","DICE","VOLSIM","HD95"]
print("# gC(topk=2) 손익 — 새 공식 eval(60765a5) vs 구 eval\n")
print("판정 규칙(사전 고정): test·val 모두 6지표 중 ≥4 개선 AND MCC 비악화 → gC 유지. (HD95 는 낮을수록 개선)\n")
verdict={}
for stage,pre,name in (("B","b1","번들 검출기 B1 (본판정)"),("A","","P3 검출기 (예비)")):
    print(f"## {name}\n")
    for sp in ("test","val"):
        on,off=L(f"{pre}gcON_{sp}"),L(f"{pre}gcOFF_{sp}")
        if not on or not off: print(f"### {sp}: 결과 없음\n"); continue
        print(f"### {sp} ({on['n_cases']}건)\n")
        print("| 지표 | 새eval gC off | 새eval gC on | Δ | 개선 | valid클래스 off→on | 구eval off | 구eval on | Δ |")
        print("|---|---|---|---|---|---|---|---|---|")
        imp=0; mcc_ok=None
        for m in M:
            a,b=off["new"][m],on["new"][m]; d=b-a
            better = (d<0) if m=="HD95" else (d>0)
            imp+=int(better)
            if m=="MCC": mcc_ok = d>=0
            oa=off.get("old",{}).get(m); ob=on.get("old",{}).get(m)
            oo=f"{oa:.4f} | {ob:.4f} | {ob-oa:+.4f}" if oa is not None and ob is not None else "— | — | —"
            print(f"| {m} | {a:.4f} | {b:.4f} | {d:+.4f} | {'○' if better else '×'} | {off['new'].get('count_valid_'+m)}→{on['new'].get('count_valid_'+m)} | {oo} |")
        print(f"\n개선 {imp}/6 · MCC 비악화 {mcc_ok}\n")
        verdict[(stage,sp)]=(imp>=4 and mcc_ok)
b=[verdict.get(("B","test")),verdict.get(("B","val"))]
print("## 판정 (본판정 = 번들 검출기 위)\n")
if None in b: print("본판정 결과 부족")
else: print("**gC 유지**" if all(b) else "**gC 재검토** — 새 eval 에서 조건 불충족")
PYEOF
log "G1 완료 · $G/RESULTS.md"
touch "$D/.done_g1"
