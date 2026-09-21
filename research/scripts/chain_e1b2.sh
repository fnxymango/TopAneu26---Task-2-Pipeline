#!/usr/bin/env bash
# E1b-2 — 292 OOF 로 **E10(예측혈관 학습) 을 심판한다** (2026-08-19). 순서 재조정.
#
# 왜 순서를 바꿨나: E10 이 test 에서 +0.0393 (t=4.68, 5/5승) 인데 val 에서 -0.0171
# (t=-1.19, 2/5승) 로 부호가 갈렸다. 둘 다 |t| 로는 결정적이지 않다.
#   val   병변 43 / 분모 33클래스 -> 병변 1개 = 0.030
#   test  병변 87 / 분모 36클래스 -> 병변 1개 = 0.028
#   292   병변 268 / 분모 43클래스 -> 병변 1개 = 0.0029   ← 10배 촘촘
# 두 거친 자가 갈렸으니 촘촘한 자로 가른다. β·τ 재판정은 그 다음이다.
#
# ⚠️ 이 숫자는 팀 표·노션에 올리지 않는다 (PROJECT_RULES.md 6-1c) — 분모가 43클래스라
#    test(36) 와 절대값 비교 불가. 용도는 **순위 하나**뿐이고 보고는 test 83 으로 한다.
#
# 판정 규칙(결과 보기 전에 못 박음):
#   292 이김(t>2.5) + test 이김 -> 채택
#   292 짐/애매                 -> 기각 (test 우세는 그 83명 운으로 본다)
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REF="$A/c10_feat_train.json"; PRD="$A/c10_feat_train_predves.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
log(){ echo "[e1b2 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

for f in "$REF" "$PRD" "$SP720"; do [ -f "$f" ] || { log "없음: $f"; exit 1; }; done
n=$(ls "$BP/vespp_train"/*.json 2>/dev/null | wc -l)
[ "$n" -ge 285 ] || { log "292 OOF 데이터 미완 (bp $n) — 중단"; exit 1; }
log "292 OOF 재료 확인 (bp $n · 필터 $(ls "$P/aneu_train_ooff"/*.nii.gz 2>/dev/null|wc -l))"

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

# 시드를 바깥 루프로 — 중간에 끊겨도 한 판은 짝이 맞는다
log "=== 1순위: 참조혈관 학습 vs 예측혈관 학습 (시드 3판) ==="
for SD in 0 1 2; do
  ev ref "$REF" "$SD"
  ev prd "$PRD" "$SD"
done

log "=== 판정 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys,re
import numpy as np
A=sys.argv[1]
def load(tag):
    o={}
    for f in glob.glob(os.path.join(A,f"c5_eval_train_e1b2_{tag}_s?.json")):
        sd=int(re.search(r"_s(\d)\.json$",f).group(1)); d=json.load(open(f))
        c=d.get("adjusted_div_present")
        if c: o[sd]=(c["MCC"],d.get("n_lesions_predicted"),d.get("n_present_classes_in_split"))
    return o
r,p=load("ref"),load("prd")
sds=sorted(set(r)&set(p))
if not sds: print("결과 없음"); raise SystemExit
print(f"\n[292 OOF] 예측병변 {r[sds[0]][1]} · 분모 {r[sds[0]][2]}클래스 "
      f"→ 병변 1개 = {1.0/r[sds[0]][2]:+.4f}  (test 는 {1.0/36:+.4f})")
print(f"\n{'시드':>4}{'참조혈관':>11}{'예측혈관':>11}{'차이':>10}")
for s in sds: print(f"{s:>4}{r[s][0]:>11.4f}{p[s][0]:>11.4f}{p[s][0]-r[s][0]:>+10.4f}")
ra=np.array([r[s][0] for s in sds]); pa=np.array([p[s][0] for s in sds]); d=pa-ra
t=d.mean()/(d.std(ddof=1)/np.sqrt(len(d))) if len(d)>1 and d.std(ddof=1)>0 else 0
print(f"\n  참조 {ra.mean():.4f}±{ra.std():.4f}   예측 {pa.mean():.4f}±{pa.std():.4f}")
print(f"  짝지은 차이 {d.mean():+.4f} ± {d.std(ddof=1):.4f}   t = {t:+.2f}   이긴 시드 {int((d>0).sum())}/{len(d)}")
print(f"\n  [참고] test 5판 +0.0393 (t=+4.68, 5/5) · val 5판 -0.0171 (t=-1.19, 2/5)")
v = "★ 292 도 이김 → test 와 일치, 채택" if d.mean()>0 and t>2.5 else \
    ("292 에서 열세 → 기각" if d.mean()<0 and t<-2.5 else "292 도 애매 → 기각 (기본값 유지)")
print(f"  [판정] {v}")
PYEOF
log "=== 완료 ==="
