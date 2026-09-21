#!/usr/bin/env bash
# E1b — 292 OOF e2e 로 **채택 결정을 다시 판정**한다 (2026-08-19, 시드 3개 판 개정).
#
# 목적은 새 성능이 아니라 두 가지다:
#   (1) 지금 쓰는 설정의 성능이 믿을 만한가
#   (2) 바뀐다면 **몇으로 수렴하는가** — 그래서 시드 하나가 아니라 여러 개 평균을 낸다
#
# 왜 여기서만 β·τ 를 판정할 수 있나: β 의 가치는 위양성을 무해한 클래스에 버리는 것이다.
# GT 병변만 보는 CV(E5)에는 위양성이 없어 β 를 못 잰다 — C31 에서 train CV 는 β=0,
# val e2e 는 β=0.5 로 정반대 결론이 났던 이유가 그것이다. 292 OOF 는 검출 위양성이 있다.
#
# 해상도: test 는 병변 87/클래스 36 이라 병변 1개 = cov.MCC 0.028.
#         292 는 병변 268/클래스 43 이라 병변 1개 = 0.0029. 3.6배 촘촘하다.
#
# ⚠️ 이 체인의 숫자는 **팀 표·노션에 올리지 않는다** (PROJECT_RULES.md 6-1c, 2026-08-19 사용자 지시).
#    팀은 동일 split · 조직위 evaluate.py · covered_gt 로 서로 비교한다. 292 OOF 는
#    분모가 present 43클래스라 test(36)와 절대값 비교가 애초에 불가능하다.
#    용도는 **순위 하나뿐** — 여기서 설정을 고르고, 고른 설정의 **test 83 점수를 보고**한다.
#
# 한계: 혈관은 V4-2 fold0 이고 그 폴드는 292 를 학습했다 -> 혈관 예측이 train-fit 이다.
#       결정규칙 축(β·τ·모델)은 혈관을 안 건드리므로 유효하고,
#       혈관 의존 피처(측지·호위치·앵커)는 여기서 판정하지 않는다(E7 소관).
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
SP720="$R/nnunet/nnUNet_preprocessed/Dataset720_TopAneuBinary417/splits_final.json"
log(){ echo "[e1b $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== E1 데이터 대기 (10분마다) ==="
for i in $(seq 1 144); do
  nb=$(ls "$BP/vespp_train"/*.json 2>/dev/null | wc -l)
  nf=$(ls "$P/aneu_train_ooff"/*.nii.gz 2>/dev/null | wc -l)
  [ "$nb" -ge 285 ] && [ "$nf" -ge 285 ] && { log "  준비 완료 (bp $nb · 필터 $nf)"; break; }
  log "  대기 $((i*10))분 — 검출OOF $(ls "$P/aneu_train_oof"/*.nii.gz 2>/dev/null|wc -l)/292 ·"\
      "혈관 $(ls "$P/vespp_train"/*.nii.gz 2>/dev/null|wc -l)/292 · 필터 $nf · bp $nb"
  sleep 600
done
[ "$(ls "$BP/vespp_train"/*.json 2>/dev/null | wc -l)" -lt 285 ] && { log "데이터 미준비 — 중단"; exit 1; }

ev(){ # $1=model $2=beta $3=tau $4=seed
  local tag="e1b_${1}_b${2}_t${3}_s${4}"
  [ -f "$A/c5_eval_train_$tag.json" ] && { log "  $tag 있음"; return; }
  log "  $tag"
  CLF_SEED=$4 $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split train \
    --cv-splits "$SP720" --vessel-dir "$P/vespp_train" --bp-dir "$BP/vespp_train" \
    --aneurysm-pred-dir "$P/aneu_train_ooff" \
    --model "$1" --use-pos --beta "$2" --conf-tau "$3" --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}

# 시드 바깥 루프 — 중간에 끊겨도 시드 하나치 전 설정이 갖춰지게 한다
log "=== 292 OOF 재판정 (설정 5 x 시드 3) ==="
for SD in 0 1 2; do
  ev rf 0.0 0.0 "$SD"     # β 가 진짜 필요한가 (C26 이전)
  ev rf 0.5 0.0 "$SD"     # C26 채택값
  ev rf 1.0 0.0 "$SD"     # C26 이 버린 값
  ev rf 0.5 0.5 "$SD"     # C31 확신 게이트
  ev et 0.5 0.5 "$SD"     # C36 현행 프로덕션
done

log "=== E1b 요약 ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,re,sys,collections
import numpy as np
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
g=collections.defaultdict(list)
meta={}
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_train_e1b_*.json"))):
    m=re.search(r"e1b_(\w+?)_b([0-9.]+)_t([0-9.]+)_s(\d)\.json$",f)
    if not m: continue
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    key=f"{m.group(1)} β={m.group(2)} τ={m.group(3)}"
    g[key].append((o['MCC'],comp(o),c['MCC'],comp(c),d.get('top1_accuracy') or 0))
    meta[key]=(d.get('n_lesions_predicted'),d.get('n_present_classes_in_split'))
if not g:
    print("결과 없음"); raise SystemExit
print(f"\n{'설정':<18}{'시드':>4}| {'off.MCC':>9}{'±':>8} | {'cov.MCC':>9}{'±':>8}"
      f"{'cov.복합':>10}{'top1':>8} | {'병변':>5}")
rows=[]
for k,v in g.items():
    a=np.array(v); rows.append((a[:,2].mean(),k,a,len(v)))
for mu,k,a,n in sorted(rows,reverse=True):
    nl,ncl=meta[k]
    print(f"{k:<18}{n:>4}| {a[:,0].mean():>9.4f}{a[:,0].std():>8.4f} | "
          f"{a[:,2].mean():>9.4f}{a[:,2].std():>8.4f}{a[:,3].mean():>10.4f}{a[:,4].mean():>8.4f} | {nl:>5}")
print(f"\n  분모클래스 {ncl} — 병변 1개의 cov.MCC 기여 ≈ {1.0/ncl:+.4f} (test 는 {1.0/36:+.4f})")
best=max(rows)[1]
print(f"  [292 OOF 최고] {best}")
print("  ⚠️ 이 숫자는 내부 판정용이다 — 분모가 43클래스라 test(36)와 비교 불가.")
print("     팀 표·노션에는 여기서 고른 설정의 **test 83** 점수를 올린다. (PROJECT_RULES.md 6-1c)")
PYEOF
log "=== 완료 ==="
