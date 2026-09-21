#!/usr/bin/env bash
# E6b — E4+E4b 전체(30건)에 케이스 부트스트랩 CI 를 붙이고 **최종 순위표**를 만든다.
# 판정: 시드 산포(재현성)와 표집 CI(일반화) 둘 다 통과한 설정 중 최고를 고른다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[e6b $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== E4+E4b 완료 대기 (30건) ==="
for i in $(seq 1 90); do
  n=$(( $(ls "$A"/c5_eval_test_e4_*.json 2>/dev/null | wc -l) + $(ls "$A"/c5_eval_test_e4b_*.json 2>/dev/null | wc -l) ))
  [ "$n" -ge 30 ] && { log "  완료 ($n/30)"; break; }
  log "  대기 $((i*5))분 — $n/30"; sleep 300
done

log "=== per-case 원자료 보충 ==="
$PY - "$A" "$S" <<'PYEOF' > /tmp/e6b_missing.sh
import glob,os,sys,re
A,S=sys.argv[1],sys.argv[2]
for f in sorted(glob.glob(os.path.join(A,"c5_eval_test_e4*.json"))):
    tag=os.path.basename(f)[len("c5_eval_test_"):-5]
    if os.path.exists(os.path.join(A,f"c5_percase_test_{tag}.json")): continue
    m=re.match(r"e4_(\w+)_s(\d)$",tag) or re.match(r"e4b_(\w+?)_b([0-9.]+)_t([0-9.]+)_s(\d)$",tag)
    if not m: continue
    g=m.groups()
    mdl,b,t,sd=(g[0],"0.5","0.5",g[1]) if len(g)==2 else (g[0],g[1],g[2],g[3])
    print(f'rm -f "{f}"; echo "  재실행 {tag}"; CLF_SEED={sd} $PY -u c5_location_v2.py eval '
          f'--train-feat "$FEAT" --split test --vessel-dir "$P/vespp_test" --bp-dir "$BP/vespp_test" '
          f'--aneurysm-pred-dir "$P/aneu_test_probavgf" --model {mdl} --use-pos --beta {b} '
          f'--conf-tau {t} --conf-beta-hi 0.0 --tag {tag} 2>&1 | grep -E \'"MCC"\' | head -1')
PYEOF
. /tmp/e6b_missing.sh || log "  일부 재실행 실패"

log "=== 부트스트랩 2000회 + 최종 순위 ==="
$PY -u e6_bootstrap.py --n 2000 --split test 2>&1 | tail -60
$PY - "$A" <<'PYEOF'
import json,glob,os,re,collections
import numpy as np
A=os.path.abspath(list(glob.glob("/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"))[0])
boot=json.load(open(os.path.join(A,"e6_bootstrap_test.json"))) if os.path.exists(os.path.join(A,"e6_bootstrap_test.json")) else {}
def cfg(tag):
    m=re.match(r"e4_(\w+)_s(\d)$",tag)
    if m: return f"{m.group(1)} β=0.5 τ=0.5"
    m=re.match(r"e4b_(\w+?)_b([0-9.]+)_t([0-9.]+)_s(\d)$",tag)
    return f"{m.group(1)} β={m.group(2)} τ={m.group(3)}" if m else None
g=collections.defaultdict(list)
for tag,v in boot.items():
    c=cfg(tag)
    if c: g[c].append((v["cov_mcc"],v["off_mcc"],v["cov_sd"]))
if not g: raise SystemExit("부트스트랩 결과 없음")
print(f"\n=== 최종: 재현성(시드) + 일반화(표집) 동시 표기 ===")
print(f"{'설정':<18}{'n':>3}| {'cov.MCC 시드평균':>17}{'시드산포':>10}{'범위':>18} | {'표집 sd':>9}")
rows=[]
for c,v in g.items():
    a=np.array(v); rows.append((a[:,0].mean(),c,a))
for mu,c,a in sorted(rows,reverse=True):
    print(f"{c:<18}{len(a):>3}| {mu:>17.4f}{a[:,0].std():>10.4f}"
          f"{f'[{a[:,0].min():.4f}, {a[:,0].max():.4f}]':>18} | {a[:,2].mean():>9.4f}")
best=max(rows)[1]; bm=max(rows)[0]
print(f"\n  [시드평균 최고] {best} = {bm:.4f}")
print(f"  표집 sd 가 시드 산포보다 훨씬 크면, 설정 간 차이는 test 83케이스로는 못 가른다는 뜻이다.")
PYEOF
log "=== 완료 ==="
