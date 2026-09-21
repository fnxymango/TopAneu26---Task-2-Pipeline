#!/usr/bin/env bash
# A8 — 검출기 합집합 (2026-08-18). 현행 A6-2 5-fold ∪ A7(Tversky 0.15/0.85).
#
# test 겹침 측정: 현행 63 · A7 65 · 둘다 60 · 현행만 3 · A7만 5 · 합집합 68/86 (0.733 -> 0.791).
# val 43병변에서는 완전히 일치했으나 test 에서는 갈렸다 — val 만 보고 A7 을 버렸으면 놓칠 뻔했다.
# 놓치던 23개 중 5개 회수. 다만 FP 가 24 -> 최대 69 로 늘어나므로 MCC 가 실제로 오르는지는 재봐야 한다.
# c7 필터를 합집합 뒤에 다시 걸어 FP 를 눌러본다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
FEAT="$A/c10_feat_train.json"
log(){ echo "[a8 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== STEP1: 합집합 마스크 생성 (c7 필터 강도별) ==="
$PY -u - <<'PYEOF'
import numpy as np, nibabel as nib, os
from scipy import ndimage as ndi
import d9xx_lib as L
ST=np.ones((3,3,3),bool)
P=L.TOPANEU_ROOT/"experiments"/"_c1_realpred"
_,va,te=L.case_ids_by_split()
def filt(pr,ves,sp,mv,md):
    lab,n=ndi.label(pr,structure=ST)
    dist=ndi.distance_transform_edt(~(ves>0),sampling=sp) if (ves>0).any() else np.full(pr.shape,1e9)
    keep=np.zeros(n+1,bool)
    for l in range(1,n+1):
        s=lab==l
        if s.sum()>=mv and dist[s].min()<=md: keep[l]=True
    return keep[lab]
for sp_name,ids,pd,ad,vd in (("val",va,"aneu_val_probavgf","aneu_val_a7","vespp_val"),
                             ("test",te,"aneu_test_probavgf","aneu_test_a7","vespp_test")):
    for mv,md,tag in ((20,3.0,"u20"),(40,2.0,"u40"),(80,2.0,"u80")):
        out=P/f"aneu_{sp_name}_union{tag}"; out.mkdir(parents=True,exist_ok=True)
        if len(list(out.glob("*.nii.gz")))>=len(ids)-5: continue
        n=0
        for cid in ids:
            a=P/pd/f"{cid}.nii.gz"; b=P/ad/f"{cid}.nii.gz"; v=P/vd/f"{cid}.nii.gz"
            if not a.exists(): continue
            ia=nib.load(a); ma=np.asanyarray(ia.dataobj)>0
            mb=np.asanyarray(nib.load(b).dataobj)>0 if b.exists() else np.zeros_like(ma)
            ves=np.asanyarray(nib.load(v).dataobj) if v.exists() else np.zeros_like(ma,np.int16)
            spc=np.array(ia.header.get_zooms()[:3],float)
            u=filt(ma|mb,ves,spc,mv,md)
            nib.save(nib.Nifti1Image(u.astype(np.int16),ia.affine,ia.header),out/f"{cid}.nii.gz")
            n+=1
        print(f"  {sp_name} {tag} (min_vox={mv}, dist={md}mm): {n}건",flush=True)
PYEOF

ev(){ local sp=$1 tag=$2 ad=$3 VD VB
  if [ "$sp" = val ]; then VD=vespp_val; VB=val_pred; else VD=vespp_test; VB=vespp_test; fi
  [ -d "$ad" ] || { log "  $ad 없음"; return; }
  [ -f "$A/c5_eval_${sp}_${tag}.json" ] && return
  log "  $sp $tag"
  $PY -u c5_location_v2.py eval --train-feat "$FEAT" --split "$sp" \
    --vessel-dir "$P/$VD" --bp-dir "$BP/$VB" --aneurysm-pred-dir "$ad" \
    --model et --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
    --tag "$tag" 2>&1 | grep -E '"MCC"' | head -1 || log "    실패"
}
log "=== STEP2: val 평가 ==="
for T in u20 u40 u80; do ev val "a8_$T" "$P/aneu_val_union$T"; done
ev val a8_base "$P/aneu_val_probavgf"

BEST=$($PY - "$A" <<'PYEOF'
import json,glob,os,sys
rows=[]
for f in glob.glob(os.path.join(sys.argv[1],"c5_eval_val_a8_*.json")):
    o=json.load(open(f)).get("official_div52") or {}
    n=os.path.basename(f)[len("c5_eval_val_"):-5]
    if o: rows.append((o.get("MCC",-1),n))
rows.sort(reverse=True)
for v,n in rows: print(f"  {n:<12} {v:.4f}",file=sys.stderr)
print(rows[0][1] if rows else "a8_base")
PYEOF
)
log "=== val 최고: $BEST ==="
log "=== STEP3: test ==="
ev test a8_base "$P/aneu_test_probavgf"
case "$BEST" in a8_u*) ev test "$BEST" "$P/aneu_test_union${BEST#a8_}";; esac

log "=== 요약 (test, MCC 순) ==="
$PY - "$A" <<'PYEOF'
import json,glob,os,sys
K=['PRECISION','RECALL','MCC','DICE','VOLSIM']
comp=lambda o:(sum(o[k] for k in K)+1-o['HD95'])/6
rows=[]
for f in sorted(glob.glob(os.path.join(sys.argv[1],"c5_eval_test_a8_*.json"))+
                [os.path.join(sys.argv[1],"c5_eval_test_c36_et.json")]):
    if not os.path.exists(f): continue
    d=json.load(open(f)); o,c=d.get("official_div52"),d.get("adjusted_div_present")
    if not (o and c): continue
    rows.append((os.path.basename(f)[len("c5_eval_test_"):-5],o,c,d.get("n_lesions_predicted")))
rows.sort(key=lambda r:-r[2]['MCC'])
print(f"\n{'설정':<14}{'예측병변':>8}| {'off.MCC':>8}{'off.복합':>9} | {'cov.P':>7}{'cov.R':>7}{'cov.MCC':>8}{'cov.복합':>9}")
for tag,o,c,n in rows:
    print(f"{tag:<14}{n or 0:>8}| {o['MCC']:>8.4f}{comp(o):>9.4f} | {c['PRECISION']:>7.4f}{c['RECALL']:>7.4f}{c['MCC']:>8.4f}{comp(c):>9.4f}")
PYEOF
log "=== 완료 ==="
