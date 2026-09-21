#!/usr/bin/env bash
# C34-b — 측지(geo)/호위치(arc) 되살리기, **train 268 기준** (2026-08-18).
# 2026-08-18 사용자 지시로 val 은 학습에 넣지 않는다(PROJECT_RULES.md §1). C32(train+val 311)는 철회.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
SRC="$A/c10_feat_train.json"; GA="$A/c34_feat_train_geoarc.json"
log(){ echo "[c34b $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== STEP1: train 268 병변에 geo + arc 채우기 ==="
if [ ! -f "$GA" ]; then
  $PY -u c34_arc_position.py --feat "$SRC" --vessel-dir "$REFVES" \
     --bp-dir "$BP/all_ref" --out "$GA" --with-geo 2>&1 | tail -4 || { log "실패"; exit 1; }
else
  log "  이미 있음"
fi

log "=== STEP2: CV 블록 제거법 (train 268, 환자단위 5-fold) ==="
$PY - "$GA" <<'PYEOF'
import json,sys,collections
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
import d9xx_lib as L, c5_location_v2 as C5, c8_classifier_cv as C8, c30_mcc_weights as C30
rows=[r for r in json.load(open(sys.argv[1])) if r.get("gt_loc")]
va,_=C5.build_feature_axes(L.vessel_dense_names())
y=np.array([r["gt_loc"] for r in rows]); pm=C8.patient_map()
g=np.array([pm.get(r["case"],r["case"]) for r in rows])
gkf=list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,g))
print(f"병변 {len(rows)} · geo {sum(1 for r in rows if r.get('geo'))} · arc {sum(1 for r in rows if r.get('arc'))}")
print(f"\n{'피처 구성':<26}{'top-1':>8}{'macroRec':>10}{'macroMCC':>10}{'재현0':>10}")
res={}
for lab,(ug,ua) in (("기준선 (pos만)",(0,0)),("+ geo (C15)",(1,0)),
                    ("+ arc (C34)",(0,1)),("+ geo + arc",(1,1))):
    C5.USE_POS=True; C5.USE_GEO=bool(ug); C5.USE_ARC=bool(ua)
    X=np.array([C5.row_to_vec(r,va) for r in rows])
    Xm=np.array([C5.row_to_vec(r,va,mirror=True) for r in rows])
    pred=np.empty(len(rows),dtype=object)
    for tr,te in gkf:
        Xa=np.concatenate([X[tr],Xm[tr]])
        ya=np.concatenate([y[tr],[C5.mirror_name(v) for v in y[tr]]])
        clf=RandomForestClassifier(n_estimators=500,class_weight="balanced",
                                   random_state=0,n_jobs=-1).fit(Xa,ya)
        Pp=clf.predict_proba(X[te])
        pri=np.array([max(collections.Counter(ya).get(c,1),1) for c in clf.classes_],float)
        mx=Pp.max(1)
        Pc=np.where((mx<0.5)[:,None], Pp/pri**0.5, Pp)      # τ=0.5, β=0.5 고정
        pred[te]=clf.classes_[np.argmax(Pc,1)]
    z=sum(1 for c in set(y) if np.sum(pred[y==c]==c)==0)
    m=C30.macro_mcc(y,pred); res[lab]=m
    print(f"{lab:<26}{np.mean(pred==y):>8.3f}{C8.macro_recall(y,pred)[0]:>10.3f}{m:>10.4f}{z:>5}/{len(set(y))}")
    # arc 가 겨냥한 클래스만 따로
    if ua:
        for fam,pat in (("ICA 3.1/3.3",("3.1","3.3")),("M1 5.1/5.2/5.3",("5.1","5.2","5.3")),
                        ("ACA 4.2/4.3/4.4",("4.2","4.3","4.4"))):
            m_=np.array([any(p in c for p in pat) for c in y])
            if m_.sum(): print(f"     [{fam}] n={int(m_.sum())} 재현율 {np.mean(pred[m_]==y[m_]):.3f}")
base=res["기준선 (pos만)"]
best=max(res,key=res.get)
print(f"\n[판정] 최고 {best} {res[best]:.4f} · 기준선 {base:.4f} · 차이 {res[best]-base:+.4f}")
print("  기준선이 최고면 측지 노선은 정식화를 바꿔도 안 살아난 것 — 두 번째 기각.")
PYEOF
log "=== 완료 ==="
