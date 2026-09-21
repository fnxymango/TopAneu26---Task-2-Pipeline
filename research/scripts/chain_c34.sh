#!/usr/bin/env bash
# C34 — 기각됐던 측지(geo) 노선 되살리기 (2026-08-18). A7 학습과 CPU/GPU 병렬.
#
# C15(중심선 측지거리 + 사행비)는 8/15 에 기각됐다(macro-recall 0.400 -> 0.368).
# 그런데 그때와 지금은 조건이 셋 다르다:
#   (1) β 이중보정(β=1.0)이 걸려 있었다 — 오늘 0.5 로 교정
#   (2) 학습이 268병변이었다 — 지금 311
#   (3) 확신 게이트(τ)가 없었다
# 게다가 핸드오프 문서(C22)에 **정식화가 틀렸다**고 이미 적어뒀다:
#   C15 = 랜드마크 3점까지의 **절대** 측지거리 / 옳은 형태 = **분기점 사이 상대 위치**.
#
# 그래서 둘 다 본다:
#   geo  C15 그대로 (6차원) — 조건이 바뀌었으니 재시험
#   arc  C34 신규 (4차원)   — 두 경계 분기점 사이 정규화 위치 t=d1/(d1+d2) 외
#
# arc 가 겨냥하는 것은 오늘 진단에서 무너진 "같은 혈관을 따라 어디인가" 클래스들이다:
#   3.1 vs 3.3 (ICA 근위/원위) · 5.1/5.2/5.3 (M1) · 4.2/4.3/4.4 (ACA)
#
# 선택 신호는 train∪val 311 병변의 환자단위 CV. 다만 CV 에는 검출 위양성이 없어
# 결정규칙(β)은 못 고른다는 걸 오늘 확인했으므로, β=0.5·τ=0.5 는 고정하고 **피처만** 바꾼다.
set -uo pipefail
R="${TOPANEU_ROOT:?}"; S="$R/code/sblee/nnunet/scripts"; E="$R/experiments"
A="$R/code/sblee/nnunet/analysis"; P="$E/_c1_realpred"; BP="$E/_c4_bpgraph"
PY=/home/sblee/miniconda3/envs/sbaneu2/bin/python
REFVES="$R/nnunet/nnUNet_raw/Dataset800_TopAneuVessel417/labelsTr"
MERGED="$A/c32_feat_trainval.json"; GA="$A/c34_feat_trainval_geoarc.json"
log(){ echo "[c34 $(date -u +'%m-%d %H:%M:%S')] $*"; }
cd "$S" || exit 1; export TOPANEU_ROOT="$R"

log "=== STEP1: 311병변에 geo + arc 채우기 (참조 혈관마스크) ==="
if [ ! -f "$GA" ]; then
  $PY -u c34_arc_position.py --feat "$MERGED" --vessel-dir "$REFVES" \
     --bp-dir "$BP/all_ref" --out "$GA" --with-geo 2>&1 | tail -6 || { log "STEP1 실패"; exit 1; }
else
  log "  이미 있음"
fi

log "=== STEP2: CV 블록 제거법 (311병변, 환자단위 5-fold) ==="
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
    print(f"{lab:<26}{np.mean(pred==y):>8.3f}{C8.macro_recall(y,pred)[0]:>10.3f}"
          f"{C30.macro_mcc(y,pred):>10.4f}{z:>5}/{len(set(y))}")
PYEOF
log "=== 완료 ==="
