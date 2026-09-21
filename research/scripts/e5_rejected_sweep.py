"""E5 — 지금까지 기각/채택한 것들을 **반복 CV** 로 전부 다시 통과시킨다 (2026-08-19).

배경: 이 프로젝트는 오늘까지 재현성 검사를 사실상 하지 않았다. 시드를 여러 개 돌린
스크립트는 0개였고(c30 의 중첩CV 하나가 유일한 정직성 장치였다), 채택된 개선은
전부 단발 실행이었다. 그런데 test 는 병변 87/분모클래스 36 이라 병변 1개가 cov.MCC 를
0.028 움직인다 — 우리 개선폭 전부가 그 근처였다.

E2/E3 에서 실제로 무너졌다: RF vs ET 는 test 3승3패·val 1승1패로 정답 수가 같았고,
268병변 CV 5시드에서는 ET 가 오히려 -0.0090(5시드 중 3패)이었다.

그래서 기각했던 노선들도 같은 의심을 받아야 한다. 전부 단발로 기각했기 때문이다.
여기서는 268병변 · 환자단위 5폴드 · 5시드로 다시 돌려 **평균과 산포를 함께** 낸다.

한계(반드시 같이 읽을 것):
  - 이 하니스는 GT 병변만 본다. 검출 위양성이 없다. β 의 가치는 "위양성을 무해한
    클래스에 버리는 것"이므로 **β·τ 는 여기서 판정할 수 없다** — E1b(292 OOF e2e) 소관.
  - 피처는 참조혈관 기반이다. 측지·호위치·앵커는 혈관이 완벽할 때의 값이라
    여기서 이겨도 예측혈관에서 이긴다는 보장이 없다. 통과하면 test 로 재확인한다.
  - 그래서 판정은 "채택"이 아니라 **"재검토 대상인가"** 다.

사용: python e5_rejected_sweep.py [--seeds 5] [--only tag1,tag2]

⚠️ 이 스크립트의 숫자는 학습코호트(268병변/43클래스) 기준이라 **test 와 비교 불가**이며
   팀 표·노션에 올리지 않는다 (PROJECT_RULES.md 6-1c). 용도는 설정 간 **순위**뿐이다.
"""
import argparse, collections, json, os
import numpy as np
from sklearn.model_selection import GroupKFold

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8
from c30_mcc_weights import macro_mcc

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"

# tag: (피처파일, 모델, 플래그dict, 환경변수dict, 합성행파일)
CFG = [
    ("base_rf",   "c10_feat_train.json",        "rf", {}, {}, None),
    ("base_et",   "c10_feat_train.json",        "et", {}, {}, None),
    ("geo",       "c34_feat_train_geoarc.json", "rf", {"USE_GEO": True}, {}, None),
    ("arc",       "c34_feat_train_geoarc.json", "rf", {"USE_ARC": True}, {}, None),
    ("geo_arc",   "c34_feat_train_geoarc.json", "rf", {"USE_GEO": True, "USE_ARC": True}, {}, None),
    ("geo_w1.0",  "c34_feat_train_geoarc.json", "rf", {"USE_GEO": True}, {"C15_GEO_WEIGHT": "1.0"}, None),
    ("miss",      "c42_feat_train_anchor.json", "rf", {"USE_MISS": True}, {}, None),
    ("imp",       "c42_feat_train_anchor.json", "rf", {"USE_IMP": True}, {}, None),
    ("miss_imp",  "c42_feat_train_anchor.json", "rf", {"USE_MISS": True, "USE_IMP": True}, {}, None),
    ("synth",     "c10_feat_train.json",        "rf", {}, {}, "c11_synth_rows.json"),
    ("tta",       "c10_feat_train.json",        "rf", {"USE_TTA": True}, {}, None),
    ("et_mf0.7",  "c10_feat_train.json",        "et", {}, {"ET_MAX_FEATURES": "0.7"}, None),
    ("et_ml2",    "c10_feat_train.json",        "et", {}, {"ET_MIN_LEAF": "2"}, None),
    ("gb",        "c10_feat_train.json",        "gb", {}, {}, None),
    ("rf_et",     "c10_feat_train.json",     "rf_et", {}, {}, None),
]
FLAGS = ("USE_GEO", "USE_ARC", "USE_MISS", "USE_IMP", "USE_TTA")


def run_one(rows, ves_axis, y, groups, kind, seeds, folds, beta, synth_rows, pm):
    accs, mccs, preds = [], [], []
    for s in range(seeds):
        os.environ["CLF_SEED"] = str(s)
        idx = np.random.RandomState(s).permutation(len(rows))
        pred = np.empty(len(rows), dtype=object)
        for tr, te in GroupKFold(n_splits=folds).split(idx, y[idx], groups[idx]):
            tri, tei = idx[tr], idx[te]
            hold = {groups[i] for i in tei}
            sub = [rows[i] for i in tri]
            syn = ([r for r in synth_rows if pm.get(r["case"], r["case"]) not in hold]
                   if synth_rows else None)
            m = C5.fit_model(sub, ves_axis, kind=kind, mirror=True, balance=True,
                             synth_rows=syn, synth_repeat=2)
            for i in tei:
                pred[i] = C5.predict_one(m, rows[i], beta)
        accs.append(float(np.mean(pred == y))); mccs.append(macro_mcc(y, pred))
        preds.append(pred.copy())
    return np.array(accs), np.array(mccs), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    C5.USE_POS = True; C5.CONF_TAU = 0.0
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())
    pm = C8.patient_map()
    only = set(a.only.split(",")) if a.only else None

    out, base_pred = {}, None
    for tag, feat, kind, flags, envs, synthf in CFG:
        if only and tag not in only:
            continue
        p = A / feat
        if not p.exists():
            print(f"[e5] {tag}: 피처파일 없음 {feat} — 건너뜀", flush=True); continue
        for f in FLAGS:
            setattr(C5, f, flags.get(f, False))
        for k in ("C15_GEO_WEIGHT", "C34_ARC_WEIGHT", "C42_MISS_WEIGHT", "C43_IMP_WEIGHT",
                  "ET_MAX_FEATURES", "ET_MIN_LEAF", "ET_TREES"):
            os.environ.pop(k, None)
        os.environ.update(envs)
        # 가중치는 import 시점에 읽히므로 모듈 상수를 직접 갱신한다
        C5.GEO_WEIGHT = float(os.environ.get("C15_GEO_WEIGHT", "0.5"))
        C5.ARC_WEIGHT = float(os.environ.get("C34_ARC_WEIGHT", "0.5"))
        C5.MISS_WEIGHT = float(os.environ.get("C42_MISS_WEIGHT", "0.5"))
        C5.IMP_WEIGHT = float(os.environ.get("C43_IMP_WEIGHT", "0.5"))

        rows = [r for r in json.load(open(p)) if r.get("gt_loc")]
        y = np.array([r["gt_loc"] for r in rows])
        groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
        synth = json.load(open(A / synthf)) if synthf else None
        acc, mcc, preds = run_one(rows, ves_axis, y, groups, kind, a.seeds, a.folds,
                                  a.beta, synth, pm)
        out[tag] = {"acc": acc.tolist(), "mcc": mcc.tolist(), "feat": feat,
                    "model": kind, "flags": flags, "envs": envs, "n": len(rows)}
        if tag == "base_rf":
            base_pred = preds
        print(f"[e5] {tag:<10} top1 {acc.mean():.4f}±{acc.std():.4f}  "
              f"macroMCC {mcc.mean():.4f}±{mcc.std():.4f}  (병변 {len(rows)})", flush=True)

    b = out.get("base_rf")
    if b:
        ba, bm = np.array(b["acc"]), np.array(b["mcc"])
        print(f"\n{'설정':<11}{'top1':>9}{'±':>8}{'Δtop1':>9}{'t':>7}  {'macroMCC':>10}{'±':>8}{'ΔMCC':>9}  판정")
        rank = sorted(out.items(), key=lambda kv: -np.mean(kv[1]["acc"]))
        for tag, r in rank:
            ac, mc = np.array(r["acc"]), np.array(r["mcc"])
            d = ac - ba                      # 같은 시드끼리 짝지은 차이
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0
            t = d.mean() / se if se > 0 else 0.0
            v = ("기준선" if tag == "base_rf" else
                 "재검토 대상" if d.mean() > 0 and abs(t) > 2.5 else
                 "확실히 열세" if d.mean() < 0 and abs(t) > 2.5 else "차이 없음")
            print(f"{tag:<11}{ac.mean():>9.4f}{ac.std():>8.4f}{d.mean():>+9.4f}{t:>7.1f}  "
                  f"{mc.mean():>10.4f}{mc.std():>8.4f}{mc.mean()-bm.mean():>+9.4f}  {v}")
        print("\n  Δ 는 **같은 시드끼리 짝지은** 차이(paired)다. t 는 그 차이의 시드간 t 통계.")
        print(f"  시드 {a.seeds}개라 |t|>2.5 를 판정선으로 쓴다. 통과해도 예측혈관 test 로 재확인해야 한다.")
        print("  β·τ 는 위양성이 없는 이 하니스로 판정 불가 — E1b(292 OOF e2e) 소관.")
    json.dump(out, open(A / "e5_rejected_sweep.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n[저장] {A / 'e5_rejected_sweep.json'}")


if __name__ == "__main__":
    main()
