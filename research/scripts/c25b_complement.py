"""C25-b — 크롭 분기와 기하 분기가 *정말* 상보적인지 (2026-08-17).

C25 에서 앙상블 이득이 0 이었다 (macro-recall 0.324 -> 0.325).
원인 후보가 둘이고, 둘은 처방이 완전히 다르다.

  (가) 두 분기가 같은 정보를 본다  -> 크롭은 버린다. 채택된 크롭 피처가 ch0,1,2 라
       혈관라벨/거리맵이 들어 있고 그건 기하 RF 가 이미 쓰는 것과 같다.
  (나) 정보는 다른데 결합 규칙이 나쁘다 -> 확률 평균 대신 스태킹으로 바꾼다.

구분법은 **일치표**다. 기하가 틀리고 크롭이 맞는 샘플 비율이 크면 (나), 작으면 (가).
오라클(둘 중 맞은 쪽을 고름) 상한을 같이 재서 결합으로 얻을 수 있는 최대치를 못박는다.

또 C25 는 크롭 분기를 **단독 macro-recall** 로 골랐는데, 앙상블에 필요한 건 단독 성능이
아니라 직교성이다. 여기서는 ch0(영상만) 도 끝까지 끌고 가 앙상블까지 비교한다.

그리고 애초 C24/C25 의 명분이었던 클래스들 — 분기점 검출률이 낮아 기하로는 원리적으로
불가능한 AChA / Pcom — 에서만 따로 본다. 전체 평균에 묻혀 안 보일 수 있다.

사용: python c25b_complement.py --crop-dir <train크롭> --feat <c10_feat_train.json>
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8
import c24_train_eval as C24
import c25_crop_mip as C25

FOCUS = ("AChA", "Pcom")          # 분기점 검출률 43% — 기하가 원리적으로 못 푸는 클래스


def norm(P, pri=None, beta=1.0):
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)
    return P / (pri ** beta) if pri is not None else P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-dir", required=True)
    ap.add_argument("--feat", required=True)
    ap.add_argument("--beta", type=float, default=1.0)
    a = ap.parse_args()

    A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
    meta, X = C25.load_crops(a.crop_dir)
    y = np.array([m["gt_loc"] for m in meta])
    pm = C8.patient_map()
    groups = np.array([pm.get(m["case"], m["case"]) for m in meta])
    keys = [f"{m['case']}__{m['lesion_idx']}" for m in meta]

    classes_g, P_geo_all, sel = C24.geo_probs(a.feat, keys)
    classes_g = np.asarray(classes_g)
    ok = [i for i, s in enumerate(sel) if s is not None]
    P_geo = P_geo_all[[sel[i] for i in ok]]
    yo = y[ok]
    go = groups[ok]
    prior = collections.Counter(yo)
    pri = np.array([max(prior.get(c, 1), 1) for c in classes_g], float)
    pred_g = classes_g[np.argmax(norm(P_geo, pri, a.beta), 1)]
    print(f"[c25b] 샘플 {len(yo)} · 클래스 {len(classes_g)}")
    print(f"  기하 단독  top-1 {np.mean(pred_g == yo):.3f} · "
          f"macroRec {C8.macro_recall(yo, pred_g)[0]:.3f}\n")

    variants = {"영상만 ch0": [0], "영상+혈관 ch0,1,2": [0, 1, 2]}
    report = {}
    for name, ch in variants.items():
        F = C25.mipfeat(X, ch)
        cls_c, P_c_full = C25.oof_probs(F, y, groups)
        Pc = C25.align(cls_c, P_c_full[ok], classes_g)
        pred_c = classes_g[np.argmax(norm(Pc, pri, a.beta), 1)]

        # ── 일치표: 결합으로 얻을 수 있는 최대치를 못박는다 ──────────────
        gr, cr = (pred_g == yo), (pred_c == yo)
        both = int((gr & cr).sum()); only_g = int((gr & ~cr).sum())
        only_c = int((~gr & cr).sum()); neither = int((~gr & ~cr).sum())
        oracle = (both + only_g + only_c) / len(yo)
        print(f"--- {name} ---")
        print(f"  크롭 단독  top-1 {cr.mean():.3f} · macroRec {C8.macro_recall(yo, pred_c)[0]:.3f}")
        print(f"  일치표   둘다맞음 {both} · 기하만 {only_g} · 크롭만 {only_c} · 둘다틀림 {neither}")
        print(f"  오라클 상한 top-1 {oracle:.3f}  (기하 단독 {gr.mean():.3f}, "
              f"크롭이 새로 맞히는 몫 {only_c / len(yo):+.3f})")

        # ── 결합 규칙 1: 확률 평균 스윕 ─────────────────────────────────
        best_avg = None
        for w in np.arange(0, 1.01, 0.05):
            P = norm(w * norm(Pc) + (1 - w) * norm(P_geo), pri, a.beta)
            p = classes_g[np.argmax(P, 1)]
            mr = C8.macro_recall(yo, p)[0]
            if best_avg is None or mr > best_avg[1]:
                best_avg = (float(w), mr, float(np.mean(p == yo)))
        print(f"  확률평균 최적 w={best_avg[0]:.2f} · macroRec {best_avg[1]:.3f} · top-1 {best_avg[2]:.3f}")

        # ── 결합 규칙 2: 스태킹 (두 확률벡터를 입력으로 메타분류기) ──────
        # 정보는 다른데 평균이 못 살리는 경우를 잡는다. 메타 학습도 환자단위 CV 로.
        Z = np.concatenate([norm(P_geo), norm(Pc)], axis=1)
        pred_s = np.empty(len(yo), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(Z, yo, go):
            mclf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            mclf.fit(Z[tr], yo[tr])
            Pm = mclf.predict_proba(Z[te])
            prim = np.array([max(collections.Counter(yo[tr]).get(c, 1), 1)
                             for c in mclf.classes_], float)
            pred_s[te] = mclf.classes_[np.argmax(Pm / (prim ** a.beta), 1)]
        mr_s = C8.macro_recall(yo, pred_s)[0]
        print(f"  스태킹(LR)  macroRec {mr_s:.3f} · top-1 {np.mean(pred_s == yo):.3f}")

        # ── 명분 클래스에서만 ───────────────────────────────────────────
        for f in FOCUS:
            m = np.array([f in c for c in yo])
            if m.sum() == 0:
                continue
            print(f"  [{f}] n={int(m.sum())}  기하 {gr[m].mean():.3f} · 크롭 {cr[m].mean():.3f} "
                  f"· 오라클 {(gr | cr)[m].mean():.3f}")
        print()
        report[name] = {"channels": ch, "crop_top1": float(cr.mean()),
                        "crop_macro_recall": float(C8.macro_recall(yo, pred_c)[0]),
                        "both": both, "only_geo": only_g, "only_crop": only_c,
                        "neither": neither, "oracle_top1": float(oracle),
                        "avg_best_w": best_avg[0], "avg_best_macro_recall": best_avg[1],
                        "stack_macro_recall": float(mr_s),
                        "focus": {f: {"n": int(np.sum([f in c for c in yo])),
                                      "geo": float(gr[[f in c for c in yo]].mean())
                                      if any(f in c for c in yo) else None,
                                      "crop": float(cr[[f in c for c in yo]].mean())
                                      if any(f in c for c in yo) else None}
                                  for f in FOCUS}}

    geo_mr = float(C8.macro_recall(yo, pred_g)[0])
    best = max(report, key=lambda k: max(report[k]["avg_best_macro_recall"],
                                         report[k]["stack_macro_recall"]))
    gain = max(report[best]["avg_best_macro_recall"], report[best]["stack_macro_recall"]) - geo_mr
    print(f"[결론] 최선 결합 {best} · macro-recall 이득 {gain:+.3f} (기하 단독 {geo_mr:.3f})")
    print("  이득이 +0.02 미만이면 크롭 노선은 여기서 접는다 — 일치표의 '크롭만' 칸이 그 근거다.")
    json.dump({"geo_macro_recall": geo_mr, "variants": report, "best": best, "gain": gain},
              open(A / "c25b_report.json", "w"), indent=1, ensure_ascii=False)
    print(f"[저장] {A / 'c25b_report.json'}")


if __name__ == "__main__":
    main()
