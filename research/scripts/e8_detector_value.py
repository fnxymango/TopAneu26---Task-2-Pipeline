"""E8 — 검출기 민감도가 오르면 e2e 점수가 얼마나 오르는가 (2026-08-19).

질문: "0.86짜리 검출기를 가져오면 성능이 얼마나 늘까". 현행은 test 63/86 = 0.733.

기존 방식(천장과 e2e 를 직선으로 잇기)은 두 가지를 섞어버린다 —
천장은 민감도 1.0 **이면서 위양성 0** 이다. 그래서 그 직선은 "민감도만 올랐을 때"의
기울기가 아니라 "민감도가 오르고 위양성도 사라졌을 때"의 기울기다. 과대추정이다.

여기서는 둘을 분리한다. 공식 지표의 TP/FP/FN 은 **케이스×클래스 단위 존재 여부**로
세므로(evaluate.py evaluation_function), 병변별 (케이스, 정답클래스, 예측클래스)만 알면
볼륨을 만들지 않고 macro-MCC 를 정확히 재구성할 수 있다. 그래서 수천 번 표집이 싸다.

두 가지 회복 모형:
  nested  현재 맞힌 63개는 그대로 두고, **놓친 23개 중 k개**를 회복한다 (현실적)
  uniform 86개에서 매번 새로 표집한다 (검출기를 통째로 갈아끼우는 경우)

Dice/VolSim/HD95 는 볼륨이 필요해 여기서 안 낸다 — MCC 만 본다(순위 기준이 MCC 다).

사용: python e8_detector_value.py [--draws 400]
"""
import argparse, collections, json, os
from pathlib import Path

import numpy as np
import nibabel as nib

import d9xx_lib as L
import c5_location_v2 as C5

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
P = L.TOPANEU_ROOT / "experiments" / "_c1_realpred"
BP = L.TOPANEU_ROOT / "experiments" / "_c4_bpgraph"


def macro_mcc(cases, present, name2id):
    """케이스별 (GT클래스집합, 예측클래스집합) 에서 클래스별 MCC 를 내고 present 로 평균."""
    N = len(cases)
    out = []
    for c in present:
        tp = fp = fn = tn = 0
        for gset, pset in cases:
            g, p = c in gset, c in pset
            if g and p: tp += 1
            elif p:     fp += 1
            elif g:     fn += 1
            else:       tn += 1
        d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        out.append(((tp * tn - fp * fn) / d) if d > 0 else 0.0)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--model", default="rf")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--conf-tau", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    C5.USE_POS = True; C5.CONF_TAU = a.conf_tau; C5.CONF_BETA_HI = 0.0
    id2name, name2id = L.official_location_names()
    ves_names = L.vessel_dense_names()
    ves_axis, _ = C5.build_feature_axes(ves_names)
    train_rows = json.load(open(A / "c10_feat_train.json"))
    _, _, test_ids = L.case_ids_by_split()

    # ── 1) test GT 병변 전부에 대해 (케이스, 정답, 예측) 을 뽑는다 (시드별) ──
    cache = A / f"e8_lesion_pred_{a.model}.json"
    if cache.exists():
        blob = json.load(open(cache))
    else:
        blob = {"gt": {}, "fp": []}
        for sd in range(a.seeds):
            os.environ["CLF_SEED"] = str(sd)
            model = C5.fit_model(train_rows, ves_axis, kind=a.model, mirror=True, balance=True)
            recs = []
            for cid in test_ids:
                vp = P / "vespp_test" / f"{cid}.nii.gz"
                if not vp.exists(): continue
                vi = nib.load(vp); ves = np.asanyarray(vi.dataobj)
                sp = np.array(vi.header.get_zooms()[:3], dtype=float)
                nodes = C5.load_bp(str(BP / "vespp_test"), cid)
                gt = np.asanyarray(nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz").dataobj)
                rows, _ = C5.extract_case_rows(gt, ves, sp, ves_names, nodes, id2name)
                for r in rows:
                    recs.append([cid, r["gt_loc"], C5.predict_one(model, r, a.beta),
                                 r["lesion_mask_idx"]])
            blob["gt"][str(sd)] = recs
            print(f"[e8] 시드{sd}: GT 병변 {len(recs)}개 분류 완료", flush=True)
        d2 = json.load(open(A / "e2_model_diff_test_rf_et.json"))
        blob["fp"] = [[r["case"], r[a.model]] for r in d2["records"] if not r["truth"]]
        blob["covered"] = [[r["case"], r["truth"]] for r in d2["records"] if r["truth"]]
        json.dump(blob, open(cache, "w"), ensure_ascii=False)
        print(f"[e8] 저장 {cache}", flush=True)

    # present 클래스: test GT 에 등장하는 것 (공식 분모와 동일하게 36 이어야 한다)
    ref = blob["gt"]["0"]
    present = sorted({name2id[t] for _, t, _, _ in ref if t in name2id})
    fps = blob["fp"]
    print(f"[e8] GT 병변 {len(ref)} · present 클래스 {len(present)} · 고정 FP {len(fps)}")

    # 현재 덮인 병변 식별 — e2 기록의 (케이스, 정답) 다중집합과 맞춘다
    cov = collections.Counter(tuple(x) for x in blob["covered"])
    covered_idx, rest_idx = [], []
    seen = collections.Counter()
    for i, (cid, truth, _, _) in enumerate(ref):
        k = (cid, truth)
        if seen[k] < cov.get(k, 0):
            covered_idx.append(i); seen[k] += 1
        else:
            rest_idx.append(i)
    print(f"[e8] 현재 덮음 {len(covered_idx)} · 놓침 {len(rest_idx)}")

    def score(sel, recs, fp_mult=1.0, rng=None):
        by = collections.defaultdict(lambda: (set(), set()))
        for cid, truth, _, _ in recs:
            by[cid][0].add(name2id.get(truth))
        for i in sel:
            cid, _, pred = recs[i][0], recs[i][1], recs[i][2]
            if pred in name2id: by[cid][1].add(name2id[pred])
        # FP 배율: 실제 FP 24개밖에 없으므로 1.0 을 넘기려면 **복원추출로 늘리고
        # 케이스도 새로 뽑아** 붙인다. (초판 버그 — min(nfp, len(fps)) 로 잘라서
        # 1.5/2.0 배가 1.0 과 같은 값이 나왔다. 2026-08-19 수정)
        nfp = int(round(len(fps) * fp_mult))
        all_cids = list(by.keys())
        for j in range(nfp):
            if j < len(fps):
                cid, pred = fps[j]
            else:                                   # 늘린 몫: 라벨은 기존 FP 분포에서,
                _, pred = fps[rng.randint(len(fps))]   # 케이스는 test 전체에서 무작위
                cid = all_cids[rng.randint(len(all_cids))]
            if pred in name2id: by[cid][1].add(name2id[pred])
        cases = [(g, p) for g, p in by.values()]
        return macro_mcc(cases, present, name2id)

    print(f"\n=== 민감도별 test cov.MCC (모형: nested — 현재 63개 유지 + 놓친 23개 중 회복) ===")
    print(f"{'민감도':>7}{'병변':>6}{'회복':>6}| " + "".join(f"{'FP×'+f'{m}':>11}" for m in (1.0, 1.5, 2.0)))
    n_all = len(ref)
    rows = []
    for s in (0.733, 0.80, 0.86, 0.90, 0.95, 1.0):
        k = max(0, int(round(s * n_all)) - len(covered_idx))
        k = min(k, len(rest_idx))
        line = []
        for mult in (1.0, 1.5, 2.0):
            vals = []
            for sd in range(a.seeds):
                recs = blob["gt"][str(sd)]
                rng = np.random.RandomState(1000 + sd)
                for _ in range(a.draws // a.seeds):
                    add = rng.choice(rest_idx, k, replace=False) if k else []
                    vals.append(score(list(covered_idx) + list(add), recs, mult, rng))
            vals = np.array(vals); line.append((vals.mean(), vals.std()))
        rows.append((s, len(covered_idx) + k, k, line))
        print(f"{s:>7.3f}{len(covered_idx)+k:>6}{k:>6}| " +
              "".join(f"{m:>7.4f}±{sd:<4.3f}" for m, sd in line))

    base = rows[0][3][0][0]
    print(f"\n  기준(0.733, FP 그대로) {base:.4f}")
    for s, n, k, line in rows[1:]:
        print(f"  민감도 {s:.2f} -> {line[0][0]:.4f} ({line[0][0]-base:+.4f})"
              f"   FP 1.5배면 {line[1][0]-base:+.4f} · 2배면 {line[2][0]-base:+.4f}")
    print(f"\n  ⚠️ 내부 판정용 근사다 — MCC 만 재구성했고 Dice/VolSim/HD95 는 빼놨다.")
    print(f"     시드 {a.seeds}판 x 표집 {a.draws}회. 팀 표에는 실제 검출기로 돌린 값을 올린다.")
    json.dump({"model": a.model, "beta": a.beta, "present": len(present),
               "rows": [(s, n, k, [[float(m), float(sd)] for m, sd in l]) for s, n, k, l in rows]},
              open(A / f"e8_detector_value_{a.model}.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
