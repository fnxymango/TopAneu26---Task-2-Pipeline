"""C18 — 공식 6지표 복합으로 하이퍼파라미터 재선택.

발견(2026-08-16, 사용자 지적): 공식 랭킹은 MCC 단독이 아니라
**Precision/Recall/MCC/Dice/VolSim/HD95 6개의 평균**이다 (eval/task2/README.md §Ranking).
시뮬레이션 표에서 완벽=HD95 0.00, 최악=HD95 1.00 이므로 HD95만 낮을수록 좋다.

우리는 지금까지 MCC(와 macro-recall)로 모든 걸 골랐다. 그 결과:
  - 5-fold 앙상블: MCC 0.2038 -> 0.2207 (+8.3%) 인데
    DICE 0.1331 -> 0.1286, HD95 0.5887 -> 0.6490 으로 악화 -> 복합으로는 사실상 동률
즉 **목적함수가 랭킹과 어긋나 있었다.** 여기서 기존 결과를 복합으로 재집계하고,
beta 를 복합 기준으로 다시 스윕한다(계산은 재평가 없이 예측만 다시 하면 됨).

복합 정의 두 가지를 모두 낸다:
  literal : README 문자 그대로 (HD95 원값을 그냥 평균)
  aligned : HD95 를 1-HD95 로 방향 보정해 평균 (해석상 이쪽이 자연스러움)
순위가 둘 사이에서 갈리면 그 사실 자체를 보고한다.

사용:
  python c18_composite_reselect.py                     # 기존 결과 재집계
  python c18_composite_reselect.py --beta-sweep --split val --...   # beta 재스윕
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np

import d9xx_lib as L

POS = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM"]     # 높을수록 좋음
A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"


def composite(o):
    """(literal, aligned). o = official_div52 또는 adjusted_div_present 딕셔너리."""
    s = sum(float(o[k]) for k in POS)
    return (s + float(o["HD95"])) / 6.0, (s + (1.0 - float(o["HD95"]))) / 6.0


def report():
    rows = []
    pats = ["c5_eval_*_*.json", "d910_eval_test_realpred_raw.json", "d910_eval_val_realpred_pp.json"]
    seen = set()
    for pat in pats:
        for f in glob.glob(str(A / pat)):
            if "SMOKE" in f or f in seen:
                continue
            seen.add(f)
            d = json.load(open(f))
            o = d.get("official_div52")
            if not o:
                continue
            lit, ali = composite(o)
            name = os.path.basename(f).replace("c5_eval_", "").replace(".json", "")
            name = name.replace("d910_eval_", "C2_")
            split = "test" if "test" in name else "val"
            rows.append((split, name, o["MCC"], o["DICE"], o["VOLSIM"], o["HD95"], lit, ali))

    for sp in ("test", "val"):
        sub = [r for r in rows if r[0] == sp]
        if not sub:
            continue
        print(f"\n=== {sp.upper()} — 공식 52클래스 평균 ===")
        print(f"{'실험':<30}{'MCC':>8}{'DICE':>8}{'VolSim':>8}{'HD95':>8}{'복합(문자)':>11}{'복합(보정)':>11}")
        for r in sorted(sub, key=lambda r: -r[7]):
            print(f"{r[1]:<30}{r[2]:>8.4f}{r[3]:>8.4f}{r[4]:>8.4f}{r[5]:>8.4f}{r[6]:>11.4f}{r[7]:>11.4f}")

        # 순위가 MCC 기준과 복합 기준에서 갈리는지
        by_mcc = [r[1] for r in sorted(sub, key=lambda r: -r[2])]
        by_cmp = [r[1] for r in sorted(sub, key=lambda r: -r[7])]
        if by_mcc != by_cmp:
            print(f"  ⚠️ MCC 순위와 복합 순위가 다름")
            print(f"     MCC : {' > '.join(by_mcc[:4])}")
            print(f"     복합: {' > '.join(by_cmp[:4])}")
        else:
            print("  MCC 순위 == 복합 순위 (MCC를 대표로 써도 방향 일치)")

    json.dump([{"split": r[0], "name": r[1], "MCC": r[2], "DICE": r[3], "VOLSIM": r[4],
                "HD95": r[5], "composite_literal": r[6], "composite_aligned": r[7]} for r in rows],
              open(A / "c18_composite_table.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n[저장] {A / 'c18_composite_table.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.parse_args()
    report()
