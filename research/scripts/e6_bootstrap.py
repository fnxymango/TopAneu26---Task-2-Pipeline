"""E6 — 보고한 e2e 숫자에 **신뢰구간**을 붙인다 (2026-08-19).

지금까지 우리는 test cov.MCC 0.3609 같은 숫자를 소수점 4자리로 비교해 왔다.
그런데 test 는 83케이스 / 86병변뿐이고, 분모클래스가 36 이라 희소 클래스 병변 1개가
cov.MCC 를 0.028 움직인다. 그러면 "83케이스가 조금만 달랐어도" 숫자가 얼마나
달라지는지를 모르고서는 0.3609 와 0.3306 을 비교하는 게 의미가 없다.

부트스트랩: 케이스를 복원추출로 83개 다시 뽑아 공식 집계를 다시 돌린다. 2000회.
공식 지표는 케이스별 TP/FP/FN/TN 을 **합산한 뒤** 클래스별 MCC 를 내므로
케이스 단위 재표집이 정확히 맞는 단위다.

분모(present 클래스 36개)는 전체 test 기준으로 **고정**한다. 재표집마다 분모가
바뀌면 지표 변동과 분모 변동이 섞여 해석이 안 된다.

사용: python e6_bootstrap.py [--n 2000] [--tags e4_et_s0,e4_rf_s0]
"""
import argparse, glob, json, os, sys
import numpy as np

import d9xx_lib as L

A = L.TOPANEU_ROOT / "code" / "sblee" / "nnunet" / "analysis"
KEYS = ["PRECISION", "RECALL", "MCC", "DICE", "HD95", "VOLSIM"]


def load_official():
    sys.path.insert(0, str(L.TOPANEU_ROOT / "code" / "TopAneu-26" / "eval" / "task2"))
    import evaluate as ev
    return ev


def score(ev, per_case, pres):
    agg = ev.evaluation_aggregation(per_case)
    off = ev.evaluation_average(agg)
    adj = {k: float(np.mean([agg[f"{k}_{i}"] for i in pres])) for k in KEYS}
    return off, adj


def comp(o):
    return (o["PRECISION"] + o["RECALL"] + o["MCC"] + o["DICE"] + o["VOLSIM"] + 1 - o["HD95"]) / 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tags", default=None, help="쉼표구분. 없으면 percase 파일 전부")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ev = load_official()

    files = sorted(glob.glob(str(A / f"c5_percase_{a.split}_*.json")))
    if a.tags:
        want = set(a.tags.split(","))
        files = [f for f in files if os.path.basename(f)[len(f"c5_percase_{a.split}_"):-5] in want]
    if not files:
        print(f"[e6] c5_percase_{a.split}_*.json 이 없다. c5 eval 을 다시 돌려야 원자료가 남는다.")
        return

    boot = {}
    for f in files:
        d = json.load(open(f)); pc, pres = d["per_case"], d["present"]
        tag = os.path.basename(f)[len(f"c5_percase_{a.split}_"):-5]
        off0, adj0 = score(ev, pc, pres)
        rng = np.random.RandomState(0); n = len(pc)
        cm, om = [], []
        for _ in range(a.n):
            idx = rng.randint(0, n, n)
            o, c = score(ev, [pc[i] for i in idx], pres)
            cm.append(c["MCC"]); om.append(o["MCC"])
        cm, om = np.array(cm), np.array(om)
        boot[tag] = {"n_cases": n, "cov_mcc": off0 and adj0["MCC"], "off_mcc": off0["MCC"],
                     "cov_ci": [float(np.percentile(cm, 2.5)), float(np.percentile(cm, 97.5))],
                     "off_ci": [float(np.percentile(om, 2.5)), float(np.percentile(om, 97.5))],
                     "cov_sd": float(cm.std()), "off_sd": float(om.std()),
                     "_cov_draws": cm.tolist()}
        print(f"[e6] {tag:<14} cov.MCC {adj0['MCC']:.4f}  95%CI [{np.percentile(cm,2.5):.4f}, "
              f"{np.percentile(cm,97.5):.4f}]  sd {cm.std():.4f}   | off.MCC {off0['MCC']:.4f} "
              f"CI [{np.percentile(om,2.5):.4f}, {np.percentile(om,97.5):.4f}]", flush=True)

    tags = list(boot)
    if len(tags) > 1:
        print(f"\n=== 쌍대 비교 (같은 부트스트랩 표본에서의 차이) ===")
        print(f"{'A vs B':<32}{'ΔcovMCC':>10}{'95%CI':>22}{'B가 이길 확률':>14}")
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                x = np.array(boot[tags[i]]["_cov_draws"]); y = np.array(boot[tags[j]]["_cov_draws"])
                d = y - x
                print(f"{tags[i]+' vs '+tags[j]:<32}{d.mean():>+10.4f}"
                      f"{f'[{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}]':>22}"
                      f"{float((d>0).mean()):>14.3f}")
        print("\n  주의 — 케이스 재표집만 짝지었을 뿐 두 설정이 독립 추정이라, 이 Δ 는 보수적이다.")
    for v in boot.values():
        v.pop("_cov_draws", None)
    json.dump(boot, open(A / f"e6_bootstrap_{a.split}.json", "w"), indent=1, ensure_ascii=False)
    print(f"\n[저장] {A / f'e6_bootstrap_{a.split}.json'}")


if __name__ == "__main__":
    main()
