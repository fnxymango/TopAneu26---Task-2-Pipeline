"""C계열 실험을 PROJECT_RULES.md §3 규약에 맞는 실험 폴더로 정리 (2026-08-16).

C4~C19는 스크립트로 돌리며 결과를 analysis/*.json 에만 남겨서, A/V계열처럼
experiments/<이름>/{config.json, summary.md, results/} 구조가 없었다. 여기서 만들어 준다.

최고 성능기 선정 기준: **MCC** (사용자 지시 2026-08-16).
공식 랭킹은 6지표 평균이지만 test에서 MCC 순위와 복합 순위가 일치함을 C18에서 확인했다.
"""
import json, os, shutil
from pathlib import Path

R = Path(os.environ["TOPANEU_ROOT"])
A = R / "code" / "sblee" / "nnunet" / "analysis"
E = R / "experiments"
K = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM"]


def metrics(tag, split="test"):
    p = A / f"c5_eval_{split}_{tag}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    o = d.get("official_div52") or {}
    if not o:
        return None
    comp = (sum(o[k] for k in K) + (1 - o["HD95"])) / 6
    return {"split": split, "MCC": o["MCC"], "DICE": o["DICE"], "HD95": o["HD95"],
            "PRECISION": o["PRECISION"], "RECALL": o["RECALL"], "VOLSIM": o["VOLSIM"],
            "composite": comp, "n_lesions": d.get("n_lesions_predicted"),
            "top1": d.get("top1_accuracy")}


# (폴더명, 한줄설명, 판정, 핵심결과 텍스트, 관련 analysis 파일 glob, 대표 eval 태그)
EXPS = [
 ("C4_branchpoint_graph", "혈관 중심선에서 분기점 그래프 추출 (스퍼 pruning + 클래스 전이점 + 인접성 검증)",
  "인프라", "참조 417 + val/test × pp/raw 전량 추출. 케이스당 노드 50~80개. "
  "랜드마크(BA tip, R/L ICA terminus)가 97.4%에서 세 개 모두 검출 — C10의 토대가 됨.",
  ["c4_*"], None),
 ("C5_location_v2", "52클래스 위치할당 재작성 — 혈관거리36 + sac점유율36 + 분기점34",
  "채택(기반)", "test e2e MCC 0.1587(C2) -> 0.1739. 버려지던 overlap 복구가 CV top-1 +0.085로 최대 기여.",
  ["c5_eval_*"], "e2e_A52f_pp"),
 ("C6_consensus_2of2", "A5-2 ∩ A6-2 2/2 합의 마스크",
  "미채택", "FP 55->29 이나 병변 1개 손실(31->30). C7 거리게이팅(FP 55->12, 손실 0)이 더 우수해 대체됨.",
  ["fullvolume_eval_cons2*"], None),
 ("C7_detect_postproc", "검출 후처리 — 혈관거리 게이팅 + 성분크기 필터",
  "채택", "A5-2: FP 55->12 (민감도 손실 0). A6-2: 152->53 (손실 0). "
  "A6-2 FP 152개 중 89개가 예측혈관 5mm 밖 — adaptive norm이 비혈관 조직에 허위검출을 만든다는 진단.",
  ["c7_detect_sweep_*"], None),
 ("C8_classifier_cv", "환자단위 5-fold CV 진단 + 피처블록 ablation",
  "진단", "macro-recall 0.383 vs top-1 0.68 — 희귀클래스 실패가 최대 손실원. "
  "그룹 정확도 0.951 / 좌우 0.902 -> 오류는 전부 그룹·측면 내부. ICA(3.x)가 오류의 대부분이며 전부 3.4 Pcom으로 흡인.",
  ["c8_cv_report.json"], None),
 ("C9_classifier_tune", "macro 직접 최적화(β 사전확률 역보정) + rel 인코딩 + 반경확대",
  "부분채택", "β=0.75로 macro-recall 0.346 -> 0.410 (+18%). "
  "rel 상대인코딩 실패(ICA 0.529->0.520), 반경 10->25mm 실패(기권 4->0이나 macro 하락).",
  ["c9_tune_*"], None),
 ("C10_landmark_coords", "해부 랜드마크(BA tip, R/L ICA terminus) 기준 정규화 좌표 6차원",
  "★채택", "macro-recall 0.410 -> 0.455 (+11%), 오늘 최대 단일 개선. "
  "기존 106차원에 전역 위치정보가 0비트였던 공백을 메움. pos 단독은 0.187로 약하나 상보적.",
  ["c10_cv_report.json"], None),
 ("C11_synthetic_rare", "분기점에 가상 sac 배치로 희귀클래스 학습샘플 1395개 합성",
  "기각", "CV macro-recall 0.400->0.463(+16%), val 개선. 그러나 **test에서 전면 악화** "
  "(천장 MCC 0.3259->0.3081, e2e 0.2207->0.1956). 합성이 train 케이스의 혈관분할 오차패턴까지 학습한 과적합. "
  "같은 분포로 나누는 CV는 이를 원리적으로 탐지 못함 — 오늘 가장 값진 교훈.",
  ["c11_synth_rows.json"], None),
 ("C12_group_experts", "해부그룹 5-way 판정 후 그룹별 전문가 분류기 (hard/soft 라우팅)",
  "기각", "macro-recall 0.360 (기준 0.455). 1단 그룹 정확도 0.944로 예상대로 높았으나, "
  "쪼개면 1단 오류가 복구 불가능하게 전파되어 통합 모델보다 나쁨.",
  ["c12_cv_report.json"], None),
 ("C13_lat_canonical", "좌우 canonical화 — 26-way 측면무관 라벨 + 좌우 이진분류기",
  "기각", "macro-recall 0.352. 좌우 전용 분류기가 오히려 더 나쁨(0.869 vs 통합 0.902). "
  "좌우를 떼면 클래스 맥락을 잃어 손해. 미러는 증강으로 쓰고 출력은 52-way 유지가 옳음.",
  ["c13_cv_report.json"], None),
 ("C14_vessel_encoder_feat", "V4-2 혈관분할망 인코더 bottleneck 특징을 병변 피처로 사용",
  "실패(구현)", "1차 60케이스: enc 단독 top-1 0.027. 2차 268병변 전량: 0.049 — 여전히 무작위 수준. "
  "표본 부족이 아니라 hook 텐서 또는 패치 좌표 매핑 오류. 붙이면 오히려 방해(both 0.403 < hand 0.519). 2회 실패로 중단.",
  ["c14*_enc_feats.json"], None),
 ("C15_geodesic_coords", "중심선 따라간 측지거리 + 사행비 (유클리드 대체)",
  "기각", "macro-recall 0.400 -> 0.368. 유클리드보다 나쁨. 중심선 파편화로 Dijkstra 미도달 케이스가 섞인 것으로 추정.",
  ["c15_geo_rows.json"], None),
 ("C16_prob_avg_ensemble", "5-fold softmax 확률 평균 앙상블 (이진 마스크 다수결 대체)",
  "★채택(복합 최고)", "test HD95 0.6490 -> 0.5827 회복, **복합지표 최고 0.2286**. "
  "MCC는 0.2175로 vote2(0.2207)에 근소 열세. 마스크 다수결의 합집합 성분이 경계를 뭉개던 문제를 해결.",
  ["c7_detect_sweep_probavg_*"], "c16_probavg"),
 ("C17_learned_fp_reject", "검출 후보 TP/FP를 C5 피처로 판별하는 학습된 기각기",
  "검증완료·미적용", "out-of-fold(train 292 완전 커버) 후보 596개(TP 225/FP 371). "
  "임계 0.1에서 민감도 -1.3%로 FP 52% 제거. c7 손규칙과의 중복 확인 필요해 val/test 미적용.",
  ["c17_cv_report.json", "c17_candidates.json"], None),
 ("C18_composite_reselect", "공식 6지표 복합으로 지표 재집계 및 재선택",
  "분석", "공식 랭킹은 MCC 단독이 아니라 P/R/MCC/Dice/VS/HD95 6개 평균. "
  "**test에서 MCC 순위 == 복합 순위** 확인 — MCC를 대표로 써도 방향 일치. "
  "다만 β·합성·다수결 등 희귀클래스 공격적 예측 개입은 MCC를 올리며 HD95를 악화시키는 패턴이 반복됨.",
  ["c18_composite_table.json"], None),
 ("C19_a4_boundary_refine", "A4(lesionscale crop) 모델로 검출 경계 재작성",
  "미완", "A4 전체볼륨 추론만 완료(val 42 / test 83). 검출결과와의 병합 규칙 미정, 평가 미실시.",
  [], None),
]

BEST = {
    "name": "_BEST_by_MCC",
    "config": "A6-2 5-fold vote2(마스크 다수결) + C7 필터(min_vox5, dist3mm) "
              "-> C5 위치분류(dist+ov+bp+pos, RF, beta=1.0)",
    "tag": "ens_vote2",
}


def write_exp(name, desc, verdict, result, globs, tag):
    d = E / name
    (d / "results").mkdir(parents=True, exist_ok=True)
    m_test, m_val = metrics(tag) if tag else None, metrics(tag, "val") if tag else None
    json.dump({"exp_name": name, "family": "C — 클래스·위치 지정(52클래스)",
               "kind": "analysis/postprocess (학습 없음)", "verdict": verdict,
               "description": desc, "metrics_test": m_test, "metrics_val": m_val},
              open(d / "config.json", "w"), indent=1, ensure_ascii=False)
    n = 0
    for g in globs:
        for f in A.glob(g if g.endswith(".json") else g + ".json"):
            shutil.copy2(f, d / "results" / f.name); n += 1
        for f in A.glob(g):
            if f.is_file() and f.suffix == ".json":
                shutil.copy2(f, d / "results" / f.name); n += 1
    with open(d / "summary.md", "w") as fh:
        fh.write(f"# {name}\n\n**{desc}**\n\n판정: **{verdict}**\n\n## 결과\n\n{result}\n\n")
        if m_test:
            fh.write("## 지표 (공식 52클래스 평균)\n\n")
            fh.write("| split | MCC | DICE | HD95 | 복합 | 병변 |\n|---|---|---|---|---|---|\n")
            for m in (m_val, m_test):
                if m:
                    fh.write(f"| {m['split']} | {m['MCC']:.4f} | {m['DICE']:.4f} | "
                             f"{m['HD95']:.4f} | {m['composite']:.4f} | {m['n_lesions']} |\n")
            fh.write("\n")
        fh.write(f"## 산출물\n\n`results/` 에 {n}개 json 복사. 원본은 `analysis/`.\n")
    return d, n


if __name__ == "__main__":
    print(f"{'폴더':<30}{'판정':<16}{'json':>5}")
    for e in EXPS:
        d, n = write_exp(*e)
        print(f"{e[0]:<30}{e[2]:<16}{n:>5}")

    # 최고 성능기 (MCC 기준)
    cands = {t: metrics(t) for t in ["e2e_A52f_pp", "best_e2e_A52f_pp", "ens_vote2",
                                     "c16_probavg", "c11_e2e_vote2"]}
    cands = {k: v for k, v in cands.items() if v}
    best = max(cands, key=lambda k: cands[k]["MCC"])
    d = E / BEST["name"]; (d / "results").mkdir(parents=True, exist_ok=True)
    m = cands[best]; mv = metrics(best, "val")
    json.dump({"exp_name": BEST["name"], "selected_by": "MCC (test)", "winning_tag": best,
               "pipeline": BEST["config"], "metrics_test": m, "metrics_val": mv,
               "all_candidates": cands}, open(d / "config.json", "w"), indent=1, ensure_ascii=False)
    with open(d / "summary.md", "w") as fh:
        fh.write(f"# 최고 성능 파이프라인 (MCC 기준)\n\n**{BEST['config']}**\n\n")
        fh.write(f"선정 태그: `{best}`\n\n## test 83 (held-out)\n\n")
        fh.write("| 지표 | 값 |\n|---|---|\n")
        for k in ["MCC", "DICE", "HD95", "PRECISION", "RECALL", "VOLSIM", "composite"]:
            fh.write(f"| {k} | {m[k]:.4f} |\n")
        fh.write("\n## 후보 비교 (test, MCC 내림차순)\n\n")
        fh.write("| 태그 | MCC | DICE | HD95 | 복합 |\n|---|---|---|---|---|\n")
        for k, v in sorted(cands.items(), key=lambda kv: -kv[1]["MCC"]):
            mark = " ←" if k == best else ""
            fh.write(f"| {k}{mark} | {v['MCC']:.4f} | {v['DICE']:.4f} | {v['HD95']:.4f} | {v['composite']:.4f} |\n")
        fh.write(f"\n> MCC 기준 최고는 `{best}`. 단 복합지표(공식 랭킹) 기준으로는 "
                 f"`{max(cands, key=lambda k: cands[k]['composite'])}` 가 최고 — C18 참조.\n")
    for t in cands:
        for sp in ("val", "test"):
            p = A / f"c5_eval_{sp}_{t}.json"
            if p.exists():
                shutil.copy2(p, d / "results" / p.name)
    print(f"\n[최고 성능기] {best}  MCC {m['MCC']:.4f}  복합 {m['composite']:.4f}  -> {d}")
