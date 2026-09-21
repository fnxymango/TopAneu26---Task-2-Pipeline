"""C계열 실험 폴더/summary 생성 v2 (2026-08-17).

v1(organize_c_experiments.py) 대비 변경:
  1. **covered_gt 를 대표 분모로** 기록 (PROJECT_RULES.md §6-1, 사용자 지시 2026-08-17).
     covered_gt = eval json 의 `adjusted_div_present` = 그 split GT 등장 클래스 수로 나눈 평균.
     official_div52 도 나란히 적는다(리더보드가 ÷52).
  2. **순위·최고 판정은 MCC** (사용자 지시 2026-08-17). 복합은 참고값으로만 병기하고,
     MCC 순위와 복합 순위가 갈리면 그 사실을 summary 에 남긴다.
  3. C1~C3, C20~C28 항목 추가. 후보 태그는 analysis/ 를 스캔해 자동 편입.
  4. 재실행 안전(idempotent) — 언제 다시 돌려도 최신 json 으로 덮어쓴다.

사용: TOPANEU_ROOT=... python organize_c_experiments_v2.py
"""
import json, os, glob, shutil, datetime
from pathlib import Path

R = Path(os.environ["TOPANEU_ROOT"])
A = R / "code" / "sblee" / "nnunet" / "analysis"
E = R / "experiments"
POS = ["PRECISION", "RECALL", "MCC", "DICE", "VOLSIM"]
K6 = POS + ["HD95"]
FAM = "C — 클래스·위치 지정(52클래스)"


def _one(d, blk):
    o = d.get(blk)
    if not o:
        return None
    comp = (sum(o[k] for k in POS) + (1 - o["HD95"])) / 6
    return {**{k: o[k] for k in K6}, "composite": comp}


def metrics(tag, split="test"):
    """covered_gt(대표) + official_div52(리더보드) 둘 다."""
    p = A / f"c5_eval_{split}_{tag}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    cov, off = _one(d, "adjusted_div_present"), _one(d, "official_div52")
    if not cov or not off:
        return None
    return {"split": split, "tag": tag, "covered_gt": cov, "official_div52": off,
            "n_present_classes": d.get("n_present_classes_in_split"),
            "n_lesions": d.get("n_lesions_predicted"), "top1": d.get("top1_accuracy"),
            "eval_mode": d.get("eval_mode", "official_e2e"),
            "detect_coverage": d.get("detect_coverage"), "beta": d.get("beta")}


# (폴더, 한줄설명, 판정, 결과서술, analysis glob, 대표 eval 태그)
EXPS = [
 ("C1_location_signature_lookup", "혈관 근접도 서명 기반 lookup 위치할당 (최초 시도)",
  "대체됨", "34차원 dist_mm 만으로 52클래스 할당. 분기점 정보가 없어 오라클 상한 61%에서 막힘. C5로 대체.",
  ["d9xx_eval_*", "d900_*"], None),
 ("C2_location_weighted_knn", "가중 kNN + 클래스 사전확률 보정 위치할당",
  "대체됨", "test e2e MCC 0.1587 — 8/11 시점 기준선. C5(분기점 피처)로 대체.",
  ["d910_eval_*"], None),
 ("C3_location_knn_a62det", "A6-2(adaptive norm) 검출기로 C2 재평가",
  "진단", "A6-2 단일은 민감도 0.814로 A5-2(0.721)보다 높은데 FP가 3배(55→152)라 e2e는 오히려 짐 "
  "(val adj MCC 0.2164 vs 0.2651). '병목은 민감도가 아니라 FP'라는 결론 → C6/C7의 출발점.",
  ["fullvolume_eval_A6-2*"], None),
 ("C4_branchpoint_graph", "혈관 중심선에서 분기점 그래프 추출 (Lee thinning + 2mm 스퍼 제거 + 클래스 전이점 + V5 인접성 검증)",
  "인프라", "참조 417 + val/test × pp/raw 전량 추출. 랜드마크(BA tip, R/L ICA terminus) 세 개가 "
  "417케이스 중 97.4%에서 모두 검출 — C10 좌표계의 토대. 5.2 vs 5.3이 분기점 절대거리(0.7 vs 5.0mm)로 분리됨을 확인.",
  ["c4_*"], None),
 ("C5_location_v2", "위치할당 재작성 — 혈관거리 36 + sac 점유율 36 + 분기점 34 (=106차원)",
  "채택(기반)", "test e2e MCC 0.1587(C2) → 0.1739. 기존 코드가 계산해놓고 버리던 overlap(ov) 복구가 "
  "CV top-1 +0.085로 최대 기여. 좌우 미러 증강(268→536)도 여기서 도입.",
  ["c5_eval_*"], "e2e_A52f_pp"),
 ("C6_consensus_2of2", "모델 합의 마스크 (2/2 교집합 → 이후 5-fold N/M 다수결로 일반화)",
  "부분채택", "2/2 합의는 FP 55→29이나 병변 1개 손실. C7 거리게이팅(55→12, 손실 0)이 더 우수해 단독으로는 미채택. "
  "다만 `--mode lesion` 5-fold 다수결은 최종 파이프라인에 채택됨.",
  ["fullvolume_eval_cons2*"], None),
 ("C7_detect_postproc", "검출 후처리 — 혈관거리 게이팅 + 성분크기 필터 (재학습 0)",
  "★채택", "A5-2: FP 55→12, 민감도 손실 0 (min_vox 20 / dist 5mm). A6-2: 152→53. "
  "근거는 GT 병변–혈관 거리 중앙값 0.30mm / p95 0.55mm. "
  "A6-2 FP 152개 중 89개가 예측혈관 5mm 밖 — adaptive norm이 비혈관 조직에 허위검출을 만든다는 진단. "
  "이득의 대부분(55→20)이 거리축이고 크기축은 20→12. max_dist 1~5mm는 결과가 동일해 튜닝 여지가 없다(과적합 위험 낮음).",
  ["c7_detect_sweep_*"], None),
 ("C8_classifier_cv", "환자단위 5-fold CV 진단 + 피처블록 ablation",
  "진단", "macro-recall 0.383 vs top-1 0.68 — 희귀클래스 실패가 최대 손실원. "
  "그룹 0.951 / 좌우 0.902 → 오류는 전부 그룹·측면 내부. ICA(3.x)가 오류 대부분이고 전부 3.4 Pcom으로 흡인(최빈 29건). "
  "ablation(kNN k5 balanced): dist 0.361 / +ov 0.379 / +bp 0.349 / dist+ov+bp 0.383.",
  ["c8_cv_report.json"], None),
 ("C9_classifier_tune", "β 사전확률 역보정 + rel 상대인코딩 + 반경확대",
  "부분채택", "β=0.75로 macro-recall 0.346 → 0.410 (+18%) — 채택. "
  "rel(분기점 절대거리를 softmax 상대비율로) 실패: 단독 0.346→0.366이나 β와 결합하면 0.351로 붕괴. "
  "원인은 rel이 5.2/5.3을 가르던 절대거리를 정규화로 지우기 때문(bp 0.146 vs 0.433 → rel 0.763 vs 0.996). "
  "반경 10→25mm 실패(기권 4→0이나 macro 0.410→0.393).",
  ["c9_tune_*"], None),
 ("C10_landmark_coords", "해부 랜드마크(BA tip, R/L ICA terminus) 기준 정규화 좌표 6차원",
  "★채택", "CV macro-recall 0.410 → 0.455 (+11%), 피처 실험 중 최대 단일 개선. "
  "기존 106차원에 전역 위치정보가 0비트였던 공백을 메움. pos 단독은 0.187로 약하나 상보적. "
  "구성: 정규화 좌표 3 + 랜드마크 거리 3. 미러 시 x부호 반전 + dR/dL swap. 블록 가중치 0.5.",
  ["c10_cv_report.json"], None),
 ("C11_synthetic_rare", "분기점에 가상 sac 배치로 희귀클래스 학습샘플 1395개 합성",
  "기각", "CV macro-recall 0.400→0.463(+16%), val 개선. 그러나 **test에서 전면 악화** "
  "(천장 MCC 0.3259→0.3081, e2e 0.2207→0.1956). 합성 위치를 train 케이스의 **예측** 분기점에서 뽑아 "
  "그 케이스들의 혈관분할 오차패턴까지 학습한 과적합. 같은 분포로 나누는 CV는 이를 원리적으로 탐지 못한다 — "
  "이번 주 가장 값진 실패.",
  ["c11_synth_rows.json"], "c11_e2e_vote2"),
 ("C12_group_experts", "해부그룹 5-way 판정 후 그룹별 전문가 분류기",
  "기각", "macro-recall 0.360 (기준 0.455). 1단 그룹 정확도 0.944로 예상대로 높았으나, "
  "쪼개면 1단 오류가 복구 불가능하게 전파되고 전문가별 학습표본이 줄어 통합 모델보다 나쁨.",
  ["c12_cv_report.json"], None),
 ("C13_lat_canonical", "좌우 canonical화 — 26-way 측면무관 라벨 + 좌우 이진분류기",
  "기각", "macro-recall 0.352. 좌우 전용 분류기가 오히려 더 나쁨(0.869 vs 통합 0.902). "
  "'오류가 그룹·측면 내부에서만 난다'는 C8 관찰이 '계층으로 쪼개면 된다'를 함의하지 않는다는 반례.",
  ["c13_cv_report.json"], None),
 ("C14_vessel_encoder_feat", "V4-2 혈관분할망 인코더 bottleneck 320차원을 병변 피처로 사용",
  "실패(구현)", "1차 73병변: enc 단독 top-1 0.027 / macro 0.023. 2차 268병변 전량: 0.049 / 0.030 — 무작위 수준. "
  "붙이면 오히려 방해(both macro 0.261 < hand 0.293). 표본 부족이 아니라 hook 텐서 또는 패치 좌표 매핑 오류로 판단. "
  "개념은 타당하나 디버깅 비용이 기대이득을 초과해 2회 실패로 중단.",
  ["c14*_enc_feats.json"], None),
 ("C15_geodesic_coords", "중심선 따라간 측지거리 3 + 사행비 3 (유클리드 거리 대체)",
  "기각(구현 결함)", "macro-recall 0.400 → 0.368. **실패 원인 규명(2026-08-17): 도달 불가를 0.0으로 인코딩**했고, "
  "0.0은 '랜드마크에 붙어 있음'과 같은 값이다. 실측 결과 측지 슬롯이 하나라도 0인 병변이 101/259(39%), "
  "BA tip만 89건(34%). 중심선이 끊기면 Dijkstra가 도달 실패하는데 유클리드는 그 실패 모드가 없다. "
  "아이디어 기각이 아니라 C14와 같은 범주의 구현 실패 — 도달 불가를 별도 플래그로 인코딩하면 재시도 가능.",
  ["c15_geo_rows.json"], None),
 ("C16_prob_avg_ensemble", "5-fold softmax 확률맵 평균 (이진 마스크 다수결 대체)",
  "채택(경계)", "test HD95 0.6490 → 0.5827. 마스크 다수결의 합집합 성분이 경계를 뭉개던 문제를 해결. "
  "**단 HD95 이득 분해(2026-08-17): 총 3.45 중 3.00이 '불가능 클래스를 7개→4개로 덜 지른 것'이고 "
  "실제 경계 개선은 0.45(13%)뿐.** MCC는 0.2175로 vote2(0.2207)에 근소 열세라 MCC 기준으로는 미채택.",
  ["c7_detect_sweep_probavg_*"], "c16_probavg"),
 ("C17_learned_fp_reject", "검출 후보 TP/FP를 C5 피처 112차원으로 판별하는 학습된 기각기",
  "검증완료·미적용", "out-of-fold(train 292 완전 커버) 후보 596개(TP 225 / FP 371). "
  "임계 0.1에서 민감도 −1.3%로 FP 52% 제거. c7 손규칙과의 중복 확인 필요해 val/test 미적용. "
  "**C28에서 '필터 완화만'이 실패했으므로 C17과 반드시 묶어야 함** — 필터를 풀어 병변을 되찾고 늘어난 FP를 학습으로 걷어내는 조합.",
  ["c17_cv_report.json", "c17_candidates.json"], None),
 ("C18_composite_reselect", "공식 6지표 복합으로 재집계·재선택",
  "분석", "공식 랭킹은 MCC 단독이 아니라 P/R/MCC/Dice/VS/HD95 6개 평균. "
  "당시 test에서 MCC 순위 == 복합 순위였으나, **이후 반례 발생**: ÷52는 확률평균 1위, ÷36(covered_gt)은 vote2 1위. "
  "C27 th0.15는 MCC 열세·복합 우세. 원인은 HD95만 분모에 비례하지 않기 때문. "
  "패턴: 희귀클래스를 공격적으로 예측하는 개입(β·합성·다수결)은 MCC를 올리며 HD95를 악화시킨다.",
  ["c18_composite_table.json"], None),
 ("C19_a4_boundary_refine", "A4(lesionscale crop) 모델로 검출 경계 재작성",
  "미완", "A4 전체볼륨 추론만 완료(val 42 / test 83). 병합 규칙 미정, 평가 미실시. "
  "DICE·VolSim·HD95 = 랭킹 6지표 중 3개를 직접 겨냥. 현재 DICE 0.135가 MCC 0.224 대비 낮아 가장 싼 구간으로 추정.",
  [], None),
 ("C20_prob_threshold", "확률맵 임계 스윕 — 검출 동작점 튜닝",
  "분석", "C16 확률평균의 동작점이 nnU-Net 기본 argmax(0.5)로 한 번도 튜닝된 적이 없었다는 지적에서 출발. "
  "민감도/FP 교환점을 MCC 기준으로 탐색. 결론은 C27이 이어받아 '임계로는 커버리지를 못 산다'로 확정.",
  ["c20_*"], None),
 ("C21_classifier_push", "분류기 밀어올리기 3종 — 케이스 내 클래스 유일성(헝가리안 할당) 등",
  "분석", "한 케이스에 같은 위치클래스 동맥류가 둘인 경우는 드물다(공식 README)는 사전지식을 "
  "확률행렬 헝가리안 할당으로 강제. 중복 클래스가 곧바로 FP가 되는 문제를 겨냥.",
  ["c21_push_report.json"], None),
 ("C22_export_for_softwindow", "3-Stage v2(soft window) 팀 핸드오프 패키지 생성",
  "핸드오프", "같은 split에서 독립 노선으로 진행 중인 3-Stage v2가 test MCC 0.323, 우리 C계열이 0.2207. "
  "우리가 앞서는 건 HD95(0.649 vs 0.747) 하나뿐이라, 우리 기하 강점을 저쪽 head에 얹는 방향으로 결정(2026-08-17).",
  ["c22_*"], None),
 ("C23_loc52_endtoend", "52클래스 직접 학습(엔드투엔드) 전처리",
  "전처리", "위치 52클래스를 분할망이 직접 예측하게 하는 경로. 전처리까지 진행.",
  ["c23_*"], None),
 ("C24_crop_classifier", "병변 중심 크롭 데이터셋 + 화소값 기반 분류기",
  "분석", "현재 112차원이 전부 마스크 기하량이고 **원본 화소값이 0개**라는 점에서 출발. "
  "AChA 분기점 검출률이 43.2%라 케이스의 57%에서는 Pcom과 AChA의 입력이 동일해 어떤 알고리즘도 구분 불가. "
  "영상에는 찍혀 있으나 분할 임계를 못 넘은 정보를 회수하려는 시도.",
  ["c24_report.json"], None),
 ("C25_crop_mip_rf", "크롭 분기를 MIP+RF로 교체하고 기하 분기와 혼합(w 스윕)",
  "기각", "앙상블 이득 사실상 0 (macro-recall 0.324 → 0.325). C25-b로 원인 분해: "
  "크롭만 맞히는 샘플이 43/268 있고 Pcom은 기하 0.300 → 크롭 0.650이라 정보는 상보적이었으나, "
  "혼합 가중 w를 올리면 val e2e MCC가 오히려 하락(w0.3 0.3375 / w0.5 0.3553 / w0.7 0.3908 < 기준 0.4413).",
  ["c25_report.json", "c25b_report.json", "c25_prob_*"], None),
 ("C26_beta_double_correction", "β 이중보정 교정 — RF class_weight='balanced' 위에 사전확률로 또 나누고 있었다",
  "★채택(→C31로 승계)", "**프로덕션 β=1.0이 이중 보정이었다.** RF가 이미 class_weight='balanced'로 균형을 맞추는데 "
  "그 위에 prior로 또 나눔. β=0.5로 내리자 test official MCC 0.2207(vote2) → **0.2238**, "
  "covered_gt(÷36) MCC 0.3188 → **0.3233**. HD95도 0.6490 → 0.5069로 크게 개선. "
  "vote2 대비 6지표 중 5개 우세(RECALL만 열세), C16 대비 PREC 동률에 나머지 5개 우세. 재학습·재추론 0.",
  ["c5_eval_*_c26_*"], "c26_b0.5"),
 ("C27_detect_threshold_mcc", "검출 확률 임계를 MCC 기준으로 재선택 (커버리지 회수 시도)",
  "기각", "동기: C26에서 official MCC 0.2238 vs covered-GT MCC 0.2863 — 차이 0.0625가 검출이 놓친 병변의 값(커버리지 0.733). "
  "**결과: 임계를 0.5→0.15로 내려도 val 커버리지가 0.7674에서 1도 안 움직인다**(covered-GT MCC도 0.6773 고정). "
  "놓친 병변은 확률이 낮아 놓친 게 아니라 확률맵에 아예 없다. 스크립트가 'test 생략'으로 자동 판정. "
  "⚠️ 단 복합지표로는 th0.15(0.3862) / th0.2(0.3854)가 기준선(0.3800)을 넘는다 — MCC 기준이라 미채택이나 갈렸다는 사실을 기록.",
  ["c5_eval_*_c27_*"], "c27_base"),
 ("C28_detect_filter_relax", "c7 필터를 느슨하게 해서 되찾은 병변이 MCC에 이득인지",
  "기각(진행중 확인)", "필터 완화는 커버리지를 실제로 산다(val 0.7674 → 0.8140, 33 → 35병변). 그런데 값을 못 한다: "
  "official MCC 0.4413 → 0.4284, **covered-GT MCC도 0.6773 → 0.6443**, top-1 0.7576 → 0.7429. "
  "검출 리콜을 분모에서 뺀 covered-GT까지 떨어졌다는 건 새로 되찾은 병변이 평균적으로 **더 어려운** 병변이라는 뜻. "
  "여기에 FP 증가로 precision까지 깎임(0.4182 → 0.3874). → 남은 경로는 C17과의 조합.",
  ["c5_eval_*_c28_*"], None),
 ("C29_beta_fine_ceiling", "β 미세 스윕(0.35~0.65) + 분류 천장 재측정 + 혈관 fold1 학습 착수",
  "기각(β 미세조정)", "val 최적이 β=0.55로 나와 그걸 test에 올렸는데 covered_gt MCC 0.3161로 "
  "β=0.5(0.3233)보다 낮았다. 원인은 분해능 — **val에서 β 0.5와 0.55가 소수 4자리까지 동률(0.2801)** 이라 "
  "42케이스로는 0.05 단위를 가를 수 없다. β 스윕은 여기서 종료. "
  "부수 소득 1 — 분류 천장 재측정: covered_gt 0.4429 / official 0.3066. e2e 0.3233과의 격차 **0.1196이 검출 몫**. "
  "부수 소득 2 — 혈관 fold1(500ep) 학습 착수: 피처 112차원이 전부 혈관마스크에서 나오는데 "
  "혈관은 fold0 단일이라 5-fold 앙상블한 검출기와 비대칭이다. ⚠ 체인 로그의 epoch 파싱이 계속 0으로 찍힌 버그 있음(학습 자체는 정상).",
  ["c5_eval_*_c29_*"], "c29_b0.55"),
 ("C30_mcc_class_weights", "전역 지수 β 하나 대신 **클래스별 가중치 w_c** 를 macro-MCC에 직접 최적화",
  "분석", "진단: 43클래스 중 **21개가 재현율 0** 이고 혼동이 거의 전부 같은 해부학적 가족 안에서 일어난다 "
  "(R-3.2 n=2 → R-3.3, R-1.3 n=2 → R-1.1, AChA는 좌우 9개 전부 오답). "
  "지금은 전역 지수 하나로 모든 클래스를 똑같이 민다 — β를 키우면 희소는 살아나나 빈발 클래스 정밀도가 같이 무너져 "
  "β=1.0이 0.5보다 나빴다. 클래스별 가중치 `argmax_c w_c·p(c|x)` 가 올바른 정식화. "
  "⚠ 43개 자유파라미터를 268샘플에 맞추므로 과적합 위험이 커서 (1) w는 작은 격자에서만 고르고(연속 최적화 금지) "
  "(2) **중첩 CV**(4폴드 OOF로 w를 맞추고 남은 폴드에서 평가)로 정직한 추정치를 함께 낸다.",
  ["c30_*"], None),
 ("C31_conf_gated_beta", "β를 전체가 아니라 **분류기 확신이 낮은 병변에만** 적용 (확신 게이트)",
  "★★채택(현재 최고)", "test covered_gt MCC **0.3306** (official 0.2289) — 직전 최고 c26_b0.5(0.3233) 대비 "
  "**6지표 전부 상회**. 재학습·재추론 없음. \n\n"
  "핵심 통찰: **β의 가치는 희소 클래스를 살리는 것이 아니라 위양성을 벌점 없는 곳에 버리는 것**이다. "
  "GT에 없는 클래스로 예측하면 tp=fn=0 이라 MCC 분자가 0 → 그 클래스 MCC는 0 그대로라 벌점이 없다. "
  "그래서 FP가 없는 train CV는 β=0이 최선(macro-MCC 0.3284 vs 0.3106), FP가 섞이는 val e2e는 β=0.5가 최선(0.2801 vs 0.2233)으로 "
  "**정반대 결론**이 나왔다. 둘을 동시에 만족시키려면 고확신 병변은 argmax로 두고 저확신만 밀면 된다.\n\n"
  "설정: τ=0.5 → 저확신 42%에만 β=0.5, 고확신 58%는 보정 없음. 근거가 된 확신 분포(train CV 268병변): "
  "τ=0.3 저확신11%/정확도0.207·고확신0.720, τ=0.4 26%/0.394·0.761, τ=0.5 42%/0.455·0.814. "
  "고확신군은 이미 0.814라 흔들 이유가 없다.\n\n"
  "검증: 선택은 val 격자, test는 1회만. 학습셋은 train 214케이스/268병변로 val·test 0건 직접 확인. "
  "C32가 대조군으로 돌린 c32_trainonly가 소수 4자리까지 동일한 0.3306 — 독립 재현됨.",
  ["c5_eval_*_c31_*"], "c31_t0.5_b0.5"),
 ("C33_trainval_ceiling", "train∪val 체제에서 분류 천장 재측정 + β/τ 재선택",
  "보류(전제 철회)", "C32(분류기 학습셋에 val 추가)를 전제로 설계된 후속이다. "
  "그런데 **2026-08-18 사용자 지시로 val을 학습에 넣지 않기로 했고**(PROJECT_RULES.md §1), "
  "C32가 철회되면서 이 실험의 전제도 함께 무효화됐다. 결과 자체는 c33_chain.log 에 남아 있다. "
  "설계 자체는 정직했다 — val이 학습에 들어가면 선택에 못 쓰므로 train∪val 안 환자단위 CV로 τ만 좁히고 "
  "test는 확인 1회로만 쓰겠다고 명시했다.",
  ["c5_eval_*_c33_*"], None),
 ("C34_geodesic_revival", "기각됐던 측지(geo) 노선을 바뀐 조건에서 재시도 + arc-length 추가",
  "분석", "C15(중심선 측지거리 + 사행비)는 8/15에 기각됐다(macro-recall 0.400 → 0.368). "
  "그러나 당시와 조건이 셋 달라졌다 — (1) β 이중보정(β=1.0)이 걸려 있었고 지금은 0.5로 교정, "
  "(2) 학습이 268병변이었고, (3) 확신 게이트(τ)가 없었다. 그래서 같은 피처를 새 조건에서 다시 재봤다. "
  "결과는 c34_chain.log 참조(geo / arc / geo+arc 세 조합 비교). "
  "⚠ C15의 실패 원인으로 2026-08-17에 규명된 것 — Dijkstra 도달 불가를 0.0으로 인코딩해 "
  "'랜드마크에 붙어있음'과 같은 값이 되고, 측지 슬롯이 하나라도 0인 병변이 101/259(39%), "
  "BA tip만 89건(34%)이었다 — 이 인코딩 결함이 C34에서 고쳐졌는지 확인 필요.",
  ["c34_*", "c15_geo_rows.json"], None),
]

# MCC 기준 최고 후보 (analysis 스캔으로 자동 편입)
BASE_CANDS = ["e2e_A52f_pp", "best_e2e_A52f_pp", "ens_vote2", "c16_probavg",
              "c11_e2e_vote2", "c26_b0.5", "c26_b1.0", "c27_base"]


def _mtable(fh, ms, title):
    fh.write(f"### {title}\n\n| split | 분모 | PREC | RECALL | **MCC** | DICE | VOLSIM | HD95↓ | 복합 | 병변 |\n")
    fh.write("|---|---|---|---|---|---|---|---|---|---|\n")
    for m in ms:
        if not m:
            continue
        c = m["covered_gt"]
        fh.write(f"| {m['split']} | covered_gt(÷{m['n_present_classes']}) | "
                 + " | ".join(f"{c[k]:.4f}" for k in ["PRECISION", "RECALL"])
                 + f" | **{c['MCC']:.4f}** | "
                 + " | ".join(f"{c[k]:.4f}" for k in ["DICE", "VOLSIM", "HD95"])
                 + f" | {c['composite']:.4f} | {m['n_lesions']} |\n")
        o = m["official_div52"]
        fh.write(f"| {m['split']} | official(÷52) | "
                 + " | ".join(f"{o[k]:.4f}" for k in ["PRECISION", "RECALL"])
                 + f" | **{o['MCC']:.4f}** | "
                 + " | ".join(f"{o[k]:.4f}" for k in ["DICE", "VOLSIM", "HD95"])
                 + f" | {o['composite']:.4f} | {m['n_lesions']} |\n")
    fh.write("\n")


def write_exp(name, desc, verdict, result, globs, tag):
    d = E / name
    (d / "results").mkdir(parents=True, exist_ok=True)
    m_test = metrics(tag) if tag else None
    m_val = metrics(tag, "val") if tag else None
    json.dump({"exp_name": name, "family": FAM, "kind": "analysis/postprocess (학습 없음)",
               "verdict": verdict, "description": desc,
               "report_basis": "covered_gt(adjusted_div_present) 대표 · 순위 기준 MCC",
               "metrics_test": m_test, "metrics_val": m_val,
               "generated": datetime.datetime.now().isoformat(timespec="seconds")},
              open(d / "config.json", "w"), indent=1, ensure_ascii=False)
    n = 0
    seen = set()
    for g in globs:
        pat = g if g.endswith(".json") else (g + ".json" if g.endswith("*") else g + "*.json")
        for f in A.glob(pat):
            if f.is_file() and f.name not in seen:
                shutil.copy2(f, d / "results" / f.name); seen.add(f.name); n += 1
    with open(d / "summary.md", "w") as fh:
        fh.write(f"# {name}\n\n**{desc}**\n\n판정: **{verdict}**\n\n")
        fh.write(f"> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**\n\n")
        fh.write(f"## 결과\n\n{result}\n\n")
        if m_test or m_val:
            _mtable(fh, [m_val, m_test], "지표")
        fh.write(f"## 산출물\n\n`results/` 에 json {n}개 복사. 원본은 `analysis/`.\n\n")
        fh.write(f"---\n_생성 {datetime.datetime.now():%Y-%m-%d %H:%M}_ · `organize_c_experiments_v2.py`\n")
    return n


def main():
    print(f"{'폴더':<34}{'판정':<22}{'json':>5}")
    for e in EXPS:
        n = write_exp(*e)
        print(f"{e[0]:<34}{e[2]:<22}{n:>5}")

    # ---- 최고 성능기 (MCC 기준, covered_gt) ----
    tags = list(dict.fromkeys(BASE_CANDS + [
        os.path.basename(p)[len("c5_eval_test_"):-len(".json")]
        for p in glob.glob(str(A / "c5_eval_test_*.json"))]))
    # 철회된 구성 — PROJECT_RULES.md §1(2026-08-18): 분류기 학습셋에 val 을 넣지 않는다.
    # c32_trainval 은 규칙 위반은 아니나 사용자 지시로 철회됐다(선택용 정직한 셋이 사라짐).
    RETRACTED = {"c32_trainval"}
    cands = {}
    for t in tags:
        m = metrics(t)
        if m and m["eval_mode"] == "official_e2e" and "ceil" not in t and t not in RETRACTED:
            cands[t] = m
    if not cands:
        print("후보 없음"); return
    best = max(cands, key=lambda k: cands[k]["covered_gt"]["MCC"])
    best_comp = max(cands, key=lambda k: cands[k]["covered_gt"]["composite"])
    d = E / "_BEST_by_MCC"; (d / "results").mkdir(parents=True, exist_ok=True)
    json.dump({"exp_name": "_BEST_by_MCC", "selected_by": "MCC (test, covered_gt 분모)",
               "winning_tag": best, "composite_winner": best_comp,
               "criteria_agree": best == best_comp,
               "metrics_test": cands[best], "metrics_val": metrics(best, "val"),
               "all_candidates": cands,
               "generated": datetime.datetime.now().isoformat(timespec="seconds")},
              open(d / "config.json", "w"), indent=1, ensure_ascii=False)
    with open(d / "summary.md", "w") as fh:
        fh.write(f"# 최고 성능 파이프라인 — MCC 기준\n\n선정 태그: **`{best}`**\n\n")
        fh.write("> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 선택 기준 **MCC**\n\n")
        _mtable(fh, [metrics(best, "val"), cands[best]], f"`{best}` 지표")
        fh.write("## 후보 비교 — test, covered_gt 분모, MCC 내림차순\n\n")
        fh.write("| 태그 | PREC | RECALL | **MCC** | DICE | VOLSIM | HD95↓ | 복합 | 병변 |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|\n")
        for k, v in sorted(cands.items(), key=lambda kv: -kv[1]["covered_gt"]["MCC"]):
            c = v["covered_gt"]
            mark = " ←최고" if k == best else ""
            fh.write(f"| {k}{mark} | {c['PRECISION']:.4f} | {c['RECALL']:.4f} | **{c['MCC']:.4f}** | "
                     f"{c['DICE']:.4f} | {c['VOLSIM']:.4f} | {c['HD95']:.4f} | {c['composite']:.4f} | {v['n_lesions']} |\n")
        if best != best_comp:
            fh.write(f"\n> ⚠️ **MCC 기준과 복합 기준이 갈린다.** MCC 최고 `{best}`, 복합 최고 `{best_comp}`. "
                     f"PROJECT_RULES.md §6-1에 따라 MCC를 따르고 이 사실을 기록한다. 원인은 HD95만 분모에 비례하지 않기 때문.\n")
        else:
            fh.write(f"\n> MCC 기준과 복합 기준이 일치한다 (`{best}`).\n")
        rt = metrics("c32_trainval")
        if rt:
            fh.write(f"\n> **제외됨 — `c32_trainval`** (covered_gt MCC {rt['covered_gt']['MCC']:.4f}): "
                     f"분류기 학습셋에 공식 val 43병변을 넣어 얻은 값. test 누출은 없으나 "
                     f"PROJECT_RULES.md §1(2026-08-18 사용자 지시)에 따라 철회 — val 이 학습에 들어가면 "
                     f"하이퍼파라미터를 고를 정직한 셋이 사라진다.\n")
        fh.write(f"\n---\n_생성 {datetime.datetime.now():%Y-%m-%d %H:%M}_\n")
    for t in cands:
        for sp in ("val", "test"):
            p = A / f"c5_eval_{sp}_{t}.json"
            if p.exists():
                shutil.copy2(p, d / "results" / p.name)
    agree = "일치" if best == best_comp else f"갈림(복합최고={best_comp})"
    print(f"\n[최고 MCC] {best}  covered_gt MCC {cands[best]['covered_gt']['MCC']:.4f} "
          f"/ official {cands[best]['official_div52']['MCC']:.4f}  · 복합기준 {agree}  -> {d}")


if __name__ == "__main__":
    main()
