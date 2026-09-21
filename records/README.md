# 실험 기록

TopAneu-26 Task 2 파이프라인을 만들며 남긴 실험 기록 원문입니다(2026-08-07 ~ 09-18, 실험 80개, 문서 162개).
최종 구성과 코드는 [저장소 README](../README.md)에 있고, 이 폴더에는 **왜 그 구성이 되었는지**의 근거가 있습니다.

기록은 작업 당시에 쓴 그대로입니다. 개인 경로와 작업용 임시 경로만 `<project>`, `<scratch>`, `~`로 바꿨습니다.
시기마다 수치 기준이 다르니 주의하세요. 2026-09-16 이전 문서는 대부분 `covered_gt` 분모를 쓰고, 이후 문서는 조직위 개정 공식 eval을 씁니다. 두 기준의 수치는 서로 비교할 수 없습니다.

## 먼저 읽을 문서

| 문서 | 내용 |
|---|---|
| [`PROJECT_RULES.md`](PROJECT_RULES.md) | 실험 공통 규칙: split·누수 방지, 채택 판정 기준, 평가 기준 변경 이력 |
| [`submission/FINAL_MODEL.md`](submission/FINAL_MODEL.md) | 제출본(2026-09-10) 구성과 제출 파일 설명 |
| [`submission/APPLY_CHANGES.md`](submission/APPLY_CHANGES.md) | 제출본 이후 변경의 반영 명세. 맨 아래 "변경 이력"이 **채택·기각 이력 표**입니다. |
| [`experiments/chain_status.md`](experiments/chain_status.md) | 자동 실험 체인의 판정 로그(시간순) |
| [`experiments/C60_metric_structure/summary.md`](experiments/C60_metric_structure/summary.md) | 공식 지표의 구조 분석. 이후 개입 설계가 여기서 출발합니다. |
| [`experiments/D1_newdata/POSTMORTEM.md`](experiments/D1_newdata/POSTMORTEM.md) | 2026-09 개정 데이터와 개정 eval로 전환한 뒤의 사후 분석 |
| [`experiments/V1_vessel_axis/NOTES.md`](experiments/V1_vessel_axis/NOTES.md) | 최종 구성에 이른 마지막 라운드의 진행 노트 |
| [`EDA_TopAneu417.md`](EDA_TopAneu417.md) | 데이터셋 탐색(케이스·모달리티·위치 클래스 분포) |

## 채택 규칙

방법은 **test 83케이스와 val 42케이스, 두 집합의 end-to-end 지표가 동시에 개선될 때만** 채택했습니다.
병변 단위 스크리닝에서 좋았던 후보 상당수가 end-to-end에서는 사라졌기 때문입니다. 스크리닝에는 허위 blob을 내는 비용과, 오답 라벨이 FP와 FN으로 이중 감점되는 효과가 반영되지 않습니다.
차이가 RF 시드 간 산포보다 작으면 근거로 보지 않았습니다. 그래서 대부분의 판정은 시드 5~10개 평균입니다.

## 주요 결론 (시간순)

| 시기 | 결론 | 근거 |
|---|---|---|
| 08월 | 초기 검출기 실험 일부는 split 누수로 test가 오염되어 **무효** 처리했습니다(폴더명 `_TESTLEAK`). 이후에는 모든 데이터셋을 공식 split에서 만들고, test가 어느 fold의 train에도 없는지 검사했습니다. | `PROJECT_RULES.md` §1 |
| 08월 | 위치 클래스 52개 중 21개가 "두 혈관이 만나는 지점"이라, 혈관 거리 피처만으로는 성능 상한이 막힙니다. 그래서 중심선에서 **분기점 그래프**를 뽑아 분류 피처로 썼습니다. | `C4`, `C5`, `C8` |
| 08월 | gC(1·2등 확률이 가까우면 2등 클래스도 조각으로 출력)를 채택했습니다. | `C26_topk_ica_margin` |
| 08월 | 검출 후보를 학습된 기각기로 지우는 방식은 검증까지 마쳤지만 적용하지 않았습니다. 진짜 병변을 잃는 손실이 허위 blob을 지우는 이득보다 커서, 요구되는 순도가 너무 높았습니다. | `C17_learned_fp_reject`, `C60` |
| 09-09 | 공동 작업자의 패치 CNN 필터는 임계를 손익 대칭으로 잡아 위 한계를 넘었습니다. 통합본에 포함했습니다. | `H1_patchfilter` |
| 09-04 | 조직위 eval이 개정되어(NaN 분모, 존재 기반 TP, HD95 벌점) 이전 수치와 비교할 수 없게 됐습니다. 그래서 주요 판정을 다시 했습니다. | `G1`, `G2`, `D1_newdata` |
| 09-16 | 10폴드 ResEncL 검출기(E9)는 **기각**했습니다. TP가 12 늘 때 오답 FP가 27 늘었고, 병변 단위 부호검정에서도 차이가 없었습니다(p = 0.50). | `E9_ensemble10`, `H4_folds`, `V1/RESULTS_DET9.md` |
| 09-16 | 개정판 라벨로 만든 학습표는 성능으로 구별되지 않았습니다(부호검정 최소 p = 0.055). 그래서 제출본과 같은 구 학습표를 유지했습니다. | `V1/RESULTS_TABLE10.md` |
| 09-16 | **FRAC 0.35 채택**: gC 2등 조각에 blob 지분을 줍니다. | `V1/RESULTS_FRAC{,_REP}.md` |
| 09-17 | **OUT_GROW 1.32 채택**: 출력 부피를 보정합니다. | `V1/RESULTS_GROW{,_REP}.md` |
| 09-17 | 곁가지 신뢰도 피처, TabICL, ICA 전용 분류기, 혈관 기시부 fine-tune은 기각했습니다. | `H3`, `V1/RESULTS_*`, `B1_vessel_contact_ft` |
| 09-18 | **min_vox 12 채택**: 3폴드 검출기가 남기는 작은 허위 blob을 지웁니다. 원판과 복제 모두 10시드에서 통과했습니다. | `V1/RESULTS_REP3F_{A,B,R}.md` |

## 실험 목록

폴더 이름의 접두가 계열을 나타냅니다. 링크를 누르면 각 실험의 대표 문서가 열립니다. 내용 칸은 그 문서의 첫 설명 줄을 옮긴 것입니다.

### A — 검출기 손실·구조 변형. `_TESTLEAK` 표시는 test 누수가 확인되어 **무효**인 실험

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`A1-2_binary_tverskytopk_a5b5_417_TESTLEAK`](experiments/A1-2_binary_tverskytopk_a5b5_417_TESTLEAK/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 0 · 시작 2026-08-08T17:18:31+00:00 | 1 |
| [`A1-f1_binary_tverskytopk_417_TESTLEAK`](experiments/A1-f1_binary_tverskytopk_417_TESTLEAK/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 1 · 시작 2026-08-09T06:52:13+00:00 | 1 |
| [`A1-f2_binary_tverskytopk_417_TESTLEAK`](experiments/A1-f2_binary_tverskytopk_417_TESTLEAK/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 2 · 시작 2026-08-09T12:16:46+00:00 | 1 |
| [`A1_binary_tverskytopk_417_TESTLEAK`](experiments/A1_binary_tverskytopk_417_TESTLEAK/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 0 · 시작 2026-08-08T10:50:21+00:00 | 1 |
| [`A2_binary_hardneg_417_TESTLEAK`](experiments/A2_binary_hardneg_417_TESTLEAK/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 0 · 시작 2026-08-09T02:05:23+00:00 | 1 |
| [`A6-2_resencl_adaptivenorm_topk_417_f0`](experiments/A6-2_resencl_adaptivenorm_topk_417_f0/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 0 · 시작 2026-08-13T16:40:16+00:00 | 1 |
| [`A6-2_resencl_adaptivenorm_topk_417_f1`](experiments/A6-2_resencl_adaptivenorm_topk_417_f1/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 1 · 시작 2026-08-14T13:44:52+00:00 | 1 |
| [`A6-2_resencl_adaptivenorm_topk_417_f2`](experiments/A6-2_resencl_adaptivenorm_topk_417_f2/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 2 · 시작 2026-08-14T13:44:52+00:00 | 1 |
| [`A6-2_resencl_adaptivenorm_topk_417_f3`](experiments/A6-2_resencl_adaptivenorm_topk_417_f3/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 3 · 시작 2026-08-15T05:01:41+00:00 | 1 |
| [`A6-2_resencl_adaptivenorm_topk_417_f4`](experiments/A6-2_resencl_adaptivenorm_topk_417_f4/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 4 · 시작 2026-08-15T05:01:41+00:00 | 1 |
| [`A7_tversky_a15b85_417_f0`](experiments/A7_tversky_a15b85_417_f0/summary.md) | 데이터셋 Dataset720_TopAneuBinary417 · config 3d_fullres · fold 0 · 시작 2026-08-17T17:01:50+00:00 | 1 |

### D5·D6·P — 검출·혈관 nnU-Net 베이스라인과 변형

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`D500_region_3dfr_f0_baseline`](experiments/D500_region_3dfr_f0_baseline/summary.md) | 데이터셋 Dataset500_TopAneuRegion · config 3d_fullres · fold 0 · 시작 2026-07-19T18:11:42+09:00 | 1 |
| [`D510_loc_3dfr_f0_baseline`](experiments/D510_loc_3dfr_f0_baseline/summary.md) | 데이터셋 Dataset510_TopAneuLoc · config 3d_fullres · fold 0 · 시작 2026-07-19T18:11:42+09:00 | 1 |
| [`D520_binary_focaltversky_canonsplit`](experiments/D520_binary_focaltversky_canonsplit/summary.md) | 데이터셋 Dataset520_TopAneuBinary · config 3d_fullres · fold 0 · 시작 2026-07-21T02:12:35+09:00 | 1 |
| [`D520_binary_tversky_canonsplit`](experiments/D520_binary_tversky_canonsplit/summary.md) | 데이터셋 Dataset520_TopAneuBinary · config 3d_fullres · fold 0 · 시작 2026-07-20T22:48:31+09:00 | 1 |
| [`D520_binary_tversky_os6_250ep`](experiments/D520_binary_tversky_os6_250ep/summary.md) | 데이터셋 Dataset520_TopAneuBinary · config 3d_fullres · fold 0 · 시작 2026-07-20T13:15:21+09:00 | 1 |
| [`D520_binary_tverskytopk_canonsplit`](experiments/D520_binary_tverskytopk_canonsplit/summary.md) | 데이터셋 Dataset520_TopAneuBinary · config 3d_fullres · fold 0 · 시작 2026-07-21T02:12:37+09:00 | 1 |
| [`D521_binary_tversky_adaptivenorm`](experiments/D521_binary_tversky_adaptivenorm/summary.md) | 데이터셋 Dataset521_TopAneuBinaryAdaptive · config 3d_fullres · fold 0 · 시작 2026-07-20T14:52:23+09:00 | 1 |
| [`D600_vessel_skelrec_resencm_250ep`](experiments/D600_vessel_skelrec_resencm_250ep/summary.md) | 데이터셋 Dataset600_TopAneuVessel · config 3d_fullres · fold 0 · 시작 2026-07-27T15:14:30+09:00 | 2 |
| [`D600_vessel_skelrec_resencm_250ep_bd0`](experiments/D600_vessel_skelrec_resencm_250ep_bd0/summary.md) | 데이터셋 Dataset600_TopAneuVessel · config 3d_fullres · fold 0 · 시작 2026-07-29T22:03:50+09:00 | 1 |
| [`D610_vessel_uzh_ft_f0`](experiments/D610_vessel_uzh_ft_f0/summary.md) | 데이터셋 Dataset610_TopAneuVesselUZH · config 3d_fullres · fold 0 · 시작 2026-07-31T21:32:10+09:00 | 1 |
| [`D610_vessel_uzh_ft_f1`](experiments/D610_vessel_uzh_ft_f1/summary.md) | 데이터셋 Dataset610_TopAneuVesselUZH · config 3d_fullres · fold 1 · 시작 2026-07-31T21:57:10+09:00 | 1 |
| [`D610_vessel_uzh_ft_f2`](experiments/D610_vessel_uzh_ft_f2/summary.md) | 데이터셋 Dataset610_TopAneuVesselUZH · config 3d_fullres · fold 2 · 시작 2026-07-31T22:22:10+09:00 | 1 |
| [`P2_pjh3cls_ourloss_iso04_f0`](experiments/P2_pjh3cls_ourloss_iso04_f0/summary.md) | 데이터셋 Dataset722_TopAneuPjh3cls417 · config 3d_fullres · fold 0 · 시작 2026-08-25T02:08:00+00:00 | 1 |

### V2–V5 — 혈관 분할 모델 (Skeleton Recall, UZH fine-tune, 클래스 가중, 후처리)

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`V2_vessel_skelrec_resencm_417`](experiments/V2_vessel_skelrec_resencm_417/summary.md) | 데이터셋 Dataset800_TopAneuVessel417 · config 3d_fullres · fold 0 · 시작 2026-08-08T18:22:37+00:00 | 1 |
| [`V3_vessel_uzh_ft_417_f0`](experiments/V3_vessel_uzh_ft_417_f0/summary.md) | 데이터셋 Dataset810_TopAneuVesselUZH417 · config 3d_fullres · fold 0 · 시작 2026-08-09T04:11:38+00:00 | 1 |
| [`V4-2_vessel_classweighted_417_500ep`](experiments/V4-2_vessel_classweighted_417_500ep/summary.md) | 데이터셋 Dataset800_TopAneuVessel417 · config 3d_fullres · fold 0 · 시작 2026-08-12T03:47:13+00:00 | 1 |
| [`V5_vessel_classweighted_postproc_417`](experiments/V5_vessel_classweighted_postproc_417/summary.md) | 데이터셋 Dataset800_TopAneuVessel417 · config 3d_fullres · fold 0 · 시작 2026-08-12T02:07:00+00:00 | 1 |

### C — 위치 분류기 (분기점 그래프 · RF · gC · 지표 구조)

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`C1_location_signature_lookup`](experiments/C1_location_signature_lookup/summary.md) | 혈관 근접도 서명 기반 lookup 위치할당 (최초 시도) | 1 |
| [`C2_location_weighted_knn`](experiments/C2_location_weighted_knn/summary.md) | 가중 kNN + 클래스 사전확률 보정 위치할당 | 1 |
| [`C3_location_knn_a62det`](experiments/C3_location_knn_a62det/summary.md) | A6-2(adaptive norm) 검출기로 C2 재평가 | 1 |
| [`C4_branchpoint_graph`](experiments/C4_branchpoint_graph/summary.md) | 혈관 중심선에서 분기점 그래프 추출 (Lee thinning + 2mm 스퍼 제거 + 클래스 전이점 + V5 인접성 검증) | 1 |
| [`C5_location_v2`](experiments/C5_location_v2/summary.md) | 위치할당 재작성 — 혈관거리 36 + sac 점유율 36 + 분기점 34 (=106차원) | 1 |
| [`C6_consensus_2of2`](experiments/C6_consensus_2of2/summary.md) | 모델 합의 마스크 (2/2 교집합 → 이후 5-fold N/M 다수결로 일반화) | 1 |
| [`C7_detect_postproc`](experiments/C7_detect_postproc/summary.md) | 검출 후처리 — 혈관거리 게이팅 + 성분크기 필터 (재학습 0) | 1 |
| [`C8_classifier_cv`](experiments/C8_classifier_cv/summary.md) | 환자단위 5-fold CV 진단 + 피처블록 ablation | 1 |
| [`C9_classifier_tune`](experiments/C9_classifier_tune/summary.md) | β 사전확률 역보정 + rel 상대인코딩 + 반경확대 | 1 |
| [`C10_landmark_coords`](experiments/C10_landmark_coords/summary.md) | 해부 랜드마크(BA tip, R/L ICA terminus) 기준 정규화 좌표 6차원 | 1 |
| [`C11_synthetic_rare`](experiments/C11_synthetic_rare/summary.md) | 분기점에 가상 sac 배치로 희귀클래스 학습샘플 1395개 합성 | 1 |
| [`C12_group_experts`](experiments/C12_group_experts/summary.md) | 해부그룹 5-way 판정 후 그룹별 전문가 분류기 | 1 |
| [`C13_lat_canonical`](experiments/C13_lat_canonical/summary.md) | 좌우 canonical화 — 26-way 측면무관 라벨 + 좌우 이진분류기 | 1 |
| [`C14_vessel_encoder_feat`](experiments/C14_vessel_encoder_feat/summary.md) | V4-2 혈관분할망 인코더 bottleneck 320차원을 병변 피처로 사용 | 1 |
| [`C15_geodesic_coords`](experiments/C15_geodesic_coords/summary.md) | 중심선 따라간 측지거리 3 + 사행비 3 (유클리드 거리 대체) | 1 |
| [`C16_prob_avg_ensemble`](experiments/C16_prob_avg_ensemble/summary.md) | 5-fold softmax 확률맵 평균 (이진 마스크 다수결 대체) | 1 |
| [`C17_learned_fp_reject`](experiments/C17_learned_fp_reject/summary.md) | 검출 후보 TP/FP를 C5 피처 112차원으로 판별하는 학습된 기각기 | 1 |
| [`C18_composite_reselect`](experiments/C18_composite_reselect/summary.md) | 공식 6지표 복합으로 재집계·재선택 | 1 |
| [`C19_a4_boundary_refine`](experiments/C19_a4_boundary_refine/summary.md) | A4(lesionscale crop) 모델로 검출 경계 재작성 | 1 |
| [`C20_prob_threshold`](experiments/C20_prob_threshold/summary.md) | 확률맵 임계 스윕 — 검출 동작점 튜닝 | 1 |
| [`C21_classifier_push`](experiments/C21_classifier_push/summary.md) | 분류기 밀어올리기 3종 — 케이스 내 클래스 유일성(헝가리안 할당) 등 | 1 |
| [`C22_export_for_softwindow`](experiments/C22_export_for_softwindow/summary.md) | 3-Stage v2(soft window) 팀 핸드오프 패키지 생성 | 1 |
| [`C23_loc52_endtoend`](experiments/C23_loc52_endtoend/summary.md) | 52클래스 직접 학습(엔드투엔드) 전처리 | 1 |
| [`C24_crop_classifier`](experiments/C24_crop_classifier/summary.md) | 병변 중심 크롭 데이터셋 + 화소값 기반 분류기 | 1 |
| [`C25_crop_mip_rf`](experiments/C25_crop_mip_rf/summary.md) | 크롭 분기를 MIP+RF로 교체하고 기하 분기와 혼합(w 스윕) | 1 |
| [`C26_beta_double_correction`](experiments/C26_beta_double_correction/summary.md) | β 이중보정 교정 — RF class_weight='balanced' 위에 사전확률로 또 나누고 있었다 | 1 |
| [`C26_topk_ica_margin`](experiments/C26_topk_ica_margin/summary.md) | C26 — ICA·마진 게이트 2등조각 방출 (gC) · 채택 ★ — 31번째 시도에서 첫 채택. 규칙(test+ AND val+, 사전 고정) 을 10/10 시드로 통과. | 1 |
| [`C27_detect_threshold_mcc`](experiments/C27_detect_threshold_mcc/summary.md) | 검출 확률 임계를 MCC 기준으로 재선택 (커버리지 회수 시도) | 1 |
| [`C28_detect_filter_relax`](experiments/C28_detect_filter_relax/summary.md) | c7 필터를 느슨하게 해서 되찾은 병변이 MCC에 이득인지 | 1 |
| [`C29_beta_fine_ceiling`](experiments/C29_beta_fine_ceiling/summary.md) | β 미세 스윕(0.35~0.65) + 분류 천장 재측정 + 혈관 fold1 학습 착수 | 1 |
| [`C30_mcc_class_weights`](experiments/C30_mcc_class_weights/summary.md) | 전역 지수 β 하나 대신 클래스별 가중치 w_c 를 macro-MCC에 직접 최적화 | 1 |
| [`C31_conf_gated_beta`](experiments/C31_conf_gated_beta/summary.md) | β를 전체가 아니라 분류기 확신이 낮은 병변에만 적용 (확신 게이트) | 1 |
| [`C33_trainval_ceiling`](experiments/C33_trainval_ceiling/summary.md) | train∪val 체제에서 분류 천장 재측정 + β/τ 재선택 | 1 |
| [`C34_geodesic_revival`](experiments/C34_geodesic_revival/summary.md) | 기각됐던 측지(geo) 노선을 바뀐 조건에서 재시도 + arc-length 추가 | 1 |
| [`C36_ET_location_52cls_417`](experiments/C36_ET_location_52cls_417/summary.md) | C36 — ExtraTrees 위치 분류기 (52클래스) — 2026-08-18 기준 C 계열 최고. 지표·모델 선택 기준은 MCC(PROJECT_RULES.md §6-1). | 1 |
| [`C60_metric_structure`](experiments/C60_metric_structure/summary.md) | C60 — 공식 지표의 구조를 실제로 재고, 그에 맞춰 개입 설계 (2026-09-02 새벽 KST) — 기준선(=`d2007`): test 0.3011 / val 0.3218 (official_div52, 1−HD95 사용). | 1 |

### B·D1·E·G·H·DR — 2026-09 개정 데이터 · 개정 eval · 통합본 이후

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`B1_bundle_newdata`](experiments/B1_bundle_newdata/RESULTS.md) | B1 — 제출 번들 구성 그대로, 데이터셋만 2026-09 개정판 — 바꾼 것은 GT/영상뿐. 모델 가중치·파라미터·코드는 번들과 동일하다. | 1 |
| [`B1_vessel_contact_ft`](experiments/B1_vessel_contact_ft/PLAN.md) | B1 — 혈관 모델 소분지 기시부 fine-tune (2026-09-15 · 사용자 지시 "b1진행") — 검출기는 건드리지 않는다. 분류기 입력인 혈관 모델만 바꾼다(추론 구조·런타임 불변 — 같은 아키텍처·같은 plans). | 2 |
| [`B2_bundle_recipe_retrain`](experiments/B2_bundle_recipe_retrain/summary.md) | 데이터셋 Dataset722_TopAneuPjh3cls417 · config 3d_fullres · fold 2 · 시작 2026-09-04T11:39:42+00:00 | 2 |
| [`D1_newdata`](experiments/D1_newdata/LINE_VERDICT.md) | 계열 판정 — covered_gt MCC (시드 0~4 평균) — 판정: resencl (두 집합 분열 — test 우선(규칙대로)) | 6 |
| [`DR1_duorf`](experiments/DR1_duorf/README.md) | DR1 · DuoRF — 추론조건 혼합 확률평균 (구 K1) — 2026-09-16 명명. 이전 이름 `K1` 은 순번일 뿐이라 무엇을 하는지 전달되지 않았다. | 1 |
| [`E9_ensemble10`](experiments/E9_ensemble10/RESULTS.md) | E9 — resencl 계열 10폴드 앙상블 (분류기: 개정판 피처) — 예측 blob 98 · 케이스 83 · GT존재클래스 36 | 1 |
| [`G1_gc_neweval`](experiments/G1_gc_neweval/RESULTS.md) | 개정 공식 eval에서 gC(2등 조각)를 켜고 끈 효과 재판정 (예비·본판정) | 1 |
| [`G2_gc_onoff_newbuild`](experiments/G2_gc_onoff_newbuild/RESULTS.md) | G2 — 개정판 빌드 gC on/off 종합 — → 개선 1/6 · 평균ΔMCC -0.0277 · 미충족 | 4 |
| [`G3_gcoff_candidates`](experiments/G3_gcoff_candidates/RESULTS.md) | G3 — 세 후보 모두 gC OFF(topk n=1) · 5시드 · test/val — 후보: B1 = 구 번들 가중치(구피처) · B2 = 번들 레시피 개정판 재학습(검출기+분류기, 신피처) · E9 = ResEncL 10폴드(P3구+P5신, 신… | 2 |
| [`H1_patchfilter`](experiments/H1_patchfilter/NOTES.md) | H1 — 통합본의 패치 CNN 환각필터를 우리 front end 에 이식 (2026-09-10) — 아무것도 재학습하지 않았다. 검출·혈관 추론 결과를 재사용하고 분류 단계부터 다시 돌렸다. | 1 |
| [`H2_featureswap`](experiments/H2_featureswap/NOTES.md) | H2 — 검출기 freeze · 분류기 피처만 개정판으로 (2026-09-10, 진행 중) — E9 는 검출기·피처·gC 세 가지가 동시에 바뀌어 있어 기여도를 귀속할 수 없다. | 1 |
| [`H3_vesconf`](experiments/H3_vesconf/NOTES.md) | H3 — 곁가지 신뢰도(vesconf) 피처 (준비 완료 · H2 판정 후 실행) — `c5_location_v2.py` 의 `TOPANEU_VESCONF` 환경변수만 켠다. 다른 건 전부 기준팔과 동일하다. | 1 |
| [`H4_folds`](experiments/H4_folds/RESULTS.md) | H4 — E9 검출기 폴드 축소 · 이득이 남는가 — 판정규칙은 H4.sh 머리말에 결과 보기 전에 고정. | 1 |

### V1·FRAC — 최종 구성에 이른 마지막 라운드 (FRAC · GROW · min_vox 12 등)

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`FRAC035_weakness_2026-09-17`](experiments/FRAC035_weakness_2026-09-17/REPORT.md) | FRAC 0.35 분류 취약점 분석 — 2026-09-17 — 대상: `b1frac035_pf` · test 83 + val 41 케이스 · RF 시드 0–9. 제출 RF 시드 3은 별도 표시한다. | 1 |
| [`V1_vessel_axis`](experiments/V1_vessel_axis/NOTES.md) | 최종 라운드 전체. 혈관 축 설계에서 출발해 분류기 판정 장치(K0), FRAC·GROW·min_vox 12 채택, E9·vesconf·TabICL 기각까지의 결과 문서 모음(`RESULTS_*.md`) | 56 |

### 기타

| 실험 | 내용 | 문서 수 |
|---|---|---|
| [`_BEST_by_MCC`](experiments/_BEST_by_MCC/summary.md) | 최고 성능 파이프라인 — MCC 기준 — 선정 태그: `c31_t0.5_b0.5` | 1 |

### 체인 로그·분석 메모

| 문서 | 내용 |
|---|---|
| [`chain_status.md`](experiments/chain_status.md) | 자동 실험 체인 판정 로그 |
| [`gC_subset_size_analysis.md`](experiments/gC_subset_size_analysis.md) | gC 이득이 평가 집합 크기에 따라 달라지는지 분석 |
