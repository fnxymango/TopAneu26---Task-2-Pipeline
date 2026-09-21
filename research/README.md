# 학습·분석 코드

실험에 쓴 학습·분석 코드 원본입니다. 추론에 필요한 파일만 추린 것은 [`app/`](../app/)에 있고, 여기에는 나머지가 모두 있습니다.
코드 속 경로는 실험 당시 서버 경로 그대로입니다. 바로 실행되지는 않으며, 참고와 재현용입니다.

| 폴더 | 내용 |
|---|---|
| `scripts/` | 파이프라인 단계, 학습 데이터 빌드, 평가, 분석 스크립트(.py 108개)와 학습 실행 셸 스크립트(.sh 132개) |
| `trainers/` | nnU-Net 커스텀 트레이너(Tversky·Focal Tversky·Top-k CE 손실, Skeleton Recall 에폭 변형) |
| `normalization/` | 모달리티별 적응 정규화 `topaneu_norm.py` |
| `preprocess/` | 강도 통계 계산 |
| `dataset_split.json` | 환자 단위 공식 split(train 291 · val 41 · test 83, 2026-09 개정판), 케이스별 center·모달리티, 클래스별 병변 수. `app/` 안의 같은 이름 파일은 클래스 이름표만 남긴 축약본입니다. |

실험별 실행 스크립트(판정 규칙이 머리말에 고정돼 있음)는 [`records/experiments/`](../records/experiments/)의 각 실험 폴더에 기록과 함께 있습니다.

## 파이썬 파일 목록

| 파일 | 설명 (docstring 첫 줄) |
|---|---|
| [`normalization/topaneu_norm.py`](normalization/topaneu_norm.py) | TopAneu 모달리티 적응형 정규화 (우리 EDA 기반, per-image). |
| [`preprocess/compute_intensity_stats.py`](preprocess/compute_intensity_stats.py) | Modality-aware foreground intensity statistics for the TopAneu dataset. |
| [`scripts/c10_landmark_coords.py`](scripts/c10_landmark_coords.py) | C10 — 해부 랜드마크 기준 정규화 좌표. |
| [`scripts/c11_synth_rare.py`](scripts/c11_synth_rare.py) | C11 — 희귀 클래스 합성 (분기점에 가상 sac 배치). |
| [`scripts/c12_group_experts.py`](scripts/c12_group_experts.py) | C12 — 해부그룹 조건부 전문가. |
| [`scripts/c13_lat_canonical.py`](scripts/c13_lat_canonical.py) | C13 — 좌우 canonical화 (증강이 아니라 출력공간 축소). |
| [`scripts/c14_vessel_encoder_feat.py`](scripts/c14_vessel_encoder_feat.py) | C14 — 혈관 분할망(V4-2)의 내부 표현을 병변 피처로 사용. |
| [`scripts/c15_geodesic_coords.py`](scripts/c15_geodesic_coords.py) | C15 — 혈관 트리 위 측지거리(geodesic) 좌표. |
| [`scripts/c17_fp_reject.py`](scripts/c17_fp_reject.py) | C17 — 학습된 위양성 기각기. |
| [`scripts/c18_composite_reselect.py`](scripts/c18_composite_reselect.py) | C18 — 공식 6지표 복합으로 하이퍼파라미터 재선택. |
| [`scripts/c19_merge_a4.py`](scripts/c19_merge_a4.py) | C19 — A4(lesionscale crop) 마스크로 검출 경계 재작성. |
| [`scripts/c20_prob_threshold.py`](scripts/c20_prob_threshold.py) | C20 — 확률맵 임계 스윕 (검출 동작점 튜닝). |
| [`scripts/c21_classifier_push.py`](scripts/c21_classifier_push.py) | C21 — 분류기 밀어올리기 3종 (2026-08-17). 목표: test MCC > 0.3. |
| [`scripts/c22_export_for_softwindow.py`](scripts/c22_export_for_softwindow.py) | C22 — 3-Stage v2 (soft window) 팀에 넘길 핸드오프 패키지 생성 (2026-08-17). |
| [`scripts/c24_build_crops.py`](scripts/c24_build_crops.py) | C24-a — 병변 중심 크롭 데이터셋 생성. |
| [`scripts/c24_train_eval.py`](scripts/c24_train_eval.py) | C24-b — 크롭 분류기: 게이트 검증 → 학습 → 기하 RF 와 앙상블. |
| [`scripts/c25_crop_mip.py`](scripts/c25_crop_mip.py) | C25 — 크롭 분류기를 3D CNN에서 MIP+RF 로 교체 (2026-08-17). |
| [`scripts/c25b_complement.py`](scripts/c25b_complement.py) | C25-b — 크롭 분기와 기하 분기가 *정말* 상보적인지 (2026-08-17). |
| [`scripts/c30_mcc_weights.py`](scripts/c30_mcc_weights.py) | C30 — 클래스별 가중치를 macro-MCC 에 직접 맞춘다 (2026-08-17). |
| [`scripts/c34_arc_position.py`](scripts/c34_arc_position.py) | C34 — 분기점 사이의 상대 호(arc) 위치. C15 측지 노선의 정식화 수정 (2026-08-18). |
| [`scripts/c41_anchor_diag.py`](scripts/c41_anchor_diag.py) | C41 — 앵커가 왜 없나: 혈관 부재인가 노드형성 실패인가 (2026-08-18). |
| [`scripts/c42_anchor_atlas.py`](scripts/c42_anchor_atlas.py) | C42/C43 — 앵커 결측 지시자 + 랜드마크 아틀라스 기반 대체값 (2026-08-18). |
| [`scripts/c4_bp40.py`](scripts/c4_bp40.py) | C4 — 혈관 중심선에서 분기점 그래프 추출. |
| [`scripts/c4_branchpoint_graph.py`](scripts/c4_branchpoint_graph.py) | C4 — 혈관 중심선에서 분기점 그래프 추출. |
| [`scripts/c5_k1.py`](scripts/c5_k1.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_k7.py`](scripts/c5_k7.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_location_v2.py`](scripts/c5_location_v2.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_neck.py`](scripts/c5_neck.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_neck2.py`](scripts/c5_neck2.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_r1a.py`](scripts/c5_r1a.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_r3.py`](scripts/c5_r3.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_t3.py`](scripts/c5_t3.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_v1b.py`](scripts/c5_v1b.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_v40.py`](scripts/c5_v40.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c5_v4i.py`](scripts/c5_v4i.py) | C5 — 분기점 기반 52클래스 위치 할당 (C1 lookup / C2 kNN의 후속). |
| [`scripts/c6_consensus.py`](scripts/c6_consensus.py) | C6 — 동맥류 검출 다수결 합의 마스크 생성 (N모델 중 M표 이상). |
| [`scripts/c7_detect_postproc.py`](scripts/c7_detect_postproc.py) | C7 — 동맥류 검출 후처리 스윕 (재학습 없이 동작점 조정). |
| [`scripts/c8_classifier_cv.py`](scripts/c8_classifier_cv.py) | C8 — 위치분류기 환자단위 교차검증 + 오류 진단. |
| [`scripts/c9_classifier_tune.py`](scripts/c9_classifier_tune.py) | C9 — 위치분류기 개선 실험 (c8 진단 결과를 표적으로). |
| [`scripts/convert_to_nnunet.py`](scripts/convert_to_nnunet.py) | TopAneu -> nnU-Net v2 raw format converter. |
| [`scripts/d740_measure_lesion_size.py`](scripts/d740_measure_lesion_size.py) | D740(병변스케일 고정crop) 준비 — 훈련셋 병변 bbox 크기 분포를 재서 |
| [`scripts/d900_build_lookup.py`](scripts/d900_build_lookup.py) | D900 방법론(signature+3단계 lookup table, 기존 방식): train split GT로부터 |
| [`scripts/d900_infer_eval.py`](scripts/d900_infer_eval.py) | D9xx: 예측(D720 aneurysm + D800 vessel) -> 위치할당 -> 공식 evaluate.py로 채점. |
| [`scripts/d910_build_index.py`](scripts/d910_build_index.py) | D910(weighted-kNN) 인덱스 빌드 — train split GT lesion feature만 사용(leakage 방지). |
| [`scripts/d910_infer_eval.py`](scripts/d910_infer_eval.py) | D910(weighted-kNN): 예측(또는 GT-ceiling용 GT) -> 위치할당 -> 공식 evaluate.py로 채점. |
| [`scripts/d910_lib.py`](scripts/d910_lib.py) | D910 방법론(weighted-kNN, 신규): D900(signature+3단계 hard-fallback lookup)의 대안. |
| [`scripts/d9xx_lib.py`](scripts/d9xx_lib.py) | D9xx 위치분류 공용 로직 — feature 추출 + signature 규칙 + lookup table. |
| [`scripts/d9xx_lib_neck.py`](scripts/d9xx_lib_neck.py) | D9xx 위치분류 공용 로직 — feature 추출 + signature 규칙 + lookup table. |
| [`scripts/d9xx_lib_neck2.py`](scripts/d9xx_lib_neck2.py) | D9xx 위치분류 공용 로직 — feature 추출 + signature 규칙 + lookup table. |
| [`scripts/d9xx_lib_v40.py`](scripts/d9xx_lib_v40.py) | D9xx 위치분류 공용 로직 — feature 추출 + signature 규칙 + lookup table. |
| [`scripts/det_filter.py`](scripts/det_filter.py) | 검출 후처리 필터 — GT 불요판 (2026-08-28, 도커/번들용). |
| [`scripts/e16_weak_classes.py`](scripts/e16_weak_classes.py) | E16 — 52클래스 중 어디가 약한가, 그 약한 것들의 공통 특징은 무엇인가 (2026-08-19). |
| [`scripts/e2_model_diff.py`](scripts/e2_model_diff.py) | E2 — 두 분류기가 실제로 **몇 개의 병변에서 갈리는지** 세어본다 (2026-08-19). |
| [`scripts/e3_model_cv.py`](scripts/e3_model_cv.py) | E3 — RF vs ET 를 268병변 환자단위 CV 로 직접 겨룬다 (2026-08-19). |
| [`scripts/e5_rejected_sweep.py`](scripts/e5_rejected_sweep.py) | E5 — 지금까지 기각/채택한 것들을 **반복 CV** 로 전부 다시 통과시킨다 (2026-08-19). |
| [`scripts/e6_bootstrap.py`](scripts/e6_bootstrap.py) | E6 — 보고한 e2e 숫자에 **신뢰구간**을 붙인다 (2026-08-19). |
| [`scripts/e8_detector_value.py`](scripts/e8_detector_value.py) | E8 — 검출기 민감도가 오르면 e2e 점수가 얼마나 오르는가 (2026-08-19). |
| [`scripts/eval_cv_5fold.py`](scripts/eval_cv_5fold.py) | 5-fold 교차검증 집계 — 각 fold를 '그 fold가 안 본 자기 val'로만 채점(유출 없음). |
| [`scripts/eval_d740_fullvolume.py`](scripts/eval_d740_fullvolume.py) | D740(lesionscale crop) 모델을 **정식 val 84케이스 전체볼륨**에서 평가. |
| [`scripts/final_infer.py`](scripts/final_infer.py) | 최종 모델(X5+gC) 단독 추론기 — 도커 제작자용 (2026-08-28). |
| [`scripts/make_5fold_split.py`](scripts/make_5fold_split.py) | Dataset600 splits_final.json: 1-fold(정식) -> 5-fold CV 확장. |
| [`scripts/make_binary.py`](scripts/make_binary.py) | Build Dataset520_TopAneuBinary: combined CTA+MRA (98), aneurysm vs background. |
| [`scripts/make_combined.py`](scripts/make_combined.py) | Build COMBINED (CTA+MRA) single-model datasets from the per-modality ones: |
| [`scripts/make_leaderboard.py`](scripts/make_leaderboard.py) | Build ONE master CSV over ALL experiments: outputs/experiments_leaderboard.csv |
| [`scripts/make_slide_figs.py`](scripts/make_slide_figs.py) | 발표자료용 데이터 그림 생성 (2026-08-16). |
| [`scripts/make_slides_0811.py`](scripts/make_slides_0811.py) | 8/11~8/16 실험 발표자료 생성 (2026-08-16, v2 — 그림 중심 / 차분한 디자인). |
| [`scripts/make_summary.py`](scripts/make_summary.py) | Generate a per-experiment summary markdown: <EXP_DIR>/summary.md |
| [`scripts/mha_nifti.py`](scripts/mha_nifti.py) | GrandChallenge .mha ↔ 우리 파이프라인(.nii.gz/nibabel) 변환 (2026-08-28, 도커용). |
| [`scripts/oracle_location_assign.py`](scripts/oracle_location_assign.py) | 동맥류 위치라벨(29클래스) 규칙기반 할당 — 오라클 상한 측정. |
| [`scripts/oracle_location_assign_417.py`](scripts/oracle_location_assign_417.py) | 동맥류 위치라벨 규칙기반 할당 — 오라클 상한 측정 (417케이스 버전). |
| [`scripts/organize_c_experiments.py`](scripts/organize_c_experiments.py) | C계열 실험을 PROJECT_RULES.md §3 규약에 맞는 실험 폴더로 정리 (2026-08-16). |
| [`scripts/organize_c_experiments_v2.py`](scripts/organize_c_experiments_v2.py) | C계열 실험 폴더/summary 생성 v2 (2026-08-17). |
| [`scripts/p_adjsmooth.py`](scripts/p_adjsmooth.py) | Q3 — 인접 분절 라벨 평활 (2026-08-26). |
| [`scripts/p_arb_val.py`](scripts/p_arb_val.py) | val 42 판정 — Q2 프로토타입 / Q3 인접평활 을 val 에서 재측정 (2026-08-26). |
| [`scripts/p_build_3cls.py`](scripts/p_build_3cls.py) | P1 — pjh Stage1 재현용 3-class 데이터셋 빌드 (2026-08-25). |
| [`scripts/p_build_crops2.py`](scripts/p_build_crops2.py) | 크롭 데이터셋 v2 — patch-CNN 용 (2026-08-26). |
| [`scripts/p_case_prior.py`](scripts/p_case_prior.py) | N2 — 케이스 수준 혈관 변이 prior (2026-08-26). |
| [`scripts/p_cprior_table.py`](scripts/p_cprior_table.py) | N2 케이스 prior 테이블 — 292 전체로 적합해 test+val 케이스에 적용 (2026-08-26). |
| [`scripts/p_crop_cnn.py`](scripts/p_crop_cnn.py) | Q4 — 크롭 3D CNN, 5폴드 CV (2026-08-26). |
| [`scripts/p_export_feat.py`](scripts/p_export_feat.py) | 피처 벡터 내보내기 (2026-08-26) — Q2(프로토타입)·Q3 공용. |
| [`scripts/p_export_proba.py`](scripts/p_export_proba.py) | 병변별 확률행렬 내보내기 (2026-08-26). Q1(전역배정)·Q2(프로토타입) 공용 재료. |
| [`scripts/p_hier_mtta.py`](scripts/p_hier_mtta.py) | M1 계층 argmax + M2 좌우 TTA 병변수준 스크리닝 (2026-08-26). |
| [`scripts/p_ladder_cost.py`](scripts/p_ladder_cost.py) | 검출 강도 사다리의 손익 — 회수 vs 환각 (2026-08-26). |
| [`scripts/p_lesion_recs.py`](scripts/p_lesion_recs.py) | e2e 병변단위 기록 (2026-08-25) — 오라클 분해용. |
| [`scripts/p_miss_char.py`](scripts/p_miss_char.py) | 미검출 병변 특성화 (2026-08-26). |
| [`scripts/p_more_mcc.py`](scripts/p_more_mcc.py) | 추가 MCC 인상 후보 3종 병변수준 스크리닝 (2026-08-26). |
| [`scripts/p_oracle_split.py`](scripts/p_oracle_split.py) | 오라클 분해 (2026-08-25) — 팀원 exp_55 분석과 같은 축으로 우리 test 83 을 가른다. |
| [`scripts/p_pos_missing.py`](scripts/p_pos_missing.py) | pos(랜드마크 좌표) 결측 병변의 정확도 진단 (2026-08-26). |
| [`scripts/p_proba_fast.py`](scripts/p_proba_fast.py) | 병변별 확률행렬 — 피처행렬 재사용판 (2026-08-26). |
| [`scripts/p_proto.py`](scripts/p_proto.py) | Q2 — 판별 메트릭 축소 + 클래스 프로토타입, RF 와 혼합 (2026-08-26). |
| [`scripts/p_shape_calib.py`](scripts/p_shape_calib.py) | S4 복셀 형태 보정 스크리닝 (2026-08-27) — Dice/VolSim/HD95 축. |
| [`scripts/p_side_noise.py`](scripts/p_side_noise.py) | S1 좌우 하드제약 + S2 학습 라벨잡음 스크리닝 (2026-08-27). |
| [`scripts/p_sinkhorn.py`](scripts/p_sinkhorn.py) | Q1b — 전역 배정(Sinkhorn) 조기 게이트 (2026-08-26). |
| [`scripts/p_size_effect.py`](scripts/p_size_effect.py) | 집합 크기 효과 검정 (2026-08-26). |
| [`scripts/p_size_feat.py`](scripts/p_size_feat.py) | ④ 크기 피처 스크리닝 (2026-08-26). |
| [`scripts/p_sliver_diag.py`](scripts/p_sliver_diag.py) | 조각 배치 진단 (2026-08-26). |
| [`scripts/p_surgical_recover.py`](scripts/p_surgical_recover.py) | 수술적 회수 — 놓친 병변의 조건에 맞는 blob 만 붙인다 (2026-08-26). |
| [`scripts/p_topk_ceiling.py`](scripts/p_topk_ceiling.py) | ③ top-k 상한 곡선 (2026-08-25). |
| [`scripts/p_ves_insample_gap.py`](scripts/p_ves_insample_gap.py) | 혈관 in-sample / out-of-sample 격차 측정 (2026-08-25). |
| [`scripts/p_vesconf.py`](scripts/p_vesconf.py) | ① 곁가지 존재 신뢰도 (2026-08-25). |
| [`scripts/p_weak_report.py`](scripts/p_weak_report.py) | 약한 클래스 전용 채점 (2026-08-25). |
| [`scripts/pipeline_case.py`](scripts/pipeline_case.py) | 단일 케이스 전체 파이프라인 — 원본 영상 1개 → 52클래스 위치 라벨맵 (2026-08-28). |
| [`scripts/plot_loss.py`](scripts/plot_loss.py) | Plot train & val loss (+ EMA pseudo-Dice) for one experiment. |
| [`scripts/postprocess_vessel.py`](scripts/postprocess_vessel.py) | 혈관 seg 후처리 — GT 통계 기반 (전역 largest-CC 금지, 클래스별 처리). |
| [`scripts/predict_queue.py`](scripts/predict_queue.py) | 작업 큐 방식 예측 워커 — GPU가 노는 것을 막는다. |
| [`scripts/predict_seq.py`](scripts/predict_seq.py) | 멀티프로세싱 없는 순차 예측기. |
| [`scripts/skeleton_from_predictions.py`](scripts/skeleton_from_predictions.py) | 예측 마스크(val_predict_out) → skeleton(centerline)만 추출해서 이미지화. |
| [`scripts/skeleton_poc.py`](scripts/skeleton_poc.py) | Skeleton PoC — GT 36-class 뇌혈관 마스크에서 centerline 추출 + 시각화. |
| [`scripts/summarize_c3.py`](scripts/summarize_c3.py) | C3 — 16칸 비교표 생성 (검출기 x 혈관후처리 x C방법 x split). |
| [`scripts/watchdog_cv_supervisor.py`](scripts/watchdog_cv_supervisor.py) | CV 감독자 — GPU당 정확히 1워커를 유지하고, 죽으면 되살리고, 끝나면 채점까지. |
| [`trainers/nnUNetTrainerFocalTverskyCE.py`](trainers/nnUNetTrainerFocalTverskyCE.py) | Focal-Tversky + CE trainer for TopAneu aneurysm (tiny-lesion detection). |
| [`trainers/nnUNetTrainerSkeletonRecallNoMirroring_1000ep.py`](trainers/nnUNetTrainerSkeletonRecallNoMirroring_1000ep.py) | 1000-epoch(nnU-Net 표준 기본값) variant — D800이 250ep로 이미 Dice 0.703을 낸 뒤, |
| [`trainers/nnUNetTrainerSkeletonRecallNoMirroring_250ep.py`](trainers/nnUNetTrainerSkeletonRecallNoMirroring_250ep.py) | 250-epoch variant of the fork's SkelRecall + NoMirroring trainer (fast fold-0 vessel check). |
| [`trainers/nnUNetTrainerTverskyCE.py`](trainers/nnUNetTrainerTverskyCE.py) | Custom nnU-Net v2 trainer for tiny-lesion detection (TopAneu aneurysm). |
| [`trainers/nnUNetTrainerTverskyTopkCE.py`](trainers/nnUNetTrainerTverskyTopkCE.py) | Tversky + TopK-CE trainer for TopAneu aneurysm (tiny-lesion detection). |
| [`trainers/nnUNetTrainerTverskyTopkCE_1000ep.py`](trainers/nnUNetTrainerTverskyTopkCE_1000ep.py) | Tversky+TopK-CE base 레시피를 250ep -> 1000ep(nnU-Net 표준 기본값)로 되돌린 변형. |
| [`trainers/nnUNetTrainerTverskyTopkCE_500ep.py`](trainers/nnUNetTrainerTverskyTopkCE_500ep.py) | Tversky+TopK-CE base 레시피를 500epoch로. 1000ep는 STEP9/10(5fold 앙상블)까지 |
| [`trainers/nnUNetTrainerTverskyTopkCE_a15b85.py`](trainers/nnUNetTrainerTverskyTopkCE_a15b85.py) | 재현율을 더 밀어붙인 Tversky 변형 — A7 (2026-08-17). |
| [`trainers/nnUNetTrainerTverskyTopkCE_a5b5.py`](trainers/nnUNetTrainerTverskyTopkCE_a5b5.py) | FP-rebalanced variant of nnUNetTrainerTverskyTopkCE for the 417-case migration. |
| [`trainers/nnUNetTrainerTverskyTopkCE_hardneg.py`](trainers/nnUNetTrainerTverskyTopkCE_hardneg.py) | Hard-negative (domain-aware) case oversampling for the 417-case migration. |
