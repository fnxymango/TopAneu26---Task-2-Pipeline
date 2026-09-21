# 실험 요약 — D610_vessel_uzh_ft_f0

> 데이터셋 **Dataset610_TopAneuVesselUZH** · config **3d_fullres** · fold **0** · 시작 2026-07-31T21:32:10+09:00
> 명령: `CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=1 nnUNetv2_train 610 3d_fullres 0 -p nnUNetPlansUZH -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -pretrained_weights /home/user/TopAneu/seg/sblee/nnunet/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **98**
- 라벨 체계: 전경 클래스 **36개** (+background)
- 클래스: 1:BA, 2:R-P1P2, 3:L-P1P2, 4:R-ICA-C6-C7, 5:R-M1, 6:L-ICA-C6-C7, 7:L-M1, 8:R-Pcom, 9:L-Pcom, 10:Acom, 11:R-A1A2, 12:L-A1A2, 13:R-A3, 14:L-A3, 15:3rd-A2, 16:3rd-A3, 17:R-M2, 18:R-M3, 19:L-M2, 20:L-M3, 21:R-P3P4, 22:L-P3P4, 23:R-VA, 24:L-VA, 25:R-SCA, 26:L-SCA, 27:R-AICA, 28:L-AICA, 29:R-PICA, 30:L-PICA, 31:R-AChA, 32:L-AChA, 33:R-OA, 34:L-OA, 35:R-ICA-C1-C5, 36:L-ICA-C1-C5

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **ZScoreNormalization** (채널명 `angiography` 기준 자동 선택)
- target spacing: **[0.6, 0.375, 0.375]** mm
  (원본 median spacing [0.6, 0.375, 0.375])
- patch size: **[128, 256, 256]** · batch size: **1**
- median image size(voxels): [196.0, 456.5, 372.0]
- 리샘플: data=`resample_data_or_seg_to_shape`, seg=`resample_data_or_seg_to_shape` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `PlainConvUNet` · stages=6 · features=[32, 64, 128, 256, 320, 320]
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **nnUNetTrainerSkeletonRecallNoMirroring_250ep** · 목표 epochs: **250** · 완료 epoch: **250**
- base 대비 변경점:
  - trainer = `nnUNetTrainerSkeletonRecallNoMirroring_250ep` (base=`nnUNetTrainer`), epochs 250 (base 1000)
  - 추가 인자: `-p nnUNetPlansUZH  -pretrained_weights /home/user/TopAneu/seg/sblee/nnunet/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 0)
- 출처: nnU-Net 내장 validation
- **foreground mean Dice: 0.6015**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | BA | 0.9478 | 7593.2 | 296.6 | 505.2 |
| 2 | R-P1P2 | 0.9048 | 4342.933333333333 | 234.06666666666666 | 714.3333333333334 |
| 3 | L-P1P2 | 0.8940 | 4289.2 | 413.93333333333334 | 647.3333333333334 |
| 4 | R-ICA-C6-C7 | 0.9245 | 6768.333333333333 | 1158.9333333333334 | 553.2666666666667 |
| 5 | R-M1 | 0.9018 | 4291.266666666666 | 521.9333333333333 | 425.26666666666665 |
| 6 | L-ICA-C6-C7 | 0.9544 | 6893.066666666667 | 244.33333333333334 | 441.8 |
| 7 | L-M1 | 0.8872 | 4024.4666666666667 | 596.1333333333333 | 479.53333333333336 |
| 8 | R-Pcom | 0.0000 | 0.0 | 0.0 | 468.0 |
| 9 | L-Pcom | 0.0000 | 0.0 | 0.0 | 670.4666666666667 |
| 10 | Acom | 0.0000 | 0.0 | 0.0 | 265.8666666666667 |
| 11 | R-A1A2 | 0.8960 | 3976.5333333333333 | 350.2 | 660.2666666666667 |
| 12 | L-A1A2 | 0.8933 | 3725.8 | 372.4 | 532.6666666666666 |
| 13 | R-A3 | 0.8590 | 2299.6666666666665 | 297.6 | 441.8666666666667 |
| 14 | L-A3 | 0.8085 | 2120.733333333333 | 436.26666666666665 | 313.2 |
| 15 | 3rd-A2 | 0.0000 | 0.0 | 0.0 | 1.3333333333333333 |
| 16 | 3rd-A3 | 0.0000 | 0.0 | 0.0 | 112.46666666666667 |
| 17 | R-M2 | 0.9028 | 15366.4 | 1184.8666666666666 | 2185.0666666666666 |
| 18 | R-M3 | 0.8432 | 6035.0 | 956.5333333333333 | 1398.9333333333334 |
| 19 | L-M2 | 0.9111 | 15438.933333333332 | 896.8 | 2168.4666666666667 |
| 20 | L-M3 | 0.8558 | 6680.4 | 1024.4666666666667 | 1188.1333333333334 |
| 21 | R-P3P4 | 0.8165 | 2904.866666666667 | 575.2 | 769.5333333333333 |
| 22 | L-P3P4 | 0.7286 | 2351.4 | 1033.0 | 476.0 |
| 23 | R-VA | 0.9215 | 13811.666666666666 | 267.73333333333335 | 1895.0666666666666 |
| 24 | L-VA | 0.8603 | 13719.066666666668 | 279.6 | 1866.8666666666666 |
| 25 | R-SCA | 0.8441 | 1941.3333333333333 | 382.2 | 368.46666666666664 |
| 26 | L-SCA | 0.8756 | 2032.1333333333334 | 267.46666666666664 | 368.0 |
| 27 | R-AICA | 0.0000 | 0.0 | 0.0 | 665.2666666666667 |
| 28 | L-AICA | 0.0000 | 0.0 | 0.0 | 555.3333333333334 |
| 29 | R-PICA | 0.6772 | 3024.2 | 536.5333333333333 | 796.2 |
| 30 | L-PICA | 0.6474 | 1993.0666666666666 | 569.4666666666667 | 613.7333333333333 |
| 31 | R-AChA | 0.0000 | 0.0 | 0.0 | 118.33333333333333 |
| 32 | L-AChA | 0.0000 | 0.0 | 0.0 | 195.6 |
| 33 | R-OA | 0.0000 | 0.0 | 0.0 | 874.1333333333333 |
| 34 | L-OA | 0.0000 | 0.0 | 0.0 | 1005.5333333333333 |
| 35 | R-ICA-C1-C5 | 0.9385 | 34063.86666666667 | 768.2666666666667 | 4356.8 |
| 36 | L-ICA-C1-C5 | 0.9599 | 32693.2 | 1188.6 | 1539.8 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.8483** @ epoch 221
- best EMA pseudo-Dice: **0.6014999747276306** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | BA | 0.9494 |
| 2 | R-P1P2 | 0.9072 |
| 3 | L-P1P2 | 0.8880 |
| 4 | R-ICA-C6-C7 | 0.9190 |
| 5 | R-M1 | 0.8958 |
| 6 | L-ICA-C6-C7 | 0.9454 |
| 7 | L-M1 | 0.8881 |
| 8 | R-Pcom | 0.0000 |
| 9 | L-Pcom | 0.0000 |
| 10 | Acom | 0.0000 |
| 11 | R-A1A2 | 0.8899 |
| 12 | L-A1A2 | 0.8983 |
| 13 | R-A3 | 0.8375 |
| 14 | L-A3 | 0.8467 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.8847 |
| 18 | R-M3 | 0.7994 |
| 19 | L-M2 | 0.8872 |
| 20 | L-M3 | 0.8271 |
| 21 | R-P3P4 | 0.8034 |
| 22 | L-P3P4 | 0.7341 |
| 23 | R-VA | 0.9335 |
| 24 | L-VA | 0.9385 |
| 25 | R-SCA | 0.8098 |
| 26 | L-SCA | 0.8218 |
| 27 | R-AICA | 0.0000 |
| 28 | L-AICA | 0.0000 |
| 29 | R-PICA | 0.8323 |
| 30 | L-PICA | 0.7799 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.0000 |
| 34 | L-OA | 0.0000 |
| 35 | R-ICA-C1-C5 | 0.9448 |
| 36 | L-ICA-C1-C5 | 0.9651 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.6, 0.375, 0.375] mm (이방성비 1.60)
- 검증 Dice=0 (검출 실패) 클래스 11/36: R-Pcom, L-Pcom, Acom, 3rd-A2, 3rd-A3, R-AICA, L-AICA, R-AChA, L-AChA, R-OA …
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
