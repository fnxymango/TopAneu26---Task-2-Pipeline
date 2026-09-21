# 실험 요약 — V3_vessel_uzh_ft_417_f0

> 데이터셋 **Dataset810_TopAneuVesselUZH417** · config **3d_fullres** · fold **0** · 시작 2026-08-09T04:11:38+00:00
> 명령: `CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=1 nnUNetv2_train 810 3d_fullres 0 -p nnUNetPlansUZH -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -pretrained_weights <project>/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`

## 0. 계보 — baseline / 직전 실험 대비 변경점
- 계열: **V — 혈관 분할(36클래스)**
- 목적: 외부 데이터 전이학습 효과 확인
- **baseline(`V2_vessel_skelrec_resencm_417`) 대비**: **UZH 외부 사전학습 가중치 fine-tune 도입**(Dataset810, nnUNetPlansUZH).
- **직전 실험(`V2_vessel_skelrec_resencm_417`) 대비**: scratch 학습 → 외부 사전학습 모델 fine-tune (새 개념 → 새 10단위)

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **417**
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
  - 추가 인자: `-p nnUNetPlansUZH  -pretrained_weights <project>/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 0)
- 출처: nnU-Net 내장 validation
- **foreground mean Dice: 0.6438**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | BA | 0.9522 | 5801.976190476191 | 170.04761904761904 | 426.35714285714283 |
| 2 | R-P1P2 | 0.9177 | 3185.095238095238 | 143.33333333333334 | 426.42857142857144 |
| 3 | L-P1P2 | 0.9169 | 3238.809523809524 | 228.97619047619048 | 399.0952380952381 |
| 4 | R-ICA-C6-C7 | 0.9407 | 4291.738095238095 | 126.52380952380952 | 424.7142857142857 |
| 5 | R-M1 | 0.8917 | 2631.7619047619046 | 184.61904761904762 | 485.0238095238095 |
| 6 | L-ICA-C6-C7 | 0.9460 | 4454.0952380952385 | 126.83333333333333 | 386.64285714285717 |
| 7 | L-M1 | 0.9131 | 2874.785714285714 | 297.42857142857144 | 316.6904761904762 |
| 8 | R-Pcom | 0.0000 | 0.0 | 0.0 | 324.0 |
| 9 | L-Pcom | 0.6140 | 227.0 | 28.976190476190474 | 72.0 |
| 10 | Acom | 0.0000 | 0.0 | 0.0 | 172.92857142857142 |
| 11 | R-A1A2 | 0.8953 | 2541.4761904761904 | 185.07142857142858 | 425.3809523809524 |
| 12 | L-A1A2 | 0.8974 | 2657.4761904761904 | 216.07142857142858 | 396.7857142857143 |
| 13 | R-A3 | 0.8423 | 1593.0714285714287 | 232.61904761904762 | 322.5 |
| 14 | L-A3 | 0.7855 | 1327.0714285714287 | 375.54761904761904 | 226.16666666666666 |
| 15 | 3rd-A2 | 0.0000 | 0.0 | 0.0 | 85.64285714285714 |
| 16 | 3rd-A3 | 0.0000 | 0.0 | 0.0 | 162.6904761904762 |
| 17 | R-M2 | 0.9061 | 9895.547619047618 | 499.0238095238095 | 1672.357142857143 |
| 18 | R-M3 | 0.8439 | 3708.1428571428573 | 629.5 | 801.7142857142857 |
| 19 | L-M2 | 0.9219 | 10251.380952380952 | 529.3571428571429 | 1307.952380952381 |
| 20 | L-M3 | 0.8423 | 3953.5238095238096 | 791.4285714285714 | 687.6190476190476 |
| 21 | R-P3P4 | 0.8166 | 1723.6190476190477 | 400.35714285714283 | 340.4761904761905 |
| 22 | L-P3P4 | 0.7560 | 1280.1904761904761 | 510.23809523809524 | 259.7857142857143 |
| 23 | R-VA | 0.9078 | 8145.238095238095 | 363.35714285714283 | 1183.3095238095239 |
| 24 | L-VA | 0.9186 | 12056.047619047618 | 553.0238095238095 | 1123.2142857142858 |
| 25 | R-SCA | 0.8497 | 992.8333333333334 | 146.76190476190476 | 213.0952380952381 |
| 26 | L-SCA | 0.8575 | 1045.952380952381 | 121.83333333333333 | 187.0 |
| 27 | R-AICA | 0.6583 | 364.6904761904762 | 87.0952380952381 | 138.16666666666666 |
| 28 | L-AICA | 0.0000 | 0.0 | 0.0 | 390.6190476190476 |
| 29 | R-PICA | 0.6934 | 1274.0 | 476.1666666666667 | 273.14285714285717 |
| 30 | L-PICA | 0.7955 | 1715.2142857142858 | 303.7857142857143 | 395.0 |
| 31 | R-AChA | 0.0000 | 0.0 | 0.0 | 139.64285714285714 |
| 32 | L-AChA | 0.0000 | 0.0 | 0.0 | 143.61904761904762 |
| 33 | R-OA | 0.0000 | 0.0 | 0.0 | 468.8809523809524 |
| 34 | L-OA | 0.0000 | 0.0 | 0.0 | 544.6190476190476 |
| 35 | R-ICA-C1-C5 | 0.9488 | 28066.54761904762 | 389.2857142857143 | 2014.2380952380952 |
| 36 | L-ICA-C1-C5 | 0.9471 | 28193.35714285714 | 334.5 | 2071.1428571428573 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.8675** @ epoch 236
- best EMA pseudo-Dice: **0.640500009059906** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | BA | 0.9388 |
| 2 | R-P1P2 | 0.9000 |
| 3 | L-P1P2 | 0.9083 |
| 4 | R-ICA-C6-C7 | 0.9164 |
| 5 | R-M1 | 0.8779 |
| 6 | L-ICA-C6-C7 | 0.9400 |
| 7 | L-M1 | 0.8988 |
| 8 | R-Pcom | 0.0000 |
| 9 | L-Pcom | 0.8212 |
| 10 | Acom | 0.0000 |
| 11 | R-A1A2 | 0.8809 |
| 12 | L-A1A2 | 0.8934 |
| 13 | R-A3 | 0.8289 |
| 14 | L-A3 | 0.7777 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.8837 |
| 18 | R-M3 | 0.7851 |
| 19 | L-M2 | 0.8791 |
| 20 | L-M3 | 0.7899 |
| 21 | R-P3P4 | 0.8032 |
| 22 | L-P3P4 | 0.7692 |
| 23 | R-VA | 0.9202 |
| 24 | L-VA | 0.9393 |
| 25 | R-SCA | 0.8147 |
| 26 | L-SCA | 0.8264 |
| 27 | R-AICA | 0.7074 |
| 28 | L-AICA | 0.0000 |
| 29 | R-PICA | 0.8250 |
| 30 | L-PICA | 0.8518 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.0000 |
| 34 | L-OA | 0.0000 |
| 35 | R-ICA-C1-C5 | 0.9542 |
| 36 | L-ICA-C1-C5 | 0.9473 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.6, 0.375, 0.375] mm (이방성비 1.60)
- 검증 Dice=0 (검출 실패) 클래스 9/36: R-Pcom, Acom, 3rd-A2, 3rd-A3, L-AICA, R-AChA, L-AChA, R-OA, L-OA
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
