# 실험 요약 — D600_vessel_skelrec_resencm_250ep

> 데이터셋 **Dataset600_TopAneuVessel** · config **3d_fullres** · fold **0** · 시작 2026-07-27T15:14:30+09:00
> 명령: `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 600 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep`

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **98**
- 라벨 체계: 전경 클래스 **36개** (+background)
- 클래스: 1:BA, 2:R-P1P2, 3:L-P1P2, 4:R-ICA-C6-C7, 5:R-M1, 6:L-ICA-C6-C7, 7:L-M1, 8:R-Pcom, 9:L-Pcom, 10:Acom, 11:R-A1A2, 12:L-A1A2, 13:R-A3, 14:L-A3, 15:3rd-A2, 16:3rd-A3, 17:R-M2, 18:R-M3, 19:L-M2, 20:L-M3, 21:R-P3P4, 22:L-P3P4, 23:R-VA, 24:L-VA, 25:R-SCA, 26:L-SCA, 27:R-AICA, 28:L-AICA, 29:R-PICA, 30:L-PICA, 31:R-AChA, 32:L-AChA, 33:R-OA, 34:L-OA, 35:R-ICA-C1-C5, 36:L-ICA-C1-C5

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **ZScoreNormalization** (채널명 `angiography` 기준 자동 선택)
- target spacing: **[0.39, 0.3, 0.39]** mm
  (원본 median spacing [0.39, 0.3, 0.39])
- patch size: **[128, 128, 128]** · batch size: **2**
- median image size(voxels): [426.0, 384.5, 357.5]
- 리샘플: data=`resample_data_or_seg_to_shape`, seg=`resample_data_or_seg_to_shape` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `ResidualEncoderUNet` · stages=6 · features=[32, 64, 128, 256, 320, 320]
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **nnUNetTrainerSkeletonRecallNoMirroring_250ep** · 목표 epochs: **250** · 완료 epoch: **250**
- base 대비 변경점:
  - trainer = `nnUNetTrainerSkeletonRecallNoMirroring_250ep` (base=`nnUNetTrainer`), epochs 250 (base 1000)
  - 추가 인자: `-p nnUNetResEncUNetMPlans`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 0)
- 출처: 별도 예측(-nps 1) 후 GT 대비 채점 — 내장 validation은 36클래스 확률맵 RAM 스파이크로 crash
- **foreground mean Dice: 0.6790** · mean clDice: **0.8947**
| id | class | Dice |
|---|---|---|
| 1 | BA | 0.9493 |
| 2 | R-P1P2 | 0.8917 |
| 3 | L-P1P2 | 0.8915 |
| 4 | R-ICA-C6-C7 | 0.9442 |
| 5 | R-M1 | 0.9031 |
| 6 | L-ICA-C6-C7 | 0.9572 |
| 7 | L-M1 | 0.8991 |
| 8 | R-Pcom | 0.7228 |
| 9 | L-Pcom | 0.0000 |
| 10 | Acom | 0.5229 |
| 11 | R-A1A2 | 0.8945 |
| 12 | L-A1A2 | 0.8875 |
| 13 | R-A3 | 0.8531 |
| 14 | L-A3 | 0.7696 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.9008 |
| 18 | R-M3 | 0.8178 |
| 19 | L-M2 | 0.8933 |
| 20 | L-M3 | 0.8108 |
| 21 | R-P3P4 | 0.0000 |
| 22 | L-P3P4 | 0.0000 |
| 23 | R-VA | 0.9212 |
| 24 | L-VA | 0.8765 |
| 25 | R-SCA | 0.8361 |
| 26 | L-SCA | 0.8608 |
| 27 | R-AICA | 0.7497 |
| 28 | L-AICA | 0.6604 |
| 29 | R-PICA | 0.6937 |
| 30 | L-PICA | 0.6902 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.8782 |
| 34 | L-OA | 0.8550 |
| 35 | R-ICA-C1-C5 | 0.9478 |
| 36 | L-ICA-C1-C5 | 0.9650 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.9261** @ epoch 244
- best EMA pseudo-Dice: **0.6809999942779541** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | BA | 0.9428 |
| 2 | R-P1P2 | 0.8667 |
| 3 | L-P1P2 | 0.8588 |
| 4 | R-ICA-C6-C7 | 0.9070 |
| 5 | R-M1 | 0.8839 |
| 6 | L-ICA-C6-C7 | 0.9397 |
| 7 | L-M1 | 0.8683 |
| 8 | R-Pcom | 0.8455 |
| 9 | L-Pcom | 0.0000 |
| 10 | Acom | 0.5502 |
| 11 | R-A1A2 | 0.8690 |
| 12 | L-A1A2 | 0.8608 |
| 13 | R-A3 | 0.8428 |
| 14 | L-A3 | 0.6620 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.8809 |
| 18 | R-M3 | 0.7630 |
| 19 | L-M2 | 0.8160 |
| 20 | L-M3 | 0.7672 |
| 21 | R-P3P4 | 0.0000 |
| 22 | L-P3P4 | 0.0000 |
| 23 | R-VA | 0.9145 |
| 24 | L-VA | 0.9370 |
| 25 | R-SCA | 0.8185 |
| 26 | L-SCA | 0.8129 |
| 27 | R-AICA | 0.7314 |
| 28 | L-AICA | 0.7453 |
| 29 | R-PICA | 0.8563 |
| 30 | L-PICA | 0.7144 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.8523 |
| 34 | L-OA | 0.8006 |
| 35 | R-ICA-C1-C5 | 0.9076 |
| 36 | L-ICA-C1-C5 | 0.9562 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.39, 0.3, 0.39] mm (이방성비 1.30)
- 검증 Dice=0 (검출 실패) 클래스 7/36: L-Pcom, 3rd-A2, 3rd-A3, R-P3P4, L-P3P4, R-AChA, L-AChA
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
