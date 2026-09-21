# 실험 요약 — D610_vessel_uzh_ft_f1

> 데이터셋 **Dataset610_TopAneuVesselUZH** · config **3d_fullres** · fold **1** · 시작 2026-07-31T21:57:10+09:00
> 명령: `CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=1 nnUNetv2_train 610 3d_fullres 1 -p nnUNetPlansUZH -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -pretrained_weights /home/user/TopAneu/seg/sblee/nnunet/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth`

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
  - 추가 인자: `-p nnUNetPlansUZH  -pretrained_weights /home/user/TopAneu/seg/sblee/nnunet/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 1)
- 출처: nnU-Net 내장 validation
- **foreground mean Dice: 0.6186**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | BA | 0.9333 | 8528.70588235294 | 1096.0 | 602.8235294117648 |
| 2 | R-P1P2 | 0.9203 | 4376.588235294118 | 239.05882352941177 | 574.8235294117648 |
| 3 | L-P1P2 | 0.9082 | 4248.117647058823 | 339.0 | 505.4117647058824 |
| 4 | R-ICA-C6-C7 | 0.9399 | 6021.0 | 263.52941176470586 | 429.5882352941176 |
| 5 | R-M1 | 0.8965 | 3991.0 | 270.6470588235294 | 699.8235294117648 |
| 6 | L-ICA-C6-C7 | 0.9403 | 5804.588235294118 | 265.88235294117646 | 529.2941176470588 |
| 7 | L-M1 | 0.9004 | 3993.4117647058824 | 406.5882352941176 | 519.0588235294117 |
| 8 | R-Pcom | 0.0000 | 0.0 | 0.0 | 342.70588235294116 |
| 9 | L-Pcom | 0.0000 | 0.0 | 0.0 | 325.8235294117647 |
| 10 | Acom | 0.0000 | 0.0 | 0.0 | 211.35294117647058 |
| 11 | R-A1A2 | 0.8551 | 3201.5882352941176 | 313.6470588235294 | 887.9411764705883 |
| 12 | L-A1A2 | 0.8134 | 3228.9411764705883 | 550.7647058823529 | 627.3529411764706 |
| 13 | R-A3 | 0.8087 | 1978.0 | 370.1764705882353 | 352.11764705882354 |
| 14 | L-A3 | 0.7447 | 1548.0 | 282.3529411764706 | 637.6470588235294 |
| 15 | 3rd-A2 | 0.0000 | 0.0 | 0.0 | 19.0 |
| 16 | 3rd-A3 | 0.0000 | 0.0 | 0.0 | 152.7058823529412 |
| 17 | R-M2 | 0.9179 | 14943.235294117647 | 995.1764705882352 | 1853.5294117647059 |
| 18 | R-M3 | 0.8475 | 5524.235294117647 | 940.4705882352941 | 1181.8823529411766 |
| 19 | L-M2 | 0.9088 | 14825.823529411764 | 1077.7058823529412 | 2053.294117647059 |
| 20 | L-M3 | 0.8339 | 5798.0 | 1084.5882352941176 | 1350.4705882352941 |
| 21 | R-P3P4 | 0.8042 | 2724.8823529411766 | 774.1176470588235 | 469.4117647058824 |
| 22 | L-P3P4 | 0.8254 | 2304.3529411764707 | 554.6470588235294 | 565.7058823529412 |
| 23 | R-VA | 0.8991 | 7972.411764705882 | 275.0 | 1540.1764705882354 |
| 24 | L-VA | 0.9251 | 11156.941176470587 | 615.3529411764706 | 1482.5882352941176 |
| 25 | R-SCA | 0.8317 | 1733.7058823529412 | 208.35294117647058 | 550.1764705882352 |
| 26 | L-SCA | 0.8388 | 1618.3529411764705 | 202.58823529411765 | 404.5882352941176 |
| 27 | R-AICA | 0.0000 | 0.0 | 0.0 | 752.2352941176471 |
| 28 | L-AICA | 0.5941 | 426.05882352941177 | 171.35294117647058 | 250.47058823529412 |
| 29 | R-PICA | 0.7445 | 2016.1176470588234 | 262.11764705882354 | 575.8823529411765 |
| 30 | L-PICA | 0.7088 | 2022.0 | 640.7058823529412 | 378.7647058823529 |
| 31 | R-AChA | 0.0000 | 0.0 | 0.0 | 215.1764705882353 |
| 32 | L-AChA | 0.0000 | 0.0 | 0.0 | 263.94117647058823 |
| 33 | R-OA | 0.0000 | 0.0 | 0.0 | 897.8823529411765 |
| 34 | L-OA | 0.0000 | 0.0 | 0.0 | 945.1764705882352 |
| 35 | R-ICA-C1-C5 | 0.9643 | 33640.294117647056 | 461.7647058823529 | 2008.8823529411766 |
| 36 | L-ICA-C1-C5 | 0.9637 | 32224.058823529413 | 658.0 | 1648.235294117647 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.8583** @ epoch 236
- best EMA pseudo-Dice: **0.6161999702453613** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | BA | 0.9158 |
| 2 | R-P1P2 | 0.9133 |
| 3 | L-P1P2 | 0.8901 |
| 4 | R-ICA-C6-C7 | 0.9213 |
| 5 | R-M1 | 0.8902 |
| 6 | L-ICA-C6-C7 | 0.9248 |
| 7 | L-M1 | 0.8905 |
| 8 | R-Pcom | 0.0000 |
| 9 | L-Pcom | 0.0000 |
| 10 | Acom | 0.0000 |
| 11 | R-A1A2 | 0.8820 |
| 12 | L-A1A2 | 0.8795 |
| 13 | R-A3 | 0.8511 |
| 14 | L-A3 | 0.8050 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.8904 |
| 18 | R-M3 | 0.8130 |
| 19 | L-M2 | 0.8919 |
| 20 | L-M3 | 0.8024 |
| 21 | R-P3P4 | 0.7851 |
| 22 | L-P3P4 | 0.7877 |
| 23 | R-VA | 0.8758 |
| 24 | L-VA | 0.9026 |
| 25 | R-SCA | 0.8123 |
| 26 | L-SCA | 0.8203 |
| 27 | R-AICA | 0.0000 |
| 28 | L-AICA | 0.6290 |
| 29 | R-PICA | 0.8275 |
| 30 | L-PICA | 0.7894 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.0000 |
| 34 | L-OA | 0.0000 |
| 35 | R-ICA-C1-C5 | 0.9622 |
| 36 | L-ICA-C1-C5 | 0.9575 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.6, 0.375, 0.375] mm (이방성비 1.60)
- 검증 Dice=0 (검출 실패) 클래스 10/36: R-Pcom, L-Pcom, Acom, 3rd-A2, 3rd-A3, R-AICA, R-AChA, L-AChA, R-OA, L-OA
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
