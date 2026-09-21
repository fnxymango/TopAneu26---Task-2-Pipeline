# 실험 요약 — D610_vessel_uzh_ft_f2

> 데이터셋 **Dataset610_TopAneuVesselUZH** · config **3d_fullres** · fold **2** · 시작 2026-07-31T22:22:10+09:00
> 명령: `CUDA_VISIBLE_DEVICES=2 nnUNet_def_n_proc=1 nnUNetv2_train 610 3d_fullres 2 -p nnUNetPlansUZH -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep -pretrained_weights /home/user/TopAneu/seg/sblee/nnunet/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth`

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
  - 추가 인자: `-p nnUNetPlansUZH  -pretrained_weights /home/user/TopAneu/seg/sblee/nnunet/pretrained/uzh_topbrain/opt/app/trained_models/Dataset525_topbrain_AM_case_01_27_wReg/Tr_rot30_DiffClusterSM_TopK_ceW_TopBrainAM_13fgCls_cowPT_LRx0p1__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 2)
- 출처: nnU-Net 내장 validation
- **foreground mean Dice: 0.6015**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | BA | 0.9239 | 7709.64705882353 | 220.94117647058823 | 899.5294117647059 |
| 2 | R-P1P2 | 0.9090 | 4468.64705882353 | 331.0 | 524.1764705882352 |
| 3 | L-P1P2 | 0.8983 | 4368.058823529412 | 166.35294117647058 | 777.9411764705883 |
| 4 | R-ICA-C6-C7 | 0.9370 | 6671.470588235294 | 516.2941176470588 | 571.7647058823529 |
| 5 | R-M1 | 0.8989 | 4736.294117647059 | 340.88235294117646 | 742.8235294117648 |
| 6 | L-ICA-C6-C7 | 0.9442 | 6441.882352941177 | 176.7058823529412 | 581.4705882352941 |
| 7 | L-M1 | 0.8902 | 4883.588235294118 | 567.0 | 561.2941176470588 |
| 8 | R-Pcom | 0.0000 | 0.0 | 0.0 | 516.7058823529412 |
| 9 | L-Pcom | 0.6561 | 325.05882352941177 | 59.529411764705884 | 190.1764705882353 |
| 10 | Acom | 0.0000 | 0.0 | 0.0 | 453.3529411764706 |
| 11 | R-A1A2 | 0.8810 | 3953.764705882353 | 458.05882352941177 | 627.0588235294117 |
| 12 | L-A1A2 | 0.8758 | 4014.235294117647 | 438.1764705882353 | 663.4117647058823 |
| 13 | R-A3 | 0.8397 | 2502.705882352941 | 277.2352941176471 | 432.8235294117647 |
| 14 | L-A3 | 0.7885 | 1888.235294117647 | 350.70588235294116 | 421.0 |
| 15 | 3rd-A2 | 0.0000 | 0.0 | 0.0 | 89.0 |
| 16 | 3rd-A3 | 0.0000 | 0.0 | 0.0 | 91.11764705882354 |
| 17 | R-M2 | 0.8992 | 15258.29411764706 | 874.6470588235294 | 2617.1176470588234 |
| 18 | R-M3 | 0.8450 | 6231.411764705882 | 855.4117647058823 | 1419.235294117647 |
| 19 | L-M2 | 0.9109 | 15821.117647058823 | 785.9411764705883 | 2320.764705882353 |
| 20 | L-M3 | 0.8370 | 6986.470588235294 | 1367.764705882353 | 1252.8823529411766 |
| 21 | R-P3P4 | 0.7774 | 3138.470588235294 | 482.8235294117647 | 885.6470588235294 |
| 22 | L-P3P4 | 0.7711 | 2628.823529411765 | 408.8235294117647 | 987.2352941176471 |
| 23 | R-VA | 0.8979 | 12512.941176470587 | 222.05882352941177 | 1765.0588235294117 |
| 24 | L-VA | 0.9270 | 18645.941176470587 | 408.94117647058823 | 2161.5882352941176 |
| 25 | R-SCA | 0.8618 | 2496.9411764705883 | 408.29411764705884 | 402.11764705882354 |
| 26 | L-SCA | 0.8280 | 1967.0588235294117 | 223.47058823529412 | 592.1764705882352 |
| 27 | R-AICA | 0.0000 | 0.0 | 0.0 | 863.6470588235294 |
| 28 | L-AICA | 0.0000 | 0.0 | 0.0 | 994.7058823529412 |
| 29 | R-PICA | 0.0000 | 0.0 | 0.0 | 2703.235294117647 |
| 30 | L-PICA | 0.7325 | 2604.176470588235 | 780.0588235294117 | 476.0 |
| 31 | R-AChA | 0.0000 | 0.0 | 0.0 | 174.1764705882353 |
| 32 | L-AChA | 0.0000 | 0.0 | 0.0 | 152.41176470588235 |
| 33 | R-OA | 0.0000 | 0.0 | 0.0 | 1102.3529411764705 |
| 34 | L-OA | 0.0000 | 0.0 | 0.0 | 1062.5882352941176 |
| 35 | R-ICA-C1-C5 | 0.9615 | 32621.58823529412 | 920.1176470588235 | 1655.0588235294117 |
| 36 | L-ICA-C1-C5 | 0.9634 | 33188.117647058825 | 1197.0 | 1338.4705882352941 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.8547** @ epoch 228
- best EMA pseudo-Dice: **0.6022999882698059** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | BA | 0.9374 |
| 2 | R-P1P2 | 0.8874 |
| 3 | L-P1P2 | 0.8988 |
| 4 | R-ICA-C6-C7 | 0.9386 |
| 5 | R-M1 | 0.8897 |
| 6 | L-ICA-C6-C7 | 0.9415 |
| 7 | L-M1 | 0.8788 |
| 8 | R-Pcom | 0.0000 |
| 9 | L-Pcom | 0.7922 |
| 10 | Acom | 0.0000 |
| 11 | R-A1A2 | 0.8881 |
| 12 | L-A1A2 | 0.8771 |
| 13 | R-A3 | 0.8567 |
| 14 | L-A3 | 0.8364 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.8923 |
| 18 | R-M3 | 0.8242 |
| 19 | L-M2 | 0.8943 |
| 20 | L-M3 | 0.8053 |
| 21 | R-P3P4 | 0.8316 |
| 22 | L-P3P4 | 0.7846 |
| 23 | R-VA | 0.9246 |
| 24 | L-VA | 0.9235 |
| 25 | R-SCA | 0.8229 |
| 26 | L-SCA | 0.8126 |
| 27 | R-AICA | 0.0000 |
| 28 | L-AICA | 0.0000 |
| 29 | R-PICA | 0.0000 |
| 30 | L-PICA | 0.7318 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.0000 |
| 34 | L-OA | 0.0000 |
| 35 | R-ICA-C1-C5 | 0.9564 |
| 36 | L-ICA-C1-C5 | 0.9568 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.6, 0.375, 0.375] mm (이방성비 1.60)
- 검증 Dice=0 (검출 실패) 클래스 11/36: R-Pcom, Acom, 3rd-A2, 3rd-A3, R-AICA, L-AICA, R-PICA, R-AChA, L-AChA, R-OA …
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
