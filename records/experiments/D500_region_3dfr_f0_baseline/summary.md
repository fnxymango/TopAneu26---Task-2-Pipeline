# 실험 요약 — D500_region_3dfr_f0_baseline

> 데이터셋 **Dataset500_TopAneuRegion** · config **3d_fullres** · fold **0** · 시작 2026-07-19T18:11:42+09:00
> 명령: `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 500 3d_fullres 0 `

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **98**
- 라벨 체계: 전경 클래스 **5개** (+background)
- 클래스: 1:VB_posterior, 2:PCA, 3:ICA, 4:ACA_Acom, 5:MCA

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **ZScoreNormalization** (채널명 `angiography` 기준 자동 선택)
- target spacing: **[0.39, 0.3, 0.39]** mm
  (원본 median spacing [0.39, 0.3, 0.39])
- patch size: **[128, 128, 128]** · batch size: **2**
- median image size(voxels): [426.0, 384.5, 357.5]
- 리샘플: data=`resample_data_or_seg_to_shape`, seg=`resample_data_or_seg_to_shape` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `PlainConvUNet` · stages=6 · features=[32, 64, 128, 256, 320, 320]
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **nnUNetTrainer** · 목표 epochs: **1000** · 완료 epoch: **1000**
- base 대비 변경점:
  - base `nnUNetTrainer` 기본 설정 그대로 (SGD lr 1e-2, mom 0.99, DiceCE, 1000 epochs) — **변경 없음**

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 0 val 5클래스)
- **foreground mean Dice: nan**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | VB_posterior | 0.0000 | 0.0 | 0.0 | 249.25 |
| 2 | PCA | nan | 0.0 | 0.0 | 0.0 |
| 3 | ICA | 0.0000 | 0.0 | 831.95 | 339.35 |
| 4 | ACA_Acom | 0.0000 | 0.0 | 0.0 | 248.95 |
| 5 | MCA | 0.0000 | 0.0 | 0.0 | 496.05 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.7844** @ epoch 748
- best EMA pseudo-Dice: **0.2935999929904938** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | VB_posterior | 0.0000 |
| 2 | PCA | nan |
| 3 | ICA | 0.0126 |
| 4 | ACA_Acom | 0.0000 |
| 5 | MCA | 0.0000 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.39, 0.3, 0.39] mm (이방성비 1.30)
- 검증 Dice=0 (검출 실패) 클래스 4/5: VB_posterior, ICA, ACA_Acom, MCA
- 검증셋에 GT 없어 평가불가(nan) 클래스 1: PCA
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
