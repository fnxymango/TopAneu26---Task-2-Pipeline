# 실험 요약 — B2_bundle_recipe_retrain

> 데이터셋 **Dataset722_TopAneuPjh3cls417** · config **3d_fullres** · fold **2** · 시작 2026-09-04T11:39:42+00:00
> 명령: `CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=1 nnUNetv2_train 722 3d_fullres 2 -p nnUNetPlans -tr nnUNetTrainer_250epochs`

## 0. 계보 — baseline / 직전 실험 대비 변경점
_(analysis/experiment_lineage.json에 항목 없음 — 새 실험이면 추가할 것)_

## 1. 데이터 / 라벨
- 모달리티/채널: **noNorm** · 학습 케이스 **415**
- 라벨 체계: 전경 클래스 **2개** (+background)
- 클래스: 1:vessel, 2:aneurysm

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **NoNormalization** (채널명 `noNorm` 기준 자동 선택)
- target spacing: **[0.4, 0.4, 0.4]** mm
  (원본 median spacing [0.5, 0.382, 0.382])
- patch size: **[112, 160, 128]** · batch size: **2**
- median image size(voxels): [219.0, 453.0, 368.0]
- 리샘플: data=`resample_data_or_seg_to_shape`, seg=`resample_data_or_seg_to_shape` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `PlainConvUNet` · stages=6 · features=[32, 64, 128, 256, 320, 320]
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **nnUNetTrainer_250epochs** · 목표 epochs: **250** · 완료 epoch: **250**
- base 대비 변경점:
  - trainer = `nnUNetTrainer_250epochs` (base=`nnUNetTrainer`), epochs 250 (base 1000)
  - 추가 인자: `-p nnUNetPlans`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 2)
- 출처: nnU-Net 내장 validation
- **foreground mean Dice: 0.7244**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | vessel | 0.8998 | 145062.90277777778 | 8761.694444444445 | 19133.555555555555 |
| 2 | aneurysm | 0.5489 | 2866.5416666666665 | 680.9027777777778 | 834.7916666666666 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.6565** @ epoch 216
- best EMA pseudo-Dice: **0.8432999849319458** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | vessel | 0.9090 |
| 2 | aneurysm | 0.8105 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.4, 0.4, 0.4] mm (이방성비 1.00)
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
