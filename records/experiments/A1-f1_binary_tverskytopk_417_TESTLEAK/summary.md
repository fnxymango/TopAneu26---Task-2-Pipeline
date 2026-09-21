# 실험 요약 — A1-f1_binary_tverskytopk_417

> 데이터셋 **Dataset720_TopAneuBinary417** · config **3d_fullres** · fold **1** · 시작 2026-08-09T06:52:13+00:00
> 명령: `CUDA_VISIBLE_DEVICES=0 nnUNet_def_n_proc=1 nnUNetv2_train 720 3d_fullres 1 -tr nnUNetTrainerTverskyTopkCE`

## 0. 계보 — baseline / 직전 실험 대비 변경점
- 계열: **A — 동맥류 검출(binary, 417코호트)**
- 목적: fold간 분산 확인
- **baseline(`A1_binary_tverskytopk_417`) 대비**: fold만 0→1. 설정 완전 동일.
- **직전 실험(`A1_binary_tverskytopk_417`) 대비**: fold 0→1

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **417**
- 라벨 체계: 전경 클래스 **1개** (+background)
- 클래스: 1:aneurysm

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **ZScoreNormalization** (채널명 `angiography` 기준 자동 선택)
- target spacing: **[0.5, 0.382, 0.382]** mm
  (원본 median spacing [0.5, 0.382, 0.382])
- patch size: **[80, 192, 160]** · batch size: **2**
- median image size(voxels): [219.0, 446.0, 368.0]
- 리샘플: data=`resample_data_or_seg_to_shape`, seg=`resample_data_or_seg_to_shape` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `PlainConvUNet` · stages=6 · features=[32, 64, 128, 256, 320, 320]
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **nnUNetTrainerTverskyTopkCE** · 목표 epochs: **1000** · 완료 epoch: **250**
- base 대비 변경점:
  - trainer = `nnUNetTrainerTverskyTopkCE` (base=`nnUNetTrainer`)

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 1)
- 출처: nnU-Net 내장 validation
- **foreground mean Dice: 0.2442**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | aneurysm | 0.2442 | 2724.178082191781 | 1894.5479452054794 | 996.5205479452055 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.5524** @ epoch 242
- best EMA pseudo-Dice: **0.7443000078201294** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | aneurysm | 0.7065 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.5, 0.382, 0.382] mm (이방성비 1.31)
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
