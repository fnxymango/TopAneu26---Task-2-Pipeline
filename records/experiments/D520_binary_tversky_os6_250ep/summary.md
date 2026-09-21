# 실험 요약 — D520_binary_tversky_os6_250ep

> 데이터셋 **Dataset520_TopAneuBinary** · config **3d_fullres** · fold **0** · 시작 2026-07-20T13:15:21+09:00
> 명령: `CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 520 3d_fullres 0 -tr nnUNetTrainerTverskyCE`

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **98**
- 라벨 체계: 전경 클래스 **1개** (+background)
- 클래스: 1:aneurysm

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
- trainer: **nnUNetTrainerTverskyCE** · 목표 epochs: **1000** · 완료 epoch: **250**
- base 대비 변경점:
  - trainer = `nnUNetTrainerTverskyCE` (base=`nnUNetTrainer`)

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 0 val 1클래스)
- **foreground mean Dice: 0.4452**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | aneurysm | 0.4452 | 822.15 | 638.05 | 511.45 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.7348** @ epoch 238
- best EMA pseudo-Dice: **0.7768999934196472** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | aneurysm | 0.7688 |

</details>

## 5. 특이점 / 특징점
- target spacing [0.39, 0.3, 0.39] mm (이방성비 1.30)
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
