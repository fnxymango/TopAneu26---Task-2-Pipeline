# 실험 요약 — D600_vessel_skelrec_resencm_250ep_bd0_f2

> 데이터셋 **Dataset600_TopAneuVessel** · config **3d_fullres** · fold **2** · 시작 2026-07-30T23:05:41+09:00
> 명령: `CUDA_VISIBLE_DEVICES=1 nnUNet_def_n_proc=1 nnUNetv2_train 600 3d_fullres 2 -p nnUNetResEncUNetMPlansBD0 -tr nnUNetTrainerSkeletonRecallNoMirroring_250ep`

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **98**
- 라벨 체계: 전경 클래스 **36개** (+background)
- 클래스: 1:BA, 2:R-P1P2, 3:L-P1P2, 4:R-ICA-C6-C7, 5:R-M1, 6:L-ICA-C6-C7, 7:L-M1, 8:R-Pcom, 9:L-Pcom, 10:Acom, 11:R-A1A2, 12:L-A1A2, 13:R-A3, 14:L-A3, 15:3rd-A2, 16:3rd-A3, 17:R-M2, 18:R-M3, 19:L-M2, 20:L-M3, 21:R-P3P4, 22:L-P3P4, 23:R-VA, 24:L-VA, 25:R-SCA, 26:L-SCA, 27:R-AICA, 28:L-AICA, 29:R-PICA, 30:L-PICA, 31:R-AChA, 32:L-AChA, 33:R-OA, 34:L-OA, 35:R-ICA-C1-C5, 36:L-ICA-C1-C5

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **?** (채널명 `angiography` 기준 자동 선택)
- target spacing: **?** mm
  (원본 median spacing [])
- patch size: **?** · batch size: **?**
- median image size(voxels): ?
- 리샘플: data=`?`, seg=`?` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `?` · stages=? · features=?
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **nnUNetTrainerSkeletonRecallNoMirroring_250ep** · 목표 epochs: **1000** · 완료 epoch: **250**
- base 대비 변경점:
  - trainer = `nnUNetTrainerSkeletonRecallNoMirroring_250ep` (base=`nnUNetTrainer`)
  - 추가 인자: `-p nnUNetResEncUNetMPlansBD0`

## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold 2 val ?클래스)
- **foreground mean Dice: (검증 전)**
_(아직 없음 — 최종 검증 미완료)_

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.9745** @ epoch 245
- best EMA pseudo-Dice: **0.678600013256073** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

| id | class | pseudo-Dice |
|---|---|---|
| 1 | BA | 0.9296 |
| 2 | R-P1P2 | 0.8821 |
| 3 | L-P1P2 | 0.8780 |
| 4 | R-ICA-C6-C7 | 0.9153 |
| 5 | R-M1 | 0.8793 |
| 6 | L-ICA-C6-C7 | 0.9313 |
| 7 | L-M1 | 0.8775 |
| 8 | R-Pcom | 0.7798 |
| 9 | L-Pcom | 0.0000 |
| 10 | Acom | 0.6965 |
| 11 | R-A1A2 | 0.8842 |
| 12 | L-A1A2 | 0.8795 |
| 13 | R-A3 | 0.7682 |
| 14 | L-A3 | 0.7537 |
| 15 | 3rd-A2 | 0.0000 |
| 16 | 3rd-A3 | 0.0000 |
| 17 | R-M2 | 0.8528 |
| 18 | R-M3 | 0.7546 |
| 19 | L-M2 | 0.8637 |
| 20 | L-M3 | 0.7339 |
| 21 | R-P3P4 | 0.0000 |
| 22 | L-P3P4 | 0.0000 |
| 23 | R-VA | 0.9091 |
| 24 | L-VA | 0.8975 |
| 25 | R-SCA | 0.8269 |
| 26 | L-SCA | 0.7822 |
| 27 | R-AICA | 0.7491 |
| 28 | L-AICA | 0.7432 |
| 29 | R-PICA | 0.7826 |
| 30 | L-PICA | 0.7072 |
| 31 | R-AChA | 0.0000 |
| 32 | L-AChA | 0.0000 |
| 33 | R-OA | 0.0000 |
| 34 | L-OA | 0.8620 |
| 35 | R-ICA-C1-C5 | 0.9444 |
| 36 | L-ICA-C1-C5 | 0.9464 |

</details>

## 5. 특이점 / 특징점
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
