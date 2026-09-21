# 실험 요약 — D510_loc_3dfr_f0_baseline

> 데이터셋 **Dataset510_TopAneuLoc** · config **3d_fullres** · fold **0** · 시작 2026-07-19T18:11:42+09:00
> 명령: `CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 510 3d_fullres 0 `

## 1. 데이터 / 라벨
- 모달리티/채널: **angiography** · 학습 케이스 **98**
- 라벨 체계: 전경 클래스 **29개** (+background)
- 클래스: 1:R-1.1 VA trunk, 2:L-1.2 PICA trunk, 3:R-1.3 VA-PICA junction, 4:L-1.3 VA-PICA junction, 5:R-1.9 BA-SCA junction, 6:L-1.9 BA-SCA junction, 7:1.10 BA tip, 8:L-2.1 P1P2, 9:R-3.1 ICA infraclinoid C1-C5, 10:L-3.1 ICA infraclinoid C1-C5, 11:R-3.2 ICA C6-OA-junction, 12:L-3.2 ICA C6-OA-junction, 13:R-3.3 ICA C6-nonOA, 14:L-3.3 ICA C6-nonOA, 15:R-3.4 ICA C7-Pcom-junction, 16:L-3.4 ICA C7-Pcom-junction, 17:R-3.5 ICA C7-AChA-junction, 18:L-3.5 ICA C7-AChA-junction, 19:R-3.6 ICA C7-nonBranch, 20:L-3.6 ICA C7-nonBranch, 21:R-3.7 ICA C7-terminus, 22:L-3.7 ICA C7-terminus, 23:4.1 Acom complex, 24:R-4.5 Distal ACA branches, 25:L-4.5 Distal ACA branches, 26:R-5.1 M1 trunk, 27:L-5.1 M1 trunk, 28:R-5.2 M1-M2 junction, 29:L-5.2 M1-M2 junction

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

### 실제 검증 성능 (전체영상 sliding-window, fold 0 val 29클래스)
- **foreground mean Dice: nan**
| id | class | Dice | TP | FP | FN |
|---|---|---|---|---|---|
| 1 | R-1.1 VA trunk | nan | 0.0 | 0.0 | 0.0 |
| 2 | L-1.2 PICA trunk | nan | 0.0 | 0.0 | 0.0 |
| 3 | R-1.3 VA-PICA junction | 0.0000 | 0.0 | 0.0 | 203.1 |
| 4 | L-1.3 VA-PICA junction | nan | 0.0 | 0.0 | 0.0 |
| 5 | R-1.9 BA-SCA junction | nan | 0.0 | 0.0 | 0.0 |
| 6 | L-1.9 BA-SCA junction | nan | 0.0 | 0.0 | 0.0 |
| 7 | 1.10 BA tip | 0.0000 | 0.0 | 0.0 | 46.15 |
| 8 | L-2.1 P1P2 | nan | 0.0 | 0.0 | 0.0 |
| 9 | R-3.1 ICA infraclinoid C1-C5 | 0.0000 | 0.0 | 0.0 | 25.3 |
| 10 | L-3.1 ICA infraclinoid C1-C5 | nan | 0.0 | 0.0 | 0.0 |
| 11 | R-3.2 ICA C6-OA-junction | 0.0000 | 0.0 | 0.0 | 8.95 |
| 12 | L-3.2 ICA C6-OA-junction | nan | 0.0 | 0.0 | 0.0 |
| 13 | R-3.3 ICA C6-nonOA | 0.0000 | 0.0 | 0.0 | 4.3 |
| 14 | L-3.3 ICA C6-nonOA | 0.0000 | 0.0 | 0.0 | 15.2 |
| 15 | R-3.4 ICA C7-Pcom-junction | 0.0000 | 0.0 | 0.0 | 102.9 |
| 16 | L-3.4 ICA C7-Pcom-junction | 0.0000 | 0.0 | 0.0 | 149.55 |
| 17 | R-3.5 ICA C7-AChA-junction | nan | 0.0 | 0.0 | 0.0 |
| 18 | L-3.5 ICA C7-AChA-junction | 0.0000 | 0.0 | 0.0 | 6.55 |
| 19 | R-3.6 ICA C7-nonBranch | 0.0000 | 0.0 | 0.0 | 7.2 |
| 20 | L-3.6 ICA C7-nonBranch | nan | 0.0 | 0.0 | 0.0 |
| 21 | R-3.7 ICA C7-terminus | 0.0000 | 0.0 | 0.0 | 19.4 |
| 22 | L-3.7 ICA C7-terminus | nan | 0.0 | 0.0 | 0.0 |
| 23 | 4.1 Acom complex | 0.0000 | 0.0 | 0.0 | 248.95 |
| 24 | R-4.5 Distal ACA branches | nan | 0.0 | 0.0 | 0.0 |
| 25 | L-4.5 Distal ACA branches | nan | 0.0 | 0.0 | 0.0 |
| 26 | R-5.1 M1 trunk | 0.0000 | 0.0 | 0.0 | 73.3 |
| 27 | L-5.1 M1 trunk | nan | 0.0 | 0.0 | 0.0 |
| 28 | R-5.2 M1-M2 junction | 0.0000 | 0.0 | 0.0 | 113.8 |
| 29 | L-5.2 M1-M2 junction | 0.0000 | 0.0 | 0.0 | 308.95 |

> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **-0.9547** @ epoch 982
- best EMA pseudo-Dice: **?** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

_(없음)_

</details>

## 5. 특이점 / 특징점
- target spacing [0.39, 0.3, 0.39] mm (이방성비 1.30)
- 검증 Dice=0 (검출 실패) 클래스 15/29: R-1.3 VA-PICA junction, 1.10 BA tip, R-3.1 ICA infraclinoid C1-C5, R-3.2 ICA C6-OA-junction, R-3.3 ICA C6-nonOA, L-3.3 ICA C6-nonOA, R-3.4 ICA C7-Pcom-junction, L-3.4 ICA C7-Pcom-junction, L-3.5 ICA C7-AChA-junction, R-3.6 ICA C7-nonBranch …
- 검증셋에 GT 없어 평가불가(nan) 클래스 14: R-1.1 VA trunk, L-1.2 PICA trunk, L-1.3 VA-PICA junction, R-1.9 BA-SCA junction, L-1.9 BA-SCA junction, L-2.1 P1P2, L-3.1 ICA infraclinoid C1-C5, L-3.2 ICA C6-OA-junction, R-3.5 ICA C7-AChA-junction, L-3.6 ICA C7-nonBranch
- 동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존

---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
