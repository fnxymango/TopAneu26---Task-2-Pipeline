# TopAneu 실험 일정 (sblee)

> 목표: **동맥류 segmentation + 어느 혈관위치인지 라벨** (location-labeled aneurysm seg).
> 제약: 추론 입력 = CTA/MRA 스캔뿐, **vessel mask는 학습전용**, 제출 = 단일 모델.
> 원칙: **한 번에 한 레버**만 바꾸고 **lesion-wise+위치정확도**로 비교. 유형(saccular 등)은 목표 아님.

---

## 현재 상태
> ⚠️ **split 표준화 (2026-07-21):** 모든 실험 = 정식 `dataset_split.json` (train69/**val15**/test14, test held-out). 이전 auto-split(78/20) 결과는 val이 달라 **비교 무효**. 상세: [results_2026-07-21_canon_loss_sweep.md](results_2026-07-21_canon_loss_sweep.md)

| | 내용 | 결과(정식 val15) |
|---|---|---|
| 베이스라인 | 500(5영역)·510(29클래스), plain nnU-Net | Dice≈0 (멀티클래스 붕괴, 구 split) |
| 정규화 | 520 plain-z vs 521 adaptive | adaptive 미채택, **plain-z 유지** (521 정식 재실행 안 함) |
| **base (canonical)** | 520 binary + Tversky(0.3/0.7)+CE+os0.6, plain-z | **Dice 0.655** · FP961 FN880 |
| **E2 loss sweep** | +Focal-Tversky(γ1.33) vs +TopK-CE(k10) | Focal **0.628 폐기** · **TopK-CE 0.685 채택** (FP↓31%) |
| 도구 | leaderboard csv/xlsx 자동생성, TopAneuAdaptiveNorm, EDA | 완료 |

**→ 현재 base recipe = 520 binary + Tversky(0.3/0.7) + TopK-CE(k10) + os0.6 (plain-z, canonical).** 이후 실험은 이 위에 한 레버씩.

## 기술 배치 맵 (레벨별로 조합)
```
전처리 : 우리 adaptive norm  → (후속) HD-BET(MRA)/HD-CTBET(CTA) brain mask
plans  : ResEnc-L
trainer: DA5 증강 + [Tversky/TopK/Focal] loss + oversampling   ← 한 클래스로 병합
         + (Phase2) 보조 vessel head + Skeleton Recall Loss
추론   : TTA (최종)
```
- 회원님 기술: adaptive-norm→P0, ResEnc-L→E1, DA5→E3, Skeleton Recall→E5, HD-BET→E8, TTA→P5
- 내 로드맵: 평가→P0, 검출강화→P1·P2, 위치→P3, 최종→P5

---

## Phase 0 — 기반 확정 (측정 도구 + base)
| ID | 작업 | 레벨 | 산출/게이트 |
|---|---|---|---|
| **N** | 521 결과로 정규화 확정 (plain z vs adaptive) | 전처리 | base 데이터셋 확정 |
| **EVAL** | `eval_lesion.py` 제작: lesion recall/precision/FP + **위치정확도(fine 29 / region 5)** | 도구 | 이후 모든 비교의 기준 |
| B0(선택) | 순수 binary 베이스라인(default trainer, DiceCE, os0.33) | trainer | Tversky/os 기여 분리 |

## Phase 1 — 검출 recipe 확정 (base 위, 한 레버씩·병렬 가능, 각 ~3h)
| ID | 변경 (한 개) | 레벨 | 겨냥 |
|---|---|---|---|
| **E1** | **ResEnc-L** (`-pl nnUNetPlannerResEncL`, patch 160³) | plans | 맥락↑·성능↑ (제일 쉬움) |
| ~~E2~~ ✅ | loss: **TopK-CE(k10) 채택 0.685** · Focal-Tversky(γ1.33) 폐기 0.628 · β0.8 미시도 | trainer | FP↓31% (recall은 os↑·β↑로 별도 공략) |
| **E3** | **DA5** 증강 | trainer | 정규화 (⚠️작은병변 소실 감시) |
| **E4** | sampling: os 0.6→0.8 / lesion-aware pool | trainer | 병변 노출↑ |
| → | eval_lesion로 recall·FP 비교 → 이긴 것 결합 = **검출 recipe A** |

## Phase 2 — vessel 활용 (학습전용): 검출↑ + 위치 토대
| ID | 변경 | 레벨 | 겨냥 |
|---|---|---|---|
| **E5** | **보조 vessel head(multi-task) + Skeleton Recall Loss**(vessel head, RSNA 동맥류 1위 사용) | trainer/net | 혈관 연결성 내재화 → FP↓, **위치 토대** |
| **E6** | vessel 분기점 hard-negative sampling | trainer | 분기점 오검출 FP↓ |

## Phase 3 — 위치(location) 할당 — 진짜 목표
| ID | 변경 | 겨냥 |
|---|---|---|
| **E7** | (A) 2-stage 검출→위치분류(좌표+국소영상) **또는** (B) multi-task [검출 head + 위치 head] | 위치라벨 생성 |
| → | vessel-aware(P2) 위에서 **위치정확도(29 / 5-region)** 측정 |

## Phase 4 — 전처리 고도화 (병렬 조사·후순위)
| ID | 변경 | 비고 |
|---|---|---|
| **E8** | brain extraction: MRA=HD-BET, CTA=**HD-CTBET/CTA-BET** → adaptive norm within brain | ⚠️CTA-BET 코드 확보 + 이미 crop된 데이터 대비 **추가 이득 검증** 후 적용 |

## Phase 5 — 최종 확정
- 이긴 레버 전부 결합(전처리+ResEnc-L+trainer recipe+위치) → **1000ep × 5-fold 앙상블**
- **TTA** 추가 (mirroring+스케일)
- 제출 포맷 = location-labeled aneurysm segmentation

---
## 평가 게이트 (매 실험 공통)
- 250ep · fold 0 고정 비교 (최종만 1000ep×5fold)
- 지표: **lesion recall(최우선) · FP/case · precision · 위치정확도 · voxel Dice(보조)**
- 판정: 이전 최고 대비 recall/위치정확도 개선 시 채택, 1항목만 변경 기록
