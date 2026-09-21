# C19_a4_boundary_refine

**A4(lesionscale crop) 모델로 검출 경계 재작성**

판정: **미완**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

A4 전체볼륨 추론만 완료(val 42 / test 83). 병합 규칙 미정, 평가 미실시. DICE·VolSim·HD95 = 랭킹 6지표 중 3개를 직접 겨냥. 현재 DICE 0.135가 MCC 0.224 대비 낮아 가장 싼 구간으로 추정.

## 산출물

`results/` 에 json 0개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
