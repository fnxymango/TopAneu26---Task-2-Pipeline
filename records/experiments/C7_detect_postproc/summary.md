# C7_detect_postproc

**검출 후처리 — 혈관거리 게이팅 + 성분크기 필터 (재학습 0)**

판정: **★채택**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

A5-2: FP 55→12, 민감도 손실 0 (min_vox 20 / dist 5mm). A6-2: 152→53. 근거는 GT 병변–혈관 거리 중앙값 0.30mm / p95 0.55mm. A6-2 FP 152개 중 89개가 예측혈관 5mm 밖 — adaptive norm이 비혈관 조직에 허위검출을 만든다는 진단. 이득의 대부분(55→20)이 거리축이고 크기축은 20→12. max_dist 1~5mm는 결과가 동일해 튜닝 여지가 없다(과적합 위험 낮음).

## 산출물

`results/` 에 json 15개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
