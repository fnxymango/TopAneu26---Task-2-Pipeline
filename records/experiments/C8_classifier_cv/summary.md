# C8_classifier_cv

**환자단위 5-fold CV 진단 + 피처블록 ablation**

판정: **진단**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

macro-recall 0.383 vs top-1 0.68 — 희귀클래스 실패가 최대 손실원. 그룹 0.951 / 좌우 0.902 → 오류는 전부 그룹·측면 내부. ICA(3.x)가 오류 대부분이고 전부 3.4 Pcom으로 흡인(최빈 29건). ablation(kNN k5 balanced): dist 0.361 / +ov 0.379 / +bp 0.349 / dist+ov+bp 0.383.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
