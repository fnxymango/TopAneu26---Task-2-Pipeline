# C3_location_knn_a62det

**A6-2(adaptive norm) 검출기로 C2 재평가**

판정: **진단**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

A6-2 단일은 민감도 0.814로 A5-2(0.721)보다 높은데 FP가 3배(55→152)라 e2e는 오히려 짐 (val adj MCC 0.2164 vs 0.2651). '병목은 민감도가 아니라 FP'라는 결론 → C6/C7의 출발점.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
