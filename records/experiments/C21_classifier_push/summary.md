# C21_classifier_push

**분류기 밀어올리기 3종 — 케이스 내 클래스 유일성(헝가리안 할당) 등**

판정: **분석**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

한 케이스에 같은 위치클래스 동맥류가 둘인 경우는 드물다(공식 README)는 사전지식을 확률행렬 헝가리안 할당으로 강제. 중복 클래스가 곧바로 FP가 되는 문제를 겨냥.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
