# C12_group_experts

**해부그룹 5-way 판정 후 그룹별 전문가 분류기**

판정: **기각**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

macro-recall 0.360 (기준 0.455). 1단 그룹 정확도 0.944로 예상대로 높았으나, 쪼개면 1단 오류가 복구 불가능하게 전파되고 전문가별 학습표본이 줄어 통합 모델보다 나쁨.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
