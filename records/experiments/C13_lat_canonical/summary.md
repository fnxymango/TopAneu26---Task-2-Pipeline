# C13_lat_canonical

**좌우 canonical화 — 26-way 측면무관 라벨 + 좌우 이진분류기**

판정: **기각**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

macro-recall 0.352. 좌우 전용 분류기가 오히려 더 나쁨(0.869 vs 통합 0.902). '오류가 그룹·측면 내부에서만 난다'는 C8 관찰이 '계층으로 쪼개면 된다'를 함의하지 않는다는 반례.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
