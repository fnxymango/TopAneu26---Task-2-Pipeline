# C14_vessel_encoder_feat

**V4-2 혈관분할망 인코더 bottleneck 320차원을 병변 피처로 사용**

판정: **실패(구현)**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

1차 73병변: enc 단독 top-1 0.027 / macro 0.023. 2차 268병변 전량: 0.049 / 0.030 — 무작위 수준. 붙이면 오히려 방해(both macro 0.261 < hand 0.293). 표본 부족이 아니라 hook 텐서 또는 패치 좌표 매핑 오류로 판단. 개념은 타당하나 디버깅 비용이 기대이득을 초과해 2회 실패로 중단.

## 산출물

`results/` 에 json 2개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
