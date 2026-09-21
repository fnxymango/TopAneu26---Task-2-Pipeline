# C20_prob_threshold

**확률맵 임계 스윕 — 검출 동작점 튜닝**

판정: **분석**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

C16 확률평균의 동작점이 nnU-Net 기본 argmax(0.5)로 한 번도 튜닝된 적이 없었다는 지적에서 출발. 민감도/FP 교환점을 MCC 기준으로 탐색. 결론은 C27이 이어받아 '임계로는 커버리지를 못 산다'로 확정.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
