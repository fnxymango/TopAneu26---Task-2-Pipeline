# C6_consensus_2of2

**모델 합의 마스크 (2/2 교집합 → 이후 5-fold N/M 다수결로 일반화)**

판정: **부분채택**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

2/2 합의는 FP 55→29이나 병변 1개 손실. C7 거리게이팅(55→12, 손실 0)이 더 우수해 단독으로는 미채택. 다만 `--mode lesion` 5-fold 다수결은 최종 파이프라인에 채택됨.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
