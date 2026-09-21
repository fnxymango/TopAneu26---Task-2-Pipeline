# C18_composite_reselect

**공식 6지표 복합으로 재집계·재선택**

판정: **분석**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

공식 랭킹은 MCC 단독이 아니라 P/R/MCC/Dice/VS/HD95 6개 평균. 당시 test에서 MCC 순위 == 복합 순위였으나, **이후 반례 발생**: ÷52는 확률평균 1위, ÷36(covered_gt)은 vote2 1위. C27 th0.15는 MCC 열세·복합 우세. 원인은 HD95만 분모에 비례하지 않기 때문. 패턴: 희귀클래스를 공격적으로 예측하는 개입(β·합성·다수결)은 MCC를 올리며 HD95를 악화시킨다.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
