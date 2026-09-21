# C28_detect_filter_relax

**c7 필터를 느슨하게 해서 되찾은 병변이 MCC에 이득인지**

판정: **기각(진행중 확인)**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

필터 완화는 커버리지를 실제로 산다(val 0.7674 → 0.8140, 33 → 35병변). 그런데 값을 못 한다: official MCC 0.4413 → 0.4284, **covered-GT MCC도 0.6773 → 0.6443**, top-1 0.7576 → 0.7429. 검출 리콜을 분모에서 뺀 covered-GT까지 떨어졌다는 건 새로 되찾은 병변이 평균적으로 **더 어려운** 병변이라는 뜻. 여기에 FP 증가로 precision까지 깎임(0.4182 → 0.3874). → 남은 경로는 C17과의 조합.

## 산출물

`results/` 에 json 10개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
