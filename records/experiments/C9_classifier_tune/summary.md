# C9_classifier_tune

**β 사전확률 역보정 + rel 상대인코딩 + 반경확대**

판정: **부분채택**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

β=0.75로 macro-recall 0.346 → 0.410 (+18%) — 채택. rel(분기점 절대거리를 softmax 상대비율로) 실패: 단독 0.346→0.366이나 β와 결합하면 0.351로 붕괴. 원인은 rel이 5.2/5.3을 가르던 절대거리를 정규화로 지우기 때문(bp 0.146 vs 0.433 → rel 0.763 vs 0.996). 반경 10→25mm 실패(기권 4→0이나 macro 0.410→0.393).

## 산출물

`results/` 에 json 2개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
