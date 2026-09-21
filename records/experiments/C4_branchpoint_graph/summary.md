# C4_branchpoint_graph

**혈관 중심선에서 분기점 그래프 추출 (Lee thinning + 2mm 스퍼 제거 + 클래스 전이점 + V5 인접성 검증)**

판정: **인프라**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

참조 417 + val/test × pp/raw 전량 추출. 랜드마크(BA tip, R/L ICA terminus) 세 개가 417케이스 중 97.4%에서 모두 검출 — C10 좌표계의 토대. 5.2 vs 5.3이 분기점 절대거리(0.7 vs 5.0mm)로 분리됨을 확인.

## 산출물

`results/` 에 json 0개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
