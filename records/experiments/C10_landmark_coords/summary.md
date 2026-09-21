# C10_landmark_coords

**해부 랜드마크(BA tip, R/L ICA terminus) 기준 정규화 좌표 6차원**

판정: **★채택**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

CV macro-recall 0.410 → 0.455 (+11%), 피처 실험 중 최대 단일 개선. 기존 106차원에 전역 위치정보가 0비트였던 공백을 메움. pos 단독은 0.187로 약하나 상보적. 구성: 정규화 좌표 3 + 랜드마크 거리 3. 미러 시 x부호 반전 + dR/dL swap. 블록 가중치 0.5.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
