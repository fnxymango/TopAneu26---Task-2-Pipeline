# C17_learned_fp_reject

**검출 후보 TP/FP를 C5 피처 112차원으로 판별하는 학습된 기각기**

판정: **검증완료·미적용**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

out-of-fold(train 292 완전 커버) 후보 596개(TP 225 / FP 371). 임계 0.1에서 민감도 −1.3%로 FP 52% 제거. c7 손규칙과의 중복 확인 필요해 val/test 미적용. **C28에서 '필터 완화만'이 실패했으므로 C17과 반드시 묶어야 함** — 필터를 풀어 병변을 되찾고 늘어난 FP를 학습으로 걷어내는 조합.

## 산출물

`results/` 에 json 2개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
