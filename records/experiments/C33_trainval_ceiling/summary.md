# C33_trainval_ceiling

**train∪val 체제에서 분류 천장 재측정 + β/τ 재선택**

판정: **보류(전제 철회)**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

C32(분류기 학습셋에 val 추가)를 전제로 설계된 후속이다. 그런데 **2026-08-18 사용자 지시로 val을 학습에 넣지 않기로 했고**(PROJECT_RULES.md §1), C32가 철회되면서 이 실험의 전제도 함께 무효화됐다. 결과 자체는 c33_chain.log 에 남아 있다. 설계 자체는 정직했다 — val이 학습에 들어가면 선택에 못 쓰므로 train∪val 안 환자단위 CV로 τ만 좁히고 test는 확인 1회로만 쓰겠다고 명시했다.

## 산출물

`results/` 에 json 1개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
