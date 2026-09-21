# C25_crop_mip_rf

**크롭 분기를 MIP+RF로 교체하고 기하 분기와 혼합(w 스윕)**

판정: **기각**

> 기록 기준: covered_gt 분모(PROJECT_RULES.md §6-1) · 순위·최고 판정은 **MCC**

## 결과

앙상블 이득 사실상 0 (macro-recall 0.324 → 0.325). C25-b로 원인 분해: 크롭만 맞히는 샘플이 43/268 있고 Pcom은 기하 0.300 → 크롭 0.650이라 정보는 상보적이었으나, 혼합 가중 w를 올리면 val e2e MCC가 오히려 하락(w0.3 0.3375 / w0.5 0.3553 / w0.7 0.3908 < 기준 0.4413).

## 산출물

`results/` 에 json 2개 복사. 원본은 `analysis/`.

---
_생성 2026-08-18 01:17_ · `organize_c_experiments_v2.py`
