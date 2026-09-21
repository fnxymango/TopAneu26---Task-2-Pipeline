# E-0 — ICA 전용 분류기 `TOPANEU_ICA_EXPERT` · LOCO OOF 관문

규칙은 `e0_icaexpert.py` 머리말에 결과 보기 전 고정. TAB0·G-0 와 같은 장치 · 시드 [0, 1, 2] · LOCO 214폴드 · ICA 102행 / 비ICA 166행.

## 1) 시드 3개 평균 정답 개수

| 팔 | ICA top1 | ICA top2 | 취약 3.5·3.6 top2 | 비ICA top1 |
|---|---|---|---|---|
| 기준 | 56.7 | **77.0** | 4.0 | 129.0 |
| ICA전문가 | 54.3 | **76.3** | 4.0 | 129.0 |

## 2) 판정

| 규칙 | 실측 | 충족 |
|---|---|---|
| ② ICA top-2 > 기준 | 77.0 → 76.3 (-0.7) | **✗** |
| ③ ICA top-1 −2 이내 | 56.7 → 54.3 (-2.3) | **✗** |
| ④ 비ICA 판정 완전 동일 | **다름 33건** | **✗** |

**판정 → 미달 — ICA 전용분류기 축 닫음**

## 3) ICA 클래스별 (시드평균 top-2)

| 클래스 | 학습행 | 기준 | ICA전문가 |
|---|---|---|---|
| L-3.1 ICA infraclinoid C1-C5 | 14 | 13.0 | 13.0 |
| L-3.2 ICA C6-OA-junction | 7 | 3.7 | 3.0 |
| L-3.3 ICA C6-nonOA | 12 | 10.0 | 10.0 |
| L-3.4 ICA C7-Pcom-junction | 7 | 6.0 | 6.0 |
| L-3.5 ICA C7-AChA-junction | 4 | 1.0 | 1.3 |
| L-3.6 ICA C7-nonBranch | 3 | 1.0 | 1.0 |
| L-3.7 ICA C7-terminus | 4 | 4.0 | 4.0 |
| R-3.1 ICA infraclinoid C1-C5 | 13 | 11.0 | 11.0 |
| R-3.2 ICA C6-OA-junction | 2 | 1.0 | 1.0 |
| R-3.3 ICA C6-nonOA | 9 | 7.7 | 7.3 |
| R-3.4 ICA C7-Pcom-junction | 13 | 11.7 | 12.0 |
| R-3.5 ICA C7-AChA-junction | 5 | 2.0 | 1.7 |
| R-3.6 ICA C7-nonBranch | 4 | 0.0 | 0.0 |
| R-3.7 ICA C7-terminus | 5 | 5.0 | 5.0 |

## 4) 총 1022s

남는 실패 기전은 C12 의 **표본 감소** 하나다 — 전문가는 ICA 92행(미러 184)·14클래스만 본다. C12 의 1단 오류 전파는 질량 보존 설계라 구조적으로 발생하지 않는다(규칙 ④ 로 실측).
