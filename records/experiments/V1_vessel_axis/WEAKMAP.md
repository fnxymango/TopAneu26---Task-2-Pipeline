# 분류 약점 지도 — 제출본 구성 `b1on_pf` · test+val · 5시드

출처 `intweak_b1on_pf.json` (2026-09-10 생성) · 재생성 `D1_newdata/weakmap.py`

검출된 GT 병변 107/130 · 시드판정 535회 · 정답 369 · **오답 166 (31.0%)**

## 영역별

| 영역 | 병변 | 시드판정 | 정답률 | 오답 수 | 전체 오답 중 |
|---|---|---|---|---|---|
| ICA | 42 | 210 | **64.3%** | 75 | 45% |
| 후순환(VA/BA/PCA) | 18 | 90 | **47.8%** | 47 | 28% |
| MCA | 27 | 135 | **81.5%** | 25 | 15% |
| ACA/Acom | 20 | 100 | **81.0%** | 19 | 11% |

## 크기별 (최대직경 mm)

| 크기 | 병변 | 시드판정 | 정답률 | 오답 수 |
|---|---|---|---|---|
| <3mm | 8 | 40 | **77.5%** | 9 |
| 3~5mm | 59 | 295 | **67.8%** | 95 |
| 5~10mm | 32 | 160 | **74.4%** | 41 |
| ≥10mm | 8 | 40 | **47.5%** | 21 |

## 재현성

- 전시드 오답(편향) **29병변** — 오답 145회 = 전체 오답의 87%
- 시드마다 갈림(분산) 8병변 — 오답 21회
- 전시드 정답 70병변

## 전시드 오답 병변 (고칠 대상)

| GT 클래스 | 직경mm | 5시드 예측 |
|---|---|---|
| 1.10 BA tip | 10.2 | L-4.2 A1×4, L-3.6 ICA C7-nonBranch×1 |
| 1.4 BA trunk | 20.4 | 1.5 VA-BA junction×5 |
| 1.5 VA-BA junction | 9.5 | R-1.3 VA-PICA junction×5 |
| L-1.1 VA trunk | 14.7 | R-1.1 VA trunk×5 |
| L-1.3 VA-PICA junction | 3.3 | L-1.1 VA trunk×5 |
| L-1.3 VA-PICA junction | 4.2 | L-1.1 VA trunk×5 |
| L-1.9 BA-SCA junction | 5.2 | 1.10 BA tip×5 |
| L-3.1 ICA infraclinoid C1-C5 | 3.4 | L-3.3 ICA C6-nonOA×5 |
| L-3.2 ICA C6-OA-junction | 4.1 | L-3.3 ICA C6-nonOA×5 |
| L-3.3 ICA C6-nonOA | 3.4 | L-3.2 ICA C6-OA-junction×5 |
| L-3.3 ICA C6-nonOA | 5.2 | L-3.1 ICA infraclinoid C1-C5×5 |
| L-3.3 ICA C6-nonOA | 3.9 | L-4.2 A1×3, L-3.2 ICA C6-OA-junction×2 |
| L-3.4 ICA C7-Pcom-junction | 4.7 | L-3.3 ICA C6-nonOA×4, L-3.2 ICA C6-OA-junction×1 |
| L-3.5 ICA C7-AChA-junction | 3.9 | L-3.4 ICA C7-Pcom-junction×5 |
| L-3.6 ICA C7-nonBranch | 4.1 | L-3.4 ICA C7-Pcom-junction×5 |
| L-3.7 ICA C7-terminus | 2.1 | L-4.2 A1×5 |
| L-4.2 A1 | 3.6 | 4.1 Acom complex×5 |
| L-4.4 A3 | 4.5 | 미할당×5 |
| L-5.3 M1-M2 junction | 3.1 | L-5.2 M1 early bifurcation×5 |
| L-5.3 M1-M2 junction | 5.5 | L-5.2 M1 early bifurcation×5 |
| L-5.3 M1-M2 junction | 3.4 | 미할당×5 |
| R-1.3 VA-PICA junction | 4.8 | 1.5 VA-BA junction×5 |
| R-1.9 BA-SCA junction | 6.2 | 1.10 BA tip×5 |
| R-3.5 ICA C7-AChA-junction | 5.2 | R-3.4 ICA C7-Pcom-junction×5 |
| R-3.6 ICA C7-nonBranch | 3.4 | R-3.4 ICA C7-Pcom-junction×5 |
| R-3.6 ICA C7-nonBranch | 3.6 | R-3.3 ICA C6-nonOA×5 |
| R-4.3 A2 | 6.3 | R-4.2 A1×5 |
| R-5.1 M1 trunk | 15.2 | R-3.7 ICA C7-terminus×5 |
| R-5.2 M1 early bifurcation | 4.5 | R-5.3 M1-M2 junction×5 |
