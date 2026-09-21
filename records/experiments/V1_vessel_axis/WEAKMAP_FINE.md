# 분류 약점 — 세부구간판 · 제출본 `b1on_pf` · test+val · 5시드

그림 `experiments/V1_vessel_axis/territories_fine.png` · 생성 `D1_newdata/{weakmap2,territory_fig2}.py`

검출된 GT 병변 107개 · 시드판정 535회 · 오답 166 (31.0%)

| 영역 | 세부구간 | 병변 | 시드판정 | 정답률 | 오답 | 전체 오답 중 |
|---|---|---|---|---|---|---|
| 후순환 | BA 몸통 · 분지 | 7 | 35 | **40.0%** | 21 | 13% |
| 후순환 | VA · PICA | 8 | 40 | **47.5%** | 21 | 13% |
| ICA | C7 (교통동맥 구간) | 15 | 75 | **50.7%** | 37 | 22% |
| ACA | A1 | 3 | 15 | **60.0%** | 6 | 4% |
| ACA | A2 이원위 | 5 | 25 | **60.0%** | 10 | 6% |
| ICA | C6 (안동맥 구간) | 14 | 70 | **65.7%** | 24 | 14% |
| 후순환 | BA tip | 3 | 15 | **66.7%** | 5 | 3% |
| ICA | C7 종말부 | 3 | 15 | **66.7%** | 5 | 3% |
| MCA | M1 몸통 | 3 | 15 | **66.7%** | 5 | 3% |
| ICA | C1-C5 (해면동 아래) | 10 | 50 | **82.0%** | 9 | 5% |
| MCA | M1 분기부 | 24 | 120 | **83.3%** | 20 | 12% |
| ACA | Acom | 12 | 60 | **95.0%** | 3 | 2% |

## 전시드 오답(편향)이 몰린 세부구간

| 영역 | 세부구간 | 전시드 오답 병변 |
|---|---|---|
| 후순환 | VA · PICA | L-1.3 VA-PICA junction×2, L-1.1 VA trunk, R-1.3 VA-PICA junction |
| 후순환 | BA 몸통 · 분지 | L-1.9 BA-SCA junction, R-1.9 BA-SCA junction, 1.4 BA trunk, 1.5 VA-BA junction |
| 후순환 | BA tip | 1.10 BA tip |
| ICA | C1-C5 (해면동 아래) | L-3.1 ICA infraclinoid C1-C5 |
| ICA | C6 (안동맥 구간) | L-3.3 ICA C6-nonOA×3, L-3.2 ICA C6-OA-junction |
| ICA | C7 (교통동맥 구간) | R-3.6 ICA C7-nonBranch×2, L-3.4 ICA C7-Pcom-junction, R-3.5 ICA C7-AChA-junction, L-3.6 ICA C7-nonBranch, L-3.5 ICA C7-AChA-junction |
| ICA | C7 종말부 | L-3.7 ICA C7-terminus |
| ACA | A1 | L-4.2 A1 |
| ACA | A2 이원위 | L-4.4 A3, R-4.3 A2 |
| MCA | M1 몸통 | R-5.1 M1 trunk |
| MCA | M1 분기부 | L-5.3 M1-M2 junction×3, R-5.2 M1 early bifurcation |
