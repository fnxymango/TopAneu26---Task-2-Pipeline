# D600 vessel 0점 클래스 판정 + 위치라벨 관점 우선순위 (2026-07-30)

> 출처: `nnUNet_preprocessed/Dataset{510,600}/*/*.pkl`의 `class_locations`(GT 존재 여부), fold0 bd0 런의 per-class pseudo-Dice.
> 결론 한 줄: **36클래스 평균은 목표와 무관한 클래스에 끌려다니고 있다. 위치라벨(29클래스)이 실제로 요구하는 vessel 클래스로 우선순위를 다시 매겨야 한다.**

## 1. 0점 4클래스의 성격이 둘로 갈린다

| vessel class | GT 존재 | fold0 Dice | 판정 |
|---|---|---|---|
| 3rd-A2 | 15/98 (15%) | 0.00 | **희소 해부변이**(azygos ACA). 학습 데이터 자체가 ~10케이스 |
| 3rd-A3 | 29/98 (30%) | 0.00 | 동일 계열 |
| R-AChA | 61/98 (62%) | 0.00 | **흔한데 실패** — 희소성 변명 불가 |
| L-AChA | 74/98 (76%) | 0.00 | 동일. 뇌혈관 중 최소 구경 |

## 2. 위치 taxonomy(29)는 혈관 이름이 아니라 **분기점 이름**이다

Dataset510의 29클래스는 `R-3.4 ICA C7-Pcom-junction`, `1.10 BA tip`, `R-5.2 M1-M2 junction`처럼
**두 혈관이 만나는 지점** 또는 `R-5.1 M1 trunk`, `R-3.6 ICA C7-nonBranch`처럼 **한 혈관의 구간**으로 정의된다.
→ 같은 `R-ICA-C6-C7` 하나 위에 3.2 / 3.3 / 3.4 / 3.5 / 3.6 / 3.7 여섯 클래스가 올라간다.

### 동맥류 위치 분포 (98케이스, 상위)
| n | 위치클래스 | 필요한 vessel class | 현재 상태 |
|---|---|---|---|
| 18 | 4.1 Acom complex | **Acom** (+A1A2) | **0.42** ⚠️ |
| 17 / 14 | R- / L-5.2 M1-M2 junction | M1 + M2 | 0.89 / 0.86 ✅ |
| 9 / 7 | L- / R-3.4 ICA C7-Pcom-junction | ICA-C6-C7 + **Pcom** | L 0.85 / **R 0.018** ❌ |
| 7 / 4 | R- / L-3.1 ICA infraclinoid | ICA-C1-C5 | 0.93 / 0.96 ✅ |
| 7 / 3 | R- / L-5.1 M1 trunk | M1 (+분기 아님 판정) | ✅ |
| 4 / 3 | R- / L-3.6 ICA C7-nonBranch | ICA (+분기 아님 판정) | ✅ |
| 4 / 1 | L- / R-3.2 ICA C6-OA-junction | ICA + OA | 0.82 / 0.86 ✅ |
| 3 / 3 | L- / R-3.5 ICA C7-AChA-junction | ICA + **AChA** | **0.00** ❌ |
| 2 / 1 | R- / L-3.7 ICA C7-terminus | ICA 종단(M1·A1 접점) | ✅ |
| 2·1·1·1·1·1 | BA tip, VA trunk, VA-PICA, BA-SCA, P1P2, PICA trunk | 후순환 | 데이터 극소 |

## 3. 판정

1. **3rd-A2 / 3rd-A3 → 추적 중단.** 29개 위치클래스 어디에도 등장하지 않는다. 목표에 기여 0.
   → **헤드라인 지표를 present-class mean으로 승격**하고 36클래스 평균은 참고치로 강등.
   (fold0 bd0: 36클래스 0.7167 vs present-class 0.8063 — 차이의 정체가 이것)
2. **AChA → 공략 대상 유지.** 62/76% 케이스에 존재하고, `3.5 ICA C7-AChA-junction`(6케이스)을
   이름 붙이려면 반드시 필요하다. 0점이면 그 위치클래스는 원리적으로 불가능.
3. **새 우선순위 — 36클래스 평균에 묻혀 있던 진짜 병목:**
   - **Acom 0.42** — 가장 흔한 동맥류 위치(18케이스)가 여기에 걸려 있다. **최우선.**
   - **R-Pcom 0.018** — `3.4`(R 7케이스)를 막는다. L-Pcom은 0.85인데 R만 붕괴 = 좌우 비대칭 원인 조사 필요.
   - AChA 양측 0.00
   → 이 셋이 위치라벨 관점에서 vessel 모델의 실질 병목. M1/M2/ICA 계열은 이미 충분하다.

## 4. 지표 변경 제안
- 1차 지표: **present-class mean Dice** + present-class clDice
- 2차: 위 3개 병목 클래스의 개별 Dice (Acom / R-Pcom / AChA)
- 36클래스 평균은 리더보드에 남기되 해석 주의 문구 첨부
