# 결과 요약 — 정식 split 도입 + loss sweep (2026-07-21)

> 이번 세션에 **새로 돌린 실험 3개** (모두 dataset 520 binary, plain-z, os0.6, 250ep, fold0).
> ⚠️ **split 표준화**: 이번부터 모든 학습 = 정식 `dataset_split.json` (train69 / **val15** / test14, test held-out).
> 이전 auto-split(78/20) 결과(구 D520 0.445 등)와는 **val set이 달라 직접 비교 불가**.

## 공통 설정 (한 레버만 = loss 만 다름)
| 항목 | 값 |
|---|---|
| dataset / 정규화 | Dataset520_TopAneuBinary / ZScoreNormalization (plain-z) |
| split | 정식 canonical, train 69 / val 15 (test 14 held-out) |
| network | PlainConvUNet s6 f[32,64,128,256,320,320] |
| spacing / patch / batch | [0.39,0.3,0.39]mm / [128,128,128] / 2 |
| oversample_fg / epochs | 0.60 / 250 |

## 결과 (정식 val 15, 케이스당 평균 · n_ref=6301 voxel로 3개 모두 동일 GT 확인)
| 순위 | 실험 | loss | **Dice** | TP | FP | FN | P* | R* |
|---|---|---|---|---|---|---|---|---|
| 🥇 | tverskytopk_canonsplit | Tversky(0.3/0.7)+**TopK-CE(k10)** | **0.6849** | 5361 | **659** | 940 | 0.891 | 0.851 |
| 🥈 | tversky_canonsplit (base) | Tversky(0.3/0.7)+CE | 0.6550 | 5421 | 961 | 880 | 0.849 | 0.860 |
| 🥉 | focaltversky_canonsplit | FocalTversky(0.3/0.7,γ1.33)+CE | 0.6281 | 5241 | 851 | 1060 | 0.860 | 0.832 |

\* P/R = aggregate voxel precision/recall (micro, TP·FP·FN 합산). Dice는 per-case 평균이라 P/R과 정확히 일치하진 않음(경향 파악용).
학습 지표(참고): pseudo-Dice EMA base 0.876 / focal 0.860 / topk 0.873. best_val_loss는 loss함수가 달라 **상호 비교 불가**(focal은 +0.11로 부호부터 다름).

## 해석
- **TopK-CE 승리** 🎯 — 가장 어려운 10% voxel에 CE를 집중 → **오검출(FP) 961→659 (−31%)** 로 크게 억제, TP 거의 유지. precision 0.849→0.891, recall은 0.860→0.851로 거의 유지. 순 Dice **+0.030**.
- **Focal-Tversky(γ1.33) 폐기** — (1−Tversky)^γ 변조가 모델을 과하게 보수적으로 만들어 **FN 880→1060 (recall↓)**, Dice −0.027. 초소형 병변 검출과 반대 방향.
- base(plain-z Tversky+CE) = **canonical 기준선 0.6550** 확정.

## 결정
1. **base loss = Tversky + TopK-CE 채택** (이후 실험의 출발점).
2. Focal-Tversky 폐기 (필요시 γ<1 또는 β↑로 recall 쪽 재튜닝 여지만 남김).
3. 정규화 = **plain-z 유지** (adaptive/521은 정식 split 재실행 안 함).

## 한계 / 주의
- **val 15케이스·fold0 단일** → ±0.03은 노이즈 가능. 최종은 5-fold 필요.
- **voxel-Dice는 초소형 병변 검출을 과소평가** → `eval_lesion.py`(lesion recall/FP·위치정확도) 로 재채점해야 확정.
- TopK-CE는 precision 위주 개선이라 **recall(FN 940)** 은 다음 레버(os↑·β↑)로 별도 공략 필요.

## 다음 (계획 확정)
base = **Tversky+TopK-CE** 위에서 한 레버씩, GPU 0~3 병렬:
- **E1 ResEnc-L** (patch 160³, plan/preprocess 1회 선행) — 최대 기대이득
- **E4 os 0.6→0.8** — 병변 노출↑ → FN(940) 공략
- **E2b β 0.7→0.8** / **k 10→5** — recall·hard-focus 튜닝 (각 1줄 trainer)
- **eval_lesion.py** — 진짜 판정 기준, 학습과 병행 제작 → 4개 실험 재채점
