# H1 — 통합본의 패치 CNN 환각필터를 우리 front end 에 이식 (2026-09-10)

## 무엇을 바꿨나

**아무것도 재학습하지 않았다.** 검출·혈관 추론 결과를 재사용하고 분류 단계부터 다시 돌렸다.

| 재사용한 것 | 어디서 |
|---|---|
| 검출 마스크 | `_c1_realpred/aneu_{test,val}_{b1ff,e9ff}` |
| 혈관 예측 (V5 후처리판) | `_c1_realpred/vespp_{test,val}` |
| 분기점 그래프 | `_c4_bpgraph/{vespp_test,val_pred}` |
| 원본 영상 | `_c1_realpred/in_{test,val}` |

**새로 가져온 것** — 통합 제출본에서 꺼냈다. md5 로 바이트 동일 확인:

| 파일 | 출처 | md5 |
|---|---|---|
| `src/patch_filter.py` | 컨테이너 이미지 레이어 `/opt/app/src/` | `9fe69c305f9f51c179b4aa6d0c1efb45` |
| `src/features.py` | 〃 | `5399ec834273e6371a6500f2e364da58` |
| `src/config.py` | 〃 | `ff05d298fa9427abea7f792884e11213` |
| `patchclf/headA.pt` + `meta.json` | 모델 tar `./patchclf` | 0.89M 파라미터 · 임계 2.0064e-4 |

**직접 쓴 것**: `pf_dir.py` — 컨테이너의 `run_patch_filter.py` 를 케이스 1개 → 디렉터리 전체로 바꾼 것.
필터 호출 순서·정규화·임계는 원본과 동일. **SimpleITK 로 읽는다** — 원본이 sitk(z,y,x)인데
nibabel(x,y,z)로 읽으면 3D CNN 에 축이 치환된 패치가 들어가고 CNN 은 축 치환에 불변이 아니다.
모달리티는 파일명 추측이 아니라 `dataset_split.json` 의 `modality` 필드에서 읽는다.

## 팔 4개 (시드 0~4 × test·val)

| 태그 | 검출 | 분류기 피처 | gC | 패치필터 |
|---|---|---|---|---|
| `b1on` | b1ff (구 번들 검출기) | 구 | ON (n=2) | ✗ |
| `b1on_pf` | 〃 | 구 | ON | ✓ ← **제출본 근사** |
| `e9off` | e9ff (ResEncL 10폴드) | 개정 | OFF (n=1) | ✗ |
| `e9off_pf` | 〃 | 개정 | OFF | ✓ |

> `b1on` 은 제출본의 **근사**다. 통합본은 검출기 5폴드 중 3폴드만 쓰는데 `b1ff` 는 5폴드다.
> 실물 제출본은 이보다 약간 낮을 가능성이 크다.

## 판정규칙 — 결과 보기 전에 `H1.sh` 머리말에 고정

신 eval 6지표 중 ≥4 개선 ∧ 평균 ΔMCC ≥ 0 을 **test·val 둘 다** 만족할 때만 채택.
필터 임계는 `meta.json` 값을 그대로 쓰고 추론 시점에 다시 고르지 않는다.

## 결과 — 패치필터 채택

| Δ(필터 on−off) | test 개선 | test ΔMCC | val 개선 | val ΔMCC |
|---|---|---|---|---|
| E9 기반 | 5/6 | **+0.0340** | 5/6 | **+0.0400** |
| 구 front end 기반 | 5/6 | +0.0222 | 5/6 | +0.0076 |

아래 줄이 그들이 보고한 +0.0148 과 같은 방향·자릿수 → 재현으로 본다.

**발견**: `meta.json` 의 `det_lost: 0` 은 우리 데이터에서 성립하지 않는다.
5시드 합계로 정답 시행 374→369, 미할당 0→10 — 시드당 TP 1개 손실.
FP 는 시드당 6개 감소라 손익비 6:1 로 여전히 유리하지만 무손실은 아니다.

## 산출물

`report/RESULTS.md` 결과표 · `pred/` 예측 40벌 · `scores/` 채점 40벌 · `logs/`
분석: `D1_newdata/intweak.py`(층 분해) · `iw_dist.py`(오분류 거리) · `iw_headroom.py`(FN/FP 귀속)
