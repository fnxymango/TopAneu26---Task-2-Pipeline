# 제출본에 변경을 반영하는 법 — 파일별·위치별 명세

**기준선이 두 개다. 섞지 않는다.**

| | 기준선 | 성격 |
|---|---|---|
| 성능 비교 | 그 시점의 **최고 모델** | 움직인다. 더 좋은 게 나오면 그걸로 비교해도 된다 |
| **구조 변경 명세 (이 문서)** | 언제나 **현재 제출본** (`~/SUBMISSION/` 의 tar 2개) | 고정. 절대 옮기지 않는다 |

제출본에서부터 구조가 달라진 부분이 있으면 **누적 차이 전부**를 적는다.
변경이 2개 쌓였으면 "제출본 → 1 → 2" 를 전부 적고 "1 → 2" 만 적지 않는다.
그래야 실제로 제출할 때 이 문서 하나만 보고 반영할 수 있다.

---

## 0. 먼저 알아야 할 구조

### 0-1. 두 아카이브의 역할

| | 안에 든 것 | 바꾸려면 |
|---|---|---|
| `final-model-sblee_2026-09-08_15-09-29.tar.gz` | 코드·RF·vendor·파이썬 환경 (컨테이너 이미지) | **이미지 재빌드 필요** |
| `topaneu-26-task2-integrated-model.tar.gz` | 검출기·혈관·패치분류기 **가중치만** | tar 다시 말면 끝, 재빌드 불필요 |

컨테이너 안 `/opt/app/topaneu/models` → `/opt/ml/model/models` 심볼릭.
즉 **가중치만 바꾸는 변경은 모델 tar 만 다시 말면 되고, 코드·RF 변경은 이미지를 건드려야 한다.**

### 0-2. 이미지 안 경로

```
/opt/app/main.py                    GC 템플릿 원본 (건드리지 말 것)
/opt/app/inference.py               모달리티 진입점
/opt/app/topaneu_integrated.py      전체 오케스트레이션 (sblee 단계 → 패치필터)
/opt/app/fast_stages.py             크롭 EDT · crop_to_vessel(미배선)
/opt/app/run_patch_filter.py        패치필터 단일 케이스 실행기
/opt/app/src/{patch_filter,features,config}.py    jslee 패치필터 본체
/opt/app/requirements.txt
/opt/app/topaneu/code/sblee/nnunet/scripts/pipeline_case.py     ← 8단계 파이프라인
/opt/app/topaneu/code/sblee/nnunet/scripts/c5_location_v2.py    ← 분류기 (우리 번들과 바이트 동일)
/opt/app/topaneu/code/sblee/nnunet/analysis/final_rf_seed3.pkl  ← RF 70.5MB
/opt/app/topaneu/code/sblee/nnunet/analysis/e11_feat_hyb_ov.json
/opt/app/topaneu/code/sblee/nnunet/analysis/c10_feat_train.json
/opt/app/topaneu/vendor/Skeleton-Recall/                        ← PYTHONPATH 로 주입
/opt/app/topaneu/models -> /opt/ml/model/models                 ← 심볼릭
```

컨테이너 ENV (`docker inspect` 로 확인):
```
TOPANEU_BUNDLE=/opt/app/topaneu   TOPANEU_DET_FOLDS=0,1,2   TOPANEU_TIMING=1
nnUNet_raw=/opt/app/topaneu/nnunet/nnUNet_raw
nnUNet_preprocessed=/opt/app/topaneu/nnunet/nnUNet_preprocessed
nnUNet_results=/opt/app/models      ← 존재하지 않는 경로. 모든 추론이 명시 경로를 쓰므로 무해
```
**`TOPANEU_ROOT` 는 설정돼 있지 않다.** 이 값을 참조하는 코드를 켜려면 ENV 를 같이 넣어야 한다.

### 0-3. 이미지를 고치는 최소 절차 (전체 재빌드 금지)

기존 이미지를 베이스로 얇은 레이어 하나만 얹는다. 5GB 를 다시 만들 이유가 없다.

```bash
cd ~/work && docker load -i ~/SUBMISSION/final-model-sblee_2026-09-08_15-09-29.tar.gz
mkdir -p patch && cp <바꿀파일들> patch/
cat > Dockerfile.patch <<'EOF'
FROM final-model-sblee:latest
COPY --chown=user:user patch/final_rf_seed3.pkl /opt/app/topaneu/code/sblee/nnunet/analysis/
# ENV 추가가 필요하면 여기에 ENV 줄을 넣는다
EOF
docker build -f Dockerfile.patch -t final-model-sblee:v2 .
```

**저장 형식 주의 — 여기서 한 번 죽었다 (2026-09-06).**
containerd 스냅샷터 환경의 `docker save` 는 OCI 레이아웃(`index.json` + `oci-layout`)을 만들고
GC 는 `Could not find manifest.json in the container image file` 로 거부한다.
저장 후 **반드시** 확인한다:
```bash
docker save final-model-sblee:v2 | gzip > out.tar.gz
tar tzf out.tar.gz | head -1     # 반드시 manifest.json 이어야 한다
```
`index.json` 이 나오면 classic 형식으로 재포장해야 한다 (manifest.json 선두 · `<id>/layer.tar`
비압축 · `repositories` 포함). 받은 제출본 tar 은 이미 classic 형식이다.

**업로드 슬롯**: 이미지 tar.gz → Algorithm ▸ **Containers** / 모델 tar.gz → Algorithm ▸ **Models**.
바꿔 올리면 같은 `manifest.json` 오류가 난다 (2026-09-05 실측).

---

## 1. 분류기 피처 교체 — 개정판 RF (H2 ①)

**상태**: H2 판정 대기. 채택되면 아래대로.
**성격**: 이미지 변경. **런타임 비용 0.**

### 고칠 것

| 파일 | 어떻게 |
|---|---|
| `/opt/app/topaneu/code/sblee/nnunet/analysis/final_rf_seed3.pkl` | 개정판 피처로 학습한 pkl 로 **내용만 교체** (파일명 유지) |
| `/opt/app/topaneu/code/sblee/nnunet/analysis/e11_feat_hyb_ov.json` | `e11_feat_hyb_ov_NEW.json` 내용으로 교체 (추론엔 안 쓰이나 기록 일치용) |

**`pipeline_case.py` 는 고치지 않는다.** 파일명을 그대로 쓰면 코드 변경이 0 이다.
그 파일 208행이 `final_rf_seed3.pkl` 을 이름으로 열고, 이어서 `M["topk"]` 로 gC 를 켠다.

### 교체용 pkl 만드는 법

```bash
cd $R/code/sblee/nnunet/scripts
TOPANEU_TOPK=<2 또는 1> TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
TOPANEU_OUT_DILATE=0 CLF_SEED=3 \
python c5_location_v2.py eval --train-feat $A/e11_feat_hyb_ov_NEW.json --split val \
  --vessel-dir $P/vespp_val --bp-dir $BP/val_pred --aneurysm-pred-dir $P/aneu_val_b1ff \
  --model rf --use-pos --beta 0.5 --conf-tau 0.5 --conf-beta-hi 0.0 \
  --save-model $A/final_rf_NEWfeat_seed3.pkl --tag mkpkl
```
`--save-model` 이 dict 를 통째로 저장한다. 저장되는 `topk` 는 **실행 시 환경변수 값**이므로,
gC 를 끄기로 했으면 `TOPANEU_TOPK=1` 로 만들어야 한다. 만든 뒤 반드시 확인:
```python
import pickle; M=pickle.load(open("final_rf_NEWfeat_seed3.pkl","rb"))
print(M["topk"], M["train_feat"], M["clf_seed"], M["beta"], M["conf_tau"])
# 기대: topk.n 이 의도한 값 · train_feat 에 _NEW · clf_seed 3 · beta 0.5 · conf_tau 0.5
```

### 검증

컨테이너를 새로 말기 전에 로컬에서 같은 pkl 로 `pipeline_case.py` 를 한 케이스 돌려
`H1_patchfilter/pred/<해당팔>_test_s3/` 의 산출과 라벨이 일치하는지 본다.

---

## 2. gC 끄기 (H2 ②)

**상태**: H2 판정 대기.
**성격**: 이미지 변경이지만 **1번과 같은 pkl 하나로 동시 처리된다.**

`pipeline_case.py` 는 pkl 안의 `M["topk"]["n"]` 을 읽어 `TOPANEU_TOPK` 를 세팅한다.
따라서 **코드를 고칠 필요가 없다** — pkl 을 만들 때 `TOPANEU_TOPK=1` 로 만들면 gC 가 꺼진다.
1번과 2번이 함께 채택되면 pkl 파일 하나 교체로 둘 다 반영된다.

---

## 3. 곁가지 신뢰도 vesconf (H3)

**상태**: H3 실행 대기.
**성격**: 이미지 변경 + **코드 수정 필요.** 아래 함정 때문에 단순 ENV 추가로는 동작하지 않는다.

### 함정 — 컨테이너에서는 조용히 무효가 된다

`pipeline_case.py` 220행 근처가 `r["case"] = CASE` 이고 `CASE = "case"` 상수다(GC 입력 파일명이
임의라서 nnU-Net 규약용으로 고정한 것). 그런데 `c5_location_v2.py` 의 `_conf()` 는
`vesconf_{split}.json` 을 **케이스 id 로 조회**한다. 컨테이너에서는 키가 항상 `"case"` 라
표에 없고, `_conf` 는 기본값 1.0 을 돌려준다 → **conf 가 전부 1.0 = 아무 효과 없음.**
게다가 오류가 나지 않아 "효과 없더라" 로 오독하기 딱 좋다.

### 고칠 것

**(a) 기준 중앙값 표를 이미지에 넣는다**

| 파일 | 어떻게 |
|---|---|
| `/opt/app/topaneu/code/sblee/nnunet/analysis/vesconf_ref.json` | 새로 COPY (628B, 36클래스 중앙값) |

**(b) `pipeline_case.py` 에 즉석 conf 계산을 추가한다.**
7단계에서 이미 `ves`(V5 후처리 혈관맵)와 `spacing` 을 들고 있으므로 표 조회가 필요 없다.
`import c5_location_v2 as C5` **앞에** `TOPANEU_VESCONF` 를 세팅해야 한다 — c5 는 모듈 최상위에서
그 값을 읽는다. 그리고 `r["case"] = CASE` 뒤에 conf 를 직접 주입한다.

```python
# [8/8] 블록, os.environ.update(...) 바로 다음 · import c5_location_v2 앞
os.environ["TOPANEU_VESCONF"] = "block"      # 또는 "gate" — H3 가 고른 모드
import c5_location_v2 as C5, d9xx_lib as L
# ... (기존) rows, lesions = C5.extract_case_rows(...)
for r in rows:
    r["case"] = CASE
# ↓ 추가: 이 케이스의 곁가지 신뢰도를 혈관맵에서 바로 계산해 표에 꽂는다
import json as _json
from scipy import ndimage as _ndi
_ref = _json.load(open(os.path.join(B, "code/sblee/nnunet/analysis/vesconf_ref.json")))
_names = L.vessel_dense_names()
_boxes = _ndi.find_objects(ves.astype(np.int32), max_label=36)
_conf = {}
for _c in range(1, 37):
    _sl = _boxes[_c - 1]
    if _sl is None:
        continue
    _lab, _k = _ndi.label(ves[_sl] == _c, structure=np.ones((3, 3, 3), bool))
    if not _k:
        continue
    _nm = _names.get(_c)
    _med = _ref.get(_nm)
    if _nm and _med:
        _conf[_nm] = min(2.0, int(np.bincount(_lab.ravel())[1:].max()) / float(_med))
C5._VESCONF = {CASE: _conf}       # 표를 직접 채운다
C5._VESCONF_LOADED = True         # 파일 로드를 막는다
```

> `_names` 의 키 타입(int vs str)과 `vessel_dense_names()` 의 반환 형태를 **먼저 확인**할 것.
> `p_vesconf.py` 는 `vessel_mapping.json` 의 라벨명을 쓴다 — 이름이 어긋나면 조용히 conf 가 비고
> 또 무효가 된다. 반드시 한 케이스 돌려 `_conf` 가 30개 안팎 채워지는지 눈으로 본다.

**(c) 1번의 pkl 도 vesconf 를 켠 상태로 학습한 것이어야 한다.**
학습과 추론의 피처 차원이 다르면 그 자리에서 죽는다(block 은 148차원, off 는 112차원).

### 검증

컨테이너 로그에 `[vesconf] block · 1케이스 로드` 가 찍혀야 한다. 안 찍히면 무효다.

---

## 4. 검출기를 E9(ResEncL 10폴드)로 교체

**상태**: 성능은 우세하나(신 eval MCC test +0.0203 · val +0.0182) **런타임 때문에 보류.**
A5000 최악 253초 → T4 환산 380~430초, GC 한도 420초 경계선.

**성격**: 이미지 + 모델 tar **양쪽** 변경.

| 파일 | 어떻게 |
|---|---|
| 모델 tar `./models/detector/Dataset722_TopAneuPjh3cls417/` | 트레이너 디렉터리를 `nnUNetTrainer_250epochs__nnUNetResEncUNetLPlans722iso04__3d_fullres` 로 교체, fold_0~9 (408MB×10) + plans.json + dataset.json. 출처 `~/e9_bundle_experiment/models/detector/` |
| `pipeline_case.py` 157행 `DTR = "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres"` | ResEncL 트레이너명으로 |
| ENV `TOPANEU_DET_FOLDS=0,1,2` | 폴드 수 결정 (10폴드 전부는 시간 초과 확실) |
| `requirements.txt` | `seaborn` 이 이미 있는지 확인 — 없으면 vendor 로거 import 로 즉사 |

시간이 모자라면 폴드를 줄인 변형을 쓴다. `TOPANEU_DET_FOLDS` 로 조절되며 재빌드가 필요 없다
(ENV 만 바꾸는 얇은 레이어 하나).

---

## 5. 분기점 그래프 고속화 (런타임 확보)

**성격**: 이미지 변경. 성능 불변, **약 31초 절감.**

`fast_stages.py` 에 `crop_to_vessel` 이 구현돼 있으나 **호출하는 곳이 없다.**
`c4_branchpoint_graph.py` 는 우리 번들과 md5 가 같아 손대지 않은 상태다.

| 파일 | 어떻게 |
|---|---|
| `/opt/app/topaneu/code/sblee/nnunet/scripts/c4_branchpoint_graph.py` | Lee thinning 대상을 혈관 바운딩박스(여유 ≥2복셀)로 크롭하고 노드 좌표를 되돌린다 |

4번을 하려면 이 절감이 필요하다. 출력이 동일한지 실제 케이스로 대조한 뒤에만 넣는다.

---

## 6. gC 2등 조각 복셀 지분 복구 — `TOPANEU_TOPK_FRAC=0.35`  ★채택 (2026-09-16)

**상태**: **채택 · 반영 권고.** 원판(시드 0~4)·복제(시드 5~9) 둘 다 사전 규칙 통과.
**성격**: 이미지 변경(ENV 한 줄). **모델 가중치·아키텍처·런타임 전부 불변.**

### 왜 — 누락 복구다

`C60_metric_structure/summary.md` §22·§24 가 `TOPK_FRAC` 을 **채택 ②** 로 판정했는데
제출본에 들어가 있지 않다. `final_rf_seed3.pkl` 의 topk 딕셔너리에 `frac` 키가 없고
`pipeline_case.py` 도 그 ENV 를 설정하지 않아 기본값 0 으로 돈다. 되돌린 기록은 어디에도 없다
(PROJECT_RULES.md · NOTES.md · APPLY_CHANGES.md · chain_status.md 검색). 지금 2등 조각은
blob 281복셀 중 **3복셀(1%)** 만 받는다.

### 왜 안전한가 — 자유도가 구조적으로 없다

TP 성립은 1복셀이면 되므로 **PRECISION·RECALL·F1·MCC 는 정의상 불변**이고,
복셀 지분에 비례하는 DICE·VOLSIM 만 움직인다. 실측으로 확인했다 —
시드 10개 × test·val 전부에서 네 지표의 **절대편차 0.00e+00 · ΔTP 0.0 · ΔFP 0.0**.

### 측정 (기준 `b1on_pf` = 제출본 구성 · 시드 10개 평균)

| split | PRECISION·RECALL·F1·MCC | DICE | VOLSIM | HD95 |
|---|---|---|---|---|
| test | **±0.0000** | +0.0077 | +0.0104 | −0.02 |
| val | **±0.0000** | +0.0059 | +0.0111 | −0.15 |

원판/복제 분리: test DICE 원판 +0.0093 · 복제 +0.0061 (부호 네 칸 전부 유지).
지분 0.50·0.65 도 통과했으나 **val 정점인 0.35** 를 택했다 — test 로 고르지 않는다는
사전 규칙이다. 0.65 는 test 에서 계속 오르고 val 에서 꺾인다(C60 이 기록한 0.80 과 같은 모양).

### 고칠 것 — 파일 1개 교체 + ENV 1줄 (둘 다 필요하다)

> **2026-09-17 정정.** 처음에 "ENV 한 줄이면 된다" 고 적었으나 **틀렸다.**
> 컨테이너 안 `c5_location_v2.py` 는 md5 `a4c1831a214ca05e1b1fb628db9ebb3e` (59,021 B) 로
> **8/28 판**이고, 거기에는 `TOPANEU_TOPK_FRAC` 도 `_per_class_vox` 도 **존재하지 않는다**(grep 0 회).
> ENV 만 넣으면 읽는 코드가 없어 **무동작**이다. 실제 이미지를 열어 확인했다.

| # | 대상 | 어떻게 | 비고 |
|---|---|---|---|
| 1 | `/opt/app/topaneu/code/sblee/nnunet/scripts/c5_location_v2.py` | **세 군데만 수정** — 추가 9줄 · 삭제 2줄 | 신규 파일 아님 |
| 2 | 컨테이너 ENV | `ENV TOPANEU_TOPK_FRAC=0.35` 추가 | Dockerfile 한 줄 |

### 최소 패치 — 고칠 세 군데 (줄번호는 컨테이너 안 8/28 판 기준)

repo 판으로 통째 교체하면 추가 511줄·삭제 10줄이고, 제출물에 쓰지도 않을 실험 스위치
10여 개(SMOTE·HIER·NCM·SEEDENS·ICA_EXPERT·RARE_EMIT…)가 딸려온다. 지분 기능만 떼어 오면 9줄이다.

**(1) 696행 뒤에 상수 1줄**
```python
TOPK_VOX = int(os.environ.get("TOPANEU_TOPK_VOX", "3"))        # 기존 696행
TOPK_FRAC = float(os.environ.get("TOPANEU_TOPK_FRAC", "0"))    # ← 추가
```

**(2) 814행 `def _emit_topk(...)` 앞에 함수 1개**
```python
def _per_class_vox(n_idx):
    """등수 하나에 줄 복셀 수. TOPK_FRAC 이 켜지면 blob 크기에 비례한다."""
    if TOPK_FRAC > 0:
        return max(1, int(round(n_idx * TOPK_FRAC)))
    return TOPK_VOX
```

**(3) `_emit_topk` 안 846행·853행 교체**
```python
# 846행  before →  budget = min(TOPK_VOX * (TOPK_N - 1), len(idx) - 1)
# 846행  after  →  per = _per_class_vox(len(idx))
#                  budget = min(per * (TOPK_N - 1), len(idx) - 1)
# 853행  before →  take = min(TOPK_VOX, budget - used)
# 853행  after  →  take = min(per, budget - used)
```

그 위의 정렬 두 줄(`c = idx.mean(axis=0)` · `order = np.argsort(...)`)은 **그대로 둔다.**
repo 판은 이를 `_topk_order()` 로 뽑았지만 기본값에서 같은 식이라 가져올 이유가 없다.

### 최소 패치 검증 — 세 방향 전부 통과 (2026-09-17 09:29 KST)

| 검증 | 묻는 것 | 대상 | 결과 |
|---|---|---|---|
| ① 지분 끔 | 최소패치 == 현재 제출본인가 | val 41×2 · test 83×2 | **다른 파일 0** |
| ② 지분 0.35 | 최소패치 == 이득을 측정한 repo 판인가 | val 41×2 · test 83×2 | **다른 파일 0** |
| ③ 켜짐 대 꺼짐 | 지분이 실제로 먹는가 | test 시드 3 | **12개 파일이 바뀜** |

③ 이 0 이면 "ENV 는 넣었는데 아무 일도 안 일어나는" 상태다 — 처음 낸 ENV-만 방법이 정확히 그랬다.

**모델 tar(`topaneu-26-task2-integrated-model.tar.gz`) 는 건드리지 않는다.** 가중치 변경이 없다.
`pipeline_case.py` · `final_rf_seed3.pkl` 도 그대로 둔다 —
`os.environ.update(...)` 가 설정하는 키 목록에 `TOPANEU_TOPK_FRAC` 이 없어 ENV 가 덮어써지지 않는다.

### 파일 교체의 위험 — 반드시 먼저 검증할 것

repo 판은 8/28 판보다 24KB 크다(추가 511줄 · 삭제 10줄). 늘어난 것은 전부 **ENV 로 잠긴 실험 스위치**다
(SMOTE · ICA_EXPERT · HIER · SEEDENS · TWOTIER · NNBOOST · NCM · OUT_GROW · TOPK_SHAPE · RARE_EMIT …).
기본값은 전부 꺼짐(`0` / `""`), `TOPK_SHAPE` 기본값 `center` 는 8/28 판의 고정 동작과 같고,
RF 하이퍼(`RF_TREES=500` · `RF_MIN_LEAF=1` · `RF_MAX_FEATURES=sqrt`)도 기존 하드코딩과 같은 값이다.
gC 배출부는 `budget = min(TOPK_VOX*(TOPK_N-1), …)` 가 `_per_class_vox()` 경유로 바뀌었는데,
`TOPK_FRAC=0` 이면 `_per_class_vox()` 가 `TOPK_VOX` 를 그대로 돌려주므로 식이 동일하다.

**추론으로 확인했다 — 2026-09-17 09:00 KST · 전부 동일.**

| 대상 | 파일 수 | 내용이 다른 파일 |
|---|---|---|
| val · 시드 3 | 41 | **0** |
| val · 시드 0 | 41 | **0** |
| test · 시드 3 | 83 | **0** |
| test · 시드 0 | 83 | **0** |

지분 기능을 끈 상태(ENV 미설정)로 두 판을 같은 입력·같은 설정으로 돌려 md5 를 전수 대조했다.
**248 개 출력 전부 바이트 동일.** 교체는 무해하다.

함수 단위 대조도 같은 결론이다. 추론은 pkl 의 **학습 완료 모델**을 쓰므로 `fit_model` (RF
하이퍼파라미터 변경)은 실행되지 않는다. 실제 호출되는 것만 보면 —
`extract_case_rows` · `load_bp` · `maha_proba` 는 바이트 동일,
`predict_one` · `predict_ranked` 는 `if TWOTIER:` 가 앞에 붙었을 뿐 기본값 0 이라 원래 분기로 낙하,
`_proba` 는 실험 가드 5 개(NNBOOST · NCM · SEEDENS · HIER · ICA_EXPERT) 추가인데 전부 기본 0,
`_emit_topk` 만 실제 식이 바뀌었으나 `FRAC=0` · `SHAPE=center` 에서 옛 수식과 동일해진다.
라이브러리 버전도 컨테이너와 실험 환경이 같다(sklearn 1.7.2 · numpy 2.2.6 · scipy 1.15.3 · nibabel 5.4.2).

**덤으로 확인된 것**: 우리 실험 기준선 `b1on_pf` 가 제출본 분류기와 **같은 동작**임이 증명됐다.
그동안의 "제출본과 같은 구성" 이라는 전제가 실측으로 뒷받침된다.

```bash
# 동일성 검증 (FRAC 없이) — 같은 케이스, 두 c5 판
for V in container repo; do
  TOPANEU_TOPK=2 TOPANEU_TOPK_VOX=3 TOPANEU_TOPK_ICA=1 TOPANEU_TOPK_MARGIN=0.7 \
  TOPANEU_TOPK_P2=0.0 TOPANEU_TOPK_OR=0 TOPANEU_TOPK_TAU=0.0 TOPANEU_TOPK_MAXN=0 \
  TOPANEU_OUT_DILATE=0 CLF_SEED=3 python <해당판>/c5_location_v2.py eval ... \
    --save-pred-dir /tmp/chk_$V
done
diff -r --brief /tmp/chk_container /tmp/chk_repo    # 차이 없어야 한다
```

**또 하나 — 우리 실험 기준선도 repo 판으로 돌았다.** `b1on_pf` 를 비롯한 모든 측정이
repo `c5_location_v2.py` 를 썼으므로, 위 동일성이 깨지면 "제출본 구성과 같다" 던 전제가
FRAC 과 무관하게 흔들린다. 그 경우 별도로 기록한다.

### 반영 후 기대 (제출본 test 83 기준 · 존재기반은 실측 불변)

| | Precision | Recall | MCC | Dice | VolSim | HD95 |
|---|---|---|---|---|---|---|
| 제출본 | 0.5613 | 0.5423 | 0.5975 | 0.2517 | 0.2683 | 177.46 |
| + FRAC 0.35 | **0.5613** | **0.5423** | **0.5975** | **0.2594** | **0.2787** | **177.44** |

TP/FN/FP/TN = 58 / 30 / 50 / 4178 → **한 개도 안 바뀐다.**
앞 네 칸은 추정이 아니라 **정의상 불변**이고, Dice·VolSim·HD95 만 우리 측정 Δ 를 더한 추정이다.

### 확인

```bash
# 컨테이너 안에서 ENV 가 읽히는지
docker run --rm final-model-sblee:v2 python -c \
  "import os;print(os.environ.get('TOPANEU_TOPK_FRAC'))"     # 0.35
# sanity 1건 돌린 뒤 기존 출력과 비교: 라벨 집합은 같고 2등 클래스 복셀 수만 늘어야 한다
```

---

## 7. 출력 부피 보정 — `TOPANEU_OUT_GROW=1.32`  ★채택 (2026-09-17)

6번(FRAC)과 **같은 계열**이다: 결정 경계를 건드리지 않고 지표 구조만 만진다.
둘은 **한 파일에 같이 들어간다** — 아래 최소 패치는 6+7 통합본이다.

### 왜

적중 병변의 (예측 부피 / GT 부피) 중앙값이 **val 0.756 · test 0.816** 이다. 병변의 대다수를
GT 보다 작게 낸다. 사전 규칙 "val 중앙 ≤ 0.85 면 진행, 배수는 val 중앙의 역수" 에 따라
**1.32 하나만** 돌렸다(스윕 없음 = 선택 자유도 0).

### 왜 안전한가 — 존재 불변

껍질 복셀(4mm 이내)을 **거리순**으로 필요한 만큼만 붙이고 라벨은 **최근접 상속**이다.
새 클래스가 생기지 않으므로 케이스별 클래스 존재가 안 바뀐다 →
PRECISION·RECALL·F1·MCC 는 정의상 불변이고 DICE·VOLSIM·HD95 만 움직인다.
**가정이 아니라 실측이다**: 10시드 · 양쪽 split 에서 네 지표 최대 절대편차 **0.00e+00**, ΔTP·ΔFP 모두 0.

> 밝기순(`OUT_GROW_BY=int`) 변형 G1 은 **기각**했다. 4mm 껍질에서 멀리 떨어진 밝은 복셀을 집어
> 최근접 상속 라벨이 달라지면서 **존재 불변이 깨졌다**(test ΔFP +1 · val ΔTP −1) 그리고
> DICE 가 오히려 −0.0177. 거리순은 인접 복셀만 집어 이 일이 없다. 컨테이너에는 **거리순만** 넣는다.

### 측정 (기준 `b1frac035_pf` = 제출본 구성 + FRAC 0.35 · 시드 **10개**)

| split | PRECISION | RECALL | F1 | MCC | DICE | VOLSIM | HD95 |
|---|---|---|---|---|---|---|---|
| test | +0.0000 | +0.0000 | +0.0000 | +0.0000 | **+0.0017** | **+0.0125** | −0.004 |
| val | +0.0000 | +0.0000 | +0.0000 | +0.0000 | **+0.0029** | **+0.0227** | −0.031 |

시드 일관성: DICE·VOLSIM 이 **10/10 시드 · 양쪽 split 전부 개선**(악화 0). 최악 시드도 플러스다.
HD95 도 10/10 개선. 원판(0~4)·복제(5~9) 둘 다 사전 규칙 충족.

### 고칠 것 — FRAC 과 **같은 파일 하나** + ENV 1줄 추가

> **2026-09-21 정정 — 위 방법만으로는 컨테이너에서 GROW 가 무동작이다 (실측).**
> 컨테이너는 `c5 eval` 을 거치지 않고 `pipeline_case.py` 가 출력 라벨맵을 직접 만든다(분류 루프 → OUT_DILATE → 저장).
> 그래서 c5 안의 GROW 블록은 호출되지 않는다. 위 "248개 동일" 검증은 c5 eval 로 돌려 이걸 못 잡았다.
> 이 저장소 이미지로 CT·MR 1건씩 돌렸을 때 총 복셀이 1825→1825 · 1403→1403 이었다(FRAC 은 동작).
> **`pipeline_case.py` 의 `_emit_topk` 루프와 `if C5.OUT_DILATE > 0` 사이에 같은 GROW 블록(`C5.OUT_GROW` 참조)을 넣어야 한다.**
> 넣은 뒤: 끔 == 제출본(복셀 단위 일치) · 켬 총 복셀 1825→2409 · 1403→1852(정확히 ×1.32) · 라벨 종류 불변.
> 반영본: `~/TopAneu26---Task-2-Pipeline/app/topaneu/code/sblee/nnunet/scripts/pipeline_case.py`
> (같은 파일에 min_vox 12 용 `TOPANEU_MIN_VOX` ENV 도 들어 있다 — `--min-vox` 기본값을 ENV 에서 읽음.)

`OUT_GROW` 도 컨테이너 안 8/28 판 c5 에 **없다**(`grep -c OUT_GROW` = 0). ENV 만으로는 안 된다.

```
/opt/app/topaneu/code/sblee/nnunet/scripts/c5_location_v2.py   ← 6+7 통합 최소패치본으로 교체
ENV TOPANEU_TOPK_FRAC=0.35
ENV TOPANEU_OUT_GROW=1.32
```

### 통합 최소 패치 — 제출본 대비 **추가 26줄 / 삭제 2줄**

6번(FRAC)의 세 군데에 더해, OUT_DILATE 블록 **바로 앞뒤** 두 군데만 늘어난다.

```python
# (가) OUT_DILATE 선언 바로 다음 — 705행 뒤
OUT_GROW    = float(os.environ.get("TOPANEU_OUT_GROW", "0"))      # S5 목표 부피 배수 (0=무동작)

# (나) `n_les += len(rows)` 와 `if OUT_DILATE > 0` 사이 — 1054행 뒤
        if OUT_GROW > 1.0 and out.any():
            fg = out > 0
            n0 = int(fg.sum())
            need = int(round(n0 * OUT_GROW)) - n0
            if need > 0:
                dist, nn = ndimage.distance_transform_edt(~fg, return_indices=True)
                shell = np.flatnonzero((dist.ravel() > 0) & (dist.ravel() <= 4.0))
                if shell.size:
                    order = shell[np.argsort(dist.ravel()[shell], kind="stable")[:need]]
                    ii = np.unravel_index(order, out.shape)
                    out[ii] = out[nn[0][ii], nn[1][ii], nn[2][ii]]
```

`ndimage` 는 8/28 판에 **이미 import 돼 있다**(31행) — 새 의존성 없음.
OUT_DILATE 블록이 같은 자리에서 같은 구조(최근접 상속)로 이미 돌고 있어, 패턴이 검증돼 있다.

### 통합 최소 패치 검증 — 세 방향 (`scratchpad/equiv3.sh`)

세 방향 전부 통과 (2026-09-17 16:43 KST · val 41×2 + test 83×2 = **248개 출력**).

| 방향 | 무엇을 보장하나 | 대상 | 결과 |
|---|---|---|---|
| ① 둘 다 끔 == 8/28 컨테이너판 | ENV 안 주면 **제출본과 완전 동일** | 248개 | **다른 파일 0** |
| ② 둘 다 켬 == repo 판 | 우리가 측정한 이득이 그대로 재현 | 248개 | **다른 파일 0** |
| ③ 끔 vs 켬 이 실제로 다름 | 조용히 무동작으로 도는 것 아님 | val_s3 / test_s3 | **28개 / 62개 바뀜** |

③ 은 FRAC 단독일 때 test_s3 에서 12개가 바뀌었던 것과 비교된다 — OUT_GROW 는 예측이 있는
케이스를 전부 건드리므로 62개로 늘었다. 0 이 나왔다면 ENV 를 넣고도 아무 일이 없는 상태였을 것이다.

### 확인

```bash
python -c "import os;print(os.environ.get('TOPANEU_OUT_GROW'))"    # 1.32
# sanity 1건: 라벨 **집합**은 같고(존재 불변) 각 라벨의 복셀 수만 약 1.32배가 되어야 한다
```

---

## 변경 이력

| 날짜 | 변경 | 근거 | 반영 여부 |
|---|---|---|---|
| 2026-09-09 | 패치 CNN 환각필터 추가 | 통합 시점에 이미 포함 | **반영됨** (현 제출본) |
| — | 개정판 피처 RF (1) | H2 ① | 판정 대기 |
| — | gC OFF (2) | H2 ② | 판정 대기 |
| — | vesconf (3) | H3 | 실행 대기 |
| — | E9 검출기 (4) | H1 · 런타임 보류 | 보류 |
| — | 분기점 고속화 (5) | 미측정 | 미착수 |
| 2026-09-16 | **gC 2등 조각 지분 FRAC 0.35 (6)** | FRAC 원판·복제 통과 · `V1_vessel_axis/RESULTS_FRAC{,_REP}.md` | **채택 · 반영 대기** |
| 2026-09-16 | DuoRF(구 K1) 두 표 확률평균 | test 미개선(seed3 MCC 0.5983→0.5707) | 미채택 |
| 2026-09-16 | 개정판 학습표 | TBL10 K0 오른4·내린6 p=0.828 | 미채택 |
| 2026-09-16 | E9 검출기 (4) 재검토 | DET9 안전① 위반 ΔTP+12/ΔFP+27 · val 1/7 | **기각**(런타임 아님) |
| 2026-09-17 | **출력 부피 보정 OUT_GROW 1.32 (7)** | 10시드 전수 개선(DICE·VOLSIM 40/40 조합) · 존재불변 실측 0 · `RESULTS_GROW{,_REP}.md` | **채택 · 반영 대기** |
| 2026-09-17 | 밝기순 성장 G1 (`OUT_GROW_BY=int`) | 존재불변 깨짐(test ΔFP+1) · DICE −0.0177 | **기각** |
| 2026-09-17 | vesconf (3) | H3 block·gate **둘 다 0/7** · test MCC −0.080/−0.069 | **기각** |
| 2026-09-17 | TabICLv2 분류기 교체 | LOCO OOF 표적 top-2 동수 · 전체 top-1 −3.6%p | **기각** |
| 2026-09-17 | ICA 전용분류기 (`ICA_EXPERT`) | LOCO OOF 규칙 3개 전부 미달 | **기각** |
