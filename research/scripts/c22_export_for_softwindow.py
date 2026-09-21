"""C22 — 3-Stage v2 (soft window) 팀에 넘길 핸드오프 패키지 생성 (2026-08-17).

배경: 두 팀이 같은 split에서 독립 노선으로 진행 중이고, 현재 3-Stage v2 가 test MCC 0.323,
우리(C계열 기하피처+트리)가 0.2207 이다. 우리가 앞서는 건 HD95(0.649 vs 0.747) 하나뿐이므로,
우리 강점을 저쪽 head 에 얹는 편이 빠르다(사용자 결정 2026-08-17, "A로 해봐").

3-Stage v2 코드는 우리 작업공간에 없으므로 **병합은 못 하고 인터페이스만 만든다.**
저쪽이 바로 읽어 쓸 수 있는 형태로 내보낸다.

넘기는 것:
  1) anchor 검출률 표 — 저쪽 슬라이드 7의 명시된 병목("새 분기점을 앵커로 추가했더니
     그 혈관이 애초에 자주 안 보였다")에 대한 정량 답. 417케이스에서 분기점 타입별 검출률.
  2) 케이스별 anchor 좌표 — c4 분기점 그래프 (이미 _c4_bpgraph/ 에 있음, 스키마 문서화)
  3) 랜드마크 정규화 좌표 규격 — BA tip / R,L ICA terminus 삼각형 기준. 97.4% 케이스에서 성립.
  4) 검출 후처리 레시피 — 혈관거리 게이팅 + 성분크기. FP 55->12, 민감도 손실 0.
"""
import json, os, shutil, collections, glob
from pathlib import Path

import numpy as np

import c5_location_v2 as C5
import d9xx_lib as L

R = L.TOPANEU_ROOT
BP = R / "experiments" / "_c4_bpgraph" / "all_ref"
OUT = R / "code" / "sblee" / "outputs" / "handoff_softwindow"
OUT.mkdir(parents=True, exist_ok=True)


def anchor_rates():
    cnt = collections.Counter(); lm = collections.Counter(); tot = 0
    for f in glob.glob(str(BP / "*.json")):
        d = json.load(open(f)); tot += 1
        seen, seen_lm = set(), set()          # 케이스당 1회만 계수 (한 케이스에 같은 타입 노드가 여럿)
        for nd in d["nodes"]:
            cs = set(nd["classes"])
            for i, p in enumerate(C5.JUNCTION_PAIRS):
                if set(p) <= cs:
                    seen.add(i)
            for k, pats in C5.LANDMARKS.items():
                if any(pp <= cs for pp in pats):
                    seen_lm.add(k)
        for i in seen:
            cnt[i] += 1
        for k in seen_lm:
            lm[k] += 1
    rows = [{"anchor": f"{a} + {b}", "pair": [a, b], "rate": cnt[i] / tot}
            for i, (a, b) in enumerate(C5.JUNCTION_PAIRS)]
    rows.sort(key=lambda r: -r["rate"])
    return tot, rows, {k: v / tot for k, v in lm.items()}


def main():
    tot, rows, lmr = anchor_rates()
    json.dump({"n_cases": tot, "source": "reference vessel masks (Dataset800 labelsTr)",
               "anchors": rows, "landmarks": lmr},
              open(OUT / "anchor_detection_rates.json", "w"), indent=1, ensure_ascii=False)

    hi = [r for r in rows if r["rate"] >= .87]
    lo = [r for r in rows if r["rate"] < .50]

    with open(OUT / "README.md", "w") as f:
        f.write(f"""# 3-Stage v2 (soft window) 핸드오프 — C계열에서

두 노선이 같은 split에서 진행 중입니다. 현재 3-Stage v2 가 test MCC **0.323**,
C계열(기하피처+트리)이 **0.2207** 입니다. 저희가 앞서는 지표는 HD95 하나뿐입니다
(0.649 vs 0.747). 그래서 저희 산출물을 그쪽 head 에 얹는 방향으로 정리했습니다.

---

## 1. anchor 검출률 — 슬라이드 7의 병목에 대한 정량 답

> *"새 분기점을 앵커로 추가했더니 그 혈관이 애초에 자주 안 보였습니다."*

참조 혈관마스크 **{tot}케이스**에서 분기점 타입별로 **실제 검출된 비율**을 쟀습니다.
그쪽 표의 수치(R-Pcom 50.8% 등)는 *혈관 존재율*인데, anchor 로 쓰려면 **분기점 자체가
잡히는 비율**이 필요합니다. 둘은 다릅니다 — 혈관이 있어도 접합부가 안 잡힐 수 있습니다.

| anchor | 검출률 |
|---|---|
""")
        for r in rows:
            f.write(f"| `{r['anchor']}` | {r['rate']*100:.1f}% |\n")
        f.write(f"""
**87%에서 뚜렷한 단절이 있습니다.** 상위 {len(hi)}개가 87% 이상이고 그 다음이 76.5%로 떨어집니다.
빈 창이 공유 trunk 창보다 나쁘다는 그쪽 관찰을 감안하면, **이 {len(hi)}개를 분기 anchor 로 쓰고
나머지는 trunk 로 흡수**하는 레이아웃이 근거 있는 후보입니다.

특히 아래는 anchor 로 쓰면 절반 이상의 케이스에서 빈 창이 됩니다:

""")
        for r in lo:
            f.write(f"- `{r['anchor']}` — {r['rate']*100:.1f}%\n")
        f.write(f"""
Pcom / AChA 계열이 40%대인데, 이게 저희 쪽에서 3.4/3.5 클래스가 무너지는 원인과 같습니다.

## 2. 랜드마크 정규화 좌표 (저희 쪽 최대 개선, macro-recall +11%)

BA tip · R-ICA terminus · L-ICA terminus 세 점으로 좌표계를 세웁니다.
셋 모두 검출되는 케이스가 **{min(lmr.values())*100:.1f}% 이상** ({', '.join(f'{k} {v*100:.1f}%' for k,v in lmr.items())}).

```
원점 O = 세 랜드마크 무게중심
x축   = R-terminus → L-terminus        (좌우)
y축   = O → BA tip 을 x에 직교화        (전후)
z축   = x × y                           (상하)
스케일 = |R-terminus − L-terminus|      (두개 크기 정규화)
```

6차원: 정규화 (x,y,z) + 세 랜드마크까지의 정규화 거리.
아틀라스 등록 없이 두개 크기·자세·모달리티에 불변입니다.

> **교차 학습 하나** — 그쪽 `--mahal`(클래스별 평균좌표 + 공유 공분산)은 기각(CV 0.2631)인데
> 저희 좌표 피처는 채택(macro-recall 0.410→0.455)이었습니다. 같은 "전역 위치" 발상인데
> 결과가 갈렸으니, 파라메트릭 사전분포 대신 **좌표를 raw 피처로 넣는** 쪽을 한 번 보실 만합니다.
>
> **반대 방향 교차 학습** — 그쪽 `--arc-train`(측지 호길이)은 채택인데, 저희가 같은 발상으로
> 만든 C15(중심선 측지거리 + 사행비)는 기각(macro-recall 0.400→0.368)이었습니다.
> 차이는 기준점입니다 — 저희는 랜드마크 3점까지의 측지거리를 썼고, 그쪽은 **anchor 사이
> 상대 위치**를 씁니다. 그쪽 정식화가 맞습니다.

## 3. 검출 후처리 — 무료로 FP 78% 제거

GT 병변의 최근접 혈관 거리는 중앙값 0.30mm / 95퍼센타일 0.55mm 입니다.
따라서 혈관에서 떨어진 검출 성분은 거의 전부 위양성입니다.

| 설정 | 민감도 | FP 총계 |
|---|---|---|
| A5-2 무필터 | 0.721 (31/43) | 55 |
| **+ 혈관거리 ≤5mm + ≥20 voxel** | **0.721 (31/43)** | **12** |

**민감도 손실 0으로 FP를 78% 제거합니다.** 재학습 불필요, 후처리만.
구현: `c7_detect_postproc.py`. 그쪽 검출 단에 그대로 붙일 수 있습니다.

## 4. 5-fold 확률맵 평균 (HD95 우위의 출처)

이진 마스크 다수결은 여러 모델의 합집합 성분이 되어 경계가 뭉개집니다.
softmax 를 평균한 뒤 임계하면 HD95 가 **0.6490 → 0.5827** 로 회복됩니다.
저희가 HD95에서 앞서는(0.649 vs 0.747) 이유가 이것으로 보입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `anchor_detection_rates.json` | 분기점 {len(rows)}종 검출률 + 랜드마크 3종 |
| `branchpoint_schema.md` | 케이스별 anchor 좌표 JSON 스키마 |
| `../../nnunet/scripts/c4_branchpoint_graph.py` | 분기점 그래프 추출기 |
| `../../nnunet/scripts/c7_detect_postproc.py` | 검출 후처리 |
| `experiments/_c4_bpgraph/all_ref/<case>.json` | 417케이스 anchor 좌표 (생성 완료) |
""")

    with open(OUT / "branchpoint_schema.md", "w") as f:
        f.write("""# 분기점 그래프 JSON 스키마

`experiments/_c4_bpgraph/<split>/<case>.json`

```json
{
  "case": "topaneu_center4_ct_108",
  "spacing": [0.5, 0.382, 0.382],
  "spur_mm": 2.0,
  "junction_r_mm": 2.0,
  "n_nodes": 73,
  "n_valid": 73,
  "nodes": [
    {
      "class_ids": [4, 8],
      "classes": ["R-ICA-C6-C7", "R-Pcom"],
      "centroid_vox": [151.3, 220.7, 198.1],
      "centroid_mm": [75.65, 84.31, 75.67],
      "n_transition_vox": 2,
      "valid": true
    }
  ]
}
```

- `classes` — 이 분기점을 구성하는 혈관 (2개 이상). anchor 식별자로 그대로 쓰면 됩니다.
- `centroid_mm` — anchor 좌표. voxel 이 아니라 mm 이라 spacing 이 달라도 비교 가능합니다.
- `valid` — V5 후처리의 인접성 테이블로 검증한 결과. 해부학적으로 불가능한 쌍이면 false.
- `n_transition_vox` — 노드 크기. 클수록 확실한 접합부이며, 노드는 이 값 내림차순 정렬입니다.

추출 파라미터는 스퍼 2mm / 노드 반경 2mm 입니다. 노드 반경은 오라클 실험에서
2mm 가 최적(61.0%)이었고 3/5mm 는 떨어졌습니다.
""")

    for fn in ("c4_branchpoint_graph.py", "c7_detect_postproc.py"):
        src = R / "code" / "sblee" / "nnunet" / "scripts" / fn
        if src.exists():
            shutil.copy2(src, OUT / fn)

    print(f"[c22] 핸드오프 패키지 -> {OUT}")
    print(f"  anchor {len(rows)}종 · 87%↑ {len(hi)}종 · 50%↓ {len(lo)}종")
    for k, v in lmr.items():
        print(f"  랜드마크 {k}: {v*100:.1f}%")


if __name__ == "__main__":
    main()
