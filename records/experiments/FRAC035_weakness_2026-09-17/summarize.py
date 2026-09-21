"""Summarize cached-mask diagnostics; no model fitting or prediction changes."""
import collections
import json
from pathlib import Path

from audit import OUT, R, NAMES, code, region

data = json.loads((OUT/'audit.json').read_text())
L = data['lesions']
D = [r for r in L if r['detected']]
features = json.loads((R/'code/sblee/nnunet/analysis/e11_feat_hyb_ov.json').read_text())
train = collections.Counter(code(r['gt_loc']) for r in features)
lines = []


def emit(s=''):
    lines.append(s)


def table(headers, rows):
    emit('| '+' | '.join(headers)+' |')
    emit('| '+' | '.join('---' for _ in headers)+' |')
    for row in rows: emit('| '+' | '.join(map(str,row))+' |')
    emit()


def agg(rows, key, detected_only=True):
    out = []
    vals = sorted({key(r) for r in rows})
    for k in vals:
        allr = [r for r in rows if key(r)==k]
        rr = [r for r in allr if r['detected']] if detected_only else allr
        n = len(rr)
        if not n:
            out.append([k,len(allr),0,'—','—',0,0]); continue
        main = sum(r['n_dominant_correct'] for r in rr)
        anyc = sum(r['n_any_correct'] for r in rr)
        s3 = sum(r['predictions'][3]['any_correct'] for r in rr)
        out.append([k,len(allr),n,f'{main/(10*n):.1%}',f'{anyc/(10*n):.1%}',10*n-anyc,f'{s3}/{n}'])
    return out


emit('# FRAC 0.35 분류 취약점 분석 — 2026-09-17')
emit()
emit('대상: `b1frac035_pf` · test 83 + val 41 케이스 · RF 시드 0–9. 제출 RF 시드 3은 별도 표시한다.')
emit('저장된 최종 마스크 1,240개를 직접 읽고, 기준선/필터 전 시드 3 마스크도 대조했다. 모델 재학습·변경 없음.')
emit('전체 52클래스의 TP/FP/FN를 모든 split×seed에서 기존 동결 공식 점수와 대조해 일치함을 확인했다.')
emit()
emit('병변 단위 진단은 GT 클래스별 6연결 성분(3복셀 이상)을 쓰며, 주변 3회 복셀 팽창 영역에서 예측을 찾는다. 공식 지표의 케이스×클래스 존재 판정과 구분해야 한다. 이 매칭 허용치는 mm 단위가 아니다.')
emit('다수 라벨 정확도는 병변 주변 예측 중 복셀이 가장 많은 라벨의 정확도다. 정답 포함률은 출력된 어느 라벨이든 GT와 같으면 정답으로 친다. gC가 복수 라벨을 출력하므로 두 수치가 다르며, 다수 라벨을 RF 원시 top-1과 동일시하지 않는다.')
emit('10시드는 같은 병변을 반복 판정한 것으로 독립 표본 10배가 아니다. 크기는 최대직경이 아니라 GT 부피의 등가구 직경이다.')
emit()
emit(f'GT 병변 {len(L)}개 중 검출 단계 이후 매칭 {len(D)}개, 미검출 {len(L)-len(D)}개.')
emit()
headers=['영역','전체 GT 병변','검출 병변','다수 라벨 정확도(10시드)','정답 포함률(10시드)','정답 미포함 판정 수','시드3 정답 포함']
emit('## 영역별')
table(headers,agg(L,lambda r:r['region']))
emit('## 세부구간별')
table(headers,agg(L,lambda r:r['subgroup']))
emit('## 출력이 남아 있는 병변만의 분류 정확도 — 미할당을 분모에서 제외')
pure=[]
for reg in ['후순환','ICA','MCA','ACA/Acom']:
    rr=[r for r in D if r['region']==reg]
    pp=[p for r in rr for p in r['predictions'] if p['labels']]
    s3=[r['predictions'][3] for r in rr if r['predictions'][3]['labels']]
    pure.append([reg,len(rr),len(pp),f"{sum(p['dominant_correct'] for p in pp)/len(pp):.1%}",
                 f"{sum(p['any_correct'] for p in pp)/len(pp):.1%}",
                 f"{sum(p['any_correct'] for p in s3)}/{len(s3)}"])
table(['영역','검출 병변','라벨이 남은 10시드 판정 수','다수 라벨 정확도','정답 포함률','시드3 정답 포함'],pure)
emit('## split별')
table(headers,agg(L,lambda r:r['split']))
emit('## 모달리티별 — 혈관 영역/크기 등의 구성 차이를 보정하지 않은 기술통계')
table(headers,agg(L,lambda r:r['modality']))
emit('## 크기별 — 등가구 직경, 검출된 병변 조건부')
table(headers,agg(L,lambda r: '<3mm' if r['equivalent_diameter_mm']<3 else '3–5mm' if r['equivalent_diameter_mm']<5 else '5–10mm' if r['equivalent_diameter_mm']<10 else '≥10mm'))

emit('## 시드3 단계별 분해')
stage = collections.Counter()
for r in L:
    p=r['predictions'][3]; before=r['before_patchfilter_seed3']
    if not r['detected']: k='검출 단계 미매칭'
    elif not p['labels']:
        k='필터 전 라벨 있음 → 최종 미할당' if before['labels'] else '필터 전부터 미할당'
    elif p['any_correct']: k='정답 라벨 포함'
    else: k='출력은 있으나 정답 클래스 없음'
    stage[k]+=1
table(['단계','병변 수'],stage.items())
emit('미할당 병변:')
table(['split','case','GT','필터 전 출력','정답이 필터 전에 있었나'],[
    [r['split'],r['case'],r['name'],', '.join(NAMES[c] for c in r['before_patchfilter_seed3']['labels']),
     r['cls'] in r['before_patchfilter_seed3']['labels']]
    for r in D if not r['predictions'][3]['labels']])

emit('## 시드 안정성')
for key in ['n_dominant_correct','n_any_correct']:
    always_bad=[r for r in D if r[key]==0]
    mixed=[r for r in D if 0<r[key]<10]
    always_good=[r for r in D if r[key]==10]
    errors=sum(10-r[key] for r in D)
    emit(f'- {key}: 전시드 오답 {len(always_bad)}병변, 혼재 {len(mixed)}병변, 전시드 정답 {len(always_good)}병변. 전시드 오답 병변에서 {10*len(always_bad)}/{errors} 오류 판정 발생.')
emit()

emit('## 정답 미포함 오류 유형 — 검출 병변, 10시드')
types=collections.Counter(p['kind'] for r in D for p in r['predictions'] if not p['any_correct'])
table(['진단용 유형','오류 판정 수'],types.most_common())
emit('인접성은 기존 intweak.py의 수동 분절 그래프를 사용한다. 그래프 정의에 영향을 받으므로 실제 클래스 쌍도 함께 제시한다.')
emit()
emit('## 혼동 쌍 — 정답이 출력에 없을 때만, 좌우를 합쳐 집계')
conf=collections.Counter((code(r['name']),code(NAMES[p['dominant']]) if p['dominant'] else '미할당')
                        for r in D for p in r['predictions'] if not p['any_correct'])
table(['GT 분절','예측 다수 분절','10시드 오류 판정 수'],[(a,b,n) for (a,b),n in conf.most_common()])

emit('## 클래스별 — 좌우 합산, 검출된 병변만')
rows=[]
for c in sorted({r['code'] for r in D}):
    rr=[r for r in D if r['code']==c]; n=len(rr)
    rows.append([c,train[c],n,f"{sum(r['n_dominant_correct'] for r in rr)/(10*n):.1%}",
                 f"{sum(r['n_any_correct'] for r in rr)/(10*n):.1%}",sum(r['n_any_correct']==0 for r in rr)])
table(['분절','현 구 학습표 병변 수','평가 검출 병변','다수 라벨 정확도','정답 포함률','전10시드 정답 미포함 병변'],rows)

emit('## 전10시드 정답이 전혀 포함되지 않는 검출 병변')
rows=[]
for r in sorted(D,key=lambda r:(r['region'],r['name'],r['case'])):
    if r['n_any_correct']!=0: continue
    preds=collections.Counter(NAMES.get(p['dominant'],'미할당') for p in r['predictions'])
    rows.append([r['split'],r['case'],r['name'],f"{r['equivalent_diameter_mm']:.1f}",
                 ', '.join(f'{k} ×{n}' for k,n in preds.most_common()),
                 ', '.join(NAMES[c] for c in r['predictions'][3]['labels']) or '미할당'])
table(['split','case','GT','등가구직경 mm','10시드 다수 라벨','시드3 모든 출력라벨'],rows)

emit('## 다수 라벨은 틀렸지만 정답이 보조 라벨에 포함되는 사례 — 시드3')
table(['split','case','GT','다수 라벨'],[[r['split'],r['case'],r['name'],NAMES[r['predictions'][3]['dominant']]]
      for r in D if r['predictions'][3]['any_correct'] and not r['predictions'][3]['dominant_correct']])
emit('## FRAC 전후 시드3의 국소 판정 변화')
changes=[]
for r in L:
    a=r['baseline_seed3']; b=r['predictions'][3]
    if a['dominant']!=b['dominant'] or a['labels']!=b['labels']:
        changes.append([r['split'],r['case'],r['name'],a['dominant'],b['dominant'],a['labels'],b['labels']])
table(['split','case','GT','기준 다수','FRAC 다수','기준 라벨','FRAC 라벨'],changes)
emit('공식 케이스×클래스 TP/FP/FN/TN/support는 20개 split×seed 모두 FRAC 전후 완전히 일치했다.')
emit()
emit('## 위양성: GT에 없는 케이스×클래스 출력')
fp=data['fps']
table(['영역','10시드 FP 합계','GT 병변 주변에서 발견','GT 주변에 없음'],[
    [reg,len(rr:= [f for f in fp if f['region']==reg]),sum(f['near_gt'] for f in rr),sum(not f['near_gt'] for f in rr)]
    for reg in ['ICA','후순환','MCA','ACA/Acom']])
emit('주변 FP는 잘못 붙인 위치라벨/보조라벨 가능성이 있고, 주변에 없는 FP는 공간적으로 떨어진 출력이다. 이 구분 자체가 원인 확정은 아니다.')
emit()
emit('## 현 구 학습표의 희소성')
table(['분절','좌우 합산 학습 병변 수'],sorted(train.items()))
emit('근거: `code/sblee/nnunet/analysis/e11_feat_hyb_ov.json` 268행. 미러 증강 전 원본 행 수이며, 부위별 수가 작은 만큼 정확도 해석에도 주의가 필요하다.')

emit()
emit('## 원인 해석과 한계')
emit('- 관측: 위 표는 현재 FRAC 최종 마스크에서 직접 확인한 오류다. FRAC는 보조 클래스에 주는 복셀 비중을 바꾸며 RF를 다시 학습하지 않는다. 케이스 단위 정답/오답 클래스 집합은 기존 모델과 같다.')
emit('- 구조적 근거: `c5_location_v2.py`의 `row_to_vec`는 혈관 거리·중첩, 분기점 거리와 위치 좌표를 사용한다. 혈관 라벨의 `ICA-C6-C7`, `A1A2`처럼 출력 위치 클래스보다 거친 구획이 있으므로 세부 위치는 주변 가지와 기하 정보에 의존한다.')
emit('- 추론: 작은 분지 정보의 불안정성, 세부 클래스의 적은 표본, 학습 혈관과 예측 혈관의 차이가 인접 클래스 혼동에 기여할 가능성이 있다. 이 분석만으로 각각의 인과 기여도를 확정할 수는 없다.')
emit('- 과거 검증: `V1_vessel_axis/RESULTS_M1CENSUS.md`에서 M1 길이만으로 early bifurcation과 M1–M2 junction을 구별하는 AUC는 0.577이었다. 길이 임계 하나로 해결된다고 보기 어렵다.')
emit('- 과거 검증: `RESULTS_ICA_DEEP_PV.md`의 실제 적용 가능한 ICA 기하 게이트는 train OOF 정확도 56.1%→54.7%로 떨어졌다. `RESULTS_V3P.md`의 예측혈관 학습표 교체도 test MCC가 -0.0166으로 미채택됐다. 이들은 현재 FRAC 구성에 대한 신규 실험이 아니라 과거 별도 구성에서 얻은 제한적 근거다.')
emit('- 평가 한계: PCA, 원위 MCA 등 검출된 평가 병변이 없는 구간은 분류 성능을 판단할 수 없다. 작은 부위별 표본과 모달리티별 구성 차이를 고려해야 한다. 이 보고서는 오류 분석이며 test 기반 임계 선택이나 재학습은 하지 않았다.')

(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
