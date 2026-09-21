#!/usr/bin/env bash
# B2EVAL — ① B2 지표 기록  ② stock 계열 vs ResEncL 계열 비교  ③ 이긴 계열 판정
#
# 판정 규칙 (결과 보기 전에 고정, 2026-09-04 09:20 KST):
#   축   : covered_gt MCC, 시드 0~4 평균
#   대상 : 개정판 데이터로 학습된 두 모델 — B2(stock) vs cmpnew(ResEncL=P5)
#   승리 : test·val 두 집합 모두에서 높으면 그 계열
#   분열 : test 를 우선한다(83건으로 더 크고 공식 held-out). 단 '분열'로 명시 기록
#   보조 : 구데이터 학습본끼리도 같이 본다 — B1(stock) vs cmpold(ResEncL=P3)
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
E=$R/experiments; D=$E/D1_newdata; B2=$E/B2_bundle_recipe_retrain; ST=$D/STATUS.log
PY=$HOME/miniconda3/envs/sbaneu2/bin/python
exec 9>"$D/b2eval.lock"; flock -n 9 || exit 0
echo $$ > "$D/B2EVAL.pid"
log(){ echo "[$(TZ=Asia/Seoul date +'%m-%d %H:%M') KST][b2eval] $*" | tee -a "$ST"; }
log "B2 완료 대기"
for i in $(seq 1 2880); do [ -f "$D/.done_b2" ] && break; sleep 60; done
[ -f "$D/.done_b2" ] || { log "★B2 미완 — 취소"; exit 1; }
mkdir -p "$B2"
{
  "$PY" "$D/record_metrics.py" b2new "B2 — 번들 레시피 · 개정판으로 검출기+분류기 모두 재학습 (프로덕션 후보)"
  echo "---"
  echo
  "$PY" "$D/record_metrics.py" b2 "B2-대조 — 검출기만 재학습, 분류기는 구 GT 피처 (레버 분리용)"
  echo "## 구성"
  echo '```'
  echo "검출기   stock nnUNetPlans · PlainConvUNet · 5폴드 · 개정판 415케이스로 재학습"
  echo "혈관     V4-2 -> V5 후처리 (번들과 동일, 재학습 안 함)"
  echo "c7      min_vox=5 · max_dist=1.0mm (번들과 동일)"
  echo "분류기   RF500 · beta 0.5 · topk n=2 vox=3 ica=1 margin=0.7 (번들과 동일)"
  echo "분류기피처 재학습됨 — e11_feat_hyb_ov_NEW.json (train 291, 개정판 GT)"
  echo "          대조군 b2 는 번들본 e11_feat_hyb_ov.json (구 GT) 사용"
  echo "혈관     재학습 안 함 — 학습데이터 414/415 가 바이트 동일해 재학습해도 같은 모델"
  echo "데이터   415케이스 · train 291 / val 41 / test 83 · 채점은 개정판 GT"
  echo '```'
  echo
  echo "---"
  echo
  echo "# 계열 비교 — stock vs ResEncL"
  echo
  echo "## 개정판 데이터 학습본끼리 (판정 기준)"
  echo
  "$PY" "$D/cmp_tags.py" cmpnew b2 "P5 ResEncL(신)" "B2 stock(신)" | tail -n +2
  echo "## 구데이터 학습본끼리 (보조)"
  echo
  "$PY" "$D/cmp_tags.py" cmpold b1 "P3 ResEncL(구)" "B1 stock(구)" | tail -n +2
} > "$B2/RESULTS.md" 2>&1
log "B2 지표·계열비교 기록 · $B2/RESULTS.md"

# ---- 판정
"$PY" - <<'PYEOF' > "$D/LINE_VERDICT.md" 2>&1
import json, os
import numpy as np
A = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/code/sblee/nnunet/analysis"
D = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata"
def m(tag, sp):
    v = [json.load(open(f"{A}/c5_eval_{sp}_{tag}_{sp}_s{s}.json"))["adjusted_div_present"]["MCC"]
         for s in range(5) if os.path.exists(f"{A}/c5_eval_{sp}_{tag}_{sp}_s{s}.json")]
    return float(np.mean(v)) if v else None
print("# 계열 판정 — covered_gt MCC (시드 0~4 평균)\n")
print("| 집합 | ResEncL(P5, 신) | stock(B2, 신) | 승 |")
print("|---|---|---|---|")
win = {}
for sp in ("test", "val"):
    r, s = m("cmpnew", sp), m("b2", sp)
    if r is None or s is None:
        print(f"| {sp} | {r} | {s} | 판정불가 |"); win[sp] = None; continue
    w = "stock" if s > r else "resencl"
    win[sp] = w
    print(f"| {sp} | {r:.4f} | {s:.4f} | **{w}** |")
if win["test"] == win["val"] and win["test"]:
    choice, note = win["test"], "두 집합 일치"
else:
    choice, note = win["test"], "두 집합 분열 — test 우선(규칙대로)"
print(f"\n**판정: {choice}** ({note})\n")
print("10폴드 구성: " + ("P1(구) + B2(신)" if choice == "stock" else "P3(구) + P5(신)"))
open(f"{D}/.line_choice", "w").write(choice + "\n")
PYEOF
cat "$D/LINE_VERDICT.md" | tee -a "$ST" > /dev/null
log "계열 판정 → $(cat "$D/.line_choice" 2>/dev/null)"
touch "$D/.done_b2eval"
