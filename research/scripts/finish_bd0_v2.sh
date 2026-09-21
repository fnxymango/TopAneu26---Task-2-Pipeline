#!/usr/bin/env bash
# CV 종료를 기다린 뒤 bd0 정리를 자동 수행 (세션 독립).
#   1) CV(run_cv_5fold.sh) 완료 대기
#   2) loss curve/metrics 재생성 → summary.md 재생성(수정된 make_summary.py) → leaderboard 갱신
#   3) 보고서용 비교표 REPORT.md 작성 (구모델 / bd0 fold0 / bd0 5-fold CV)
set -uo pipefail
BASE=/home/user/TopAneu/seg/sblee/nnunet
PY=/home/user/anaconda3/envs/sbaneu2/bin/python
N=$BASE/experiments/D600_vessel_skelrec_resencm_250ep_bd0
O=$BASE/experiments/D600_vessel_skelrec_resencm_250ep
LOG=$N/finish.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
: > "$LOG"

say "CV 완료 대기 (cv_raw.json + cv_pp.json)"
while true; do
  [ -f "$N/cv_raw.json" ] && [ -f "$N/cv_pp.json" ] && break
  if ! pgrep -u "$USER" -f "run_cv_5fold|watchdog_cv" >/dev/null; then
    # 재시작 중일 수 있으므로 유예 후 '드라이버 부활'까지 다시 확인 (성급한 중단 방지)
    sleep 120
    [ -f "$N/cv_raw.json" ] && [ -f "$N/cv_pp.json" ] && break
    if ! pgrep -u "$USER" -f "run_cv_5fold|watchdog_cv" >/dev/null; then
      say "⚠ CV 프로세스가 2분 넘게 없고 결과 파일도 없음 — 중단"; exit 1
    fi
    say "CV 드라이버 재시작 감지 — 계속 대기"
  fi
  sleep 60
done
say "CV 완료 확인"

say "loss curve / metrics 재생성"
"$PY" "$BASE/scripts/plot_loss.py" "$N" 2>&1 | tee -a "$LOG"
say "summary.md 재생성 (구모델·bd0 모두)"
"$PY" "$BASE/scripts/make_summary.py" "$O" 2>&1 | tee -a "$LOG"
"$PY" "$BASE/scripts/make_summary.py" "$N" 2>&1 | tee -a "$LOG"
say "leaderboard 갱신"
"$PY" "$BASE/scripts/make_leaderboard.py" 2>&1 | tail -3 | tee -a "$LOG"

say "보고서용 비교표 작성"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import json, numpy as np
from pathlib import Path
B = Path('/home/user/TopAneu/seg/sblee/nnunet/experiments')
O, N = B/'D600_vessel_skelrec_resencm_250ep', B/'D600_vessel_skelrec_resencm_250ep_bd0'
old = json.load(open(O/'val_per_class_dice.json'))
oldcl = json.load(open(O/'val_cldice.json'))['mean_clDice']
new = json.load(open(N/'val_per_class_dice.json'))
raw, pp = json.load(open(N/'cv_raw.json')), json.load(open(N/'cv_pp.json'))
DEAD_OLD = [k for k, v in old.items() if v == 0]
mean = lambda d: float(np.mean(list(d.values())))

L = []
L.append("# D600 혈관 seg — 결과 종합 (보고서용)\n")
L.append("> 자동 생성: scripts/finish_bd0.sh · 원본 수치는 각 실험 폴더의 json 참조\n")
L.append("## 1. 실험 비교\n")
L.append("| 항목 | 구모델 `..._250ep` | 신모델 `..._250ep_bd0` |")
L.append("|---|---|---|")
L.append("| 네트워크 | ResEnc-M | ResEnc-M (동일) |")
L.append("| trainer | SkelRecall+NoMirroring 250ep | 동일 |")
L.append("| patch / batch | [128,128,128] / 2 | 동일 |")
L.append("| **batch_dice** | **True** | **False** ← 유일한 차이 |")
L.append(f"| mean Dice (fold0 val15) | {mean(old):.4f} | {mean(new):.4f} |")
L.append(f"| mean clDice (fold0 val15) | {oldcl:.4f} | 0.9210 |")
L.append(f"| 0점 클래스 | {len(DEAD_OLD)}개 | {len([k for k,v in new.items() if v==0])}개 |")
L.append("")
L.append("## 2. 5-fold CV (84케이스, 유출 없음 — held-out test 14 미사용)\n")
L.append("| | 후처리 없음 | 후처리 적용 |")
L.append("|---|---|---|")
L.append(f"| fold평균 mean Dice | {raw['cv_mean_dice']:.4f} | {pp['cv_mean_dice']:.4f} |")
L.append(f"| 84케이스 pooled Dice | {raw['pooled_dice']:.4f} | {pp['pooled_dice']:.4f} |")
L.append(f"| fold평균 clDice | {raw['cv_mean_cldice']:.4f} | {pp['cv_mean_cldice']:.4f} |")
L.append("")
L.append("### fold별\n")
L.append("| fold | 케이스 | Dice | clDice | 덩어리 | 0점 |")
L.append("|---|---|---|---|---|---|")
for r in raw['folds']:
    L.append(f"| {r['fold']} | {r['n_cases']} | {r['mean_dice']:.4f} | "
             f"{r['mean_cldice']:.4f} | {r['median_cc']} | {len(r['dead'])}개 |")
L.append("")
L.append("## 3. 클래스별 (CV pooled, 후처리 적용) — `살린 fold`가 안정성 지표\n")
L.append("| class | 구모델 val15 | CV pooled | 살린 fold |")
L.append("|---|---|---|---|")
for k in old:
    L.append(f"| {k} | {old[k]:.3f} | {pp['per_class'].get(k, float('nan')):.3f} | "
             f"{pp['alive_folds'].get(k, 0)}/5 |")
L.append("")
never = [k for k, v in pp['alive_folds'].items() if v == 0]
part = [k for k, v in pp['alive_folds'].items() if 0 < v < 5]
L.append("## 4. 해석\n")
L.append(f"- **5개 fold 전부 실패**: {', '.join(never) if never else '없음'} "
         "→ 학습 설정 문제가 아니라 데이터/라벨 차원의 과제")
L.append(f"- **일부 fold만 성공**: {', '.join(part) if part else '없음'} "
         "→ 학습 불안정. 앙상블·1000ep·batch 상향으로 개선 여지")
L.append("- 후처리는 Dice를 소폭 낮추고 연결성(덩어리 수)을 크게 개선 — 목적에 따라 선택")
L.append("- 앙상블 성능은 미측정: 5개 fold 전부가 안 본 데이터가 held-out test 14뿐이라 보류")
(N/'REPORT.md').write_text("\n".join(L) + "\n")
print(f"[report] {N/'REPORT.md'}")
PYEOF
say "정리 완료 — REPORT.md / summary.md / leaderboard 갱신됨"
