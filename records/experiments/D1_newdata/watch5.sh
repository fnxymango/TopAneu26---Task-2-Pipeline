#!/usr/bin/env bash
# watch5 — 돌고 있는 작업의 생사와 '살아있는데 멈춤' 을 함께 본다.
#
# 죽음만 보면 충분하지 않다. nnUNet 추론은 프로세스가 살아 있어도 워커가 죽어 멈출 수 있고,
# 채점은 파일만 만들어 놓고 진전이 없을 수 있다. 그래서 **진행 카운터를 이전 호출과 비교**한다.
# 이전 값은 watch5.state 에 남긴다.
set -uo pipefail
R=/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee
D=$R/experiments/D1_newdata; H=$R/experiments/H1_patchfilter; G=$R/experiments/H4_folds; P=$R/experiments/_c1_realpred
ST=$D/watch5.state
now=$(date +%s)

alive(){ # $1 = 정확한 argv 접두 (공백 포함). pgrep 금지 — 느슨한 매칭으로 예전에 셸을 죽였다.
  local n=0 p c
  for p in /proc/[0-9]*; do
    [ -r "$p/cmdline" ] || continue
    c=$(tr '\0' '\n' < "$p/cmdline" | head -3 | tr '\n' ' ')
    case "$c" in "$1"*) n=$((n+1));; esac
  done
  echo "$n"
}
cnt_scores(){ local n=0 t sp sd
  for t in b1Non b1Non_pf b1Noff b1Noff_pf e9f3P5 e9f3P5_pf e9f3P3 e9f3P3_pf; do
    for sp in test val; do for sd in 0 1 2 3 4; do
      grep -q '"label"' "$H/scores/${t}_${sp}_s${sd}.json" 2>/dev/null && n=$((n+1))
    done; done
  done; echo "$n"
}
cnt_pred(){ local n=0 d
  for d in "$G"/pred_*; do [ -d "$d" ] && n=$((n + $(ls -1 "$d"/*.nii.gz 2>/dev/null | wc -l))); done
  echo "$n"
}
cnt_c7(){ local n=0 d      # 2단계 산출. 이게 없으면 c7 도는 동안 진전을 못 본다.
  for d in "$P"/aneu_test_e9f3P5ff "$P"/aneu_val_e9f3P5ff "$P"/aneu_test_e9f3P3ff "$P"/aneu_val_e9f3P3ff; do
    [ -d "$d" ] && n=$((n + $(ls -1 "$d" 2>/dev/null | wc -l)))
  done; echo "$n"
}
cnt_c5(){ local n=0 d
  for d in "$H"/pred/e9f3*_s*; do [ -d "$d" ] && n=$((n + $(ls -1 "$d" 2>/dev/null | wc -l))); done
  echo "$n"
}

sc=$(cnt_scores); pr=$(cnt_pred); c5=$(cnt_c5); c7=$(cnt_c7)
h4=$(alive "bash ./H4.sh "); h2=$(alive "bash ./H2.sh ")
pdt=$(alive "/home/sblee/miniconda3/envs/sbaneu2/bin/python -u /home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata/h4_predict.py ")
nev=$(alive "/home/sblee/miniconda3/envs/sbaneu2/bin/python /home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata/neweval.py ")
pf=$(alive "bash ./promote_fill.sh ")
fill=$(ls -1 "$H"/scores/.fill_* 2>/dev/null | wc -l)

osc=0; opr=0; oc5=0; oc7=0; ots=0
[ -f "$ST" ] && . "$ST"
dt=$(( now - ots )); [ "$ots" = 0 ] && dt=0
printf '%s\n' "osc=$sc" "opr=$pr" "oc5=$c5" "oc7=$c7" "ots=$now" > "$ST"

echo "== $(TZ=Asia/Seoul date +%H:%M:%S) KST (직전 확인 ${dt}초 전) =="
printf "  H4.sh %s · H2.sh %s · h4_predict %s · neweval %s · promote_fill %s\n" \
  "$([ "$h4" -gt 0 ] && echo 살아있음 || echo ★종료)" \
  "$([ "$h2" -gt 0 ] && echo 살아있음 || echo 종료)" "$pdt" "$nev" "$pf"
printf "  검출예측 %s (+%s) · c5산출 %s (+%s) · 채점 %s/%s (+%s)\n" \
  "$pr" "$((pr-opr))" "$c5" "$((c5-oc5))" "$sc" 80 "$((sc-osc))"
printf "  c7산출 %s/248 (+%s)\n" "$c7" "$((c7-oc7))"
[ "$fill" -gt 0 ] && echo "  임시 채점파일 $fill개 대기 중"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  GPU/'
echo "  $(uptime | sed 's/.*load/load/')"

# --- 이상 징후 판정 -------------------------------------------------------
warn=0
[ "$h4" -eq 0 ] && [ ! -f "$D/.done_h4" ] && { echo "  ★H4.sh 가 완료 표시 없이 사라졌다"; warn=1; }
[ "$h2" -eq 0 ] && [ ! -f "$D/.done_h2" ] && { echo "  ★H2.sh 가 완료 표시 없이 사라졌다"; warn=1; }
if [ "$dt" -gt 120 ]; then
  if [ "$pdt" -gt 0 ] && [ "$((pr-opr))" -eq 0 ]; then echo "  ★검출 추론이 살아있는데 ${dt}초간 진전 0"; warn=1; fi
  pst=$(alive "/home/sblee/miniconda3/envs/sbaneu2/bin/python /home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee/experiments/D1_newdata/h4_post.py ")
  if [ "$pst" -gt 0 ] && [ "$((c7-oc7))" -eq 0 ] && [ "$c7" -lt 248 ]; then echo "  ★c7 이 살아있는데 ${dt}초간 진전 0"; warn=1; fi
  if [ "$nev" -gt 0 ] && [ "$((sc-osc))" -eq 0 ] && [ "$c5" = "$oc5" ]; then
    echo "  (채점 진전 0 — test 한 판이 ~19분이라 정상일 수 있음)"
  fi
fi
for f in "$G"/predict_*.log "$H"/logs/*.err; do
  [ -e "$f" ] || continue
  # 이전 실패 회차의 로그를 다시 잡지 않도록, H4.sh 시작보다 새 로그만 본다
  [ -f "$D/H4.pid" ] && [ "$f" -ot "$D/H4.pid" ] && continue
  grep -lqE "Traceback|RuntimeError|CUDA out of memory|Killed" "$f" 2>/dev/null && { echo "  ★오류 흔적: ${f##*/}"; warn=1; }
done
[ "$warn" -eq 0 ] && echo "  이상 없음"
