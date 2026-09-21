#!/usr/bin/env python3
"""CV 감독자 — GPU당 정확히 1워커를 유지하고, 죽으면 되살리고, 끝나면 채점까지.

설계 원칙(오늘 겪은 실패들을 그대로 반영):
  · RAM 사고 → 워커는 GPU당 1개, 총 4개로 고정. 여유가 없으면 그보다 적게.
  · export 워커 데드락 → nnUNet CLI 대신 predict_queue.py(멀티프로세싱 없음) 사용.
  · GPU 놀기 → 작업 큐 방식이라 빈 워커가 즉시 다음 케이스를 가져감.
  · 죽은 프로세스의 선점 방치 → 매 주기 고아 선점을 자동 해제.
  · 부분 데이터 채점 → 69개 전부 확인된 뒤에만 후처리·채점 진행.

60초 주기. 완료되면 cv_raw.json / cv_pp.json 생성 후 종료.
"""
import os, sys, json, time, subprocess, signal

B = '/home/user/TopAneu/seg/sblee/nnunet'
N = f'{B}/experiments/D600_vessel_skelrec_resencm_250ep_bd0'
CV, CLAIMS = f'{N}/cv', f'{N}/cv/claims'
LOG = f'{N}/supervisor.log'
PY = '/home/user/anaconda3/envs/sbaneu2/bin/python'
GPUS = [0, 1, 2, 3]
MIN_FREE_GB = 20          # 이보다 여유가 적으면 새 워커를 띄우지 않음
INTERVAL = 60

os.makedirs(CLAIMS, exist_ok=True)
splits = json.load(open(f'{B}/nnUNet_preprocessed/Dataset600_TopAneuVessel/splits_final.json'))
ENV = dict(os.environ, nnUNet_raw=f'{B}/nnUNet_raw',
           nnUNet_preprocessed=f'{B}/nnUNet_preprocessed',
           nnUNet_results=f'{N}/results')


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def remaining():
    return [(f, c) for f in range(1, 5) for c in sorted(splits[f]['val'])
            if not os.path.exists(f'{CV}/fold{f}/out/{c}.nii.gz')]


def free_gb():
    for ln in open('/proc/meminfo'):
        if ln.startswith('MemAvailable'):
            return int(ln.split()[1]) // 1024 // 1024
    return 0


def live_workers():
    """{gpu: pid} — 실제로 돌고 있는 예측 워커."""
    out = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', 'predict_queue.py'],
                         capture_output=True, text=True).stdout.split()
    res = {}
    for pid in out:
        try:
            env = open(f'/proc/{pid}/environ', 'rb').read().decode(errors='ignore')
            stat = open(f'/proc/{pid}/stat').read().split()
            if stat[2] in ('Z',):        # 좀비 제외
                continue
        except Exception:
            continue
        for kv in env.split('\0'):
            if kv.startswith('CUDA_VISIBLE_DEVICES='):
                g = kv.split('=', 1)[1]
                if g.isdigit():
                    res.setdefault(int(g), int(pid))
    return res


def release_orphan_claims():
    """소유 PID가 죽었는데 결과물이 없는 선점 → 큐로 반환."""
    alive = set(os.listdir('/proc'))
    freed = []
    for fn in os.listdir(CLAIMS):
        if not fn.startswith('f'):
            continue
        fold = fn[1:fn.index('_')]
        cid = fn[len(fold) + 2:]
        if os.path.exists(f'{CV}/fold{fold}/out/{cid}.nii.gz'):
            continue
        try:
            pid = open(f'{CLAIMS}/{fn}').read().strip()
        except Exception:
            continue
        if pid not in alive:
            try:
                os.remove(f'{CLAIMS}/{fn}'); freed.append(f'fold{fold}/{cid}')
            except FileNotFoundError:
                pass
    if freed:
        say(f"[!] 고아 선점 {len(freed)}개 해제 → 큐 반환: {', '.join(freed[:4])}"
            + (" …" if len(freed) > 4 else ""))
    return freed


def spawn(gpu):
    log = open(f'{N}/qw_sup_gpu{gpu}.log', 'a')
    p = subprocess.Popen([PY, f'{B}/scripts/predict_queue.py'],
                         env=dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu)),
                         stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    say(f"  GPU{gpu} 워커 기동 (pid {p.pid})")


say("=== 감독자 시작 — GPU당 1워커, 60초 주기 ===")
stall, last_done = 0, -1
while True:
    release_orphan_claims()
    rem = remaining()
    done = 69 - len(rem)
    workers = live_workers()
    fg = free_gb()

    if not rem:
        say(f"예측 69/69 완료 — 워커 종료 대기")
        break

    say(f"예측 {done}/69 · 워커 {sorted(workers)} · RAM여유 {fg}GB · 선점 {len(os.listdir(CLAIMS))}")

    if done == last_done:
        stall += 1
    else:
        stall, last_done = 0, done

    for g in GPUS:
        if g in workers:
            continue
        if fg < MIN_FREE_GB:
            say(f"  GPU{g} 비었지만 RAM 여유 {fg}GB < {MIN_FREE_GB}GB — 기동 보류")
            break
        unclaimed = [x for x in rem if not os.path.exists(f'{CLAIMS}/f{x[0]}_{x[1]}')]
        if not unclaimed:
            break
        spawn(g)
        fg -= 16                          # 워커 1개당 대략적 소요
        time.sleep(10)

    if stall > 60:                        # 60분간 진전 없음
        say("[!] 60분간 진전 없음 — 감독자 종료(수동 확인 필요)"); sys.exit(1)
    time.sleep(INTERVAL)

# ---- 예측 완료 → 후처리 + 채점 ----
while live_workers():
    time.sleep(15)
say("후처리 시작 (fold1~4)")
for f in range(1, 5):
    r = subprocess.run([PY, '-u', f'{B}/scripts/postprocess_vessel.py', 'apply',
                        f'{CV}/fold{f}/out', f'{CV}/fold{f}/pp'],
                       env=ENV, capture_output=True, text=True)
    say(f"  fold{f} 후처리 {'완료' if r.returncode == 0 else '실패: ' + r.stderr[-200:]}")

for label, sub in (('raw', 'out'), ('pp', 'pp')):
    say(f"집계: {label}")
    args = [PY, '-u', f'{B}/scripts/eval_cv_5fold.py', label,
            f'{N}/val_predict_out' + ('_pp' if label == 'pp' else '')] + \
           [f'{CV}/fold{f}/{sub}' for f in range(1, 5)]
    r = subprocess.run(args, env=ENV, capture_output=True, text=True)
    say(r.stdout[-1500:] if r.returncode == 0 else f"실패: {r.stderr[-500:]}")
say("=== 감독자 완료 ===")
