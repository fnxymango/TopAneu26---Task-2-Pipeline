#!/usr/bin/env python3
"""
Generate a per-experiment summary markdown: <EXP_DIR>/summary.md

Pulls objective facts from:
  - <EXP_DIR>/config.json                    (what was run)
  - <EXP_DIR>/metrics.json                   (train/val loss, best)
  - nnUNet_raw/<DS>/dataset.json             (labels, channels)
  - nnUNet_preprocessed/<DS>/nnUNetPlans.json + dataset_fingerprint.json  (auto preprocessing plan)
  - results/.../fold_X/training_log_*.txt    (final per-class pseudo-Dice, best EMA)

Auto-diffs training setup vs base nnUNetTrainer defaults, and auto-collects
data-driven "특이점". Re-runnable anytime. Usage: make_summary.py <EXP_DIR>
"""
import sys, os, re, json, glob
from pathlib import Path

exp = Path(sys.argv[1])
def jload(p, default=None):
    try: return json.load(open(p))
    except Exception: return default if default is not None else {}

cfg = jload(exp / "config.json")
met = jload(exp / "metrics.json")

BASE = Path(os.environ.get("TOPANEU_ROOT", str(Path.home() / "topaneu_sblee"))) / "nnunet"
# NOTE: 전역 nnUNet_* env가 다른 프로젝트를 가리키므로 무시하고 우리 레이아웃 고정
RAW  = BASE / "nnUNet_raw"
PRE  = BASE / "nnUNet_preprocessed"

ds_id  = str(cfg.get("dataset_id", "")).zfill(3)
config = cfg.get("config", "3d_fullres")
fold   = cfg.get("fold", "0")

# resolve dataset folder name (DatasetXXX_*)
ds_dirs = sorted(RAW.glob(f"Dataset{ds_id}_*"))
ds_name = ds_dirs[0].name if ds_dirs else f"Dataset{ds_id}"
dsjson  = jload((ds_dirs[0] / "dataset.json") if ds_dirs else "", {})

# plans: 학습이 results/에 남긴 스냅샷이 가장 정확(플랜 이름·데이터셋 리네임에 무관).
# 없으면 config의 -p 인자로 preprocessed에서 찾고, 마지막에 기본 nnUNetPlans.
_snap = sorted(exp.glob("results/*/*/plans.json"))
plans = jload(_snap[-1], {}) if _snap else {}
if not plans:
    _pm = re.search(r"-p\s+(\S+)", cfg.get("extra_args") or "")
    _cand = [PRE / ds_name / f"{_pm.group(1)}.json"] if _pm else []
    _cand.append(PRE / ds_name / "nnUNetPlans.json")
    for _c in _cand:
        plans = jload(_c, {})
        if plans: break
finger  = jload(PRE / ds_name / "dataset_fingerprint.json", {})
pc      = plans.get("configurations", {}).get(config, {})
# dataset.json도 results 스냅샷 우선(학습 당시 라벨 체계)
_dsnap = sorted(exp.glob("results/*/*/dataset.json"))
if _dsnap:
    dsjson = jload(_dsnap[-1], dsjson)

# ---- training log: final per-class dice, best EMA, trainer ----
fold_dirs = sorted(exp.glob(f"results/*/*/fold_{fold}")) or sorted(exp.glob("results/*/*/fold_*"))
last_dice, best_ema, trainer_from_dir = None, None, None
if fold_dirs:
    fd = fold_dirs[-1]
    trainer_from_dir = fd.parent.name.split("__")[0]
    logs = sorted(fd.glob("training_log_*.txt"))
    if logs:
        logs = sorted(logs, key=lambda f: open(f, errors="ignore").read().count("Pseudo dice"))
        for line in open(logs[-1], errors="ignore"):
            m = re.search(r"Pseudo dice\s+\[([^\]]*)\]", line)
            if m:
                vals = re.findall(r"np\.float\d+\(([^)]+)\)", m.group(1)) or re.findall(r"-?\d+\.?\d*", m.group(1))
                last_dice = [float(x) for x in vals]   # 'nan' -> nan
            m = re.search(r"New best EMA pseudo Dice:\s*([-\d.]+)", line)
            if m: best_ema = float(m.group(1))

# ---- REAL validation metrics (full-image sliding-window inference) ----
val_dice, val_fg_mean = None, None
if fold_dirs:
    vs = fold_dirs[-1] / "validation" / "summary.json"
    if vs.exists():
        vj = jload(vs, {})
        val_dice = {int(k): v for k, v in vj.get("mean", {}).items()}   # label id -> {Dice,TP,FP,FN}
        val_fg_mean = vj.get("foreground_mean", {}).get("Dice")

val_src, val_cldice, val_by_name = "nnU-Net 내장 validation", None, None
if val_fg_mean is None:
    _pcj = jload(exp / "val_per_class_dice.json", {})
    if _pcj:
        val_by_name = _pcj
        val_fg_mean = sum(_pcj.values()) / len(_pcj)
        val_src = "별도 예측(-nps 1) 후 GT 대비 채점 — 내장 validation은 36클래스 확률맵 RAM 스파이크로 crash"
val_cldice = jload(exp / "val_cldice.json", {}).get("mean_clDice")

# ---- labels ----
labels = dsjson.get("labels", {})              # name -> id
fg = sorted([(v, k) for k, v in labels.items() if v != 0])   # (id, name)
channel = list(dsjson.get("channel_names", {"0": "?"}).values())[0]
n_train = dsjson.get("numTraining", "?")

# ---- base-diff for training ----
extra = (cfg.get("extra_args") or "").strip()
trainer = trainer_from_dir or "nnUNetTrainer"
m = re.search(r"-tr\s+(\S+)", extra)
if m: trainer = m.group(1)
epochs_target = 1000
# nnUNetTrainer_XXXepochs / 우리 커스텀 ..._250ep 양쪽 지원
me = re.search(r"nnUNetTrainer_(\d+)epochs", trainer) or re.search(r"_(\d+)ep(?:ochs)?$", trainer)
if me: epochs_target = int(me.group(1))
deltas = []
if trainer not in ("nnUNetTrainer", None):
    deltas.append(f"trainer = `{trainer}` (base=`nnUNetTrainer`)"
                  + (f", epochs {epochs_target} (base 1000)" if me else ""))
if extra and not (m and extra == m.group(0)):
    # list any extra args beyond -tr
    rest = re.sub(r"-tr\s+\S+", "", extra).strip()
    if rest: deltas.append(f"추가 인자: `{rest}`")
if not deltas:
    deltas.append("base `nnUNetTrainer` 기본 설정 그대로 (SGD lr 1e-2, mom 0.99, DiceCE, 1000 epochs) — **변경 없음**")

def fmt_list(x):
    return "[" + ", ".join(str(round(v, 3)) if isinstance(v, float) else str(v) for v in x) + "]" if isinstance(x, (list, tuple)) else str(x)

# ---- data-driven 특이점 ----
notes = []
sp = pc.get("spacing")
if sp:
    ratio = max(sp) / min(sp) if min(sp) else 0
    notes.append(f"target spacing {fmt_list([round(x,3) for x in sp])} mm (이방성비 {ratio:.2f})")
# 실제 검증 기준 검출 실패 / 평가불가
if val_by_name and fg:
    _z = [nm for i, nm in fg if val_by_name.get(nm) == 0.0]
    if _z:
        notes.append(f"검증 Dice=0 (검출 실패) 클래스 {len(_z)}/{len(fg)}: " + ", ".join(_z))
if val_dice and fg:
    zero = [nm for i, nm in fg if isinstance(val_dice.get(i, {}).get("Dice"), (int, float)) and val_dice[i]["Dice"] == 0.0]
    nogt = [nm for i, nm in fg if val_dice.get(i, {}).get("Dice") != val_dice.get(i, {}).get("Dice")]  # nan
    if zero:
        notes.append(f"검증 Dice=0 (검출 실패) 클래스 {len(zero)}/{len(fg)}: " + ", ".join(zero[:10]) + (" …" if len(zero) > 10 else ""))
    if nogt:
        notes.append(f"검증셋에 GT 없어 평가불가(nan) 클래스 {len(nogt)}: " + ", ".join(nogt[:10]))
notes.append("동맥류는 수~십 voxel 초소형 병변 → voxel-Dice가 검출능력을 과소평가, foreground oversampling 의존")

# ---- per-class dice table (patch pseudo-Dice, training proxy) ----
dice_rows = ""
if last_dice is not None and fg:
    for (i, nm), d in zip(fg, last_dice):
        dice_rows += f"| {i} | {nm} | {d:.4f} |\n"

# ---- REAL validation dice table (full-image) ----
def _f(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) and x == x else "nan"
val_rows, val_hdr = "", "| id | class | Dice | TP | FP | FN |\n|---|---|---|---|---|---|\n"
if val_dice and fg:
    for i, nm in fg:
        d = val_dice.get(i, {})
        val_rows += f"| {i} | {nm} | {_f(d.get('Dice'))} | {d.get('TP','?')} | {d.get('FP','?')} | {d.get('FN','?')} |\n"
elif val_by_name and fg:
    val_hdr = "| id | class | Dice |\n|---|---|---|\n"
    for i, nm in fg:
        val_rows += f"| {i} | {nm} | {_f(val_by_name.get(nm))} |\n"

arch = pc.get("architecture", {}).get("arch_kwargs", {})
norm = ", ".join(pc.get("normalization_schemes", [])) or "?"

# ---- 계보(lineage): baseline 대비 / 직전 실험 대비 변경점 ----
# 사용자 지시(2026-08-13): 실험마다 무엇이 달라졌는지 summary에 명시할 것.
# analysis/experiment_lineage.json에 항목을 추가해두면 여기에 자동 반영된다.
_lin_all = jload(Path(__file__).resolve().parents[1] / "analysis" / "experiment_lineage.json", {})
_lin = _lin_all.get(cfg.get("exp_name", exp.name), {})
if _lin:
    _rows = []
    if _lin.get("family"):
        _rows.append(f"- 계열: **{_lin['family']}**")
    if _lin.get("purpose"):
        _rows.append(f"- 목적: {_lin['purpose']}")
    _b = _lin.get("baseline")
    _rows.append(f"- **baseline(`{_b}`) 대비**: {_lin.get('baseline_diff','?')}" if _b
                 else f"- **baseline 위치**: {_lin.get('baseline_diff','이 계열의 기준 실험')}")
    if _lin.get("prev"):
        _rows.append(f"- **직전 실험(`{_lin['prev']}`) 대비**: {_lin.get('prev_diff','?')}")
    lineage_md = "\n## 0. 계보 — baseline / 직전 실험 대비 변경점\n" + "\n".join(_rows) + "\n"
else:
    lineage_md = ("\n## 0. 계보 — baseline / 직전 실험 대비 변경점\n"
                  "_(analysis/experiment_lineage.json에 항목 없음 — 새 실험이면 추가할 것)_\n")

md = f"""# 실험 요약 — {cfg.get('exp_name', exp.name)}

> 데이터셋 **{ds_name}** · config **{config}** · fold **{fold}** · 시작 {cfg.get('started','?')}
> 명령: `{cfg.get('cmd','?')}`
{lineage_md}
## 1. 데이터 / 라벨
- 모달리티/채널: **{channel}** · 학습 케이스 **{n_train}**
- 라벨 체계: 전경 클래스 **{len(fg)}개** (+background)
- 클래스: {", ".join(f"{i}:{nm}" for i, nm in fg) if fg else "?"}

## 2. 전처리 방식 (nnU-Net 자동 계획)
- 정규화 스킴: **{norm}** (채널명 `{channel}` 기준 자동 선택)
- target spacing: **{fmt_list([round(x,3) for x in sp]) if sp else '?'}** mm
  (원본 median spacing {fmt_list([round(x,3) for x in plans.get('original_median_spacing_after_transp',[])])})
- patch size: **{fmt_list(pc.get('patch_size','?'))}** · batch size: **{pc.get('batch_size','?')}**
- median image size(voxels): {fmt_list(pc.get('median_image_size_in_voxels','?'))}
- 리샘플: data=`{pc.get('resampling_fn_data','?')}`, seg=`{pc.get('resampling_fn_seg','?')}` · crop-to-nonzero + foreground oversampling 33%(기본)
- 네트워크: `{pc.get('architecture',{}).get('network_class_name','?').split('.')[-1]}` · stages={arch.get('n_stages','?')} · features={fmt_list(arch.get('features_per_stage','?'))}
- 손 전처리 없음 — 원본이 이미 NIfTI/defacing/LPS+/brain-crop, 여기에 nnU-Net 자동 계획만 적용

## 3. 학습 방식 & base 대비 차이
- trainer: **{trainer}** · 목표 epochs: **{epochs_target}** · 완료 epoch: **{(met.get('last_epoch','?')+1) if isinstance(met.get('last_epoch'), int) else '?'}**
- base 대비 변경점:
""" + "".join(f"  - {d}\n" for d in deltas) + f"""
## 4. 결과

### 실제 검증 성능 (전체영상 sliding-window, fold {fold})
- 출처: {val_src}
- **foreground mean Dice: {_f(val_fg_mean) if val_fg_mean is not None else '(검증 전)'}**{f" · mean clDice: **{val_cldice:.4f}**" if val_cldice else ""}
""" + (val_hdr + val_rows if val_rows else "_(아직 없음 — 최종 검증 미완료)_\n") + f"""
> ⚠️ 초소형 병변 task에서는 voxel-Dice가 검출능력을 과소평가함(수 voxel만 어긋나도 0). 검출 지표(per-lesion sensitivity/FP)로도 별도 평가 권장.

### 학습 지표 (참고)
- best val loss: **{met.get('best_val_loss','?')}** @ epoch {met.get('best_val_epoch','?')}
- best EMA pseudo-Dice: **{best_ema if best_ema is not None else '?'}** _(패치 기반 proxy — 전체영상 성능과 다를 수 있음)_
- loss 곡선:

![loss curve](loss_curve.png)

<details><summary>per-class 패치 pseudo-Dice (마지막 epoch, 참고용)</summary>

""" + ("| id | class | pseudo-Dice |\n|---|---|---|\n" + dice_rows if dice_rows else "_(없음)_\n") + """
</details>
""" + f"""
## 5. 특이점 / 특징점
""" + "".join(f"- {n}\n" for n in notes) + """
---
_이 파일은 실험 종료 시 자동 생성됩니다(make_summary.py). 필요 시 내용 보강 가능._
"""

out = exp / "summary.md"
out.write_text(md)
print(f"[make_summary] {out}")
