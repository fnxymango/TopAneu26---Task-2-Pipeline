#!/usr/bin/env python3
"""
Build ONE master CSV over ALL experiments: outputs/experiments_leaderboard.csv

한 실험 = 한 행. 점수(실제 검증 Dice/TP/FP/FN·pseudo-Dice·val loss), 모델(network),
전처리(spacing/patch/batch), 정규화(norm scheme), 학습방식(trainer·loss·oversample·
epochs·split) 를 한 곳에 모은다. 매 실험 종료 시 run_experiment.sh 가 호출 →
전체를 다시 스캔해 통째로 재생성하므로 중복행이 없고 항상 최신 상태.

소스: experiments/<EXP>/{config.json, metrics.json, train.log,
      results/*/*/fold_*/validation/summary.json}
      nnUNet_raw/<DS>/dataset.json, nnUNet_preprocessed/<DS>/nnUNetPlans.json

Usage: make_leaderboard.py [EXPERIMENTS_DIR] [OUTPUT_CSV]
"""
import sys, os, re, json, csv, math, glob
from pathlib import Path

ROOT = Path(os.environ.get("TOPANEU_ROOT", str(Path.home() / "topaneu_sblee")))
BASE = ROOT / "nnunet"
RAW  = BASE / "nnUNet_raw"
PRE  = BASE / "nnUNet_preprocessed"
EXPS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments"
OUT  = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "code/sblee/outputs/experiments_leaderboard.csv"

# 알려진 trainer -> (loss 설명, oversample_fg, target_epochs). 새 trainer는 여기 추가.
TRAINER_META = {
    "nnUNetTrainer":               ("DiceCE",                           0.33, 1000),
    "nnUNetTrainerTverskyCE":      ("Tversky(a0.3,b0.7)+CE",            0.60, 250),
    "nnUNetTrainerFocalTverskyCE": ("FocalTversky(a0.3,b0.7,g1.33)+CE", 0.60, 250),
    "nnUNetTrainerTverskyTopkCE":  ("Tversky(a0.3,b0.7)+TopK-CE(k10)",  0.60, 250),
    "nnUNetTrainerSkeletonRecallNoMirroring":       ("Dice+CE+SkelRecall (NoMirror)", 0.33, 1000),
    "nnUNetTrainerSkeletonRecallNoMirroring_250ep": ("Dice+CE+SkelRecall (NoMirror)", 0.33, 250),
}


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default if default is not None else {}


def loss_from_trainer(trainer):
    """알려진 매핑 우선, 없으면 trainer 클래스 attr(ALPHA/BETA/GAMMA/K)로 best-effort 추론."""
    if trainer in TRAINER_META:
        return TRAINER_META[trainer]
    loss, os_fg, ep = "?", "?", 1000
    me = re.search(r"_(\d+)epochs", trainer or "")
    if me:
        ep = int(me.group(1))
    try:  # sbaneu env에서 실행되므로 import 가능
        import importlib
        cls = getattr(importlib.import_module(f"nnunetv2.training.nnUNetTrainer.{trainer}"), trainer)
        a, b = getattr(cls, "ALPHA", None), getattr(cls, "BETA", None)
        g, k = getattr(cls, "GAMMA", None), getattr(cls, "K", None)
        if a is not None:
            core = f"Tversky(a{a},b{b})" if g is None else f"FocalTversky(a{a},b{b},g{g})"
            loss = core + ("+TopK-CE(k%s)" % k if k is not None else "+CE")
    except Exception:
        pass
    return loss, os_fg, ep


def fmt(x):
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(str(round(v, 3)) if isinstance(v, float) else str(v) for v in x) + "]"
    return "" if x is None else str(x)


def num(x, nd=4):
    return round(x, nd) if isinstance(x, (int, float)) and x == x else ""


def row_for(exp):
    cfg = jload(exp / "config.json")
    if not cfg:
        return None
    met = jload(exp / "metrics.json")

    ds_id = str(cfg.get("dataset_id", "")).zfill(3)
    config = cfg.get("config", "3d_fullres")
    fold = str(cfg.get("fold", "0"))
    ds_dirs = sorted(RAW.glob(f"Dataset{ds_id}_*"))
    ds_name = ds_dirs[0].name if ds_dirs else f"Dataset{ds_id}"
    dsjson = jload((ds_dirs[0] / "dataset.json") if ds_dirs else "", {})
    plans = jload(PRE / ds_name / "nnUNetPlans.json", {})
    pc = plans.get("configurations", {}).get(config, {})

    # trainer: 결과 폴더명 우선 -> extra_args -tr -> 기본
    fold_dirs = sorted(exp.glob(f"results/*/*/fold_{fold}")) or sorted(exp.glob("results/*/*/fold_*"))
    trainer = None
    if fold_dirs:
        trainer = fold_dirs[-1].parent.name.split("__")[0]
    m = re.search(r"-tr\s+(\S+)", cfg.get("extra_args", "") or "")
    if m:
        trainer = m.group(1)
    trainer = trainer or "nnUNetTrainer"
    loss, os_fg, ep_target = loss_from_trainer(trainer)

    # 실제 split (해당 run의 train.log에서 정확히 추출)
    split = ""
    tl = exp / "train.log"
    if tl.exists():
        sm = re.search(r"split has (\d+) training and (\d+) validation", tl.read_text(errors="ignore"))
        if sm:
            split = f"{sm.group(1)}/{sm.group(2)}"

    # 실제 검증 성능 (full-image sliding-window)
    fg_mean = tp = fp = fn = None
    per_class = {}
    if fold_dirs:
        vj = jload(fold_dirs[-1] / "validation" / "summary.json", {})
        fm = vj.get("foreground_mean", {})
        fg_mean, tp, fp, fn = fm.get("Dice"), fm.get("TP"), fm.get("FP"), fm.get("FN")
        per_class = {int(k): round(v["Dice"], 4) if v.get("Dice") == v.get("Dice") else None
                     for k, v in vj.get("mean", {}).items()}

    # 학습 pseudo-Dice (참고)
    best_ema = None
    fd = fold_dirs[-1] if fold_dirs else None
    if fd:
        logs = sorted(fd.glob("training_log_*.txt"))
        if logs:
            for line in open(logs[-1], errors="ignore"):
                mm = re.search(r"New best EMA pseudo Dice:\s*([-\d.]+)", line)
                if mm:
                    best_ema = float(mm.group(1))

    last_ep = met.get("last_epoch")
    ep_done = (last_ep + 1) if isinstance(last_ep, int) else None
    if fg_mean is not None:
        status = "완료(검증됨)"
    elif ep_done:
        status = "진행중/미검증"
    else:
        status = "시작"

    arch = pc.get("architecture", {}).get("arch_kwargs", {})
    net = pc.get("architecture", {}).get("network_class_name", "").split(".")[-1]
    network = f"{net} s{arch.get('n_stages','?')} f{fmt(arch.get('features_per_stage'))}" if net else ""
    labels = dsjson.get("labels", {})
    n_fg = len([v for v in labels.values() if v != 0])
    channel = list(dsjson.get("channel_names", {"0": "?"}).values())[0]

    return {
        "exp_name": cfg.get("exp_name", exp.name),
        "status": status,
        "dataset": f"{ds_id} {ds_name.replace('Dataset'+ds_id+'_','')}",
        "config": config,
        "fold": fold,
        "split_tr/val": split,
        "trainer": trainer,
        "loss": loss,
        "norm": ", ".join(pc.get("normalization_schemes", [])) or "",
        "oversample_fg": os_fg,
        "epochs_done/target": f"{ep_done or '?'}/{ep_target}",
        "val_fg_Dice": num(fg_mean),
        "val_TP": num(tp, 1), "val_FP": num(fp, 1), "val_FN": num(fn, 1),
        "best_ema_pseudoDice": num(best_ema),
        "best_val_loss": met.get("best_val_loss", ""),
        "best_val_epoch": met.get("best_val_epoch", ""),
        "n_fg_classes": n_fg,
        "channel": channel,
        "target_spacing": fmt([round(x, 3) for x in pc.get("spacing", [])]) if pc.get("spacing") else "",
        "patch_size": fmt(pc.get("patch_size")),
        "batch_size": pc.get("batch_size", ""),
        "network": network,
        "per_class_Dice": json.dumps(per_class, ensure_ascii=False) if per_class else "",
        "started": cfg.get("started", ""),
        "extra_args": cfg.get("extra_args", ""),
        "cmd": cfg.get("cmd", ""),
    }


COLS = ["exp_name", "status", "dataset", "config", "fold", "split_tr/val",
        "trainer", "loss", "norm", "oversample_fg", "epochs_done/target",
        "val_fg_Dice", "val_TP", "val_FP", "val_FN",
        "best_ema_pseudoDice", "best_val_loss", "best_val_epoch",
        "n_fg_classes", "channel", "target_spacing", "patch_size", "batch_size",
        "network", "per_class_Dice", "started", "extra_args", "cmd"]


def write_xlsx(rows, cols, path):
    """서식 있는 진짜 엑셀표: 헤더 고정+색, 첫 열 고정, 자동필터, 열너비, val Dice 색상 스케일."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
        from openpyxl.formatting.rule import ColorScaleRule
    except Exception:
        return False
    wb = Workbook()
    ws = wb.active
    ws.title = "leaderboard"
    ws.append(cols)
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    hf = Font(bold=True, color="FFFFFF")
    hfill = PatternFill("solid", fgColor="374151")
    for j in range(1, len(cols) + 1):
        c = ws.cell(1, j)
        c.font, c.fill = hf, hfill
        c.alignment = Alignment(vertical="center", horizontal="center")
    ws.freeze_panes = "B2"                 # 헤더행 + 첫 열(exp_name) 고정
    ws.auto_filter.ref = ws.dimensions     # 엑셀 정렬/필터 버튼
    for j, name in enumerate(cols, 1):
        maxlen = max([len(str(name))] + [len(str(r.get(name, ""))) for r in rows])
        width = min(max(maxlen + 2, 8), 46)
        if name in ("per_class_Dice", "cmd", "extra_args", "started"):
            width = min(width, 22)
        ws.column_dimensions[get_column_letter(j)].width = width
    if "val_fg_Dice" in cols and rows:
        col = get_column_letter(cols.index("val_fg_Dice") + 1)
        ws.conditional_formatting.add(
            f"{col}2:{col}{len(rows) + 1}",
            ColorScaleRule(start_type="num", start_value=0, start_color="F8696B",
                           mid_type="num", mid_value=0.3, mid_color="FFEB84",
                           end_type="num", end_value=0.6, end_color="63BE7B"))
    wb.save(path)
    return True


def main():
    rows = []
    for exp in sorted(EXPS.iterdir()):
        if exp.is_dir() and (exp / "config.json").exists():
            r = row_for(exp)
            if r:
                rows.append(r)
    # 점수 높은 순, 미검증은 아래로
    rows.sort(key=lambda r: r["val_fg_Dice"] if isinstance(r["val_fg_Dice"], (int, float)) else -1, reverse=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel 한글 정상
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    xlsx = OUT.with_suffix(".xlsx")
    ok = write_xlsx(rows, COLS, xlsx)
    print(f"[make_leaderboard] {OUT}  ({len(rows)} experiments)"
          + (f" + {xlsx.name}" if ok else "  (xlsx 생략: openpyxl 없음)"))


if __name__ == "__main__":
    main()
