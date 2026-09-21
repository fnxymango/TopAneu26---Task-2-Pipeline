#!/usr/bin/env python3
"""D740(lesionscale crop) 모델을 **정식 val 84케이스 전체볼륨**에서 평가.

왜 필요한가 (2026-08-13 발견):
  Dataset740은 train split 케이스만 잘라 만든 crop window 528개로 구성되고,
  nnU-Net 내부 validation도 그 중 106개 window에서 잰다. 즉 기존 D740 Dice 0.5131은
  '병변이 이미 창 한가운데 놓인' 쉬운 조건의 숫자라, 전체볼륨에서 잰 D750(0.5273)과
  같은 잣대가 아니다. 빌더 주석도 "테스트는 원본 전체볼륨 sliding-window"로 명시.

  또한 학습은 crop(케이스당 음성창 1~2개)만 보고 추론은 전체볼륨에서 하므로
  **위양성 폭발** 위험이 구조적으로 존재한다. voxel Dice만으로는 이게 안 보여서
  병변 단위 검출지표(sensitivity / 케이스당 FP)를 같이 계산한다.

지표:
  - case Dice: 케이스별 binary Dice (GT/pred 모두 비면 1.0로 보지 않고 제외 — 편향 방지)
  - lesion sensitivity: GT 병변 컴포넌트 중 pred와 1voxel이라도 겹친 비율
  - FP per case: pred 컴포넌트 중 어떤 GT 병변과도 안 겹친 개수
"""
import json, os, sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

ST = np.ones((3, 3, 3))
ROOT = Path(os.environ["TOPANEU_ROOT"])
PRE = ROOT / "nnunet" / "nnUNet_preprocessed" / "Dataset720_TopAneuBinary417"
LAB = ROOT / "nnunet" / "nnUNet_raw" / "Dataset720_TopAneuBinary417" / "labelsTr"


def main():
    pred_dir = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else pred_dir.name

    val_ids = json.load(open(PRE / "splits_final.json"))[0]["val"]
    print(f"[eval] val {len(val_ids)}케이스, pred={pred_dir}", flush=True)

    dices, sens_hit, sens_tot, fps, per_case = [], 0, 0, [], []
    for i, cid in enumerate(val_ids, 1):
        pp = pred_dir / f"{cid}.nii.gz"
        gp = LAB / f"{cid}.nii.gz"
        if not pp.exists() or not gp.exists():
            print(f"  스킵 {cid}: 파일 없음")
            continue
        p = np.asanyarray(nib.load(pp).dataobj) > 0
        g = np.asanyarray(nib.load(gp).dataobj) > 0

        s = p.sum() + g.sum()
        d = None
        if s > 0:
            d = float(2.0 * (p & g).sum() / s)
            dices.append(d)

        glab, gn = ndimage.label(g, structure=ST)
        plab, pn = ndimage.label(p, structure=ST)
        hit = sum(1 for l in range(1, gn + 1) if (p & (glab == l)).any())
        fp = sum(1 for l in range(1, pn + 1) if not (g & (plab == l)).any())
        sens_hit += hit; sens_tot += gn; fps.append(fp)
        per_case.append({"case": cid, "dice": d, "gt_lesions": int(gn),
                         "detected": int(hit), "fp_components": int(fp),
                         "pred_vox": int(p.sum()), "gt_vox": int(g.sum())})
        if i % 20 == 0:
            print(f"  {i}/{len(val_ids)}", flush=True)

    res = {
        "tag": tag, "pred_dir": str(pred_dir), "n_cases": len(per_case),
        "mean_case_dice": float(np.mean(dices)) if dices else None,
        "n_cases_scored_dice": len(dices),
        "lesion_sensitivity": sens_hit / sens_tot if sens_tot else None,
        "lesions_detected": sens_hit, "lesions_total": sens_tot,
        "fp_per_case_mean": float(np.mean(fps)) if fps else None,
        "fp_per_case_median": float(np.median(fps)) if fps else None,
        "fp_total": int(np.sum(fps)),
        "per_case": per_case,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "per_case"},
                     indent=2, ensure_ascii=False))
    out = ROOT / "code" / "sblee" / "nnunet" / "analysis" / f"fullvolume_eval_{tag}.json"
    json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()
