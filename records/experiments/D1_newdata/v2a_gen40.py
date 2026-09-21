#!/usr/bin/env python3
"""V2-A 1단계 — GT 혈관마스크 전 케이스(train·val·test)를 40클래스로 분할해 저장한다.

규칙은 v2_rule.json (큐 3단계에서 train 파라미터로만 정해짐). 규칙 자체가 라벨을 안 보거나
train 에서만 보정됐으므로 test 케이스에 적용해도 누수가 아니다 — fine GT 는 전혀 읽지 않는다.
출력: experiments/_v2_ves40/gt/{case}.nii.gz (원본 header·affine 유지) + info.json
검증: 케이스마다 ICA-C6-C7 밖 복셀이 원본과 같은지, ICA 복셀 총수가 같은지 assert.
"""
import json, os, sys
import numpy as np, nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ica_split_lib as IL

R = IL.R
OUT = f"{R}/experiments/_v2_ves40/gt"
RULE = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/v2_rule.json"))["rule"]


def one(cid):
    src = f"{R}/dataset/TopAneu/vessel_masks/{cid}.nii.gz"
    dst = f"{OUT}/{cid}.nii.gz"
    if not os.path.exists(src):
        return cid, None
    im = nib.load(src)
    v = np.asanyarray(im.dataobj)
    sp = np.array(im.header.get_zooms()[:3], float)
    o, info = IL.split_volume(v, sp, RULE)
    ica = np.isin(v, [4, 6])
    assert (o[~ica] == v[~ica]).all(), f"{cid}: ICA 밖이 바뀜"
    assert int(ica.sum()) == int(np.isin(o, [4, 6, 37, 38, 39, 40]).sum()), f"{cid}: ICA 복셀 수 불일치"
    hdr = im.header.copy(); hdr.set_data_dtype(np.uint8)
    tmp = dst + f".tmp{os.getpid()}.nii.gz"
    nib.save(nib.Nifti1Image(o.astype(np.uint8), im.affine, hdr), tmp)
    os.replace(tmp, dst)
    return cid, info


def main():
    import multiprocessing as mp
    os.makedirs(OUT, exist_ok=True)
    S = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))
    cases = [c for sp in ("train", "val", "test") for c in S["splits"][sp]]
    todo = [c for c in cases if not os.path.exists(f"{OUT}/{c}.nii.gz")]
    print(f"[gen40] 규칙 {RULE['kind']} · 대상 {len(cases)} · 남은 {len(todo)}", flush=True)
    info = {}
    ip = f"{OUT}/../info.json"
    if os.path.exists(ip):
        info = json.load(open(ip))
    with mp.Pool(int(os.environ.get("V2_WORKERS", "10"))) as pool:
        for i, (c, inf) in enumerate(pool.imap_unordered(one, todo), 1):
            info[c] = inf
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
    json.dump(info, open(ip, "w"))
    n = sum(1 for c in cases if os.path.exists(f"{OUT}/{c}.nii.gz"))
    fail = {s: sum(1 for c in cases if (info.get(c) or {}).get(s) is None) for s in ("R", "L")}
    full = sum(1 for c in cases for s in ("R", "L") if ((info.get(c) or {}).get(s) or {}).get("branch_full"))
    print(f"[gen40] 완료 {n}/{len(cases)} · 축 없음 R {fail['R']} L {fail['L']} · 두 절단 모두 분지로 결정 {full} side")


if __name__ == "__main__":
    main()
