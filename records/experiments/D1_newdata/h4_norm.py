#!/usr/bin/env python
"""h4_norm.py — 추론용 Dataset722 정규화 영상만 만든다 (라벨맵 생략 · 병렬).

p_build_3cls.py 는 **학습용 데이터셋 빌더**라 imagesTr 와 labelsTr 를 함께 만든다.
우리는 추론만 하므로 labelsTr 는 쓰이지 않는다(nnUNetv2_predict 는 라벨을 읽지 않는다).

정규화 수식은 p_build_3cls.robust_z 를 **그대로 import 해서** 쓴다 — 여기서 다시 구현하면
학습 때와 미세하게 달라질 수 있고, 그러면 검출기 비교가 아니라 전처리 비교가 된다.

이미 만들어진 파일은 건너뛴다. 중단됐던 것을 이어받을 수 있다.
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, nibabel as nib

R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
sys.path.insert(0, f"{R}/code/sblee/nnunet/scripts")
from p_build_3cls import robust_z, modality, OUT      # 수식·경로를 원본에서 가져온다

IMG = f"{R}/dataset/TopAneu/images"
DST = OUT / "imagesTr"
DST.mkdir(parents=True, exist_ok=True)

def one(cid):
    dst = DST / f"{cid}_0000.nii.gz"
    if dst.exists():
        return (cid, "skip")
    img = nib.load(f"{IMG}/{cid}_0000.nii.gz")
    arr = np.asanyarray(img.dataobj)
    norm, *_ = robust_z(arr, modality(cid))
    tmp = DST / f".tmp_{cid}_0000.nii.gz"
    nib.save(nib.Nifti1Image(norm.astype(np.float32), img.affine, img.header), tmp)
    os.replace(tmp, dst)                                # 중간에 죽어도 반쪽 파일이 남지 않는다
    return (cid, "made")

if __name__ == "__main__":
    sp = json.load(open(f"{R}/dataset/TopAneu/dataset_split.json"))["splits"]
    ids = sorted(set(sp["test"]) | set(sp["val"]))
    ids = [c for c in ids if os.path.exists(f"{IMG}/{c}_0000.nii.gz")]
    nw = int(os.environ.get("NW", "6"))
    print(f"[norm] 대상 {len(ids)}케이스 · 워커 {nw}", flush=True)
    made = skip = 0
    with ProcessPoolExecutor(max_workers=nw) as ex:
        for i, (cid, st) in enumerate(ex.map(one, ids), 1):
            made += st == "made"; skip += st == "skip"
            if i % 20 == 0:
                print(f"  {i}/{len(ids)} (생성 {made} · 기존 {skip})", flush=True)
    print(f"[norm] 완료 · 생성 {made} · 기존 {skip} · 총 {len(list(DST.glob('*_0000.nii.gz')))}", flush=True)
