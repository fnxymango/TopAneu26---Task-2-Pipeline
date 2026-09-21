#!/usr/bin/env python3
"""COMPARE/학습 실행 전제조건 점검. 실패하면 '무엇이 없어서 못 도는지'를 이름으로 찍는다.
   지난 사고(Dataset720 격리 → c7 이 GT 를 못 찾음)를 이 검사로 잡을 수 있어야 한다."""
import json, os, sys, glob, shutil
R = "/home/sblee/topaneu_sblee_2026-08-07/topaneu_sblee"
E, P, BP = f"{R}/experiments", f"{R}/experiments/_c1_realpred", f"{R}/experiments/_c4_bpgraph"
RAW = f"{R}/nnunet/nnUNet_raw"
bad = []
def need_dir(path, minfiles, why):
    n = len(glob.glob(f"{path}/*")) if os.path.isdir(path) else -1
    okk = n >= minfiles
    print(f"  [{'OK ' if okk else 'NG '}] {why}\n         {path}  ({'없음' if n < 0 else str(n)+'개'}, 최소 {minfiles})")
    if not okk: bad.append(why)
def need_file(path, why):
    okk = os.path.isfile(path)
    print(f"  [{'OK ' if okk else 'NG '}] {why}\n         {path}")
    if not okk: bad.append(why)

print("=== 전제조건 점검 ===")
need_dir(f"{RAW}/Dataset720_TopAneuBinary417/labelsTr", 400,
         "c7/c17/c20 이 읽는 GT 병변마스크 (Dataset720 labelsTr)")
need_file(f"{R}/dataset/TopAneu/dataset_split.json", "공식 split 정의")
need_dir(f"{R}/dataset/TopAneu/location_masks", 400, "위치 GT 마스크")
for sp, n in (("test", 83), ("val", 41)):
    need_dir(f"{P}/in_{sp}_722", n, f"검출기 입력 ({sp})")
    need_dir(f"{P}/vespp_{sp}", n, f"혈관 예측 ({sp})")
need_dir(f"{BP}/vespp_test", 83, "분기점 그래프 (test)")
need_dir(f"{BP}/val_pred", 41, "분기점 그래프 (val)")
need_file(f"{R}/code/sblee/nnunet/analysis/e11_feat_hyb_ov.json", "분류기 학습피처")
for arm in ("old", "new"):
    need_dir(f"{E}/P3_pjh3cls_resencl_iso04_f0/results" if arm == "old"
             else f"{E}/P5_newdata_detector/results", 1, f"검출기 체크포인트 ({arm})")
free = shutil.disk_usage("/").free / 2**30
okk = free > 10
print(f"  [{'OK ' if okk else 'NG '}] 디스크 여유 {free:.0f}G (최소 10G)")
if not okk: bad.append("디스크 여유 부족")
print()
if bad:
    print("★ 실패 전제조건:")
    for b in bad: print(f"    - {b}")
    sys.exit(1)
print("전 항목 통과")
