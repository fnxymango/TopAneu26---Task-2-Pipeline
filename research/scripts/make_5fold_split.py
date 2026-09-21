"""Dataset600 splits_final.json: 1-fold(정식) -> 5-fold CV 확장.

제약:
  - fold 0 = 기존 정식 split(train69/val15)과 완전히 동일 (기존 fold0 결과 비교 유효성 보존)
  - fold 1~4의 val = 기존 fold0의 train 69케이스를 4등분 (17/17/17/18)
  - 84케이스(=69+15)가 정확히 한 번씩만 val에 등장, test14는 절대 등장 안 함
"""
import json, random, shutil, sys
from pathlib import Path

PRE = Path("/home/user/TopAneu/seg/sblee/nnunet/nnUNet_preprocessed/Dataset600_TopAneuVessel")
SPLIT = PRE / "splits_final.json"
BAK = PRE / "splits_final.json.bak_1fold_canonical"

orig = json.load(open(SPLIT))
assert len(orig) == 1, f"예상과 다름: splits {len(orig)}개"
canon_train = list(orig[0]["train"])
canon_val = list(orig[0]["val"])
pool84 = sorted(set(canon_train) | set(canon_val))
assert len(pool84) == 84, len(pool84)

# fold 1~4 val = canon_train 69개를 4등분 (deterministic)
rng = random.Random(20260730)
rest = sorted(canon_train)
rng.shuffle(rest)
sizes = [17, 17, 17, 18]
vals, i = [canon_val], 0
for s in sizes:
    vals.append(sorted(rest[i:i + s]))
    i += s
assert i == 69

splits = [{"train": sorted(set(pool84) - set(v)), "val": sorted(v)} for v in vals]

# --- 검증 ---
# 1) fold0가 원본과 동일 (순서 무시, 집합으로)
assert set(splits[0]["train"]) == set(canon_train), "fold0 train 변경됨!"
assert set(splits[0]["val"]) == set(canon_val), "fold0 val 변경됨!"
# 2) val들이 84를 정확히 한 번씩 덮음
seen = [c for f in splits for c in f["val"]]
assert len(seen) == 84 and len(set(seen)) == 84, "val 중복/누락"
assert set(seen) == set(pool84), "val 합집합이 84와 다름"
# 3) train/val 겹침 없음, 모든 케이스가 pool84 안
for k, f in enumerate(splits):
    assert not (set(f["train"]) & set(f["val"])), f"fold{k} train/val 겹침"
    assert set(f["train"]) | set(f["val"]) == set(pool84), f"fold{k} 커버리지 이상"

if "--write" not in sys.argv:
    print("[DRY RUN] --write 없으면 저장 안 함")
else:
    if not BAK.exists():
        shutil.copy2(SPLIT, BAK)
        print(f"백업 생성: {BAK.name}")
    json.dump(splits, open(SPLIT, "w"), indent=2)
    print(f"저장 완료: {SPLIT}")

for k, f in enumerate(splits):
    tag = " (= 정식 split, 불변)" if k == 0 else ""
    print(f"  fold {k}: train {len(f['train']):2d} / val {len(f['val']):2d}{tag}")
print("검증 통과: 84케이스가 val에 정확히 1회씩, test14 미포함")
