"""C14 — 혈관 분할망(V4-2)의 내부 표현을 병변 피처로 사용.

발상: V4-2는 36클래스를 pseudo-Dice 0.9까지 맞힌다. 그러려면 이미 풍부한 해부학적
표현을 인코더에 학습해 뒀다는 뜻이다. 우리가 손으로 만든 106차원(거리/점유율/분기점/좌표)
대신 **그 표현을 병변 위치에서 뽑아 쓰면** 새 라벨 없이 훨씬 풍부한 피처를 얻는다.

구현 제약(2026-08-15): GPU 2장이 A6-2 5-fold 학습에 19GB씩 물려 있어 여유가 ~5GB뿐이다.
여기서 GPU를 잡으면 13시간짜리 학습을 OOM으로 죽일 수 있으므로 **CPU 전용으로 돌린다.**
병변 중심 패치만 forward 하므로 전체볼륨 추론보다 훨씬 싸다.

절차:
  1) nnUNetPredictor로 V4-2 fold0 모델 복원 (plans/architecture 재구성 포함)
  2) 인코더 마지막 stage에 forward hook — bottleneck 특징맵 포획
  3) 케이스별로 nnU-Net 전처리를 한 번 돌리고, 병변 중심 패치를 잘라 forward
  4) 포획된 특징맵을 global average pooling -> 병변당 특징벡터
  5) 기존 피처와 붙여 CV 비교

사용:
  python c14_vessel_encoder_feat.py --feat <c10_feat_train.json> --max-cases 60
"""
import argparse, json, collections, os
from pathlib import Path

import numpy as np
import nibabel as nib
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier

import d9xx_lib as L
import c5_location_v2 as C5
import c8_classifier_cv as C8

V42 = ("V4-2_vessel_classweighted_417_500ep/results/Dataset800_TopAneuVessel417/"
       "nnUNetTrainerSkeletonRecallNoMirroringClassWeightedV2_500ep__nnUNetResEncUNetMPlans__3d_fullres")
PATCH = (96, 96, 96)


def load_network():
    """CPU로 V4-2 fold0 네트워크 복원. nnUNetPredictor가 plans->architecture 재구성을 처리한다."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    model_dir = L.TOPANEU_ROOT / "experiments" / V42
    pred = nnUNetPredictor(device=torch.device("cpu"), allow_tqdm=False,
                           perform_everything_on_device=False)
    pred.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,),
                                              checkpoint_name="checkpoint_best.pth")
    net = pred.network.eval()
    feats = {}

    def hook(_m, _i, o):
        feats["z"] = o[0] if isinstance(o, (list, tuple)) else o

    # 인코더의 가장 깊은 stage에 hook — 구현체마다 이름이 달라 encoder.stages[-1]을 우선 시도
    enc = getattr(net, "encoder", None)
    target = None
    if enc is not None and hasattr(enc, "stages"):
        target = enc.stages[-1]
    if target is None:                    # 폴백: 가장 깊은 Conv3d
        convs = [m for m in net.modules() if m.__class__.__name__ == "Conv3d"]
        target = convs[len(convs) // 2]
    target.register_forward_hook(hook)
    return pred, net, feats


def preprocess_case(pred, cid):
    """nnU-Net 전처리기로 케이스 1건 전처리 (정규화/리샘플). 반환 (data, 원본->전처리 스케일)"""
    raw = L.TOPANEU_ROOT / "nnunet" / "nnUNet_raw" / "Dataset800_TopAneuVessel417" / "imagesTr"
    f = raw / f"{cid}_0000.nii.gz"
    if not f.exists():
        return None, None
    pp = pred.configuration_manager.preprocessor_class(verbose=False)
    data, _, props = pp.run_case([str(f)], None, pred.plans_manager,
                                 pred.configuration_manager, pred.dataset_json)
    orig = np.array(props["shape_before_cropping"], dtype=float)
    scale = np.array(data.shape[1:], dtype=float) / orig
    return data, (scale, props)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-cases", type=int, default=60)
    ap.add_argument("--cache", default=None)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    C5.USE_POS = True
    rows = [r for r in json.load(open(args.feat)) if r.get("gt_loc")]
    ves_axis, _ = C5.build_feature_axes(L.vessel_dense_names())

    cache = Path(args.cache) if args.cache else Path(args.feat).parent / "c14_enc_feats.json"
    if cache.exists():
        enc_map = json.load(open(cache))
        print(f"[c14] 캐시 로드 {len(enc_map)}")
    else:
        pred, net, feats = load_network()
        print("[c14] 네트워크 복원 완료 (CPU)")
        by_case = collections.defaultdict(list)
        for i, r in enumerate(rows):
            by_case[r["case"]].append((i, r))
        enc_map = {}
        cases = list(by_case)[:args.max_cases]
        for n, cid in enumerate(cases, 1):
            data, sc = preprocess_case(pred, cid)
            if data is None:
                continue
            scale, props = sc
            img = torch.from_numpy(data[None]).float()
            for i, r in by_case[cid]:
                cen = np.array(r["_cen"], dtype=float)
                vi = nib.load(L.DATA / "location_masks" / f"{cid}.nii.gz")
                sp = np.array(vi.header.get_zooms()[:3], dtype=float)
                vox = (cen / sp) * scale
                lo = np.round(vox - np.array(PATCH) / 2).astype(int)
                lo = np.clip(lo, 0, np.maximum(np.array(img.shape[2:]) - np.array(PATCH), 0))
                sl = (slice(None), slice(None)) + tuple(
                    slice(int(a), int(a) + p) for a, p in zip(lo, PATCH))
                patch = img[sl]
                if any(s < 16 for s in patch.shape[2:]):
                    continue
                with torch.no_grad():
                    net(patch)
                z = feats.get("z")
                if z is None:
                    continue
                enc_map[str(i)] = [float(x) for x in z.mean(dim=(2, 3, 4))[0].numpy()]
            if n % 5 == 0 or n == len(cases):
                print(f"  {n}/{len(cases)} 케이스, 특징 {len(enc_map)}", flush=True)
        json.dump(enc_map, open(cache, "w"))
        print(f"[c14] 인코더 특징 {len(enc_map)} -> {cache}")

    sub = [i for i in range(len(rows)) if str(i) in enc_map]
    if len(sub) < 40:
        print(f"[c14] 특징 보유 병변 {len(sub)}개 — 너무 적어 CV 생략"); return
    rows = [rows[i] for i in sub]
    E = np.array([enc_map[str(i)] for i in sub], dtype=float)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    print(f"[c14] CV 대상 병변 {len(rows)}, 인코더 차원 {E.shape[1]}\n")

    pm = C8.patient_map()
    groups = np.array([pm.get(r["case"], r["case"]) for r in rows])
    y = np.array([r["gt_loc"] for r in rows])

    def vec(k, mode):
        h = C5.row_to_vec(rows[k], ves_axis, mirror=False)
        if mode == "hand":
            return h
        if mode == "enc":
            return E[k]
        out = np.concatenate([h, E[k] * 0.5]); n = np.linalg.norm(out)
        return out / n if n > 0 else out

    print(f"{'피처':<20}{'top-1':>8}{'macroRec':>10}")
    for mode in ("hand", "enc", "both"):
        pred_ = np.empty(len(rows), dtype=object)
        for tr, te in GroupKFold(n_splits=args.folds).split(np.zeros(len(rows)), y, groups):
            X = np.array([vec(i, mode) for i in tr]); Y = y[tr]
            clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                         random_state=0, n_jobs=-1).fit(X, Y)
            prior = collections.Counter(Y)
            pri = np.array([prior[c] for c in clf.classes_], dtype=float)
            P = clf.predict_proba(np.array([vec(i, mode) for i in te])) / pri
            pred_[te] = clf.classes_[np.argmax(P, axis=1)]
        t1 = float(np.mean(y == pred_)); mr, _ = C8.macro_recall(y, pred_)
        print(f"{mode:<20}{t1:>8.3f}{mr:>10.3f}")
    print("[c14] 완료")


if __name__ == "__main__":
    main()
