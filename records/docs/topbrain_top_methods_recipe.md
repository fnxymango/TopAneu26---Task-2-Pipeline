# TopBrain / TopCoW 상위팀 방법론 — 복제용 정리 (2026-07-27, 웹조사)

> 목적: 우리 **98 GT 혈관마스크로 vessel seg 학습 → skeleton 추출**을 DIY 말고 **상위팀 검증 방식 그대로** 복제.
> 출처: TopCoW summary(arXiv 2312.17670, 2023+2024), CoWBenchmark 공식 eval repo(코드 직접 확인), 상위팀 repo 3개. TopBrain 2025 per-team은 논문 접근제한(403)이라 coarse — TopCoW 2024와 연속으로 취급.
> ⚠️ 우리가 받은 `Team_KDH_2025_...tar.gz`는 **truncated(gzip EOF)** → 사용 불가. **하지만 우승팀 코드/가중치가 공개돼 있어 불필요.**

## 1. Architecture
- **모든 상위팀이 nnU-Net 기반** (summary 원문: *"All based their network architecture on nnUNet"*). 트랜스포머/MedNeXt/STU-Net 우승 없음.
- **CLAIM (2024 우승, MRA 1위/CTA 2위)**: **2-stage YOLO 검출→ nnU-Net `ResEncM` 3D full-res, 5-fold 앙상블**. → **ResEnc 방향 검증됨.**
- 상위팀 절반 이상이 **2-stage localize→segment**. 앙상블=nnU-Net 5-fold 표준.

## 2. Preprocessing / normalization
- **우승팀: 커스텀 전처리 없음** — raw CTA/TOF-MRA를 nnU-Net **fingerprint**에 맡김(CT=clip+global z-score, MR=per-image z-score).
- **CTA+MRA를 한 풀에 섞어 modality-agnostic 학습** — 2023→2024 최대 개선점.
- 👉 우리 adaptive-norm이 열세였던 것과 일치: **plain-z + 혼합학습이 정석.**

## 3. Data augmentation
- 조직위 baseline: **inter-modal registration**(CTA↔MRA 등록)으로 데이터 2배.
- **NoMirroring 필수** — 미러링은 좌/우 혈관 라벨을 뒤집음(vessel-specific correctness). DA5 같은 heavy-aug 불필요.

## 4. Loss (핵심 — topology/centerline-aware)
전원 **Dice+CE + topology 항**:
| loss | 팀 | 재사용 코드 |
|---|---|---|
| **Skeleton Recall Loss** | DKFZ(원조) + CLAIM(우승)·NIC-VICOROB·UZH | **`nnUNetTrainerSkeletonRecall`** — [MIC-DKFZ/Skeleton-Recall](https://github.com/MIC-DKFZ/Skeleton-Recall), arXiv 2404.03010. GT skeleton을 data-loading 때 미리 tube화 → soft recall. 저비용, `-tr` 한 플래그. |
| **cbDice** (centerline boundary Dice) | HITSZ(2023 CTA 2위) | `nnUNetTrainer_CE_DC_CBDC_NoMirroring_3d_fullres` — [PengchengShi1220/cbDice](https://github.com/PengchengShi1220/cbDice), arXiv 2407.01517. clDice+B-DoU도 포함. |

## 5. Post-processing
- **전역 largest-CC 금지**(라벨된 혈관 삭제됨). **클래스별 largest-CC + 인접성(valid-neighbor) 강제.**
- TopBrain은 "invalid neighbors" 오류를 점수화 → adjacency table 맞추면 직접 보상. (`cls_avg_invalid_neighbors.py` in TopBrain_Eval_Metrics)

## 6. Skeleton / centerline 추출 (공식 metric 코드 확인)
- clDice metric: **모든 혈관 라벨을 하나의 binary로 병합** → `skimage.morphology.skeletonize_3d`(Lee/Zhang thinning). **VMTK/kimimaro 아님.**
- Betti-0: `skimage.measure.label(connectivity=3)` (3D 26-connectivity).
- 👉 **우리 skeleton도 `skeletonize_3d`(binary mask)** 로 하면 챌린지와 완전 일치. (train-time soft skeleton은 SkelRecall/cbDice의 것 사용)

## 7. TTA / inference
- 표준 nnU-Net sliding-window + Gaussian + 5-fold. **미러링 TTA 끄기**(좌우 라벨). 주 트릭 = 2-stage ROI crop.

## 8. 재사용 repo
- **CLAIM 2024 우승(YOLO→ResEncM+SkelRecall) + 가중치**: [github](https://github.com/claim-berlin/TopCoW_2024_MRA_winning_solution) · [Zenodo 14191592](https://zenodo.org/records/14191592)
- **Skeleton Recall trainer**: https://github.com/MIC-DKFZ/Skeleton-Recall
- **cbDice/clDice trainer**: https://github.com/PengchengShi1220/cbDice
- **조직위 baseline(nnDetection→nnU-Net)**: https://github.com/fmusio/TopCoWSubmissions
- **공식 eval(clDice/Betti-0/HD95/F1/invalid-neighbors)**: TopBrain → https://github.com/CoWBenchmark/TopBrain_Eval_Metrics
- **데이터**: TopCoW [Zenodo 15692630] · TopBrain2025 [Zenodo 16878417]

---

## ▶ 우리 케이스 복제 레시피 (CLAIM 우승 방식 축소판, 98 CTA+MRA GT + 3090)
**그대로 채택(off-the-shelf):**
1. nnU-Net v2, **3D full-res, ResEnc(M/L) preset**, **CTA+MRA 혼합 단일 데이터셋**, 단일채널, **5-fold**.
2. **커스텀 전처리 없음** — nnU-Net fingerprint에 위임.
3. **loss = Skeleton Recall** (`-tr nnUNetTrainerSkeletonRecall`, repo `pip install -e .`). 대안 cbDice.
4. **미러링 TTA off**(좌/우 라벨이면). sliding-window+Gaussian+5-fold 유지.
5. **평가/skeleton**: 공식 `topbrain25_eval`(또는 topcow24) 설치 → clDice/Betti-0/HD95/F1 무료. skeleton = `skimage.skeletonize_3d`(병합 binary).

**직접 짜야(소량):**
6. (선택) stage-1 ROI localizer — whole-brain이면 생략 가능.
7. **클래스별 CC 후처리** + adjacency 강제 (~30–50줄, `skimage.measure.label` + eval repo adjacency JSON). turnkey 없음.
8. skeleton 산출물 = `skeletonize_3d`. 분기/반경 분석 필요시에만 kimimaro(TEASAR)/VMTK 추가(챌린지는 안 씀).

**불확실**: TopBrain 2025 per-team 방법·우승자명·리더보드(논문 접근제한). TopCoW 2024와 연속 취급.
