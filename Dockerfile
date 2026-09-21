# TopAneu-26 Task 2 — 최고 성능 구성 (제출 통합본 + FRAC 0.35 + OUT_GROW 1.32 + min_vox 12)
#
# Grand Challenge 공식 템플릿(templates/task2/Dockerfile) 기반.
# 베이스는 검출기·혈관 모델을 검증한 torch 2.5.1 + cu121 로 고정한다
# (템플릿 기본 2.9.1/cu12.6 으로 바꾸면 nnU-Net 추론이 미세하게 달라질 수 있다).
#
# 빌드 전에 준비할 것 (저장소에는 없다 — README "가중치" 참고):
#   app/topaneu/code/sblee/nnunet/analysis/final_rf_seed3.pkl   (위치 분류 RF, 70 MB)
# 가중치(검출기·혈관·패치필터)는 이미지에 넣지 않고 GC "Models" 탈볼로 /opt/ml/model 에 마운트한다.
FROM --platform=linux/amd64 pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

COPY --chown=user:user app/requirements.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt

COPY --chown=user:user app/ /opt/app/
# 모델 탈볼은 /opt/ml/model 에 풀린다(최상위 = models/). 번들이 기대하는 위치에 링크한다.
RUN ln -s /opt/ml/model/models /opt/app/topaneu/models

# PYTHONPATH 는 전역으로 걸지 않는다 — topaneu_integrated.py 가 sblee 단계 서브프로세스에만
# vendor/Skeleton-Recall 을 넣고, 패치필터 서브프로세스에서는 뺀다.
ENV TOPANEU_BUNDLE=/opt/app/topaneu \
    nnUNet_raw=/opt/app/topaneu/nnunet/nnUNet_raw \
    nnUNet_preprocessed=/opt/app/topaneu/nnunet/nnUNet_preprocessed \
    nnUNet_results=/opt/app/models \
    TOPANEU_TIMING=1 \
    TOPANEU_DET_FOLDS=0,1,2 \
    TOPANEU_MIN_VOX=12 \
    TOPANEU_TOPK_FRAC=0.35 \
    TOPANEU_OUT_GROW=1.32

ENTRYPOINT ["python", "main.py"]
