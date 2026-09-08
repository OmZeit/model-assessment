FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime@sha256:c16f4c749e2d9e96878875cdf6cc45cddda1d1a36fddd371dd6f2360f1b6e2a2

WORKDIR /app
COPY requirements.lock.txt /app/requirements.lock.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.lock.txt

COPY weights /model
ARG TASK_HEAD_DIR=task_head
COPY ${TASK_HEAD_DIR} /head
COPY runtime.py train_head.py predict.py /app/

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/huggingface \
    HF_MODULES_CACHE=/tmp/huggingface/modules \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["python", "/app/predict.py"]
