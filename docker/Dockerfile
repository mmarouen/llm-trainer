FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310
WORKDIR /app
ARG TASK_ARG
ARG PROJECT_ID_ARG
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV PYTHONUNBUFFERED=1
ENV TASK_DESCR=$TASK_ARG
ENV PYTHONPATH=/app
COPY requirements/ requirements/

# Override index
RUN pip install flash-attn --no-build-isolation
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    -r requirements/train-reqs.txt

COPY . .

ENTRYPOINT ["sh", "-c", "torchrun --nproc_per_node=1 scripts/train_chain.py --train-step ${TASK_DESCR}"]