# Reference only (local builds). CI uses crane append, see .github/workflows/build-image.yml.
# Mandated base image (shared League rules, Section 2): checked by LAYER IDENTITY.
# Do not change it, do not squash or flatten the image.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code model shipped inside the image, so startup never depends on a download.
ARG MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct
RUN python3 -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_ID}', local_dir='/models/llm', allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'])" \
 && rm -rf /root/.cache

ENV MC5_MODEL_DIR=/models/llm HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
COPY app/ /app/
RUN mkdir -p /app/input /app/output && chmod -R a+rwX /app/input /app/output
# No service, no port: the harness execs into the running container.
# The model server starts with the container (weights load inside the 10-minute
# startup budget) so no per-task process ever waits for a cold load.
CMD ["/bin/bash", "-c", "python3 /app/mc5/server.py >>/tmp/mc5-server.out 2>&1 & exec sleep infinity"]
