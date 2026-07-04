# Use SGLang as the inference engine.
FROM lmsysorg/sglang:latest

USER root

# Force Docker to rebuild this layer whenever the GitHub repo has a new commit.
ADD https://api.github.com/repos/eshaoliu/serverless-quickstart/commits?sha=main&per_page=1 /tmp/latest-commit.json

WORKDIR /app

# Ensure pip is usable and upgrade core tools.
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python dependencies.
# The sglang image marks the system env as externally-managed, so we must use
# --break-system-packages to install the RunPod SDK into it.
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

# Build-time verification (non-fatal): confirm handler contains SGLang markers.
RUN grep -E "sglang.launch_server|SGLANG_PORT" /app/handler.py && \
    echo "handler.py is SGLang version" || \
    echo "WARNING: handler.py SGLang marker not found"

# Use the RunPod cached model instead of baking weights into the image.
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark
ENV MODEL_FILE=""
ENV SGLANG_PORT=30000
ENV TENSOR_PARALLEL_SIZE=1
ENV TRUST_REMOTE_CODE=true
# NOTE: set HF_TOKEN via RunPod endpoint env vars, not here.

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
