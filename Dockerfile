# Use the official pre-built llama.cpp CUDA server image as the base.
# This avoids compiling llama.cpp from source inside RunPod's build environment,
# which is prone to network/OOM failures.
FROM ghcr.io/ggerganov/llama.cpp:server-cuda

USER root

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

# The official image places llama-server at /llama-server; make it available
# at the path handler.py expects by default.
RUN ln -s /llama-server /app/llama-server

# Clear the inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

ENV PYTHONUNBUFFERED=1
ENV LLAMA_SERVER_PORT=8080
ENV MODEL_PATH=/mnt/models/model.gguf
ENV N_GPU_LAYERS=999
ENV CONTEXT_SIZE=32768

CMD ["python3", "-u", "handler.py"]
