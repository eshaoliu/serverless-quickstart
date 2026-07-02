# Use the official pre-built llama.cpp CUDA server image as the base.
# This avoids compiling llama.cpp from source inside RunPod's build environment,
# which is prone to network/OOM failures.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b6795

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

# Find the pre-built llama-server binary inside the image and expose it
# at the path handler.py expects by default.
RUN set -e; \
    LLAMA_BIN=$(find / -name "llama-server" -type f 2>/dev/null | head -n1); \
    if [ -z "$LLAMA_BIN" ]; then \
        echo "ERROR: llama-server binary not found in base image"; \
        exit 1; \
    fi; \
    echo "Found llama-server at: $LLAMA_BIN"; \
    ln -s "$LLAMA_BIN" /app/llama-server

# Clear the inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

ENV PYTHONUNBUFFERED=1
ENV LLAMA_SERVER_PORT=8080
ENV MODEL_PATH=/mnt/models/model.gguf
ENV N_GPU_LAYERS=999
ENV CONTEXT_SIZE=32768

CMD ["python3", "-u", "handler.py"]
