# Build llama.cpp server from source with CUDA support.
# Using split RUN steps so that if one fails, the exact failing step is clear in the logs.
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

# Install build and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake python3 python3-pip curl ca-certificates libcurl4-openssl-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp with retries to tolerate transient network issues.
RUN set -e; \
    for i in 1 2 3; do \
        echo "Cloning llama.cpp (attempt $i)..."; \
        rm -rf /tmp/llama.cpp; \
        git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && break; \
        echo "Clone failed, retrying in 5s..."; \
        sleep 5; \
    done

# Configure the build for CUDA.
# Compile for common RunPod GPU architectures: A100(sm_80), A10G(sm_86), RTX 4090(sm_89), H100(sm_90)
RUN cd /tmp/llama.cpp && \
    cmake -B build \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

# Build only the llama-server target. Use -j4 to avoid OOM in the build environment.
RUN cd /tmp/llama.cpp && \
    cmake --build build --config Release --target llama-server -j4

# Install the binary and shared libraries into /app
RUN cd /tmp/llama.cpp && \
    cp build/bin/llama-server /app/llama-server && \
    (cp build/src/libllama.so /app/libllama.so 2>/dev/null || true) && \
    (cp build/ggml/src/libggml.so /app/libggml.so 2>/dev/null || true) && \
    rm -rf /tmp/llama.cpp

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1
ENV LLAMA_SERVER_PORT=8080
ENV MODEL_PATH=/mnt/models/model.gguf
ENV N_GPU_LAYERS=999
ENV CONTEXT_SIZE=32768

CMD ["python3", "-u", "handler.py"]
