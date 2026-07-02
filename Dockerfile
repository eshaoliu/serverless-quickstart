FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

WORKDIR /app

# Install build and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git cmake python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp server with CUDA support.
# Compile for common RunPod GPU architectures: A100(sm_80), A10G(sm_86), RTX 4090(sm_89), H100(sm_90)
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp \
    && cd /tmp/llama.cpp \
    && cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
    && cmake --build build --config Release --target llama-server -j$(nproc) \
    && cp build/bin/llama-server /app/llama-server \
    && rm -rf /tmp/llama.cpp

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

ENV PYTHONUNBUFFERED=1
ENV LLAMA_SERVER_PORT=8080
ENV MODEL_PATH=/mnt/models/model.gguf
ENV N_GPU_LAYERS=999
ENV CONTEXT_SIZE=32768

CMD ["python3", "-u", "handler.py"]
