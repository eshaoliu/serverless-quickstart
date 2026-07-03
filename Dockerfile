# Use Ollama as the inference engine.
# Ollama is based on llama.cpp and supports GGUF models via `ollama create` from a Modelfile.
# This avoids compiling llama.cpp from source inside RunPod's build environment.
FROM ollama/ollama:latest

USER root

# Install Python and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies. Debian 12 marks the system environment as
# externally-managed, so we need --break-system-packages in this single-purpose
# container.
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

# Bake the GGUF model into the image so the endpoint works without a Network Volume.
# The Q4_K_M quant matches the Ollama registry model that was being pulled at runtime.
ARG HF_MODEL_REPO=DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF
ARG HF_MODEL_FILE=Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf
RUN set -eux && \
    curl -L --fail --retry 3 --retry-delay 5 --connect-timeout 30 \
         -o /app/model.gguf \
         "https://huggingface.co/${HF_MODEL_REPO}/resolve/main/${HF_MODEL_FILE}" && \
    ls -lh /app/model.gguf

# Ollama listens on 127.0.0.1:11434 by default.
# Keep Ollama models on local image storage, not on the (optional) Network Volume.
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/app/.ollama
ENV MODEL_PATH=/app/model.gguf
ENV OLLAMA_MODEL_NAME=runpod-model
# Disable runtime Ollama registry pulling; we already ship the GGUF.
ENV OLLAMA_PULL_MODEL=

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
