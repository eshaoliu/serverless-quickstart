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

# Force Docker to rebuild this layer whenever the GitHub repo has a new commit.
ADD https://api.github.com/repos/eshaoliu/serverless-quickstart/commits?sha=main&per_page=1 /tmp/latest-commit.json

# Install Python dependencies. Debian 12 marks the system environment as
# externally-managed, so we need --break-system-packages in this single-purpose
# container.
COPY requirements.txt .
RUN python3 -m pip install --break-system-packages --no-cache-dir -r requirements.txt

# Copy the RunPod handler
COPY handler.py .

# Verify at build time that the handler is the expected Ollama version.
RUN grep -E "ollama serve|OLLAMA_MODELS" /app/handler.py && echo "handler.py is Ollama version"

# Use the RunPod cached model instead of baking the GGUF into the image.
# The model is resolved at runtime from /runpod-volume/huggingface-cache/hub.
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=127.0.0.1:11434
# OLLAMA_MODELS is set dynamically by handler.py next to the resolved GGUF
# so hard-links can share the same filesystem and avoid duplicating weights.
ENV MODEL_NAME=DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF
ENV MODEL_FILE=Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf
ENV OLLAMA_MODEL_NAME=runpod-model
# Disable runtime Ollama registry pulling; we load the cached GGUF instead.
ENV OLLAMA_PULL_MODEL=

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
