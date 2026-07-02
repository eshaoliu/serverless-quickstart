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

# Ollama listens on 127.0.0.1:11434 by default.
# Store imported models on the mounted Network Volume so they survive worker restarts.
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=127.0.0.1:11434
ENV OLLAMA_MODELS=/mnt/models/.ollama
ENV MODEL_PATH=/mnt/models/model.gguf
ENV OLLAMA_MODEL_NAME=runpod-model

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
