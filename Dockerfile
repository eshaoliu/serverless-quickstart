# Use vLLM as the inference engine with experimental GGUF support.
# vLLM is optimized for throughput and supports GGUF via vllm-gguf-plugin.
FROM vllm/vllm-openai:latest

USER root

# Install Python dependencies.
# The base image already contains vLLM and CUDA; we add the experimental GGUF
# plugin plus the RunPod serverless SDK.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt vllm-gguf-plugin

WORKDIR /app

# Copy the RunPod handler
COPY handler.py .

# Use the RunPod cached model instead of baking the GGUF into the image.
# The model is resolved at runtime from /runpod-volume/huggingface-cache/hub.
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF
ENV MODEL_FILE=Qwen3.6-40B-Deck-Opus-NEO-CODE-HERE-2T-OT-Q4_K_M.gguf
# Optional vLLM tuning knobs.
ENV TENSOR_PARALLEL_SIZE=1
ENV GPU_MEMORY_UTILIZATION=0.9

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
