# Use SGLang as the inference engine.
FROM lmsysorg/sglang:latest

USER root

# Force Docker to rebuild this layer whenever the GitHub repo has a new commit.
ADD https://api.github.com/repos/eshaoliu/serverless-quickstart/commits?sha=main&per_page=1 /tmp/latest-commit.json

# Install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

# Copy the RunPod handler
COPY handler.py .

# Verify at build time that the handler is the expected SGLang version.
RUN grep -E "sglang.launch_server|SGLANG_PORT" /app/handler.py && echo "handler.py is SGLang version"

# Use the RunPod cached model instead of baking weights into the image.
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark
ENV MODEL_FILE=""
ENV SGLANG_PORT=30000
ENV TENSOR_PARALLEL_SIZE=1
# HuggingFace token for gated/private models.
ENV HF_TOKEN=""

# Clear any inherited ENTRYPOINT so CMD is interpreted as a plain command.
ENTRYPOINT []

CMD ["python3", "-u", "handler.py"]
