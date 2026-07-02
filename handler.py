import os
import subprocess
import tempfile
import time

import requests
import runpod

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/model.gguf")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}"
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", "/mnt/models/.ollama")
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "runpod-model")

_ollama_process = None


def _wait_for_ollama(timeout: int = 120):
    for _ in range(timeout):
        try:
            response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            if response.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _model_exists() -> bool:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        return any(m.get("name") == OLLAMA_MODEL_NAME for m in models)
    except Exception:
        return False


def _create_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Make sure the Network Volume is mounted to /mnt/models and MODEL_PATH is correct."
        )

    os.makedirs(OLLAMA_MODELS, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="Modelfile", delete=False
    ) as modelfile:
        modelfile.write(f"FROM {MODEL_PATH}\n")
        modelfile_path = modelfile.name

    try:
        result = subprocess.run(
            ["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            timeout=300,
        )
        print(result.stdout, flush=True)
    finally:
        os.unlink(modelfile_path)


def start_ollama():
    """Start the local Ollama server and ensure the model is imported."""
    global _ollama_process

    log_file = open("/tmp/ollama.log", "w", buffering=1)
    _ollama_process = subprocess.Popen(
        ["ollama", "serve"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "OLLAMA_MODELS": OLLAMA_MODELS},
    )

    if not _wait_for_ollama(120):
        raise RuntimeError(
            "Ollama server did not start within 120 seconds. "
            "Check /tmp/ollama.log for details."
        )

    if not _model_exists():
        print(f"Importing {MODEL_PATH} into Ollama as {OLLAMA_MODEL_NAME}...", flush=True)
        _create_model()
    else:
        print(f"Ollama model {OLLAMA_MODEL_NAME} already exists.", flush=True)


def handler(event):
    """RunPod Serverless handler that proxies requests to Ollama."""
    if _ollama_process is None:
        start_ollama()

    input_data = event.get("input", {})

    messages = input_data.get("messages", [])
    if not messages and input_data.get("prompt"):
        messages = [{"role": "user", "content": input_data["prompt"]}]

    payload = {
        "model": OLLAMA_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": input_data.get("temperature", 0.7),
            "top_p": input_data.get("top_p", 1.0),
            "num_predict": input_data.get("max_tokens", 512),
        },
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()

    # Normalize to an OpenAI-like response shape.
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": data.get("message", {}).get("role", "assistant"),
                    "content": data.get("message", {}).get("content", ""),
                },
                "finish_reason": "stop",
            }
        ],
        "model": OLLAMA_MODEL_NAME,
    }


if __name__ == "__main__":
    start_ollama()
    runpod.serverless.start({"handler": handler})
