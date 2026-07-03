import glob
import os
import shutil
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
OLLAMA_PULL_MODEL = os.environ.get("OLLAMA_PULL_MODEL", "")

# Minimum free space (in GB) required before attempting an Ollama registry pull.
MIN_PULL_FREE_GB = int(os.environ.get("MIN_PULL_FREE_GB", "30"))

_ollama_process = None
_model_ready = False


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


def _free_space_gb(path: str) -> float:
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024**3)
    except Exception:
        return 0.0


def _find_gguf() -> str | None:
    """Return the GGUF file to use.

    Priority:
    1. MODEL_PATH if it exists.
    2. The only .gguf file under /mnt/models (recursively).
    3. None.
    """
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH

    ggufs = glob.glob("/mnt/models/**/*.gguf", recursive=True)
    ggufs = [f for f in ggufs if os.path.isfile(f)]
    if len(ggufs) == 1:
        print(
            f"MODEL_PATH {MODEL_PATH} not found; using discovered GGUF: {ggufs[0]}",
            flush=True,
        )
        return ggufs[0]
    if len(ggufs) > 1:
        print(
            f"Warning: found multiple GGUFs under /mnt/models: {ggufs}. "
            f"Set MODEL_PATH explicitly to choose one.",
            flush=True,
        )
    return None


def _pull_ollama_model() -> bool:
    """Pull a model from the Ollama registry if OLLAMA_PULL_MODEL is set."""
    if not OLLAMA_PULL_MODEL:
        return False

    free_gb = _free_space_gb(OLLAMA_MODELS)
    if free_gb < MIN_PULL_FREE_GB:
        print(
            f"Skipping ollama pull: only {free_gb:.1f} GB free at "
            f"{OLLAMA_MODELS}, need at least {MIN_PULL_FREE_GB} GB. "
            f"Mount a larger Network Volume at /mnt/models.",
            flush=True,
        )
        return False

    print(f"Pulling Ollama model {OLLAMA_PULL_MODEL}...", flush=True)
    result = subprocess.run(
        ["ollama", "pull", OLLAMA_PULL_MODEL],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=1800,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(
            f"ollama pull failed with exit code {result.returncode}",
            flush=True,
        )
        return False
    return True


def _create_model() -> bool:
    """Import or pull the model. Returns True on success, False otherwise."""
    global _model_ready

    gguf_path = _find_gguf()
    if gguf_path:
        os.makedirs(OLLAMA_MODELS, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="Modelfile", delete=False
        ) as modelfile:
            modelfile.write(f"FROM {gguf_path}\n")
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
            _model_ready = True
            return True
        except subprocess.CalledProcessError as exc:
            print(f"ollama create failed: {exc.output}", flush=True)
            return False
        finally:
            os.unlink(modelfile_path)

    if _pull_ollama_model():
        _model_ready = True
        return True

    print(
        "No model is available. Options:\n"
        "  1. Mount a Network Volume at /mnt/models with a .gguf file "
        "     (and set MODEL_PATH if the filename is not /mnt/models/model.gguf).\n"
        "  2. Set OLLAMA_PULL_MODEL to an Ollama registry model and ensure "
        "     /mnt/models has enough free space (>= 30 GB recommended).\n"
        "  3. If you already uploaded a GGUF with a different name, "
        "     the handler will auto-discover a single .gguf under /mnt/models.",
        flush=True,
    )
    return False


def start_ollama():
    """Start the local Ollama server and ensure the model is imported."""
    global _ollama_process, _model_ready

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

    if _model_exists():
        print(f"Ollama model {OLLAMA_MODEL_NAME} already exists.", flush=True)
        _model_ready = True
    else:
        gguf_path = _find_gguf()
        if gguf_path:
            print(
                f"Importing {gguf_path} into Ollama as {OLLAMA_MODEL_NAME}...",
                flush=True,
            )
        elif OLLAMA_PULL_MODEL:
            print(
                f"No local GGUF; attempting to pull {OLLAMA_PULL_MODEL}...",
                flush=True,
            )
        _create_model()


def handler(event):
    """RunPod Serverless handler that proxies requests to Ollama."""
    if _ollama_process is None:
        start_ollama()

    if not _model_ready:
        return {
            "error": (
                "Model is not loaded. Mount a Network Volume at /mnt/models "
                "containing the GGUF file, or set OLLAMA_PULL_MODEL with enough "
                "free disk space (>= 30 GB)."
            )
        }

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
