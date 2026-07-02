import os
import subprocess
import time

import requests
import runpod

MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/models/model.gguf")
LLAMA_SERVER_PORT = int(os.environ.get("LLAMA_SERVER_PORT", "8080"))
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
N_GPU_LAYERS = os.environ.get("N_GPU_LAYERS", "999")
CONTEXT_SIZE = os.environ.get("CONTEXT_SIZE", "32768")
CHAT_TEMPLATE = os.environ.get("CHAT_TEMPLATE", "")

_llama_process = None


def start_llama_server():
    """Start the local llama-server and wait until it is ready."""
    global _llama_process

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Make sure the Network Volume is mounted to /mnt/models and MODEL_PATH is correct."
        )

    cmd = [
        "/app/llama-server",
        "-m", MODEL_PATH,
        "--host", "127.0.0.1",
        "--port", str(LLAMA_SERVER_PORT),
        "-ngl", N_GPU_LAYERS,
        "-c", CONTEXT_SIZE,
        "--parallel", "1",
        "--slots", "1",
    ]

    if CHAT_TEMPLATE:
        cmd.extend(["--chat-template", CHAT_TEMPLATE])

    # Redirect server output to a log file so the pipe cannot fill and block.
    log_file = open("/tmp/llama-server.log", "w", buffering=1)
    _llama_process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for llama-server to become healthy.
    for _ in range(120):
        try:
            response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print("llama-server is ready", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)

    # If we get here, the server did not start in time.
    raise RuntimeError(
        "llama-server failed to start within 120 seconds. "
        "Check /tmp/llama-server.log for details."
    )


def handler(event):
    """RunPod Serverless handler that proxies requests to llama-server."""
    if _llama_process is None:
        start_llama_server()

    input_data = event.get("input", {})

    payload = {
        "model": "local-model",
        "messages": input_data.get("messages", []),
        "temperature": input_data.get("temperature", 0.7),
        "top_p": input_data.get("top_p", 1.0),
        "max_tokens": input_data.get("max_tokens", 512),
        "stream": False,
    }

    response = requests.post(
        f"{LLAMA_SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    start_llama_server()
    runpod.serverless.start({"handler": handler})
