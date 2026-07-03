import glob
import hashlib
import os
import shutil
import subprocess
import tempfile
import time

import requests
import runpod

# Model resolution priority:
# 1. MODEL_PATH, if set and the file exists.
# 2. A RunPod cached Hugging Face model (MODEL_NAME).
# 3. A single .gguf discovered under /mnt/models.
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_NAME = os.environ.get(
    "MODEL_NAME",
    "DavidAU/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-NEO-CODE-Di-IMatrix-MAX-GGUF",
)
MODEL_FILE = os.environ.get("MODEL_FILE", "")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}"
OLLAMA_MODELS = os.environ.get("OLLAMA_MODELS", "/runpod-volume/.ollama")
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


def _sha256_file(path: str) -> str:
    """Return the hex sha256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_gguf_as_blob(gguf_path: str) -> str:
    """Make the GGUF reachable as an Ollama blob without copying it.

    Ollama stores model weights under $OLLAMA_MODELS/blobs/sha256-<digest>.
    By pre-computing the digest and hard-linking the original GGUF into that
    location, `ollama create` sees the blob already exists and skips the copy,
    so the Hugging Face cache GGUF is not duplicated on disk.
    """
    digest = _sha256_file(gguf_path)
    blob_dir = os.path.join(OLLAMA_MODELS, "blobs")
    os.makedirs(blob_dir, exist_ok=True)
    blob_path = os.path.join(blob_dir, f"sha256-{digest}")

    if os.path.exists(blob_path):
        print(f"Ollama blob already exists: {blob_path}", flush=True)
        return digest

    try:
        # Hard link shares the inode; no extra disk space is used.
        os.link(gguf_path, blob_path)
        print(
            f"Hard-linked GGUF to Ollama blob (no extra space): "
            f"{gguf_path} -> {blob_path}",
            flush=True,
        )
    except OSError as exc:
        # Fall back to a symlink if hard links are not possible (e.g. cross-fs).
        os.symlink(os.path.abspath(gguf_path), blob_path)
        print(
            f"Could not hard-link ({exc}); symlinked GGUF to Ollama blob: "
            f"{blob_path} -> {gguf_path}",
            flush=True,
        )
    return digest


def resolve_snapshot_path(model_id: str) -> str:
    """Resolve the local snapshot path for a RunPod cached Hugging Face model."""
    if "/" not in model_id:
        raise ValueError(f"MODEL_ID '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            return candidate

    if os.path.isdir(snapshots_dir):
        versions = [
            d
            for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            return os.path.join(snapshots_dir, versions[0])

    raise RuntimeError(f"Cached model not found: {model_id}")


def _pick_gguf_from_dir(directory: str) -> str | None:
    """Return a single .gguf from *directory*, preferring MODEL_FILE if set."""
    ggufs = glob.glob(os.path.join(directory, "*.gguf"))
    ggufs = [f for f in ggufs if os.path.isfile(f)]
    if not ggufs:
        return None

    if MODEL_FILE:
        for f in ggufs:
            if os.path.basename(f) == MODEL_FILE:
                return f
        print(
            f"MODEL_FILE {MODEL_FILE} not found in {directory}; "
            "using other GGUF(s).",
            flush=True,
        )

    if len(ggufs) == 1:
        return ggufs[0]

    print(
        f"Warning: found multiple GGUFs in {directory}: {ggufs}. "
        "Set MODEL_FILE to choose one explicitly.",
        flush=True,
    )
    return ggufs[0]


def _find_cached_gguf() -> str | None:
    """Return the GGUF path from a RunPod cached Hugging Face model, if available."""
    try:
        snapshot_dir = resolve_snapshot_path(MODEL_NAME)
    except Exception as exc:
        print(f"Could not resolve cached model {MODEL_NAME}: {exc}", flush=True)
        return None

    print(f"Resolved cached model snapshot: {snapshot_dir}", flush=True)
    return _pick_gguf_from_dir(snapshot_dir)


def _find_gguf() -> str | None:
    """Return the GGUF file to use.

    Priority:
    1. MODEL_PATH if it exists.
    2. The RunPod cached model for MODEL_NAME.
    3. The only .gguf file under /mnt/models (recursively).
    4. None.
    """
    if MODEL_PATH and os.path.isfile(MODEL_PATH):
        return MODEL_PATH

    cached_path = _find_cached_gguf()
    if cached_path:
        return cached_path

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

        # Stage the GGUF as an Ollama blob first so `ollama create` does not
        # duplicate the multi-GB weights file in $OLLAMA_MODELS/blobs.
        _stage_gguf_as_blob(gguf_path)

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
                timeout=1800,
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
        "  1. Set MODEL_PATH to a local GGUF file.\n"
        "  2. Configure a RunPod cached model (MODEL_NAME) in the endpoint "
        "     Model section and ensure /runpod-volume is mounted.\n"
        "  3. Mount a Network Volume at /mnt/models with a .gguf file.\n"
        "  4. Set OLLAMA_PULL_MODEL to an Ollama registry model with enough disk space.",
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
                "Model is not loaded. Set MODEL_PATH, configure a RunPod cached model, "
                "mount a Network Volume at /mnt/models, or set OLLAMA_PULL_MODEL."
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
