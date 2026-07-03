import glob
import os

import runpod
from vllm import LLM, SamplingParams

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

# vLLM loading options.
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

_llm = None


def _resolve_snapshot_path(model_id: str) -> str:
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

    try:
        snapshot_dir = _resolve_snapshot_path(MODEL_NAME)
        print(f"Resolved cached model snapshot: {snapshot_dir}", flush=True)
        cached_path = _pick_gguf_from_dir(snapshot_dir)
        if cached_path:
            return cached_path
    except Exception as exc:
        print(f"Could not resolve cached model {MODEL_NAME}: {exc}", flush=True)

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


def _load_model() -> LLM:
    """Load the GGUF model with vLLM."""
    global _llm

    gguf_path = _find_gguf()
    if not gguf_path:
        raise RuntimeError(
            "No GGUF model is available. Options:\n"
            "  1. Set MODEL_PATH to a local GGUF file.\n"
            "  2. Configure a RunPod cached model (MODEL_NAME).\n"
            "  3. Mount a Network Volume at /mnt/models with a .gguf file."
        )

    print(f"Loading GGUF with vLLM: {gguf_path}", flush=True)
    print(
        f"vLLM options: tensor_parallel_size={TENSOR_PARALLEL_SIZE}, "
        f"gpu_memory_utilization={GPU_MEMORY_UTILIZATION}",
        flush=True,
    )

    _llm = LLM(
        model=gguf_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    print("vLLM model loaded.", flush=True)
    return _llm


def handler(event):
    """RunPod Serverless handler that uses vLLM for inference."""
    global _llm
    if _llm is None:
        _load_model()

    input_data = event.get("input", {})

    messages = input_data.get("messages", [])
    if not messages and input_data.get("prompt"):
        messages = [{"role": "user", "content": input_data["prompt"]}]

    sampling_params = SamplingParams(
        temperature=input_data.get("temperature", 0.7),
        top_p=input_data.get("top_p", 1.0),
        max_tokens=input_data.get("max_tokens", 512),
    )

    outputs = _llm.chat(messages, sampling_params)
    content = outputs[0].outputs[0].text

    # Normalize to an OpenAI-like response shape.
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
