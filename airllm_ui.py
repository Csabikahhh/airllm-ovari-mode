"""
Local AirLLM web UI with hardware-aware defaults.

Run:
    python airllm_ui.py

Then open:
    http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import mimetypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
AIRLLM_SRC = PROJECT_ROOT / "air_llm"
UI_DIST = PROJECT_ROOT / "ui" / "dist"
if AIRLLM_SRC.exists():
    sys.path.insert(0, str(AIRLLM_SRC))

MODEL_PRESETS = [
    {
        "label": "Qwen 2.5 Coder 3B Instruct",
        "model_id": "Qwen/Qwen2.5-Coder-3B-Instruct",
        "family": "Qwen2",
        "size": "3B",
    },
    {
        "label": "Qwen 2.5 Coder 7B Instruct",
        "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "family": "Qwen2",
        "size": "7B",
    },
    {
        "label": "Qwen 2.5 3B Instruct",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "family": "Qwen2",
        "size": "3B",
    },
    {
        "label": "Qwen 2.5 7B Instruct",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "Qwen2",
        "size": "7B",
    },
    {
        "label": "Qwen 2.5 14B Instruct",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "family": "Qwen2",
        "size": "14B",
    },
    {
        "label": "Qwen 2.5 32B Instruct",
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "family": "Qwen2",
        "size": "32B",
    },
    {
        "label": "Qwen 2.5 72B Instruct",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "family": "Qwen2",
        "size": "72B",
    },
    {
        "label": "Llama 3.1 8B Instruct",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "Llama",
        "size": "8B",
    },
    {
        "label": "Mistral 7B Instruct v0.3",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "family": "Mistral",
        "size": "7B",
    },
    {
        "label": "Mixtral 8x7B Instruct",
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "family": "Mixtral",
        "size": "8x7B",
    },
    {
        "label": "ChatGLM3 6B Base",
        "model_id": "THUDM/chatglm3-6b-base",
        "family": "ChatGLM",
        "size": "6B",
    },
    {
        "label": "Baichuan2 7B Chat",
        "model_id": "baichuan-inc/Baichuan2-7B-Chat",
        "family": "Baichuan",
        "size": "7B",
    },
    {
        "label": "InternLM2.5 7B Chat",
        "model_id": "internlm/internlm2_5-7b-chat",
        "family": "InternLM",
        "size": "7B",
    },
]

SUPPORTED_FAMILIES = [
    "Qwen2 / Qwen2.5",
    "QWen",
    "Llama",
    "Mistral",
    "Mixtral",
    "ChatGLM",
    "Baichuan",
    "InternLM",
]

PROVIDER_PRESETS = [
    {
        "label": "Custom OpenAI-compatible",
        "base_url": "",
        "model": "",
        "needs_key": True,
    },
    {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "",
        "needs_key": True,
    },
    {
        "label": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "",
        "needs_key": True,
    },
    {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "",
        "needs_key": True,
    },
    {
        "label": "Together AI",
        "base_url": "https://api.together.xyz/v1",
        "model": "",
        "needs_key": True,
    },
    {
        "label": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "model": "",
        "needs_key": True,
    },
    {
        "label": "LM Studio",
        "base_url": "http://127.0.0.1:1234/v1",
        "model": "local-model",
        "needs_key": False,
    },
    {
        "label": "Ollama OpenAI-compatible",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "",
        "needs_key": False,
    },
]

MODEL_LOCK = threading.RLock()
MODEL_STATE: Dict[str, Any] = {
    "model": None,
    "config": None,
    "loaded_at": None,
    "load_seconds": None,
}
GENERATION_CANCEL = threading.Event()

AGENT_EXCLUDED_DIRS = {
    ".git",
    ".github",
    ".idea",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    "dist",
    "build",
}

AGENT_IMPORTANT_FILES = [
    "README.md",
    "README_UI.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "airllm_ui.py",
    "ui/src/App.tsx",
    "ui/package.json",
]

_TORCH = None
_TORCH_ERROR: Optional[str] = None
_TORCH_CHECKED = False


def get_torch() -> Tuple[Any, Optional[str]]:
    global _TORCH, _TORCH_ERROR, _TORCH_CHECKED
    if not _TORCH_CHECKED:
        try:
            import torch  # type: ignore

            _TORCH = torch
            _TORCH_ERROR = None
        except Exception as exc:  # pragma: no cover - environment dependent
            _TORCH = None
            _TORCH_ERROR = str(exc)
        _TORCH_CHECKED = True
    return _TORCH, _TORCH_ERROR


def import_bitsandbytes_ok() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except Exception:
        return False


def round_gb(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value / 1024 / 1024 / 1024, 2)


def get_memory_info() -> Dict[str, Optional[float]]:
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        return {"total_gb": round_gb(mem.total), "available_gb": round_gb(mem.available)}
    except Exception:
        pass

    if platform.system().lower() == "windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "total_gb": round_gb(status.ullTotalPhys),
                "available_gb": round_gb(status.ullAvailPhys),
            }

    return {"total_gb": None, "available_gb": None}


def get_disk_info() -> Dict[str, Any]:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    probe = hf_home
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "hf_home": str(hf_home),
        "total_gb": round_gb(usage.total),
        "free_gb": round_gb(usage.free),
    }


def run_command(args: list[str], timeout: int = 5) -> str:
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return ""
    return (result.stdout or result.stderr or "").strip()


def get_power_info() -> Dict[str, Any]:
    if platform.system().lower() != "windows":
        return {"active_scheme": None, "raw": None}
    output = run_command(["powercfg", "/getactivescheme"])
    active = None
    if ":" in output:
        active = output.split(":", 1)[1].strip()
    return {"active_scheme": active, "raw": output or None}


def get_network_info() -> Dict[str, Any]:
    if platform.system().lower() != "windows":
        return {"interface": None, "radio": None, "receive_mbps": None, "transmit_mbps": None, "signal": None}

    output = run_command(["netsh", "wlan", "show", "interfaces"])
    parsed: Dict[str, Any] = {
        "interface": None,
        "radio": None,
        "receive_mbps": None,
        "transmit_mbps": None,
        "signal": None,
    }
    key_map = {
        "name": "interface",
        "radio type": "radio",
        "receive rate (mbps)": "receive_mbps",
        "transmit rate (mbps)": "transmit_mbps",
        "signal": "signal",
    }
    for line in output.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        target = key_map.get(key.strip().lower())
        if not target:
            continue
        cleaned = value.strip()
        if target in {"receive_mbps", "transmit_mbps"}:
            try:
                parsed[target] = float(cleaned.replace(",", "."))
            except ValueError:
                parsed[target] = cleaned
        else:
            parsed[target] = cleaned
    return parsed


def get_cuda_info() -> Dict[str, Any]:
    torch, torch_error = get_torch()
    if torch is None:
        return {"available": False, "error": torch_error, "devices": []}

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        return {"available": False, "error": str(exc), "devices": []}

    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_gb": round_gb(props.total_memory),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
    return {"available": available, "error": None, "devices": devices}


def hardware_profile() -> Dict[str, Any]:
    torch, torch_error = get_torch()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": {
            "name": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count() or 1,
        },
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "power": get_power_info(),
        "network": get_network_info(),
        "cuda": get_cuda_info(),
        "torch": {
            "available": torch is not None,
            "version": getattr(torch, "__version__", None) if torch is not None else None,
            "error": torch_error,
        },
        "bitsandbytes": {"available": import_bitsandbytes_ok()},
        "supported_families": SUPPORTED_FAMILIES,
        "recommendation": recommended_settings(),
    }


def recommended_settings() -> Dict[str, Any]:
    torch, _ = get_torch()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    bnb_available = import_bitsandbytes_ok()
    vram_gb = 0.0

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            vram_gb = max(vram_gb, props.total_memory / 1024 / 1024 / 1024)

    if not cuda_available:
        return {
            "device": "cpu",
            "dtype": "float32",
            "compression": "none",
            "prefetching": False,
            "cleanup_interval": 1,
            "prefetch_workers": 1,
            "reinitialize_model_each_forward": False,
            "max_seq_len": 512,
            "max_new_tokens": 64,
        }

    max_seq_len = 512
    max_new_tokens = 96
    if vram_gb >= 7.5:
        max_seq_len = 1024
        max_new_tokens = 128
    if vram_gb >= 16:
        max_seq_len = 2048
        max_new_tokens = 192
    if vram_gb >= 24:
        max_seq_len = 4096
        max_new_tokens = 256

    compression = "4bit" if bnb_available else "none"
    return {
        "device": "cuda:0",
        "dtype": "float16",
        "compression": compression,
        "prefetching": compression == "none",
        "cleanup_interval": 4,
        "prefetch_workers": 1,
        "reinitialize_model_each_forward": False,
        "max_seq_len": max_seq_len,
        "max_new_tokens": max_new_tokens,
    }


def apply_runtime_optimizations() -> Dict[str, Any]:
    cpu_threads = max(1, os.cpu_count() or 1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(cpu_threads))

    applied = {
        "cpu_threads": cpu_threads,
        "interop_threads": None,
        "tf32": False,
        "cudnn_benchmark": False,
    }

    torch, _ = get_torch()
    if torch is None:
        return applied

    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass

    interop_threads = max(1, min(8, cpu_threads // 2 or 1))
    try:
        torch.set_num_interop_threads(interop_threads)
        applied["interop_threads"] = interop_threads
    except Exception:
        pass

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        applied["tf32"] = True
    except Exception:
        pass

    try:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        applied["cudnn_benchmark"] = True
    except Exception:
        pass

    return applied


def run_local_benchmark(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    applied = apply_runtime_optimizations()
    result: Dict[str, Any] = {
        "applied": applied,
        "gpu_matmul_ms": None,
        "cpu_matmul_ms": None,
        "disk_write_mbps": None,
        "disk_read_mbps": None,
        "model_probe": None,
    }

    torch, _ = get_torch()
    if torch is not None:
        try:
            if torch.cuda.is_available():
                device = "cuda:0"
                dtype = torch.float16
                size = 1536
                a = torch.randn((size, size), device=device, dtype=dtype)
                b = torch.randn((size, size), device=device, dtype=dtype)
                torch.cuda.synchronize()
                for _ in range(2):
                    _ = a @ b
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    _ = a @ b
                torch.cuda.synchronize()
                result["gpu_matmul_ms"] = round((time.perf_counter() - start) * 1000 / 5, 2)
                del a, b
                torch.cuda.empty_cache()

            size = 768
            a = torch.randn((size, size), dtype=torch.float32)
            b = torch.randn((size, size), dtype=torch.float32)
            for _ in range(2):
                _ = a @ b
            start = time.perf_counter()
            for _ in range(3):
                _ = a @ b
            result["cpu_matmul_ms"] = round((time.perf_counter() - start) * 1000 / 3, 2)
        except Exception as exc:
            result["compute_error"] = str(exc)

    try:
        cache_root = Path(payload.get("benchmark_path") or os.environ.get("HF_HOME") or tempfile.gettempdir())
        cache_root.mkdir(parents=True, exist_ok=True)
        probe = cache_root / "airllm_benchmark.tmp"
        chunk = os.urandom(8 * 1024 * 1024)
        repeats = 8
        total_bytes = len(chunk) * repeats

        start = time.perf_counter()
        with probe.open("wb") as handle:
            for _ in range(repeats):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        result["disk_write_mbps"] = round(total_bytes / 1024 / 1024 / max(time.perf_counter() - start, 1e-6), 2)

        start = time.perf_counter()
        with probe.open("rb") as handle:
            while handle.read(8 * 1024 * 1024):
                pass
        result["disk_read_mbps"] = round(total_bytes / 1024 / 1024 / max(time.perf_counter() - start, 1e-6), 2)
        probe.unlink(missing_ok=True)
    except Exception as exc:
        result["disk_error"] = str(exc)

    if parse_bool(payload.get("model_probe"), False) and current_status()["loaded"]:
        probe_payload = {
            "prompt": "Valaszolj egy rovid OK uzenettel.",
            "autoload": False,
            "max_length": 64,
            "max_new_tokens": 8,
            "temperature": 0,
            "use_cache": True,
            "use_chat_template": True,
        }
        try:
            result["model_probe"] = run_generation(probe_payload, probe_payload["prompt"])
        except Exception as exc:
            result["model_probe"] = {"error": str(exc)}

    return result


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(lower, min(upper, parsed))


def parse_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(lower, min(upper, parsed))


def resolve_device(value: str) -> str:
    torch, _ = get_torch()
    recommendation = recommended_settings()
    requested = (value or "auto").strip()
    if requested == "auto":
        return recommendation["device"]
    if requested.startswith("cuda") and not (torch is not None and torch.cuda.is_available()):
        raise RuntimeError("CUDA nem elerheto ebben a Python kornyezetben.")
    return requested


def resolve_dtype(value: str, device: str) -> Any:
    torch, torch_error = get_torch()
    if torch is None:
        raise RuntimeError(f"PyTorch nem importalhato: {torch_error}")

    requested = (value or "auto").strip()
    if requested == "auto":
        requested = "float16" if device.startswith("cuda") else "float32"

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested not in dtype_map:
        raise ValueError(f"Ismeretlen dtype: {requested}")
    return dtype_map[requested]


def resolve_compression(value: str, device: str) -> Optional[str]:
    requested = (value or "auto").strip().lower()
    if requested == "auto":
        recommended = recommended_settings()["compression"]
        requested = recommended
    if requested in {"none", "", "null"}:
        return None
    if requested not in {"4bit", "8bit"}:
        raise ValueError(f"Ismeretlen compression: {value}")
    if not device.startswith("cuda"):
        raise RuntimeError("A 4bit/8bit compression CUDA modban tamogatott.")
    if not import_bitsandbytes_ok():
        raise RuntimeError("A compression hasznalatahoz telepitheto/elerheto bitsandbytes kell.")
    return requested


def safe_public_config(config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if config is None:
        return None
    public = dict(config)
    public.pop("hf_token", None)
    public["hf_token_set"] = bool(config.get("hf_token"))
    return public


def current_status() -> Dict[str, Any]:
    with MODEL_LOCK:
        return {
            "loaded": MODEL_STATE["model"] is not None,
            "config": safe_public_config(MODEL_STATE["config"]),
            "loaded_at": MODEL_STATE["loaded_at"],
            "load_seconds": MODEL_STATE["load_seconds"],
        }


def selected_provider(payload: Dict[str, Any]) -> str:
    provider = str(payload.get("provider") or "local").strip().lower()
    if provider in {"", "airllm"}:
        return "local"
    if provider not in {"local", "openai_compatible"}:
        raise ValueError(f"Ismeretlen provider: {provider}")
    return provider


def is_external_provider(payload: Dict[str, Any]) -> bool:
    return selected_provider(payload) != "local"


def external_provider_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loaded": True,
        "config": {
            "provider": "openai_compatible",
            "base_url": payload.get("external_base_url"),
            "model_id": payload.get("external_model"),
            "api_key_set": bool(payload.get("external_api_key")),
        },
        "loaded_at": None,
        "load_seconds": None,
    }


def unload_model() -> Dict[str, Any]:
    with MODEL_LOCK:
        MODEL_STATE["model"] = None
        MODEL_STATE["config"] = None
        MODEL_STATE["loaded_at"] = None
        MODEL_STATE["load_seconds"] = None

    gc.collect()
    torch, _ = get_torch()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return current_status()


def cancel_generation() -> Dict[str, Any]:
    GENERATION_CANCEL.set()
    return {"cancel_requested": True}


def normalized_load_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("Adj meg egy Hugging Face model ID-t vagy helyi modell utvonalat.")

    device = resolve_device(payload.get("device", "auto"))
    dtype_name = (payload.get("dtype") or "auto").strip()
    compression = resolve_compression(payload.get("compression", "auto"), device)
    max_seq_len = parse_int(payload.get("max_seq_len"), recommended_settings()["max_seq_len"], 128, 32768)

    prefetch_value = (payload.get("prefetching") or "auto")
    if str(prefetch_value).strip().lower() == "auto":
        prefetching = device.startswith("cuda") and compression is None
    else:
        prefetching = parse_bool(prefetch_value)

    recommendation = recommended_settings()
    cleanup_interval = parse_int(
        payload.get("cleanup_interval"),
        int(recommendation.get("cleanup_interval", 4)),
        0,
        64,
    )
    prefetch_workers = parse_int(
        payload.get("prefetch_workers"),
        int(recommendation.get("prefetch_workers", 1)),
        1,
        4,
    )

    layer_path = (payload.get("layer_shards_saving_path") or "").strip() or None
    hf_token = (payload.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None

    return {
        "model_id": model_id,
        "device": device,
        "dtype": dtype_name,
        "compression": compression,
        "max_seq_len": max_seq_len,
        "prefetching": prefetching,
        "cleanup_interval": cleanup_interval,
        "prefetch_workers": prefetch_workers,
        "reinitialize_model_each_forward": parse_bool(payload.get("reinitialize_model_each_forward"), False),
        "profiling_mode": parse_bool(payload.get("profiling_mode"), False),
        "delete_original": parse_bool(payload.get("delete_original"), False),
        "layer_shards_saving_path": layer_path,
        "hf_token": hf_token,
    }


def load_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    apply_runtime_optimizations()
    config = normalized_load_config(payload)

    with MODEL_LOCK:
        existing = MODEL_STATE["config"]
        if MODEL_STATE["model"] is not None and existing == config:
            status = current_status()
            status["reused"] = True
            return status

        MODEL_STATE["model"] = None
        MODEL_STATE["config"] = None
        MODEL_STATE["loaded_at"] = None
        MODEL_STATE["load_seconds"] = None

        gc.collect()
        torch, _ = get_torch()
        if torch is not None:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        dtype = resolve_dtype(config["dtype"], config["device"])

        from airllm import AutoModel

        start = time.perf_counter()
        kwargs = {
            "device": config["device"],
            "dtype": dtype,
            "max_seq_len": config["max_seq_len"],
            "compression": config["compression"],
            "profiling_mode": config["profiling_mode"],
            "prefetching": config["prefetching"],
            "cleanup_interval": config["cleanup_interval"],
            "prefetch_workers": config["prefetch_workers"],
            "reinitialize_model_each_forward": config["reinitialize_model_each_forward"],
            "delete_original": config["delete_original"],
        }
        if config["layer_shards_saving_path"]:
            kwargs["layer_shards_saving_path"] = config["layer_shards_saving_path"]
        if config["hf_token"]:
            kwargs["hf_token"] = config["hf_token"]

        model = AutoModel.from_pretrained(config["model_id"], **kwargs)
        elapsed = time.perf_counter() - start

        MODEL_STATE["model"] = model
        MODEL_STATE["config"] = config
        MODEL_STATE["loaded_at"] = time.time()
        MODEL_STATE["load_seconds"] = round(elapsed, 2)

        status = current_status()
        status["reused"] = False
        return status


def normalize_messages(messages: Any) -> list[Dict[str, str]]:
    if not isinstance(messages, list):
        return []

    normalized = []
    for item in messages[-30:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def messages_to_prompt(messages: list[Dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}:\n{message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def tokenize_prompt(
    model: Any,
    prompt: str,
    max_length: int,
    use_chat_template: bool,
    messages: Optional[list[Dict[str, str]]] = None,
) -> Any:
    tokenizer = model.tokenizer
    torch, _ = get_torch()

    if messages and use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if input_ids.shape[-1] > max_length:
                input_ids = input_ids[:, -max_length:]
            return input_ids
        except Exception:
            pass

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if input_ids.shape[-1] > max_length:
                input_ids = input_ids[:, -max_length:]
            return input_ids
        except Exception:
            pass

    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = inputs["input_ids"]
    if torch is not None and input_ids.shape[-1] > max_length:
        input_ids = input_ids[:, -max_length:]
    return input_ids


def generation_settings(payload: Dict[str, Any], model: Any, config: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    max_model_len = int(getattr(model, "max_seq_len", config.get("max_seq_len", 512)))
    max_length = parse_int(payload.get("max_length"), min(512, max_model_len), 16, max_model_len)
    max_new_tokens = parse_int(payload.get("max_new_tokens"), recommended_settings()["max_new_tokens"], 1, 4096)
    temperature = parse_float(payload.get("temperature"), 0.7, 0.0, 2.0)
    top_p = parse_float(payload.get("top_p"), 0.9, 0.05, 1.0)
    top_k = parse_int(payload.get("top_k"), 50, 0, 1000)
    repetition_penalty = parse_float(payload.get("repetition_penalty"), 1.05, 0.8, 2.0)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "use_cache": parse_bool(payload.get("use_cache"), True),
        "return_dict_in_generate": True,
        "repetition_penalty": repetition_penalty,
    }
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class CancelStoppingCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):  # type: ignore[no-untyped-def]
                return GENERATION_CANCEL.is_set()

        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([CancelStoppingCriteria()])
    except Exception:
        pass
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
    else:
        generation_kwargs["do_sample"] = False
    return max_length, generation_kwargs


def run_generation(
    payload: Dict[str, Any],
    prompt: str,
    messages: Optional[list[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    if parse_bool(payload.get("autoload"), True):
        load_payload = payload.get("load_config") or payload
        load_model(load_payload)

    with MODEL_LOCK:
        model = MODEL_STATE["model"]
        config = MODEL_STATE["config"]
        if model is None or config is None:
            raise RuntimeError("Nincs betoltott modell.")

        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Adj meg promptot.")

        torch, torch_error = get_torch()
        if torch is None:
            raise RuntimeError(f"PyTorch nem importalhato: {torch_error}")

        use_chat_template = parse_bool(payload.get("use_chat_template"), True)

        max_length, generation_kwargs = generation_settings(payload, model, config)
        input_ids = tokenize_prompt(model, prompt, max_length, use_chat_template, messages=messages)
        input_ids = input_ids.to(config["device"])

        GENERATION_CANCEL.clear()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(input_ids, **generation_kwargs)
        elapsed = time.perf_counter() - start
        cancelled = GENERATION_CANCEL.is_set()
        GENERATION_CANCEL.clear()

        sequence = output.sequences[0]
        new_tokens = sequence[input_ids.shape[-1] :]
        text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_text = model.tokenizer.decode(sequence, skip_special_tokens=True)

        return {
            "text": text.strip() or full_text.strip(),
            "full_text": full_text.strip(),
            "seconds": round(elapsed, 2),
            "input_tokens": int(input_ids.shape[-1]),
            "output_tokens": int(sequence.shape[-1] - input_ids.shape[-1]),
            "cancelled": cancelled,
            "status": current_status(),
        }


def external_chat_request(payload: Dict[str, Any], messages: list[Dict[str, str]]) -> Dict[str, Any]:
    base_url = str(payload.get("external_base_url") or "").strip()
    model = str(payload.get("external_model") or "").strip()
    api_key = str(
        payload.get("external_api_key")
        or os.environ.get("AI_PROVIDER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()

    if not base_url:
        raise ValueError("Add meg a kulso provider base URL-t.")
    if not model:
        raise ValueError("Add meg a kulso provider model nevet.")

    if base_url.rstrip("/").endswith("/chat/completions"):
        endpoint = base_url.rstrip("/")
    else:
        endpoint = base_url.rstrip("/") + "/chat/completions"

    max_tokens = parse_int(payload.get("max_new_tokens"), recommended_settings()["max_new_tokens"], 1, 4096)
    temperature = parse_float(payload.get("temperature"), 0.7, 0.0, 2.0)
    top_p = parse_float(payload.get("top_p"), 0.9, 0.05, 1.0)
    timeout = parse_int(payload.get("external_timeout"), 120, 10, 600)

    request_body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }
    encoded = json.dumps(request_body).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(endpoint, data=encoded, headers=headers, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Kulso provider HTTP {exc.code}: {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Kulso provider nem elerheto: {exc.reason}") from exc

    elapsed = time.perf_counter() - start
    data = json.loads(raw)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("A kulso provider nem adott vissza choices mezot.")
    first = choices[0]
    message = first.get("message") or {}
    text = message.get("content") or first.get("text") or ""
    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)

    usage = data.get("usage") or {}
    return {
        "text": str(text).strip(),
        "full_text": str(text).strip(),
        "seconds": round(elapsed, 2),
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "status": external_provider_status({**payload, "external_api_key": api_key}),
        "provider": "openai_compatible",
        "raw_model": data.get("model") or model,
    }


def external_generate_text(payload: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    return external_chat_request(payload, messages)


def generate_text(payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Adj meg promptot.")
    if is_external_provider(payload):
        return external_generate_text(payload, prompt)
    return run_generation(payload, prompt)


def chat_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = normalize_messages(payload.get("messages"))
    if not messages:
        raise ValueError("A chathez legalabb egy user uzenet kell.")
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("A chathez legalabb egy user uzenet kell.")

    if messages[0]["role"] != "system":
        system_prompt = str(
            payload.get(
                "system_prompt",
                "You are a concise AI assistant. Answer in Hungarian unless the user asks otherwise.",
            )
        ).strip()
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

    if is_external_provider(payload):
        result = external_chat_request(payload, messages)
        return {
            **result,
            "message": {"role": "assistant", "content": result["text"]},
        }

    prompt = messages_to_prompt(messages)
    result = run_generation(payload, prompt, messages=messages)
    return {
        **result,
        "message": {"role": "assistant", "content": result["text"]},
    }


def resolve_workspace(path_value: Any) -> Path:
    raw_path = str(path_value or PROJECT_ROOT).strip() or str(PROJECT_ROOT)
    workspace = Path(raw_path)
    if not workspace.is_absolute():
        workspace = (PROJECT_ROOT / workspace).resolve()
    else:
        workspace = workspace.resolve()
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError(f"A workspace nem letezik vagy nem konyvtar: {workspace}")
    return workspace


def safe_read_text(path: Path, limit: int = 6000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[Nem olvashato: {exc}]"
    if len(data) > limit:
        return data[:limit] + "\n[...truncated...]"
    return data


def workspace_tree(workspace: Path, max_files: int = 120) -> list[str]:
    files: list[str] = []
    for root, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [dirname for dirname in dirnames if dirname not in AGENT_EXCLUDED_DIRS]
        relative_root = Path(root).relative_to(workspace)
        for filename in sorted(filenames):
            if filename.endswith((".pyc", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ipynb")):
                continue
            rel = relative_root / filename if str(relative_root) != "." else Path(filename)
            files.append(rel.as_posix())
            if len(files) >= max_files:
                return files
    return files


def git_status_text(workspace: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return f"[git status nem elerheto: {exc}]"
    return result.stdout.strip() or "clean"


def collect_agent_context(workspace: Path, max_chars: int) -> Dict[str, Any]:
    tree = workspace_tree(workspace)
    sections = [
        f"WORKSPACE: {workspace}",
        "GIT STATUS:",
        git_status_text(workspace),
        "FILES:",
        "\n".join(tree),
    ]

    used_files = []
    remaining = max_chars - sum(len(section) for section in sections)
    for rel in AGENT_IMPORTANT_FILES:
        if remaining <= 1000:
            break
        path = workspace / rel
        if not path.exists() or not path.is_file():
            continue
        limit = max(1000, min(6000, remaining // 2))
        content = safe_read_text(path, limit=limit)
        block = f"\n\nFILE: {rel}\n```text\n{content}\n```"
        sections.append(block)
        used_files.append(rel)
        remaining -= len(block)

    text = "\n".join(sections)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...context truncated...]"
    return {"text": text, "files": tree, "included_files": used_files}


def coding_agent_prompt(objective: str, context: str) -> str:
    return f"""You are a coding agent running inside an AirLLM desktop project.

Rules:
- Answer in Hungarian unless code or filenames require English.
- Be precise and practical.
- Use the provided workspace context only; say when more file content is needed.
- Do not claim that you executed commands or changed files.
- Prefer small, reviewable changes.
- If you propose code edits, include unified diff style patches or exact file sections.

User objective:
{objective}

Workspace context:
{context}

Return sections:
1. Rovid helyzetkep
2. Terv
3. Javasolt modositasok vagy patch
4. Ellenorzes / tesztek
"""


def run_coding_agent(payload: Dict[str, Any]) -> Dict[str, Any]:
    objective = str(payload.get("objective") or "").strip()
    if not objective:
        raise ValueError("Add meg, mit csinaljon a coding agent.")

    workspace = resolve_workspace(payload.get("workspace_path"))
    max_context_chars = parse_int(payload.get("max_context_chars"), 16000, 4000, 64000)
    context = collect_agent_context(workspace, max_context_chars)
    prompt = coding_agent_prompt(objective, context["text"])

    agent_payload = {
        **payload,
        "prompt": prompt,
        "autoload": parse_bool(payload.get("autoload"), True),
        "use_chat_template": parse_bool(payload.get("use_chat_template"), True),
        "temperature": payload.get("temperature", 0.2),
        "top_p": payload.get("top_p", 0.9),
        "max_new_tokens": payload.get("max_new_tokens", 900),
    }
    if is_external_provider(agent_payload):
        result = external_chat_request(
            agent_payload,
            [
                {
                    "role": "system",
                    "content": "You are a precise coding agent. Answer in Hungarian and do not claim to edit files.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    else:
        result = run_generation(agent_payload, prompt)
    return {
        **result,
        "workspace": str(workspace),
        "context_files": context["files"],
        "included_files": context["included_files"],
        "objective": objective,
    }


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, payload: str, content_type: str = "text/html; charset=utf-8") -> None:
    body = payload.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def file_response(handler: BaseHTTPRequestHandler, path: Path) -> None:
    body = path.read_bytes()
    content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def try_static_response(handler: BaseHTTPRequestHandler, request_path: str) -> bool:
    if not (UI_DIST / "index.html").exists():
        return False

    relative = request_path.lstrip("/") or "index.html"
    candidate = (UI_DIST / relative).resolve()
    root = UI_DIST.resolve()
    if root != candidate and root not in candidate.parents:
        json_response(handler, {"error": "Not found"}, 404)
        return True

    if candidate.is_file():
        file_response(handler, candidate)
        return True

    file_response(handler, UI_DIST / "index.html")
    return True


def read_json(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length > 5 * 1024 * 1024:
        raise ValueError("Tul nagy keres.")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


class AirLLMHandler(BaseHTTPRequestHandler):
    server_version = "AirLLMUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/"):
                if path == "/api/hardware":
                    json_response(self, hardware_profile())
                elif path == "/api/presets":
                    json_response(self, {"presets": MODEL_PRESETS, "families": SUPPORTED_FAMILIES})
                elif path == "/api/providers":
                    json_response(self, {"providers": PROVIDER_PRESETS})
                elif path == "/api/status":
                    json_response(self, current_status())
                else:
                    json_response(self, {"error": "Not found"}, 404)
            elif try_static_response(self, path):
                return
            elif path == "/":
                text_response(self, HTML)
            else:
                json_response(self, {"error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"error": str(exc), "traceback": traceback.format_exc()}, 500)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            payload = read_json(self)
            if path == "/api/load":
                json_response(self, load_model(payload))
            elif path == "/api/generate":
                json_response(self, generate_text(payload))
            elif path == "/api/chat":
                json_response(self, chat_completion(payload))
            elif path == "/api/agent/run":
                json_response(self, run_coding_agent(payload))
            elif path == "/api/unload":
                json_response(self, unload_model())
            elif path == "/api/cancel":
                json_response(self, cancel_generation())
            elif path == "/api/optimize":
                json_response(self, apply_runtime_optimizations())
            elif path == "/api/benchmark":
                json_response(self, run_local_benchmark(payload))
            else:
                json_response(self, {"error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"error": str(exc), "traceback": traceback.format_exc()}, 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


HTML = r"""<!doctype html>
<html lang="hu">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AirLLM Control</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #eef2f4;
      --ink: #15191d;
      --muted: #626c77;
      --line: #d7dde3;
      --accent: #0f766e;
      --accent-2: #155e75;
      --danger: #b42318;
      --ok: #167647;
      --warn: #a15c07;
      --shadow: 0 14px 40px rgba(18, 31, 45, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 14px 22px;
      border-bottom: 1px solid var(--line);
      background: rgba(246, 247, 249, .94);
      backdrop-filter: blur(14px);
    }
    h1 {
      margin: 0;
      font-size: 19px;
      font-weight: 760;
      letter-spacing: 0;
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 18px;
      width: min(1480px, 100%);
      margin: 0 auto;
      padding: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .left, .right {
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .block {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .block:last-child { border-bottom: 0; }
    .block-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 12px;
      font-size: 13px;
      font-weight: 760;
      text-transform: uppercase;
      color: #2a333c;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
      font-size: 12px;
      font-weight: 650;
      color: #303943;
    }
    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 14px;
      min-height: 38px;
      padding: 8px 10px;
      outline: none;
    }
    textarea {
      min-height: 220px;
      resize: vertical;
      line-height: 1.45;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .12);
    }
    button {
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      font: inherit;
      font-size: 14px;
      font-weight: 720;
      min-height: 38px;
      padding: 8px 12px;
      cursor: pointer;
    }
    button.secondary { background: var(--accent-2); }
    button.ghost {
      background: transparent;
      color: var(--accent-2);
      border: 1px solid var(--line);
    }
    button.danger { background: var(--danger); }
    button:disabled {
      opacity: .58;
      cursor: wait;
    }
    .actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--warn);
      flex: 0 0 auto;
    }
    .dot.ok { background: var(--ok); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      padding: 9px 10px;
      min-height: 58px;
    }
    .metric b {
      display: block;
      margin-bottom: 3px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 680;
    }
    .metric span {
      display: block;
      overflow-wrap: anywhere;
      font-size: 14px;
      font-weight: 720;
    }
    .output {
      min-height: 320px;
      padding: 16px;
      background: #111820;
      color: #eff6ff;
      border-radius: 7px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.5;
      font-size: 15px;
    }
    .log {
      min-height: 90px;
      max-height: 170px;
      overflow: auto;
      padding: 10px;
      border-radius: 6px;
      background: #f0f3f6;
      color: #27323c;
      font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      line-height: 1.45;
      white-space: pre-wrap;
    }
    .toggle-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 14px;
    }
    .check {
      flex-direction: row;
      align-items: center;
      gap: 8px;
      min-height: 32px;
    }
    .check input {
      width: 17px;
      min-height: 17px;
    }
    @media (max-width: 920px) {
      main { grid-template-columns: 1fr; padding: 12px; }
      header { padding: 12px; align-items: flex-start; flex-direction: column; }
    }
    @media (max-width: 560px) {
      .grid, .metrics, .toggle-row { grid-template-columns: 1fr; }
      textarea { min-height: 170px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>AirLLM Control</h1>
    <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Betoltes...</span></div>
  </header>
  <main>
    <section class="left">
      <div class="block">
        <div class="block-title">Hardver</div>
        <div id="hardware" class="metrics"></div>
      </div>
      <div class="block">
        <div class="block-title">Modell</div>
        <div class="grid">
          <label style="grid-column: 1 / -1;">Preset
            <select id="preset"></select>
          </label>
          <label style="grid-column: 1 / -1;">Model ID / utvonal
            <input id="model_id" spellcheck="false" />
          </label>
          <label>Device
            <select id="device">
              <option value="auto">auto</option>
              <option value="cuda:0">cuda:0</option>
              <option value="cpu">cpu</option>
            </select>
          </label>
          <label>Dtype
            <select id="dtype">
              <option value="auto">auto</option>
              <option value="float16">float16</option>
              <option value="bfloat16">bfloat16</option>
              <option value="float32">float32</option>
            </select>
          </label>
          <label>Compression
            <select id="compression">
              <option value="auto">auto</option>
              <option value="none">none</option>
              <option value="4bit">4bit</option>
              <option value="8bit">8bit</option>
            </select>
          </label>
          <label>Prefetching
            <select id="prefetching">
              <option value="auto">auto</option>
              <option value="true">on</option>
              <option value="false">off</option>
            </select>
          </label>
          <label>Max seq len
            <input id="max_seq_len" type="number" min="128" max="32768" step="128" />
          </label>
          <label>Layer cache
            <input id="layer_shards_saving_path" placeholder="" />
          </label>
          <label style="grid-column: 1 / -1;">HF token
            <input id="hf_token" type="password" autocomplete="off" />
          </label>
        </div>
        <div class="toggle-row" style="margin-top: 10px;">
          <label class="check"><input id="profiling_mode" type="checkbox" />Profiling</label>
          <label class="check"><input id="delete_original" type="checkbox" />Delete original</label>
        </div>
        <div class="actions" style="margin-top: 12px;">
          <button id="loadBtn">Betolt</button>
          <button id="unloadBtn" class="danger">Kiurit</button>
          <button id="optBtn" class="ghost">Optimalizal</button>
        </div>
      </div>
      <div class="block">
        <div class="block-title">Napló</div>
        <div id="log" class="log"></div>
      </div>
    </section>
    <section class="right">
      <div class="block">
        <div class="block-title">Prompt</div>
        <textarea id="prompt" spellcheck="true">Szia! Foglald ossze roviden, mire jo az AirLLM.</textarea>
      </div>
      <div class="block">
        <div class="block-title">Generálás</div>
        <div class="grid">
          <label>Input max
            <input id="max_length" type="number" min="16" max="32768" step="16" value="512" />
          </label>
          <label>New tokens
            <input id="max_new_tokens" type="number" min="1" max="4096" step="1" value="128" />
          </label>
          <label>Temperature
            <input id="temperature" type="number" min="0" max="2" step="0.05" value="0.7" />
          </label>
          <label>Top p
            <input id="top_p" type="number" min="0.05" max="1" step="0.01" value="0.9" />
          </label>
          <label>Top k
            <input id="top_k" type="number" min="0" max="1000" step="1" value="50" />
          </label>
          <label>Repeat penalty
            <input id="repetition_penalty" type="number" min="0.8" max="2" step="0.01" value="1.05" />
          </label>
        </div>
        <div class="toggle-row" style="margin-top: 10px;">
          <label class="check"><input id="autoload" type="checkbox" checked />Autoload</label>
          <label class="check"><input id="use_cache" type="checkbox" checked />KV cache</label>
          <label class="check"><input id="use_chat_template" type="checkbox" checked />Chat template</label>
        </div>
        <div class="actions" style="margin-top: 12px;">
          <button id="generateBtn" class="secondary">General</button>
        </div>
      </div>
      <div class="block">
        <div class="block-title">Kimenet</div>
        <div id="output" class="output"></div>
      </div>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const logEl = $("log");
    let presets = [];

    function log(message) {
      const stamp = new Date().toLocaleTimeString();
      logEl.textContent = `[${stamp}] ${message}\n` + logEl.textContent;
    }

    function setBusy(busy) {
      for (const id of ["loadBtn", "unloadBtn", "optBtn", "generateBtn"]) {
        $(id).disabled = busy;
      }
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options,
      });
      const data = await res.json();
      if (!res.ok) {
        const err = new Error(data.error || "Hiba");
        err.traceback = data.traceback;
        throw err;
      }
      return data;
    }

    function metric(label, value) {
      return `<div class="metric"><b>${label}</b><span>${value ?? "-"}</span></div>`;
    }

    async function refreshHardware() {
      const hw = await api("/api/hardware");
      const gpu = hw.cuda.available && hw.cuda.devices.length
        ? hw.cuda.devices.map(d => `${d.name} (${d.total_memory_gb} GB)`).join(", ")
        : "nincs CUDA";
      $("hardware").innerHTML = [
        metric("CPU", `${hw.cpu.logical_cores} szal`),
        metric("RAM", `${hw.memory.available_gb ?? "?"} / ${hw.memory.total_gb ?? "?"} GB`),
        metric("GPU", gpu),
        metric("Torch", hw.torch.version || "nem elerheto"),
        metric("bitsandbytes", hw.bitsandbytes.available ? "elerheto" : "nem elerheto"),
        metric("HF cache", `${hw.disk.free_gb ?? "?"} GB szabad`),
      ].join("");

      const rec = hw.recommendation;
      $("device").value = "auto";
      $("dtype").value = "auto";
      $("compression").value = "auto";
      $("prefetching").value = "auto";
      $("max_seq_len").value = rec.max_seq_len;
      $("max_new_tokens").value = rec.max_new_tokens;
      $("max_length").value = Math.min(512, rec.max_seq_len);
      log(`ajanlott: ${rec.device}, ${rec.dtype}, compression=${rec.compression}, max_seq_len=${rec.max_seq_len}`);
    }

    async function refreshPresets() {
      const data = await api("/api/presets");
      presets = data.presets;
      $("preset").innerHTML = `<option value="">Egyedi model ID</option>` +
        presets.map((p, i) => `<option value="${i}">${p.label} - ${p.family}</option>`).join("");
      $("preset").value = "0";
      $("model_id").value = presets[0].model_id;
    }

    async function refreshStatus() {
      const status = await api("/api/status");
      $("statusDot").className = status.loaded ? "dot ok" : "dot";
      $("statusText").textContent = status.loaded
        ? `Betoltve: ${status.config.model_id} (${status.load_seconds}s)`
        : "Nincs betoltott modell";
    }

    function loadConfig() {
      return {
        model_id: $("model_id").value,
        device: $("device").value,
        dtype: $("dtype").value,
        compression: $("compression").value,
        prefetching: $("prefetching").value,
        max_seq_len: $("max_seq_len").value,
        layer_shards_saving_path: $("layer_shards_saving_path").value,
        hf_token: $("hf_token").value,
        profiling_mode: $("profiling_mode").checked,
        delete_original: $("delete_original").checked,
      };
    }

    function generationConfig() {
      return {
        ...loadConfig(),
        load_config: loadConfig(),
        prompt: $("prompt").value,
        max_length: $("max_length").value,
        max_new_tokens: $("max_new_tokens").value,
        temperature: $("temperature").value,
        top_p: $("top_p").value,
        top_k: $("top_k").value,
        repetition_penalty: $("repetition_penalty").value,
        autoload: $("autoload").checked,
        use_cache: $("use_cache").checked,
        use_chat_template: $("use_chat_template").checked,
      };
    }

    $("preset").addEventListener("change", () => {
      const idx = $("preset").value;
      if (idx !== "") $("model_id").value = presets[Number(idx)].model_id;
    });

    $("loadBtn").addEventListener("click", async () => {
      setBusy(true);
      log("modell betoltese indul");
      try {
        const status = await api("/api/load", {method: "POST", body: JSON.stringify(loadConfig())});
        log(status.reused ? "mar be volt toltve" : `betoltes kesz: ${status.load_seconds}s`);
        await refreshStatus();
      } catch (err) {
        log(err.message);
        if (err.traceback) console.error(err.traceback);
      } finally {
        setBusy(false);
      }
    });

    $("unloadBtn").addEventListener("click", async () => {
      setBusy(true);
      try {
        await api("/api/unload", {method: "POST", body: "{}"});
        $("output").textContent = "";
        log("modell kiuritve");
        await refreshStatus();
      } catch (err) {
        log(err.message);
      } finally {
        setBusy(false);
      }
    });

    $("optBtn").addEventListener("click", async () => {
      setBusy(true);
      try {
        const data = await api("/api/optimize", {method: "POST", body: "{}"});
        log(`optimalizalva: cpu_threads=${data.cpu_threads}, tf32=${data.tf32}`);
      } catch (err) {
        log(err.message);
      } finally {
        setBusy(false);
      }
    });

    $("generateBtn").addEventListener("click", async () => {
      setBusy(true);
      $("output").textContent = "Dolgozom...";
      log("generalas indul");
      try {
        const data = await api("/api/generate", {method: "POST", body: JSON.stringify(generationConfig())});
        $("output").textContent = data.text;
        log(`kesz: ${data.seconds}s, input=${data.input_tokens}, output=${data.output_tokens}`);
        await refreshStatus();
      } catch (err) {
        $("output").textContent = err.message;
        log(err.message);
        if (err.traceback) console.error(err.traceback);
      } finally {
        setBusy(false);
      }
    });

    (async function boot() {
      try {
        await refreshPresets();
        await refreshHardware();
        await refreshStatus();
      } catch (err) {
        log(err.message);
      }
    })();
  </script>
</body>
</html>
"""

HTML = """<!doctype html>
<html lang="hu">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AirLLM Control</title>
  <style>
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; font-family: system-ui, sans-serif; background: #f8fafc; color: #0f172a; }
    main { max-width: 640px; padding: 32px; }
    code { background: #e2e8f0; border-radius: 6px; padding: 2px 6px; }
  </style>
</head>
<body>
  <main>
    <h1>AirLLM Control</h1>
    <p>A React UI build nem talalhato. Futtasd: <code>cd ui</code>, <code>npm install</code>, <code>npm run build</code>, majd inditsd ujra a backend szervert.</p>
  </main>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local AirLLM web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    applied = apply_runtime_optimizations()
    server = ThreadingHTTPServer((args.host, args.port), AirLLMHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"AirLLM UI: {url}")
    print(f"Runtime: cpu_threads={applied['cpu_threads']}, tf32={applied['tf32']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping AirLLM UI...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
