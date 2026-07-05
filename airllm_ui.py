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
from urllib.parse import parse_qs, urlparse


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
    "mode": None,
    "loaded_at": None,
    "load_seconds": None,
}
GENERATION_CANCEL = threading.Event()
DOWNLOAD_LOCK = threading.RLock()
DOWNLOAD_CANCEL = threading.Event()
DOWNLOAD_THREAD: Optional[threading.Thread] = None
DOWNLOAD_STATE: Dict[str, Any] = {
    "active": False,
    "status": "idle",
    "model_id": None,
    "path": None,
    "current_file": None,
    "total_bytes": 0,
    "downloaded_bytes": 0,
    "files_total": 0,
    "files_cached": 0,
    "files_to_download": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}

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


def mps_available(torch: Any) -> bool:
    try:
        return bool(
            torch is not None
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def get_mps_info() -> Dict[str, Any]:
    torch, torch_error = get_torch()
    if torch is None:
        return {"available": False, "built": False, "error": torch_error}

    try:
        built = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())
        available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        return {"available": available, "built": built, "error": None}
    except Exception as exc:
        return {"available": False, "built": False, "error": str(exc)}


def get_mlx_info() -> Dict[str, Any]:
    if platform.system().lower() != "darwin":
        return {"available": False, "version": None, "error": None}
    try:
        import mlx.core as mx  # type: ignore

        return {"available": True, "version": getattr(mx, "__version__", None), "error": None}
    except Exception as exc:
        return {"available": False, "version": None, "error": str(exc)}


def available_device_options() -> list[str]:
    torch, _ = get_torch()
    devices = ["auto", "cpu"]
    if torch is not None:
        try:
            if torch.cuda.is_available():
                devices.insert(1, "cuda:0")
        except Exception:
            pass
        if mps_available(torch):
            insert_at = 1 if "cuda:0" not in devices else 2
            devices.insert(insert_at, "mps")
    return devices


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
        "mps": get_mps_info(),
        "mlx": get_mlx_info(),
        "torch": {
            "available": torch is not None,
            "version": getattr(torch, "__version__", None) if torch is not None else None,
            "error": torch_error,
        },
        "bitsandbytes": {"available": import_bitsandbytes_ok()},
        "supported_families": SUPPORTED_FAMILIES,
        "device_options": available_device_options(),
        "recommendation": recommended_settings(),
    }


_REC_SETTINGS_CACHE: Optional[Dict[str, Any]] = None


def recommended_settings() -> Dict[str, Any]:
    # Hardware (CUDA availability, VRAM, bf16 support) is static within a run, but this is
    # called on every load/generate path (and used to CUDA-probe each time). Compute once,
    # then hand back a copy so callers can mutate freely without poisoning the cache.
    global _REC_SETTINGS_CACHE
    if _REC_SETTINGS_CACHE is None:
        _REC_SETTINGS_CACHE = _compute_recommended_settings()
    return dict(_REC_SETTINGS_CACHE)


def _compute_recommended_settings() -> Dict[str, Any]:
    torch, _ = get_torch()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    mps_is_available = mps_available(torch)
    bnb_available = import_bitsandbytes_ok()
    vram_gb = 0.0

    if cuda_available:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            vram_gb = max(vram_gb, props.total_memory / 1024 / 1024 / 1024)

    if mps_is_available and not cuda_available:
        memory = get_memory_info()
        total_ram = float(memory.get("total_gb") or 0.0)
        max_seq_len = 512
        max_new_tokens = 96
        if total_ram >= 16:
            max_seq_len = 1024
            max_new_tokens = 128
        if total_ram >= 32:
            max_seq_len = 2048
            max_new_tokens = 192
        if total_ram >= 64:
            max_seq_len = 4096
            max_new_tokens = 256
        return {
            "device": "mps",
            "dtype": "float16",
            "compression": "none",
            "prefetching": False,
            "cleanup_interval": 4,
            "prefetch_workers": 1,
            "reinitialize_model_each_forward": False,
            "max_seq_len": max_seq_len,
            "max_new_tokens": max_new_tokens,
        }

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
    # Prefer bf16 on capable GPUs (Ampere+/Blackwell): same memory + speed as fp16 but a
    # wider exponent range, which avoids overflow->NaN on bf16-trained models (Qwen, Llama).
    rec_dtype = "float16"
    try:
        if torch is not None and torch.cuda.is_bf16_supported():
            rec_dtype = "bfloat16"
    except Exception:
        rec_dtype = "float16"
    return {
        "device": "cuda:0",
        "dtype": rec_dtype,
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
                empty_accelerator_cache(torch)
            elif mps_available(torch):
                device = "mps"
                dtype = torch.float16
                size = 1024
                a = torch.randn((size, size), device=device, dtype=dtype)
                b = torch.randn((size, size), device=device, dtype=dtype)
                torch.mps.synchronize()
                for _ in range(2):
                    _ = a @ b
                torch.mps.synchronize()
                start = time.perf_counter()
                for _ in range(5):
                    _ = a @ b
                torch.mps.synchronize()
                result["gpu_matmul_ms"] = round((time.perf_counter() - start) * 1000 / 5, 2)
                del a, b
                empty_accelerator_cache(torch)

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
    if requested == "mps" and not mps_available(torch):
        raise RuntimeError("Apple Metal/MPS nem elerheto ebben a Python kornyezetben.")
    if requested not in {"cpu", "mps"} and not requested.startswith("cuda"):
        raise ValueError(f"Ismeretlen device: {requested}")
    return requested


def resolve_dtype(value: str, device: str) -> Any:
    torch, torch_error = get_torch()
    if torch is None:
        raise RuntimeError(f"PyTorch nem importalhato: {torch_error}")

    requested = (value or "auto").strip()
    if requested == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            try:
                requested = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            except Exception:
                requested = "float16"
        elif device == "mps":
            requested = "float16"
        else:
            requested = "float32"

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
    # Lock-free read: MODEL_STATE entries are plain references swapped atomically
    # under MODEL_LOCK in load_model/unload_model, so a status read never needs the
    # lock. This keeps /api/status responsive while a (slow) generation holds the lock.
    return {
        "loaded": MODEL_STATE["model"] is not None,
        "config": safe_public_config(MODEL_STATE["config"]),
        "mode": MODEL_STATE.get("mode"),
        "loaded_at": MODEL_STATE["loaded_at"],
        "load_seconds": MODEL_STATE["load_seconds"],
    }


def empty_accelerator_cache(torch: Any) -> None:
    if torch is None:
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
    except Exception:
        pass


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
        MODEL_STATE["mode"] = None
        MODEL_STATE["loaded_at"] = None
        MODEL_STATE["load_seconds"] = None

    gc.collect()
    torch, _ = get_torch()
    if torch is not None:
        empty_accelerator_cache(torch)
    return current_status()


def cancel_generation() -> Dict[str, Any]:
    GENERATION_CANCEL.set()
    return {"cancel_requested": True}


def bytes_to_gb(value: int) -> float:
    return round(value / 1024 / 1024 / 1024, 2)


def hf_hub_cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE).resolve()
    except Exception:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        return (hf_home / "hub").resolve()


def cache_dir_for_model_id(model_id: str) -> Path:
    cleaned = model_id.strip()
    if not cleaned or cleaned.startswith(("/", "\\")) or ":" in cleaned or "\\" in cleaned:
        raise ValueError("Csak Hugging Face model ID torolheto a cache-bol.")
    if Path(cleaned).exists():
        raise ValueError("Helyi modell utvonal torlese UI-bol nincs engedelyezve.")

    cache_root = hf_hub_cache_dir()
    candidate = (cache_root / ("models--" + cleaned.replace("/", "--"))).resolve()
    if cache_root != candidate and cache_root not in candidate.parents:
        raise ValueError("A cel kikerulne a Hugging Face cache konyvtarbol.")
    return candidate


def directory_stats(path: Path) -> Tuple[int, Optional[float]]:
    total = 0
    latest: Optional[float] = None
    seen: set[Any] = set()

    def walk(directory: Path) -> None:
        nonlocal total, latest
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            try:
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            latest = stat.st_mtime if latest is None else max(latest, stat.st_mtime)
            if entry.is_dir(follow_symlinks=False):
                walk(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                key = (stat.st_dev, stat.st_ino) if stat.st_ino else str(Path(entry.path).resolve())
                if key not in seen:
                    seen.add(key)
                    total += stat.st_size

    walk(path)
    return total, latest


def list_cached_models() -> Dict[str, Any]:
    cache_dir = hf_hub_cache_dir()
    models = []
    total_size = 0
    if cache_dir.exists():
        for child in cache_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("models--"):
                continue
            model_id = child.name[len("models--") :].replace("--", "/")
            size_bytes, modified_at = directory_stats(child)
            snapshots_dir = child / "snapshots"
            snapshots = []
            if snapshots_dir.exists():
                try:
                    snapshots = [item.name for item in snapshots_dir.iterdir() if item.is_dir()]
                except OSError:
                    snapshots = []
            total_size += size_bytes
            models.append(
                {
                    "model_id": model_id,
                    "path": str(child),
                    "size_bytes": size_bytes,
                    "size_gb": bytes_to_gb(size_bytes),
                    "modified_at": modified_at,
                    "snapshots": len(snapshots),
                }
            )

    models.sort(key=lambda item: item.get("modified_at") or 0, reverse=True)
    return {
        "cache_dir": str(cache_dir),
        "models": models,
        "total_size_bytes": total_size,
        "total_size_gb": bytes_to_gb(total_size),
        "download": download_status(),
    }


def _model_datetime(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _hf_model_to_dict(model: Any) -> Dict[str, Any]:
    model_id = getattr(model, "id", None) or getattr(model, "modelId", None) or ""
    tags = getattr(model, "tags", None) or []
    if not isinstance(tags, list):
        tags = list(tags)
    return {
        "model_id": model_id,
        "downloads": getattr(model, "downloads", None),
        "likes": getattr(model, "likes", None),
        "pipeline_tag": getattr(model, "pipeline_tag", None),
        "library_name": getattr(model, "library_name", None),
        "private": bool(getattr(model, "private", False)),
        "gated": getattr(model, "gated", None),
        "tags": [str(tag) for tag in tags[:12]],
        "last_modified": _model_datetime(getattr(model, "last_modified", None)),
    }


def list_huggingface_models(params: Dict[str, Any]) -> Dict[str, Any]:
    query = str(params.get("q") or params.get("query") or "Qwen2.5 Instruct").strip()
    if len(query) > 120:
        query = query[:120]
    limit = parse_int(params.get("limit"), 20, 1, 50)
    task = str(params.get("task") or "text-generation").strip() or None

    token = (
        params.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    token = str(token).strip() or None

    import inspect
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    supported = inspect.signature(api.list_models).parameters
    kwargs: Dict[str, Any] = {}
    if "search" in supported:
        kwargs["search"] = query or None
    if "pipeline_tag" in supported:
        kwargs["pipeline_tag"] = task
    elif "task" in supported:
        kwargs["task"] = task
    if "sort" in supported:
        kwargs["sort"] = "downloads"
    if "direction" in supported:
        kwargs["direction"] = -1
    if "limit" in supported:
        kwargs["limit"] = limit
    if "full" in supported:
        kwargs["full"] = False
    if "gated" in supported:
        kwargs["gated"] = False
    if "token" in supported:
        kwargs["token"] = token

    models_iter = api.list_models(**kwargs)
    models = [_hf_model_to_dict(model) for model in models_iter]
    models = [model for model in models if model.get("model_id")]
    return {
        "query": query,
        "task": task,
        "limit": limit,
        "models": models,
    }


def delete_cached_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("Add meg a torlendo model ID-t.")

    target = cache_dir_for_model_id(model_id)
    if not target.exists():
        result = list_cached_models()
        result["deleted"] = False
        result["message"] = "A modell nincs a Hugging Face cache-ben."
        return result

    loaded_model = (MODEL_STATE.get("config") or {}).get("model_id")
    if loaded_model == model_id:
        if parse_bool(payload.get("unload_if_loaded"), True):
            unload_model()
        else:
            raise ValueError("A modell jelenleg be van toltve. Elobb uritsd ki.")

    size_bytes, _ = directory_stats(target)
    shutil.rmtree(target)
    result = list_cached_models()
    result["deleted"] = True
    result["deleted_model_id"] = model_id
    result["deleted_bytes"] = size_bytes
    result["deleted_gb"] = bytes_to_gb(size_bytes)
    return result


def _download_state_snapshot() -> Dict[str, Any]:
    with DOWNLOAD_LOCK:
        state = dict(DOWNLOAD_STATE)

    now = time.time()
    started_at = state.get("started_at")
    finished_at = state.get("finished_at")
    if started_at:
        elapsed = (finished_at or now) - started_at
    else:
        elapsed = 0.0

    downloaded = int(state.get("downloaded_bytes") or 0)
    total = int(state.get("total_bytes") or 0)
    percent = None
    eta = None
    if total > 0:
        percent = round(min(100.0, downloaded / total * 100), 1)
        if state.get("active") and downloaded > 0 and downloaded < total and elapsed > 0:
            rate = downloaded / elapsed
            if rate > 0:
                eta = round((total - downloaded) / rate)
    elif state.get("status") in {"cached", "done"}:
        percent = 100.0

    state["elapsed_seconds"] = round(elapsed)
    state["eta_seconds"] = eta
    state["percent"] = percent
    state["downloaded_gb"] = bytes_to_gb(downloaded)
    state["total_gb"] = bytes_to_gb(total)
    return state


def download_status() -> Dict[str, Any]:
    return _download_state_snapshot()


def _update_download_state(**updates: Any) -> None:
    with DOWNLOAD_LOCK:
        DOWNLOAD_STATE.update(updates)


class DownloadCancelled(RuntimeError):
    pass


def _raise_if_download_cancelled() -> None:
    if DOWNLOAD_CANCEL.is_set():
        raise DownloadCancelled("A modell letoltese megszakitva.")


def _add_download_bytes(value: int) -> None:
    if value <= 0:
        return
    with DOWNLOAD_LOCK:
        total = int(DOWNLOAD_STATE.get("total_bytes") or 0)
        current = int(DOWNLOAD_STATE.get("downloaded_bytes") or 0) + value
        DOWNLOAD_STATE["downloaded_bytes"] = min(current, total) if total > 0 else current


def _download_tqdm_class() -> Any:
    from tqdm.auto import tqdm

    class DownloadProgressBar(tqdm):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _raise_if_download_cancelled()
            desc = kwargs.get("desc")
            if desc:
                _update_download_state(current_file=str(desc))
            super().__init__(*args, **kwargs)

        def update(self, n: int = 1) -> Any:
            _raise_if_download_cancelled()
            _add_download_bytes(int(n or 0))
            result = super().update(n)
            _raise_if_download_cancelled()
            return result

        def close(self) -> None:
            _update_download_state(current_file=None)
            super().close()

    return DownloadProgressBar


def start_model_download(payload: Dict[str, Any]) -> Dict[str, Any]:
    global DOWNLOAD_THREAD

    model_id = str(payload.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("Add meg a letoltendo model ID-t.")
    cache_dir_for_model_id(model_id)

    hf_token = (
        payload.get("hf_token")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or ""
    )
    hf_token = str(hf_token).strip() or None
    max_workers = parse_int(payload.get("download_workers"), 4, 1, 8)

    with DOWNLOAD_LOCK:
        if DOWNLOAD_STATE.get("active"):
            raise ValueError("Mar fut egy modell letoltes.")
        DOWNLOAD_CANCEL.clear()
        DOWNLOAD_STATE.update(
            {
                "active": True,
                "status": "preparing",
                "model_id": model_id,
                "path": None,
                "current_file": None,
                "total_bytes": 0,
                "downloaded_bytes": 0,
                "files_total": 0,
                "files_cached": 0,
                "files_to_download": 0,
                "started_at": time.time(),
                "finished_at": None,
                "error": None,
            }
        )

    def worker() -> None:
        try:
            from huggingface_hub import snapshot_download

            _raise_if_download_cancelled()
            _update_download_state(status="preparing")
            dry_run = snapshot_download(model_id, token=hf_token, dry_run=True)
            _raise_if_download_cancelled()
            files_total = len(dry_run)
            files_to_download = [item for item in dry_run if getattr(item, "will_download", False)]
            files_cached = files_total - len(files_to_download)
            total_bytes = sum(int(getattr(item, "file_size", 0) or 0) for item in files_to_download)
            _update_download_state(
                status="cached" if total_bytes <= 0 else "downloading",
                total_bytes=total_bytes,
                files_total=files_total,
                files_cached=files_cached,
                files_to_download=len(files_to_download),
            )

            path = snapshot_download(
                model_id,
                token=hf_token,
                max_workers=max_workers,
                tqdm_class=_download_tqdm_class(),
            )
            _raise_if_download_cancelled()
            with DOWNLOAD_LOCK:
                if total_bytes > 0:
                    DOWNLOAD_STATE["downloaded_bytes"] = total_bytes
                DOWNLOAD_STATE.update(
                    {
                        "active": False,
                        "status": "done",
                        "path": path,
                        "current_file": None,
                        "finished_at": time.time(),
                    }
                )
        except DownloadCancelled as exc:
            _update_download_state(
                active=False,
                status="cancelled",
                error=str(exc),
                current_file=None,
                finished_at=time.time(),
            )
        except Exception as exc:
            _update_download_state(
                active=False,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                current_file=None,
                finished_at=time.time(),
            )

    DOWNLOAD_THREAD = threading.Thread(target=worker, daemon=True)
    DOWNLOAD_THREAD.start()
    return download_status()


def cancel_model_download() -> Dict[str, Any]:
    with DOWNLOAD_LOCK:
        if not DOWNLOAD_STATE.get("active"):
            return download_status()
        DOWNLOAD_CANCEL.set()
        DOWNLOAD_STATE.update(
            {
                "status": "cancelling",
                "error": None,
            }
        )
    return download_status()


def normalized_load_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("Adj meg egy Hugging Face model ID-t vagy helyi modell utvonalat.")

    device = resolve_device(payload.get("device", "auto"))
    dtype_name = (payload.get("dtype") or "auto").strip()
    load_mode = str(payload.get("load_mode") or "auto").strip().lower()
    if load_mode not in {"auto", "airllm", "direct", "hybrid"}:
        load_mode = "auto"
    if load_mode == "hybrid" and not device.startswith("cuda"):
        raise RuntimeError("A CPU+GPU hybrid mod jelenleg CUDA/NVIDIA device mellett tamogatott.")

    compression_value = payload.get("compression", "auto")
    if load_mode == "hybrid" and str(compression_value).strip().lower() == "auto":
        compression_value = "none"
    compression = resolve_compression(compression_value, device)
    if load_mode == "hybrid" and compression is not None:
        raise RuntimeError("A CPU+GPU hybrid modot compression nelkul hasznald.")
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
        "load_mode": load_mode,
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


def estimate_model_weight_gb(config: Any, bytes_per_param: int = 2) -> Optional[float]:
    """Estimate resident weight size in GB from a HF config.

    Counts attention (q/k/v/o, GQA-aware), gated MLP (3 matrices, MoE-aware),
    embeddings and lm_head (untied unless tie_word_embeddings). Used only to decide
    whether a model can be loaded resident instead of disk-streamed by AirLLM, so a
    conservative over-estimate (e.g. MoE) is the safe direction.
    """
    try:
        hidden = int(getattr(config, "hidden_size"))
        layers = int(getattr(config, "num_hidden_layers"))
        vocab = int(getattr(config, "vocab_size"))
    except Exception:
        return None
    if hidden <= 0 or layers <= 0 or vocab <= 0:
        return None
    inter = int(getattr(config, "intermediate_size", 0) or (4 * hidden))
    n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
    n_kv = int(getattr(config, "num_key_value_heads", n_heads) or n_heads)
    if n_heads > 0:
        head_dim = int(getattr(config, "head_dim", 0) or (hidden // n_heads))
        kv_dim = head_dim * n_kv if n_kv > 0 else hidden
    else:
        kv_dim = hidden
    n_experts = int(getattr(config, "num_local_experts", 0) or getattr(config, "num_experts", 0) or 1)
    q_o = 2 * hidden * hidden
    k_v = 2 * hidden * kv_dim
    mlp = 3 * hidden * inter * max(1, n_experts)
    per_layer = q_o + k_v + mlp + 2 * hidden
    tied = bool(getattr(config, "tie_word_embeddings", False))
    embed = vocab * hidden * (1 if tied else 2)
    total_params = layers * per_layer + embed + hidden
    return total_params * bytes_per_param / (1024 ** 3)


def free_vram_gb(device: str) -> Optional[float]:
    torch, _ = get_torch()
    if torch is None:
        return None
    if device == "mps":
        available = get_memory_info().get("available_gb")
        if available is None:
            return None
        # Apple Silicon uses unified memory. Keep a conservative OS/app reserve so a
        # resident load does not pressure the whole desktop session.
        return max(0.0, float(available) - 4.0)
    try:
        return torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
    except Exception:
        try:
            return torch.cuda.mem_get_info()[0] / (1024 ** 3)
        except Exception:
            return None


def _quant_bits(quant_cfg: Any) -> Optional[int]:
    """Best-effort bit-width from a HF quantization_config (dict or object: GPTQ/AWQ/...)."""
    for get in (
        lambda: getattr(quant_cfg, "bits", None),
        lambda: quant_cfg.get("bits") if isinstance(quant_cfg, dict) else None,
        lambda: getattr(quant_cfg, "w_bit", None),
        lambda: quant_cfg.get("w_bit") if isinstance(quant_cfg, dict) else None,
    ):
        try:
            value = get()
            if value:
                return int(value)
        except Exception:
            pass
    return None


def plan_load_strategy(config: Dict[str, Any], dtype: Any) -> Tuple[str, Dict[str, Any]]:
    """Decide between AirLLM disk-streaming and a resident direct load.

    Returns (strategy, info) where strategy is one of:
      - "airllm":         existing layer-by-layer disk streaming (default/fallback)
      - "direct_gpu":     plain transformers model fully resident on CUDA/MPS
      - "direct_offload": plain transformers model with accelerate CPU/disk offload
    """
    torch, _ = get_torch()
    requested = config.get("load_mode", "auto")
    device = config.get("device", "cpu")
    info: Dict[str, Any] = {"requested": requested}

    if requested == "airllm":
        info["reason"] = "explicit airllm"
        return "airllm", info
    if requested == "hybrid":
        if torch is None or not str(device).startswith("cuda") or config.get("compression") is not None:
            info["reason"] = "hybrid requires CUDA without AirLLM compression"
            return "airllm", info
        vram = free_vram_gb(device)
        ram = get_memory_info().get("available_gb")
        info.update(
            {
                "free_accelerator_gb": round(vram, 2) if vram else None,
                "free_ram_gb": ram,
            }
        )
        info["reason"] = "explicit CPU+GPU hybrid"
        return "direct_offload", info
    accelerator = str(device).startswith("cuda") or str(device) == "mps"
    # Direct load only makes sense on a Torch accelerator, without AirLLM-only compression.
    if torch is None or not accelerator or config.get("compression") is not None:
        info["reason"] = "no supported accelerator / compression set"
        return "airllm", info

    bytes_pp = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    from transformers import AutoConfig

    try:
        hf_config = AutoConfig.from_pretrained(
            config["model_id"], trust_remote_code=True,
            **({"token": config["hf_token"]} if config.get("hf_token") else {}),
        )
    except Exception as exc:
        info["reason"] = f"config read failed: {exc}"
        # If the user explicitly forced direct, still try a GPU load; else stay on AirLLM.
        return ("direct_gpu", info) if requested == "direct" else ("airllm", info)

    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is not None:
        # Prequantized (GPTQ/AWQ/...) checkpoints can load resident via transformers IF a
        # matching kernel lib is installed (gptqmodel/autoawq) -- far faster than AirLLM
        # disk-streaming. Only attempt it on explicit load_mode="direct" (auto stays on the
        # safe AirLLM path); build_direct_model()'s exception is caught by load_model() and
        # falls back to AirLLM if the kernel is missing or it OOMs.
        if requested != "direct":
            info["reason"] = "prequantized checkpoint -> airllm (set load_mode=direct to try resident)"
            return "airllm", info
        bits = _quant_bits(quant_cfg)
        est_q = estimate_model_weight_gb(hf_config, bytes_per_param=2)
        if est_q is not None and bits:
            est_q = est_q * (bits / 16.0)  # weights shrink ~bits/16 vs bf16 (rough; embeds excluded)
        vram_q = free_vram_gb(device)
        info.update({"quant_bits": bits, "est_gb": round(est_q, 2) if est_q else None,
                     "free_accelerator_gb": round(vram_q, 2) if vram_q else None})
        if est_q is not None and vram_q is not None and est_q <= max(0.0, vram_q - 1.0):
            info["reason"] = f"prequantized {bits}bit fits accelerator memory -> direct_gpu"
            return "direct_gpu", info
        info["reason"] = "prequantized too large for accelerator memory -> airllm"
        return "airllm", info

    est_gb = estimate_model_weight_gb(hf_config, bytes_pp)
    vram = free_vram_gb(device)
    ram = get_memory_info().get("available_gb")
    info.update({"est_gb": round(est_gb, 2) if est_gb else None,
                 "free_accelerator_gb": round(vram, 2) if vram else None,
                 "free_ram_gb": ram})

    # Accelerator budget: reserve ~1 GB for the runtime context, KV cache and activation
    # peak. On MPS free_vram_gb() already subtracts a larger unified-memory OS reserve.
    if est_gb is not None and vram is not None and est_gb <= max(0.0, vram - 1.0):
        info["reason"] = "fits accelerator memory"
        return "direct_gpu", info
    # CPU/disk offload is far faster than AirLLM disk streaming, but only auto-pick it
    # when the user explicitly asked for "direct" on CUDA (it can still be slow / thrash).
    if (
        requested == "direct"
        and str(device).startswith("cuda")
        and est_gb is not None
        and ram is not None
        and est_gb <= max(0.0, ram - 2.0)
    ):
        info["reason"] = "offload to RAM"
        return "direct_offload", info
    info["reason"] = "too large -> airllm/mlx streaming"
    return "airllm", info


def build_direct_model(config: Dict[str, Any], dtype: Any, strategy: str) -> Any:
    """Load a plain transformers model resident (GPU or accelerate offload) and make it
    look enough like an AirLLM model for the serving layer (model.tokenizer / .max_seq_len /
    .generate)."""
    torch, _ = get_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = config.get("hf_token")
    common: Dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if token:
        common["token"] = token

    if strategy == "direct_offload":
        vram = free_vram_gb(config["device"]) or 6.0
        ram = get_memory_info().get("available_gb") or 8.0
        gpu_cap = max(1, int(vram * 0.8))
        cpu_cap = max(1, int(ram * 0.6))
        common["device_map"] = "auto"
        common["max_memory"] = {0: f"{gpu_cap}GiB", "cpu": f"{cpu_cap}GiB"}
        common["offload_folder"] = os.path.join(tempfile.gettempdir(), "airllm_offload")
        common["low_cpu_mem_usage"] = True
    elif str(config["device"]).startswith("cuda"):  # direct_gpu on CUDA
        common["device_map"] = {"": config["device"]}
    else:  # direct_gpu on MPS/Metal
        common["low_cpu_mem_usage"] = True

    # Prefer the SDPA attention kernel; retry without it for models that reject the kwarg.
    try:
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], attn_implementation="sdpa", **common)
    except (ValueError, TypeError):
        model = AutoModelForCausalLM.from_pretrained(config["model_id"], **common)

    if strategy == "direct_gpu" and config["device"] == "mps":
        model.to("mps")

    model.eval()
    tok_kwargs = {"trust_remote_code": True}
    if token:
        tok_kwargs["token"] = token
    model.tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **tok_kwargs)
    model.max_seq_len = config["max_seq_len"]
    model._airllm_direct = True
    return model


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
            empty_accelerator_cache(torch)

        dtype = resolve_dtype(config["dtype"], config["device"])

        # Decide whether the model can run resident (fast) or must be disk-streamed by
        # AirLLM. Resident inference is ~1-2 orders of magnitude faster for models that fit.
        strategy, plan_info = plan_load_strategy(config, dtype)
        print(f"load strategy: {strategy} ({plan_info})")

        start = time.perf_counter()
        model = None
        if strategy in ("direct_gpu", "direct_offload"):
            try:
                model = build_direct_model(config, dtype, strategy)
            except Exception as exc:
                print(f"direct load failed ({type(exc).__name__}: {exc}); falling back to AirLLM streaming")
                model = None
                gc.collect()
                if torch is not None:
                    empty_accelerator_cache(torch)
                strategy = "airllm"

        if model is None:
            strategy = "airllm"
            from airllm import AutoModel

            kwargs = {
                "device": config["device"],
                "dtype": dtype,
                "max_seq_len": config["max_seq_len"],
                "compression": config["compression"],
                "profiling_mode": config["profiling_mode"],
                "prefetching": config["prefetching"],
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
        MODEL_STATE["mode"] = strategy
        MODEL_STATE["loaded_at"] = time.time()
        MODEL_STATE["load_seconds"] = round(elapsed, 2)

        status = current_status()
        status["reused"] = False
        status["plan"] = plan_info
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
            # transformers 5.x returns a BatchEncoding (dict-like), NOT a bare tensor, so
            # the old `input_ids.shape` raised AttributeError, was swallowed below, and the
            # chat template was silently skipped (model got a raw prompt -> worse answers).
            # Ask for the dict explicitly and pull out input_ids.
            msgs = list(messages)

            def _render(ms: list[Dict[str, str]]) -> Any:
                return tokenizer.apply_chat_template(
                    ms,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )["input_ids"]

            input_ids = _render(msgs)
            # If over budget, drop the OLDEST non-system turns and re-render, preserving the
            # system prompt (persona/instructions) and the most recent turns. Left-truncating
            # the raw token stream (old behavior) would chop the system prompt and split a
            # message mid-token.
            while input_ids.shape[-1] > max_length and len(msgs) > 1:
                drop = next((i for i, m in enumerate(msgs) if m.get("role") != "system"), None)
                if drop is None:
                    break
                del msgs[drop]
                input_ids = _render(msgs)
            if input_ids.shape[-1] > max_length:
                # Last resort (e.g. the system prompt alone exceeds the budget).
                input_ids = input_ids[:, -max_length:]
            return input_ids
        except Exception:
            pass

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_ids = encoded["input_ids"]
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


TASK_PRESETS: Dict[str, Dict[str, float]] = {
    # chat / creative: lively but controlled (historical defaults)
    "chat":    {"temperature": 0.7, "top_p": 0.9,  "top_k": 50, "repetition_penalty": 1.05},
    # factual Q&A: low temperature, mild repetition penalty
    "factual": {"temperature": 0.2, "top_p": 0.9,  "top_k": 40, "repetition_penalty": 1.1},
    # code: greedy, NO repetition penalty (1.2 would punish legitimate indentation/braces)
    "code":    {"temperature": 0.0, "top_p": 0.95, "top_k": 0,  "repetition_penalty": 1.0},
}


def generation_settings(payload: Dict[str, Any], model: Any, config: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    max_model_len = int(getattr(model, "max_seq_len", config.get("max_seq_len", 512)))
    is_airllm = not bool(getattr(model, "_airllm_direct", False))

    # Parse output length first so the prompt budget can be sized around it.
    max_new_tokens = parse_int(payload.get("max_new_tokens"), recommended_settings()["max_new_tokens"], 1, 4096)
    if is_airllm:
        # AirLLM preallocates fixed max_seq_len attention-mask / position-id buffers, so
        # prompt_tokens + max_new_tokens must never exceed max_seq_len (else it crashes
        # mid-generation). Clamp both defensively, independent of UI input.
        max_new_tokens = min(max_new_tokens, max(1, max_model_len - 16))
        prompt_budget = max(16, max_model_len - max_new_tokens)
        max_length = parse_int(payload.get("max_length"), prompt_budget, 16, max_model_len)
        max_length = max(16, min(max_length, max_model_len - max_new_tokens))
    else:
        # Resident transformers models have no fixed buffer; only truncate the prompt.
        max_length = parse_int(payload.get("max_length"), min(512, max_model_len), 16, max_model_len)

    # Task-aware sampling defaults (chat / factual / code); explicit payload values win.
    task_mode = str(payload.get("task_mode") or "chat").strip().lower()
    preset = TASK_PRESETS.get(task_mode, TASK_PRESETS["chat"])
    temperature = parse_float(payload.get("temperature"), preset["temperature"], 0.0, 2.0)
    top_p = parse_float(payload.get("top_p"), preset["top_p"], 0.05, 1.0)
    top_k = parse_int(payload.get("top_k"), int(preset["top_k"]), 0, 1000)
    repetition_penalty = parse_float(payload.get("repetition_penalty"), preset["repetition_penalty"], 0.8, 2.0)

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

    # Quality: some checkpoints ship an incomplete generation_config (no eos id), which
    # causes run-on answers. Fall back to the tokenizer's eos/pad ONLY when the model
    # itself defines none, so we never clobber a model's richer eos set (e.g. Llama 3.1's
    # <|eot_id|>).
    gen_cfg = getattr(model, "generation_config", None)
    if getattr(gen_cfg, "eos_token_id", None) is None:
        tok = getattr(model, "tokenizer", None)
        tok_eos = getattr(tok, "eos_token_id", None)
        if tok_eos is not None:
            generation_kwargs["eos_token_id"] = tok_eos
            if getattr(gen_cfg, "pad_token_id", None) is None:
                generation_kwargs["pad_token_id"] = getattr(tok, "pad_token_id", None) or tok_eos

    # Lossless speedup on the RESIDENT path only: prompt-lookup decoding verifies several
    # context-matched candidate tokens per forward (1.5-3x on echo-heavy code/RAG/refactor
    # output that quotes its context, ~1x otherwise). Greedy => bit-identical output;
    # sampling => distribution-preserving. Never enable on the AirLLM streaming path: its
    # fixed-size buffers + per-token disk re-stream make a multi-token candidate forward
    # both unsafe and pointless. Default on for the greedy 'code' preset; opt-in elsewhere.
    if not is_airllm and generation_kwargs["use_cache"]:
        if task_mode == "code" or parse_bool(payload.get("prompt_lookup"), False):
            generation_kwargs["prompt_lookup_num_tokens"] = parse_int(payload.get("prompt_lookup_num_tokens"), 10, 1, 32)
            generation_kwargs["max_matching_ngram_size"] = parse_int(payload.get("max_matching_ngram_size"), 2, 1, 8)

    return max_length, generation_kwargs


def resolve_input_device(model: Any, config: Dict[str, Any]) -> Any:
    """Where to put input_ids. For a resident (direct) model with accelerate offload the
    inputs must go to the model's input/embedding device, not the configured device."""
    if getattr(model, "_airllm_direct", False):
        dev = getattr(model, "device", None)
        if dev is not None:
            return dev
    return config["device"]


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
        input_ids = input_ids.to(resolve_input_device(model, config))
        if getattr(model, "_airllm_direct", False):
            # Resident HF models infer padding from the attention mask; pass an explicit
            # all-ones mask (single unpadded sequence) so a trailing eos==pad isn't misread.
            generation_kwargs["attention_mask"] = torch.ones_like(input_ids)

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


def prepare_chat_messages(payload: Dict[str, Any]) -> list[Dict[str, str]]:
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
    return messages


def chat_completion(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages = prepare_chat_messages(payload)

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


def run_generation_stream(payload: Dict[str, Any], prompt: str, messages: Optional[list[Dict[str, str]]] = None):
    """Yield {'token': str} events as the local model produces them, then a final
    {'done': True, ...} event. Streams via a background generate() thread feeding a
    TextIteratorStreamer; MODEL_LOCK is held for the whole generation (releasing it
    per token would corrupt the shared meta-model / KV cache)."""
    if parse_bool(payload.get("autoload"), True):
        load_model(payload.get("load_config") or payload)

    from transformers import TextIteratorStreamer

    with MODEL_LOCK:
        model = MODEL_STATE["model"]
        config = MODEL_STATE["config"]
        if model is None or config is None:
            raise RuntimeError("Nincs betoltott modell.")

        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("Adj meg promptot.")

        torch, torch_error = get_torch()
        if torch is None:
            raise RuntimeError(f"PyTorch nem importalhato: {torch_error}")

        use_chat_template = parse_bool(payload.get("use_chat_template"), True)
        max_length, generation_kwargs = generation_settings(payload, model, config)
        input_ids = tokenize_prompt(model, prompt, max_length, use_chat_template, messages=messages)
        input_ids = input_ids.to(resolve_input_device(model, config))

        streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs["streamer"] = streamer
        generation_kwargs.pop("return_dict_in_generate", None)
        if getattr(model, "_airllm_direct", False):
            generation_kwargs["attention_mask"] = torch.ones_like(input_ids)

        GENERATION_CANCEL.clear()
        start = time.perf_counter()
        worker_state: Dict[str, Any] = {}

        def _run() -> None:
            try:
                with torch.inference_mode():
                    out = model.generate(input_ids, **generation_kwargs)
                # generate() (no return_dict_in_generate here) returns the full sequence
                # tensor; record real new-token count for accurate tok/s in the done event.
                try:
                    worker_state["output_tokens"] = int(out.shape[-1] - input_ids.shape[-1])
                except Exception:
                    pass
            except Exception as exc:  # surfaced as a final error event
                worker_state["error"] = f"{type(exc).__name__}: {exc}"

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        output_chars = 0
        for chunk in streamer:
            if chunk:
                output_chars += len(chunk)
                yield {"token": chunk}
            if GENERATION_CANCEL.is_set():
                break

        worker.join()
        elapsed = time.perf_counter() - start
        cancelled = GENERATION_CANCEL.is_set()
        GENERATION_CANCEL.clear()

        output_tokens = worker_state.get("output_tokens")
        done: Dict[str, Any] = {
            "done": True,
            "seconds": round(elapsed, 2),
            "input_tokens": int(input_ids.shape[-1]),
            "output_chars": output_chars,
            "output_tokens": output_tokens,
            "tokens_per_second": round(output_tokens / elapsed, 1) if output_tokens and elapsed > 0 else None,
            "cancelled": cancelled,
            "status": current_status(),
        }
        if "error" in worker_state:
            done["error"] = worker_state["error"]
        yield done


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


def sse_response(handler: BaseHTTPRequestHandler, events: Any) -> None:
    """Stream an iterable of dict events as Server-Sent Events. Because the 200 headers
    are sent before iteration begins, any error raised while iterating is emitted as a
    data event (we cannot switch to a 500 afterwards)."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Accel-Buffering", "no")
    handler.send_header("Connection", "close")
    handler.end_headers()
    try:
        for event in events:
            chunk = "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            handler.wfile.write(chunk.encode("utf-8"))
            handler.wfile.flush()
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except Exception as exc:
        try:
            err = "data: " + json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n\n"
            handler.wfile.write(err.encode("utf-8"))
            handler.wfile.flush()
        except Exception:
            pass


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
    # Disable Nagle's algorithm so SSE token frames flush immediately instead of being
    # coalesced (can otherwise add up to ~40ms latency per small frame off-loopback).
    disable_nagle_algorithm = True

    def do_GET(self) -> None:  # noqa: N802
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = {key: values[-1] if values else "" for key, values in parse_qs(parsed_url.query).items()}
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
                elif path == "/api/models":
                    json_response(self, list_cached_models())
                elif path == "/api/hf-models":
                    json_response(self, list_huggingface_models(query))
                elif path == "/api/download/status":
                    json_response(self, download_status())
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
                if parse_bool(payload.get("stream"), False) and not is_external_provider(payload):
                    sse_response(self, run_generation_stream(payload, (payload.get("prompt") or "").strip()))
                else:
                    json_response(self, generate_text(payload))
            elif path == "/api/chat":
                if parse_bool(payload.get("stream"), False) and not is_external_provider(payload):
                    messages = prepare_chat_messages(payload)
                    sse_response(self, run_generation_stream(payload, messages_to_prompt(messages), messages=messages))
                else:
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
            elif path == "/api/models/delete":
                json_response(self, delete_cached_model(payload))
            elif path == "/api/download":
                json_response(self, start_model_download(payload))
            elif path == "/api/download/cancel":
                json_response(self, cancel_model_download())
            else:
                json_response(self, {"error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"error": str(exc), "traceback": traceback.format_exc()}, 500)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")


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
