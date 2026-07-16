# CLAUDE.md

Guidance for Claude Code working in this repo. Keep it accurate — update it when the facts below change.

## What this is

A fork of **AirLLM** (layer-by-layer disk-streaming inference for large models on small VRAM) with a
custom **hardware-aware local web UI** added on top. The UI is the main thing being developed here;
upstream AirLLM lives under `air_llm/`.

- `airllm_ui.py` — the entire backend: a single-file server built on the Python **stdlib**
  `http.server.ThreadingHTTPServer` (no Flask/FastAPI). Serves the built React app + a JSON/SSE API.
- `ui/` — the frontend: **React 19 + Vite 8 + Tailwind 4 + shadcn-style** components (`ui/src/App.tsx`
  is the bulk). Built output goes to `ui/dist`.
- `air_llm/` — vendored upstream AirLLM source (the `airllm` package). `airllm_ui.py` adds it to
  `sys.path` and imports `from airllm import AutoModel` only for the streaming fallback path.

## Running it

**Use the project venv's Python, never the bare `python` on PATH.**
Bare `python` resolves to `C:\Python314\python.exe` (system Python 3.14) which has **none** of the
ML deps installed — probing it gives misleading "package absent" results. The real environment is:

```powershell
.\.venv\Scripts\python.exe airllm_ui.py            # serves http://127.0.0.1:7860
.\.venv\Scripts\python.exe airllm_ui.py --host 0.0.0.0 --port 8000
```

For probes/checks always run `.venv\Scripts\python.exe -c "..."` (Git Bash: `.venv/Scripts/python.exe`).

The server serves `ui/dist` if it exists; otherwise it serves a tiny fallback page telling you to build.

## Building the UI (important gotcha)

`ui/dist` is **gitignored** (`.gitignore` line `dist`). The backend serves the *built* files, so
**source edits in `ui/src` do nothing until you rebuild**:

```powershell
cd ui
npm install          # first time only
npm run build        # tsc -b && vite build  -> writes ui/dist
npm run dev          # optional: Vite dev server on :5173, proxies /api/* to :7860
```

## Architecture map of `airllm_ui.py` (~2300 lines; line numbers approximate, verify before editing)

- **Hardware probing**: `hardware_profile()`, `recommended_settings()` (picks device/dtype/seq-len),
  `get_memory_info()`/`get_cuda_info()`/`get_power_info()`/`get_network_info()`. These run CUDA +
  `psutil` + `powercfg`/`netsh` subprocesses; they are recomputed on several request paths.
- **Load strategy** (`plan_load_strategy`, `build_direct_model`, `load_model`): decides between
  - `direct_gpu` — plain resident `transformers` model on the GPU (fast path, ~16 tok/s on a 3B),
  - `direct_offload` — `device_map="auto"` accelerate CPU/disk offload (only when `load_mode="direct"`),
  - `airllm` — upstream layer-by-layer disk streaming (fallback; **seconds per token**).
  Gated on free VRAM via `mem_get_info` reserving ~1 GB, with a guarded OOM fallback to `airllm`.
  Resident models get `.tokenizer` / `.max_seq_len` / `._airllm_direct=True` attached so the rest of
  the serving code can treat them uniformly. `estimate_model_weight_gb` is MoE-aware (over-estimates → safe).
- **Generation**: `generation_settings` (TASK_PRESETS chat/factual/code + sampling kwargs),
  `tokenize_prompt` (prefers `apply_chat_template`), `run_generation` (blocking),
  `run_generation_stream` (SSE: `TextIteratorStreamer` + background `generate()` thread).
- **Concurrency**: one global `MODEL_LOCK` (RLock) held for the *entire* generation. `current_status()`
  is intentionally **lock-free** (atomic reference reads) so `/api/status` stays responsive during a
  slow generation. `GENERATION_CANCEL` (threading.Event) + a `StoppingCriteria` implement cooperative cancel.
- **HTTP**: `AirLLMHandler.do_GET/do_POST`. Endpoints: `/api/hardware`, `/api/presets`, `/api/providers`,
  `/api/status`, `/api/load`, `/api/generate`, `/api/chat`, `/api/agent/run`, `/api/unload`,
  `/api/cancel`, `/api/optimize`, `/api/benchmark`. `/api/generate` and `/api/chat` switch to SSE when
  `stream:true` and the provider is local. `sse_response` sends 200 headers up front (errors mid-stream
  surface as a `data:` event, not a 500).
- **External providers**: `external_chat_request` forwards to any OpenAI-compatible `/chat/completions`.
  API keys are per-request only, never persisted.
- **Coding agent**: `run_coding_agent` builds a workspace-context prompt (tree + `git status` + key
  files) and asks the loaded model — it never executes commands or writes files.

### Dead code to be aware of
There are **two** `HTML = ...` literals near the end of the file. The first (~540 lines, a full inline
UI) is immediately **overwritten** by a small fallback page. The real UI is the React build in `ui/dist`;
the large inline literal is dead.

## Stack constraints (this machine — see also the persistent memory files)

- GPU: **RTX 5070 Laptop, ~8 GB VRAM, Blackwell sm_120 (cc 12.0), bf16 supported.** Often only ~6.8 GB free.
- `torch 2.11.0+cu128`, `transformers 5.10.2`, `accelerate 1.13.0`, `optimum 2.1.0`, Python 3.14.
- **transformers 5.x uses `dtype=`, not `torch_dtype=`** (the latter is deprecated/BC-only).
- **No quantization/kernel libs installed**: `bitsandbytes`, `torchao`, `triton`/`triton-windows`,
  `autoawq`, `auto_gptq`/`gptqmodel`, `hqq`, `flash_attn` are all **absent** (no reliable Windows +
  sm_120 wheels). So: bnb 4/8-bit paths are unusable; **`torch.compile`/inductor is unavailable**
  (no Triton); **do not force `flash_attention_2`** (this wheel has no flash SDPA backend → crashes).
  **SDPA (mem-efficient/cuDNN) is the correct attention backend.**
- Default dtype is **bf16** on bf16-capable GPUs (was fp16 → could NaN on bf16-trained models like Qwen/Llama).

## Performance reality (the core finding driving all perf work)

The dominant cost is **AirLLM re-streaming the whole model from disk every token**. Per-layer
micro-optimizations (attn-mask/position-id caching, prefetch tuning) are <0.01% of forward time and are
**not worth doing**. The only real levers are: (1) keep the model **resident** (avoid disk streaming),
(2) model size, (3) `max_seq_len`/`max_new_tokens`. For a resident 3B bf16 the decode ceiling is
~28 tok/s (VRAM-bandwidth bound); the rest is fixed per-token Python/launch overhead that
`torch.compile`/CUDA-graphs *would* remove but can't here (no Triton). See the persistent memory file
`airllm-optimization-decisions` for the audited list of quality-preserving levers and dead-ends.

## Conventions

- UI strings and user-facing text are in **Hungarian**; code identifiers stay English.
- Keep the backend dependency-free (stdlib only); the heavy deps live in the venv via `requirements.txt`.
- After any `ui/src` change, **rebuild** (`npm run build`) or the running server won't reflect it.
