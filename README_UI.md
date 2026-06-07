# AirLLM UI

This project now includes a local browser UI built with React, Vite, Tailwind,
and shadcn/ui-style components.

```powershell
.\.venv\Scripts\python.exe airllm_ui.py
```

Open `http://127.0.0.1:7860`.

The Python server serves the production frontend from `ui/dist` when it exists.
To rebuild it after UI changes:

```powershell
cd ui
npm install
npm run build
```

For frontend-only development, keep the Python backend running on `7860`, then:

```powershell
cd ui
npm run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api/*` to the Python backend.

The UI is responsive and adapts from mobile-width screens to desktop
workstations. It detects CPU, RAM, CUDA GPU, PyTorch, bitsandbytes, Hugging Face
cache disk space, active Windows power mode, and WiFi link details. It applies
hardware-aware runtime defaults, then lets you tune:

- supported model family / Hugging Face model ID
- Qwen2.5 Coder presets for local coding-agent experiments
- device, dtype, 4bit/8bit compression, prefetching
- `cleanup_interval`, `prefetch_workers`, and the compatibility-only
  `reinitialize_model_each_forward` switch
- max sequence length, layer shard cache path, Hugging Face token
- generation parameters such as temperature, top-p, top-k, max new tokens
- ChatUI conversation mode backed by the loaded local model
- Coding Agent mode that can be started with one button
- external AI provider mode via OpenAI-compatible `/chat/completions` APIs
- local Stop action for interrupting AirLLM generation
- benchmark action for quick GPU/CPU and cache disk throughput checks

On small screens, the layout stacks into a single column, tabs become full-width
rows, chat/output panels reduce their minimum height, and primary action buttons
expand to the available width.

Supported families are taken from the project's `AutoModel` routing:
Qwen2/Qwen2.5, QWen, Llama, Mistral, Mixtral, ChatGLM, Baichuan, and InternLM.

On CUDA systems the UI enables PyTorch CPU thread settings, TF32 where available,
cuDNN benchmarking, and AirLLM prefetching or compression according to the
selected mode. The optimized AirLLM path keeps the meta model alive between
forward passes, throttles expensive memory cleanup with `cleanup_interval`, pins
prefetched CPU tensors correctly, and uses Transformers `DynamicCache` for
modern rotary decoder models when `KV cache` is enabled. Unsupported or batched
cache cases fall back to cache-free inference.

The first model load can take a long time because AirLLM downloads and splits
model layers into the Hugging Face cache. For best results on a laptop/desktop
with NVMe storage, place `HF_HOME` or `Layer cache` on the fastest SSD with
enough free space. Use a performance Windows power profile when benchmarking or
running long generations, and prefer a stable WiFi 6/7 or wired connection for
initial model downloads.

If you start it with a Python that cannot import `torch`, the UI will still open,
but model loading will fail until you use the project virtual environment or
install the runtime dependencies there.

## Chat and Coding Agent

The Chat tab keeps message history in the browser and sends it to `/api/chat`.
If `Autoload` is enabled, the selected local model is loaded before the first
response.

The Coding Agent tab sends a task plus workspace context to `/api/agent/run`.
It reads the project tree, `git status --short`, and a small set of important
files, then asks the local model for a coding-agent style answer. It does not
run shell commands or write files automatically; it returns a plan, proposed
changes, and test suggestions for review.

## Benchmark and Stop

The `Benchmark` button runs a short local test without loading a model: CUDA
matrix multiplication when available, CPU matrix multiplication, and read/write
throughput against the selected cache location. If a local model is already
loaded, the benchmark can also run a tiny generation probe.

The `Stop` button requests cancellation for local AirLLM generation. It is wired
through Transformers stopping criteria, so the request is cooperative and stops
at the next generation check. External provider requests cannot be interrupted
once the HTTP request has been sent.

## External Providers

Use the `AI szolgaltato` panel to switch from `Local AirLLM` to
`OpenAI-compatible`. This works with services that expose a compatible
`/v1/chat/completions` endpoint, including many cloud providers and local
servers such as LM Studio or Ollama's OpenAI-compatible API.

Configure:

- `Base URL`, for example `https://api.example.com/v1`
- `Model`, using the provider's model name
- `API key`, if the provider requires one

The API key is only sent with the current request. It is not written to the
project files or stored in backend state. You can also provide keys through
environment variables such as `AI_PROVIDER_API_KEY` or `OPENAI_API_KEY`.
