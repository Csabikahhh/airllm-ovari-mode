import { useEffect, useMemo, useRef, useState } from "react"
import Swal from "sweetalert2"
import "sweetalert2/dist/sweetalert2.min.css"
import {
  Bot,
  Code2,
  LoaderCircle,
  MessageSquare,
  Play,
  Send,
  Settings,
  Trash2,
} from "lucide-react"
import { ConfigPanel } from "@/components/ConfigPanel"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { Field, pretty, SelectField, ToggleField } from "@/lib/ui-primitives"
import { configHints } from "@/lib/config-hints"
import { cn } from "@/lib/utils"
import type {
  AgentResponse,
  BenchmarkResult,
  ChatMessage,
  GenerateForm,
  HardwareProfile,
  HuggingFaceModelsResponse,
  LoadForm,
  ModelCacheInfo,
  ModelDownloadStatus,
  Preset,
  ProviderForm,
  ProviderPreset,
  Status,
} from "@/types/app"

const defaultLoadForm: LoadForm = {
  model_id: "Qwen/Qwen2.5-3B-Instruct",
  device: "auto",
  dtype: "auto",
  compression: "auto",
  load_mode: "auto",
  prefetching: "auto",
  cleanup_interval: "4",
  prefetch_workers: "1",
  reinitialize_model_each_forward: false,
  max_seq_len: "512",
  layer_shards_saving_path: "",
  hf_token: "",
  profiling_mode: false,
  delete_original: false,
}

const defaultGenerateForm: GenerateForm = {
  prompt: "Szia! Foglald ossze roviden, mire jo az AirLLM.",
  task_mode: "chat",
  max_length: "512",
  max_new_tokens: "96",
  temperature: "0.7",
  top_p: "0.9",
  top_k: "50",
  repetition_penalty: "1.05",
  autoload: true,
  use_cache: true,
  use_chat_template: true,
}

// Sampling presets mirror the server-side TASK_PRESETS (airllm_ui.py). Selecting a task
// mode repopulates the sampling fields so the user sees and can still tweak the values.
const TASK_PRESETS: Record<string, { temperature: string; top_p: string; top_k: string; repetition_penalty: string }> = {
  chat: { temperature: "0.7", top_p: "0.9", top_k: "50", repetition_penalty: "1.05" },
  factual: { temperature: "0.2", top_p: "0.9", top_k: "40", repetition_penalty: "1.1" },
  code: { temperature: "0", top_p: "0.95", top_k: "0", repetition_penalty: "1.0" },
}

const defaultProviderForm: ProviderForm = {
  provider: "local",
  provider_preset: "0",
  external_base_url: "",
  external_model: "",
  external_api_key: "",
  external_timeout: "120",
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  const payload = await response.json()
  if (!response.ok) {
    const message = typeof payload?.error === "string" ? payload.error : "API hiba"
    throw new Error(message)
  }
  return payload as T
}

async function confirmModelDelete(modelId: string) {
  const result = await Swal.fire({
    title: "Modell törlése?",
    text: `Töröljem ezt a modellt a Hugging Face cache-ből?\n\n${modelId}`,
    icon: "warning",
    showCancelButton: true,
    confirmButtonText: "Törlés",
    cancelButtonText: "Mégse",
    reverseButtons: true,
    focusCancel: true,
    buttonsStyling: false,
    customClass: {
      popup: "rounded-lg border border-border bg-card text-foreground shadow-xl",
      title: "text-lg font-semibold text-foreground",
      htmlContainer: "whitespace-pre-wrap break-words text-sm text-muted-foreground",
      actions: "gap-2",
      confirmButton:
        "inline-flex h-10 items-center justify-center rounded-md bg-destructive px-4 text-sm font-medium text-white shadow-xs transition-colors hover:bg-destructive/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
      cancelButton:
        "inline-flex h-10 items-center justify-center rounded-md border bg-background px-4 text-sm font-medium text-foreground shadow-xs transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
    },
  })
  return result.isConfirmed
}

type StreamSummary = {
  seconds?: number
  cancelled?: boolean
  output_chars?: number
  output_tokens?: number
  tokens_per_second?: number
}

// Coalesce high-frequency token updates into at most one React state write per animation
// frame (~60/s). Without this, fast / prompt-lookup-bursty streams trigger one re-render per
// token (hundreds per response). apply() always receives the latest accumulated text.
function rafBatcher(apply: (text: string) => void) {
  let acc = ""
  let scheduled = false
  let done = false
  const flush = () => {
    scheduled = false
    if (!done) apply(acc)
  }
  return {
    push(token: string) {
      acc += token
      if (!scheduled && !done) {
        scheduled = true
        requestAnimationFrame(flush)
      }
    },
    finish() {
      done = true
      apply(acc)
      return acc
    },
  }
}

// POST a request and consume the Server-Sent Events token stream. Calls onToken for each
// incremental token and returns the final summary event (or null). An optional AbortSignal
// tears the stream down on cancel/unmount.
async function streamPost(
  path: string,
  body: unknown,
  onToken: (token: string) => void,
  signal?: AbortSignal,
): Promise<StreamSummary | null> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok || !response.body) {
    let message = "API hiba"
    try {
      const payload = await response.json()
      if (typeof payload?.error === "string") message = payload.error
    } catch {
      // non-JSON error body; keep the generic message
    }
    throw new Error(message)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let summary: StreamSummary | null = null

  try {
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let sep = buffer.indexOf("\n\n")
      while (sep !== -1) {
        const frame = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        sep = buffer.indexOf("\n\n")
        const data = frame.startsWith("data: ") ? frame.slice(6) : frame
        if (!data || data === "[DONE]") continue
        let event: Record<string, unknown>
        try {
          event = JSON.parse(data) as Record<string, unknown>
        } catch {
          continue
        }
        if (typeof event.token === "string") {
          onToken(event.token)
        } else if (typeof event.error === "string") {
          throw new Error(event.error)
        } else if (event.done) {
          summary = {
            seconds: typeof event.seconds === "number" ? event.seconds : undefined,
            cancelled: typeof event.cancelled === "boolean" ? event.cancelled : undefined,
            output_chars: typeof event.output_chars === "number" ? event.output_chars : undefined,
            output_tokens: typeof event.output_tokens === "number" ? event.output_tokens : undefined,
            tokens_per_second: typeof event.tokens_per_second === "number" ? event.tokens_per_second : undefined,
          }
        }
      }
    }
  } catch (error) {
    // A client-side abort (cancel / unmount) is expected; surface what we have instead of
    // throwing so the UI shows the partial answer rather than an error.
    if (signal?.aborted) return summary ?? { cancelled: true }
    throw error
  }
  return summary
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"
  return (
    <div className={cn("flex gap-2.5", isUser ? "flex-row-reverse" : "flex-row")}>
      <div
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
        )}
      >
        {isUser ? "Te" : "AI"}
      </div>
      <div className={cn("flex min-w-0 max-w-[85%] flex-col gap-1", isUser ? "items-end" : "items-start")}>
        <span className="text-xs font-medium text-muted-foreground">{isUser ? "Te" : "Asszisztens"}</span>
        <div
          className={cn(
            "whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-sm leading-relaxed shadow-sm sm:text-[15px]",
            isUser
              ? "rounded-tr-md bg-primary text-primary-foreground"
              : "rounded-tl-md border bg-card text-foreground",
          )}
        >
          {message.content}
        </div>
      </div>
    </div>
  )
}

function ChatEmptyState() {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 py-12 text-center">
      <div className="flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
        <MessageSquare className="size-7" />
      </div>
      <div className="space-y-1">
        <p className="text-base font-medium text-foreground">Kezdj el beszelgetni</p>
        <p className="max-w-sm text-sm leading-relaxed text-muted-foreground">
          Ird be az elso uzenetet lent, es a betoltott lokalis modell valaszol.
        </p>
      </div>
    </div>
  )
}

function App() {
  const [hardware, setHardware] = useState<HardwareProfile | null>(null)
  const [presets, setPresets] = useState<Preset[]>([])
  const [providerPresets, setProviderPresets] = useState<ProviderPreset[]>([])
  const [status, setStatus] = useState<Status | null>(null)
  const [presetIndex, setPresetIndex] = useState("custom")
  const [loadForm, setLoadForm] = useState<LoadForm>(defaultLoadForm)
  const [providerForm, setProviderForm] = useState<ProviderForm>(defaultProviderForm)
  const [generateForm, setGenerateForm] = useState<GenerateForm>(defaultGenerateForm)
  const [output, setOutput] = useState("")
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState("")
  const [agentObjective, setAgentObjective] = useState("Nezd at ezt a projektet es javasolj kovetkezo fejlesztesi lepeseket.")
  const [agentWorkspace, setAgentWorkspace] = useState("")
  const [agentContextChars, setAgentContextChars] = useState("16000")
  const [agentOutput, setAgentOutput] = useState<AgentResponse | null>(null)
  const [benchmark, setBenchmark] = useState<BenchmarkResult | null>(null)
  const [modelCache, setModelCache] = useState<ModelCacheInfo | null>(null)
  const [downloadStatus, setDownloadStatus] = useState<ModelDownloadStatus | null>(null)
  const [hfModels, setHfModels] = useState<HuggingFaceModelsResponse | null>(null)
  const [hfModelQuery, setHfModelQuery] = useState("Qwen2.5 Instruct")
  const [hfModelsBusy, setHfModelsBusy] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [cacheBusy, setCacheBusy] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [initializing, setInitializing] = useState(true)
  const streamAbortRef = useRef<AbortController | null>(null)

  const gpuLabel = useMemo(() => {
    if (hardware?.cuda.available && hardware.cuda.devices.length) {
      return hardware.cuda.devices
        .map((device) => `${device.name} (${device.total_memory_gb ?? "?"} GB)`)
        .join(", ")
    }
    if (hardware?.mps.available) return "Apple Metal / MPS"
    if (hardware?.mps.built) return "MPS telepitve, nem elerheto"
    if (hardware?.platform.toLowerCase().includes("darwin")) return "Mac CPU fallback"
    return "nincs CUDA/MPS"
  }, [hardware])

  function addLog(message: string) {
    const stamp = new Date().toLocaleTimeString()
    setLogs((current) => [`[${stamp}] ${message}`, ...current].slice(0, 80))
  }

  function updateLoad<K extends keyof LoadForm>(key: K, value: LoadForm[K]) {
    setLoadForm((current) => ({ ...current, [key]: value }))
  }

  function updateGenerate<K extends keyof GenerateForm>(key: K, value: GenerateForm[K]) {
    setGenerateForm((current) => ({ ...current, [key]: value }))
  }

  function applyTaskMode(mode: string) {
    const preset = TASK_PRESETS[mode] ?? TASK_PRESETS.chat
    setGenerateForm((current) => ({ ...current, task_mode: mode, ...preset }))
  }

  function updateProvider<K extends keyof ProviderForm>(key: K, value: ProviderForm[K]) {
    setProviderForm((current) => ({ ...current, [key]: value }))
  }

  async function refreshStatus() {
    const nextStatus = await api<Status>("/api/status")
    setStatus(nextStatus)
  }

  async function refreshHardware() {
    const nextHardware = await api<HardwareProfile>("/api/hardware")
    setHardware(nextHardware)
    setLoadForm((current) => ({
      ...current,
      max_seq_len: String(nextHardware.recommendation.max_seq_len),
      cleanup_interval: String(nextHardware.recommendation.cleanup_interval),
      prefetch_workers: String(nextHardware.recommendation.prefetch_workers),
      reinitialize_model_each_forward: nextHardware.recommendation.reinitialize_model_each_forward,
    }))
    setGenerateForm((current) => ({
      ...current,
      max_length: String(Math.min(512, nextHardware.recommendation.max_seq_len)),
      max_new_tokens: String(nextHardware.recommendation.max_new_tokens),
    }))
    addLog(
      `ajanlott: ${nextHardware.recommendation.device}, ${nextHardware.recommendation.dtype}, compression=${nextHardware.recommendation.compression}`,
    )
  }

  async function refreshPresets() {
    const data = await api<{ presets: Preset[]; families: string[] }>("/api/presets")
    setPresets(data.presets)
    if (data.presets[0]) {
      setPresetIndex("0")
      updateLoad("model_id", data.presets[0].model_id)
    }
  }

  async function refreshProviders() {
    const data = await api<{ providers: ProviderPreset[] }>("/api/providers")
    setProviderPresets(data.providers)
    if (data.providers[0]) {
      setProviderForm((current) => ({
        ...current,
        provider_preset: "0",
        external_base_url: data.providers[0].base_url,
        external_model: data.providers[0].model,
      }))
    }
  }

  async function refreshModelCache() {
    const data = await api<ModelCacheInfo>("/api/models")
    setModelCache(data)
    setDownloadStatus(data.download)
  }

  async function refreshDownloadStatus() {
    const data = await api<ModelDownloadStatus>("/api/download/status")
    setDownloadStatus(data)
  }

  async function refreshHfModels(query = hfModelQuery) {
    setHfModelsBusy(true)
    try {
      const params = new URLSearchParams({
        q: query.trim() || "Qwen2.5 Instruct",
        limit: "20",
      })
      const data = await api<HuggingFaceModelsResponse>(`/api/hf-models?${params}`)
      setHfModels(data)
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Hugging Face lista hiba")
    } finally {
      setHfModelsBusy(false)
    }
  }

  function requestPayload() {
    return {
      ...loadForm,
      ...providerForm,
      load_config: loadForm,
    }
  }

  async function handleLoad() {
    setBusy(true)
    addLog("modell betoltese indul")
    try {
      const nextStatus = await api<Status & { reused: boolean }>("/api/load", {
        method: "POST",
        body: JSON.stringify(loadForm),
      })
      setStatus(nextStatus)
      addLog(nextStatus.reused ? "a modell mar be volt toltve" : `betoltes kesz: ${nextStatus.load_seconds}s`)
      await refreshModelCache()
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Betoltesi hiba")
    } finally {
      setBusy(false)
    }
  }

  async function handleUnload() {
    setBusy(true)
    try {
      const nextStatus = await api<Status>("/api/unload", {
        method: "POST",
        body: "{}",
      })
      setStatus(nextStatus)
      setOutput("")
      setAgentOutput(null)
      addLog("modell kiuritve")
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Kiuritesi hiba")
    } finally {
      setBusy(false)
    }
  }

  async function handleDownloadModel(modelIdOverride?: string) {
    const modelId = (modelIdOverride ?? loadForm.model_id).trim()
    if (modelIdOverride) {
      setPresetIndex("custom")
      updateLoad("model_id", modelId)
    }
    setCacheBusy(true)
    addLog(`letoltes inditasa: ${modelId}`)
    try {
      const next = await api<ModelDownloadStatus>("/api/download", {
        method: "POST",
        body: JSON.stringify({
          model_id: modelId,
          hf_token: loadForm.hf_token,
        }),
      })
      setDownloadStatus(next)
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Letoltesi hiba")
    } finally {
      setCacheBusy(false)
    }
  }

  async function handleCancelDownload() {
    addLog("modell letoltes leallitasa")
    try {
      const next = await api<ModelDownloadStatus>("/api/download/cancel", {
        method: "POST",
        body: "{}",
      })
      setDownloadStatus(next)
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Letoltes megszakitasi hiba")
    }
  }

  async function handleDeleteCachedModel(modelId: string) {
    if (!(await confirmModelDelete(modelId))) return
    setCacheBusy(true)
    addLog(`cache torles indul: ${modelId}`)
    try {
      const next = await api<ModelCacheInfo & { deleted?: boolean; deleted_gb?: number }>("/api/models/delete", {
        method: "POST",
        body: JSON.stringify({ model_id: modelId, unload_if_loaded: true }),
      })
      setModelCache(next)
      setDownloadStatus(next.download)
      await Promise.allSettled([refreshStatus(), refreshHardware()])
      addLog(next.deleted ? `torolve: ${modelId}, ${next.deleted_gb ?? "?"} GB` : "nincs torolheto cache")
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Torlesi hiba")
    } finally {
      setCacheBusy(false)
    }
  }

  function handleUseCachedModel(modelId: string) {
    setPresetIndex("custom")
    updateLoad("model_id", modelId)
    addLog(`kivalasztva: ${modelId}`)
  }

  function handleUseHfModel(modelId: string) {
    setPresetIndex("custom")
    updateLoad("model_id", modelId)
    addLog(`HF modell kivalasztva: ${modelId}`)
  }

  async function handleCancel() {
    try {
      // Cooperative server-side stop first (lets the worker thread halt cleanly)...
      await api<{ cancel_requested: boolean }>("/api/cancel", {
        method: "POST",
        body: "{}",
      })
      addLog("megszakitas kerese elkuldve")
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Megszakitasi hiba")
    } finally {
      // ...then tear down the client stream so the UI reacts instantly.
      streamAbortRef.current?.abort()
    }
  }

  async function handleOptimize() {
    setBusy(true)
    try {
      const result = await api<{ cpu_threads: number; tf32: boolean; cudnn_benchmark: boolean }>("/api/optimize", {
        method: "POST",
        body: "{}",
      })
      addLog(`optimalizalva: cpu_threads=${result.cpu_threads}, tf32=${result.tf32}`)
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Optimalizalasi hiba")
    } finally {
      setBusy(false)
    }
  }

  async function handleBenchmark() {
    setBusy(true)
    addLog("benchmark indul")
    try {
      const result = await api<BenchmarkResult>("/api/benchmark", {
        method: "POST",
        body: JSON.stringify({ model_probe: Boolean(status?.loaded) }),
      })
      setBenchmark(result)
      addLog(
        `benchmark: gpu=${pretty(result.gpu_matmul_ms, "?")}ms, cpu=${pretty(result.cpu_matmul_ms, "?")}ms, disk write=${pretty(result.disk_write_mbps, "?")} MB/s`,
      )
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Benchmark hiba")
    } finally {
      setBusy(false)
    }
  }

  async function handleGenerate() {
    setBusy(true)
    setOutput("")
    addLog("generalas indul")
    try {
      if (providerForm.provider === "local") {
        // Local models stream token-by-token over SSE so output appears as it is produced.
        const controller = new AbortController()
        streamAbortRef.current = controller
        const batch = rafBatcher(setOutput)
        let summary: StreamSummary | null = null
        try {
          summary = await streamPost(
            "/api/generate",
            { ...requestPayload(), ...generateForm, stream: true },
            (token) => batch.push(token),
            controller.signal,
          )
        } finally {
          streamAbortRef.current = null
        }
        const acc = batch.finish()
        if (!acc) setOutput("(ures valasz)")
        await refreshStatus()
        const tps = summary?.tokens_per_second ? `, ${summary.tokens_per_second} tok/s` : ""
        addLog(`kesz: ${summary?.seconds ?? "?"}s, tokens=${summary?.output_tokens ?? "?"}${tps}`)
      } else {
        const result = await api<{
          text: string
          seconds: number
          input_tokens: number
          output_tokens: number
          status: Status
        }>("/api/generate", {
          method: "POST",
          body: JSON.stringify({ ...requestPayload(), ...generateForm }),
        })
        setOutput(result.text)
        setStatus(result.status)
        addLog(`kesz: ${result.seconds}s, input=${result.input_tokens}, output=${result.output_tokens}`)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Generalasi hiba"
      setOutput(message)
      addLog(message)
    } finally {
      setBusy(false)
    }
  }

  async function handleChatSend() {
    const content = chatInput.trim()
    if (!content) return

    const nextMessages: ChatMessage[] = [...chatMessages, { role: "user", content }]
    setChatMessages(nextMessages)
    setChatInput("")
    setBusy(true)
    addLog("chat uzenet kuldese")
    try {
      if (providerForm.provider === "local") {
        // Stream the reply into a live assistant bubble token by token.
        setChatMessages((current) => [...current, { role: "assistant", content: "" }])
        const controller = new AbortController()
        streamAbortRef.current = controller
        const applyToBubble = (text: string) =>
          setChatMessages((current) => {
            const copy = current.slice()
            copy[copy.length - 1] = { role: "assistant", content: text }
            return copy
          })
        const batch = rafBatcher(applyToBubble)
        let summary: StreamSummary | null = null
        try {
          summary = await streamPost(
            "/api/chat",
            { ...requestPayload(), ...generateForm, messages: nextMessages, stream: true },
            (token) => batch.push(token),
            controller.signal,
          )
        } finally {
          streamAbortRef.current = null
        }
        batch.finish()
        await refreshStatus()
        const tps = summary?.tokens_per_second ? `, ${summary.tokens_per_second} tok/s` : ""
        addLog(`chat kesz: ${summary?.seconds ?? "?"}s${tps}`)
      } else {
        const result = await api<{
          message: ChatMessage
          seconds: number
          input_tokens: number
          output_tokens: number
          status: Status
        }>("/api/chat", {
          method: "POST",
          body: JSON.stringify({ ...requestPayload(), ...generateForm, messages: nextMessages }),
        })
        setChatMessages((current) => [...current, result.message])
        setStatus(result.status)
        addLog(`chat kesz: ${result.seconds}s, input=${result.input_tokens}, output=${result.output_tokens}`)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat hiba"
      setChatMessages((current) => {
        const copy = current.slice()
        if (copy.length && copy[copy.length - 1].role === "assistant") {
          copy[copy.length - 1] = { role: "assistant", content: message }
        } else {
          copy.push({ role: "assistant", content: message })
        }
        return copy
      })
      addLog(message)
    } finally {
      setBusy(false)
    }
  }

  async function handleAgentRun() {
    setBusy(true)
    setAgentOutput(null)
    addLog("coding agent inditasa")
    try {
      const result = await api<AgentResponse>("/api/agent/run", {
        method: "POST",
        body: JSON.stringify({
          ...requestPayload(),
          ...generateForm,
          objective: agentObjective,
          workspace_path: agentWorkspace,
          max_context_chars: agentContextChars,
          temperature: "0.2",
          max_new_tokens: Math.max(Number(generateForm.max_new_tokens) || 0, 900),
        }),
      })
      setAgentOutput(result)
      setStatus(result.status)
      addLog(`agent kesz: ${result.seconds}s, context files=${result.included_files.length}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Coding agent hiba"
      setAgentOutput({
        text: message,
        seconds: 0,
        input_tokens: 0,
        output_tokens: 0,
        workspace: agentWorkspace,
        included_files: [],
        context_files: [],
        status: status ?? { loaded: false, config: null, loaded_at: null, load_seconds: null },
      })
      addLog(message)
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => {
    void (async () => {
      // These four boot calls are independent -> run them concurrently (was sequential, so
      // boot latency was the SUM of all four incl. the ~134ms hardware probe). allSettled so
      // one failure does not block the others.
      const results = await Promise.allSettled([
        refreshPresets(),
        refreshProviders(),
        refreshHardware(),
        refreshStatus(),
        refreshModelCache(),
        refreshHfModels(),
      ])
      for (const result of results) {
        if (result.status === "rejected") {
          addLog(result.reason instanceof Error ? result.reason.message : "Inditasi hiba")
        }
      }
      setInitializing(false)
    })()
    return () => {
      // Abort any in-flight stream if the component unmounts.
      streamAbortRef.current?.abort()
    }
  }, [])

  useEffect(() => {
    if (!downloadStatus?.active) return
    const timer = window.setInterval(() => {
      void refreshDownloadStatus()
    }, 1000)
    return () => window.clearInterval(timer)
  }, [downloadStatus?.active])

  useEffect(() => {
    if (downloadStatus?.status !== "done") return
    void refreshModelCache()
    addLog(`letoltes kesz: ${downloadStatus.model_id}`)
  }, [downloadStatus?.status])

  const statusLabel = initializing
    ? "Betoltes..."
    : providerForm.provider === "local"
      ? status?.loaded
        ? "Betoltve"
        : "Ures"
      : "Kulso provider"

  const modelLabel =
    providerForm.provider === "local"
      ? status?.loaded
        ? status.config?.model_id ?? "Nincs betoltott modell"
        : "Nincs betoltott modell"
      : providerForm.external_model || "Nincs kulso model megadva"

  const configPanelProps = {
    hardware,
    gpuLabel,
    benchmark,
    busy,
    cacheBusy,
    logs,
    modelCache,
    downloadStatus,
    hfModels,
    hfModelQuery,
    hfModelsBusy,
    providerForm,
    providerPresets,
    updateProvider,
    presetIndex,
    setPresetIndex,
    presets,
    loadForm,
    updateLoad,
    onBenchmark: () => void handleBenchmark(),
    onLoad: () => void handleLoad(),
    onUnload: () => void handleUnload(),
    onOptimize: () => void handleOptimize(),
    onCancel: () => void handleCancel(),
    onRefreshModels: () => void refreshModelCache(),
    onDownloadModel: (modelId?: string) => void handleDownloadModel(modelId),
    onCancelDownload: () => void handleCancelDownload(),
    onDeleteCachedModel: (modelId: string) => void handleDeleteCachedModel(modelId),
    onUseCachedModel: handleUseCachedModel,
    onUseHfModel: handleUseHfModel,
    onSearchHfModels: () => void refreshHfModels(),
    setHfModelQuery,
  }

  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden bg-background text-foreground">
      <header className="z-20 shrink-0 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="flex w-full flex-col gap-2 px-3 py-2 sm:flex-row sm:items-center sm:justify-between sm:px-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm sm:size-10">
              <Bot className="size-5" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-semibold tracking-normal">AirLLM Control</h1>
              <p className="truncate text-sm text-muted-foreground">lokalis chat es coding agent</p>
            </div>
          </div>
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="xl:hidden"
              onClick={() => setSettingsOpen(true)}
            >
              <Settings className="size-4" />
              Beallitasok
            </Button>
            <Badge variant={initializing ? "secondary" : status?.loaded ? "default" : "secondary"}>
              {statusLabel}
            </Badge>
            <span
              className="hidden rounded-full border bg-muted/50 px-3 py-1 text-xs text-muted-foreground sm:inline-block sm:max-w-[520px] sm:truncate"
              title={modelLabel}
            >
              {modelLabel}
            </span>
          </div>
        </div>
      </header>

      <main className="grid min-h-0 w-full flex-1 grid-cols-1 gap-3 overflow-hidden p-3 xl:grid-cols-[380px_minmax(0,1fr)]">
        <div className="hidden min-h-0 min-w-0 overflow-y-auto pr-1 xl:block">
          <ConfigPanel {...configPanelProps} layout="sidebar" />
        </div>

        <Tabs defaultValue="chat" className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden">
          <TabsList className="mb-3 grid h-auto w-full shrink-0 grid-cols-3 gap-1 rounded-lg border bg-muted p-1 shadow-sm">
            <TabsTrigger className="h-10 w-full justify-center gap-2 rounded-lg px-3" value="chat">
              <MessageSquare className="size-4 shrink-0" />
              Chat
            </TabsTrigger>
            <TabsTrigger className="h-10 w-full justify-center gap-2 rounded-lg px-3" value="agent">
              <Code2 className="size-4 shrink-0" />
              Agent
            </TabsTrigger>
            <TabsTrigger className="h-10 w-full justify-center gap-2 rounded-lg px-3" value="generate">
              <Play className="size-4 shrink-0" />
              Generate
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="mt-0 flex min-h-0 flex-1 flex-col data-[state=inactive]:hidden">
            <Card className="flex min-h-0 flex-1 flex-col overflow-hidden shadow-sm">
              <CardHeader className="shrink-0 border-b bg-muted/20 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <CardTitle>Beszelgetes</CardTitle>
                    <CardDescription className="mt-1">
                      A betoltott lokalis modell valaszol a beszelgetesben
                    </CardDescription>
                  </div>
                  {status?.loaded && (
                    <Badge variant="outline" className="shrink-0">
                      {providerForm.provider === "local"
                        ? `${status.config?.device} / ${status.config?.dtype}${status.mode ? ` · ${status.mode}` : ""}`
                        : providerForm.external_model}
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="flex min-h-0 flex-1 flex-col p-0">
                <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto bg-muted/10 p-4">
                  {chatMessages.length ? (
                    chatMessages.map((message, index) => <MessageBubble key={index} message={message} />)
                  ) : (
                    <ChatEmptyState />
                  )}
                </div>
                <div className="shrink-0 border-t bg-background/95 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] backdrop-blur">
                  <div className="grid gap-2">
                    <Textarea
                      className="min-h-20 resize-none bg-background text-[15px] leading-relaxed"
                      placeholder="Uzenet... (Ctrl+Enter a kuldeshez)"
                      value={chatInput}
                      onChange={(event) => setChatInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                          event.preventDefault()
                          void handleChatSend()
                        }
                      }}
                    />
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                      <Button
                        className="w-full sm:w-auto"
                        onClick={() => void handleChatSend()}
                        disabled={busy || !chatInput.trim()}
                      >
                        {busy ? <LoaderCircle className="animate-spin" /> : <Send />}
                        Kuldes
                      </Button>
                      <Button
                        className="w-full sm:w-auto"
                        variant="outline"
                        onClick={() => setChatMessages([])}
                        disabled={busy}
                      >
                        <Trash2 />
                        Chat torles
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="agent" className="mt-0 grid min-h-0 flex-1 grid-rows-[auto_minmax(0,1fr)] gap-3 overflow-hidden data-[state=inactive]:hidden">
            <Card className="shrink-0">
              <CardHeader>
                <CardTitle>Coding Agent</CardTitle>
                <CardDescription>Egy gombbal indithato lokalis kodos asszisztens</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <Field label="Feladat" hint={configHints.agentObjective}>
                  <Textarea
                    className="min-h-28 resize-y"
                    value={agentObjective}
                    onChange={(event) => setAgentObjective(event.target.value)}
                  />
                </Field>
                <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_180px]">
                  <Field label="Workspace utvonal" hint={configHints.agentWorkspace}>
                    <Input
                      value={agentWorkspace}
                      placeholder="uresen hagyva: projekt root"
                      onChange={(event) => setAgentWorkspace(event.target.value)}
                    />
                  </Field>
                  <Field label="Context karakter" hint={configHints.agentContextChars}>
                    <Input
                      type="number"
                      min={4000}
                      max={64000}
                      step={1000}
                      value={agentContextChars}
                      onChange={(event) => setAgentContextChars(event.target.value)}
                    />
                  </Field>
                </div>
                <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
                  <Button className="w-full sm:w-auto" size="lg" onClick={handleAgentRun} disabled={busy || !agentObjective.trim()}>
                    {busy ? <LoaderCircle className="animate-spin" /> : <Code2 />}
                    Agent inditasa
                  </Button>
                  <Badge variant="secondary">nem ir fajlt automatikusan</Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="flex min-h-0 flex-col overflow-hidden">
              <CardHeader>
                <CardTitle>Agent valasz</CardTitle>
                <CardDescription>
                  {agentOutput
                    ? `${agentOutput.seconds}s, input=${agentOutput.input_tokens}, output=${agentOutput.output_tokens}`
                    : "A coding agent eredmenye itt jelenik meg"}
                </CardDescription>
              </CardHeader>
              <CardContent className="flex min-h-0 flex-1 flex-col gap-4">
                <div className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border bg-slate-950 p-3 text-sm leading-relaxed text-slate-50 sm:p-4 sm:text-[15px]">
                  {agentOutput?.text ?? ""}
                </div>
                {agentOutput && (
                  <div className="grid gap-3 rounded-lg border bg-muted/35 p-3 text-sm">
                    <div>
                      <span className="font-medium">Workspace: </span>
                      <span className="break-all text-muted-foreground">{agentOutput.workspace}</span>
                    </div>
                    <Separator />
                    <div>
                      <span className="font-medium">Beolvasott fontos fajlok: </span>
                      <span className="text-muted-foreground">
                        {agentOutput.included_files.length ? agentOutput.included_files.join(", ") : "nincs"}
                      </span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="generate" className="mt-0 grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,0.95fr)_minmax(0,1.05fr)] gap-3 overflow-hidden data-[state=inactive]:hidden lg:grid-cols-[minmax(340px,0.85fr)_minmax(0,1.15fr)] lg:grid-rows-1">
            <Card className="flex min-h-0 flex-col overflow-hidden">
              <CardHeader>
                <CardTitle>Prompt</CardTitle>
                <CardDescription>Egyszeri inference es sampling kontrollok</CardDescription>
              </CardHeader>
              <CardContent className="grid min-h-0 flex-1 gap-4 overflow-y-auto">
                <Textarea
                  className="min-h-48 resize-y text-base leading-relaxed lg:min-h-64"
                  value={generateForm.prompt}
                  onChange={(event) => updateGenerate("prompt", event.target.value)}
                />
                <GenerationControls generateForm={generateForm} updateGenerate={updateGenerate} onTaskMode={applyTaskMode} />
                <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
                  <Button className="w-full sm:w-auto" size="lg" onClick={handleGenerate} disabled={busy}>
                    {busy ? <LoaderCircle className="animate-spin" /> : <Play />}
                    General
                  </Button>
                  {status?.loaded && (
                    <Badge variant="outline">
                      {providerForm.provider === "local"
                        ? `${status.config?.device} / ${status.config?.dtype}${status.mode ? ` · ${status.mode}` : ""}`
                        : providerForm.external_model}
                    </Badge>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card className="flex min-h-0 flex-col overflow-hidden">
              <CardHeader>
                <CardTitle>Kimenet</CardTitle>
                <CardDescription>
                  {status?.loaded && status.load_seconds
                    ? `Aktualis modell betoltesi ideje: ${status.load_seconds}s`
                    : "A valasz itt jelenik meg"}
                </CardDescription>
              </CardHeader>
              <CardContent className="flex min-h-0 flex-1">
                <div className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap break-words rounded-lg border bg-slate-950 p-3 text-sm leading-relaxed text-slate-50 sm:p-4 sm:text-[15px]">
                  {output}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      <Sheet open={settingsOpen} onOpenChange={setSettingsOpen}>
        <SheetContent side="right" className="w-full overflow-y-auto sm:max-w-lg">
          <SheetHeader>
            <SheetTitle>Beallitasok</SheetTitle>
            <SheetDescription>Hardver, provider, modell es naplo</SheetDescription>
          </SheetHeader>
          <div className="overflow-y-auto pb-[max(1rem,env(safe-area-inset-bottom))]">
            <ConfigPanel {...configPanelProps} layout="stack" />
          </div>
        </SheetContent>
      </Sheet>
    </div>
  )
}

function GenerationControls({
  generateForm,
  updateGenerate,
  onTaskMode,
}: {
  generateForm: GenerateForm
  updateGenerate: <K extends keyof GenerateForm>(key: K, value: GenerateForm[K]) => void
  onTaskMode: (mode: string) => void
}) {
  return (
    <>
      <Field
        label="Feladat mod"
        hint="Chat: kreativ alapertekek. Tenyszeru: alacsony homerseklet, enyhe ismetles-buntetes. Kod: greedy dekodolas, nincs ismetles-buntetes."
      >
        <SelectField
          value={generateForm.task_mode}
          onValueChange={onTaskMode}
          options={[
            { value: "chat", label: "Chat / kreativ" },
            { value: "factual", label: "Tenyszeru" },
            { value: "code", label: "Kod" },
          ]}
        />
      </Field>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="Input max" hint={configHints.inputMax}>
          <Input
            type="number"
            min={16}
            max={32768}
            step={16}
            value={generateForm.max_length}
            onChange={(event) => updateGenerate("max_length", event.target.value)}
          />
        </Field>
        <Field label="New tokens" hint={configHints.newTokens}>
          <Input
            type="number"
            min={1}
            max={4096}
            value={generateForm.max_new_tokens}
            onChange={(event) => updateGenerate("max_new_tokens", event.target.value)}
          />
        </Field>
        <Field label="Temperature" hint={configHints.temperature}>
          <Input
            type="number"
            min={0}
            max={2}
            step={0.05}
            value={generateForm.temperature}
            onChange={(event) => updateGenerate("temperature", event.target.value)}
          />
        </Field>
        <Field label="Top p" hint={configHints.topP}>
          <Input
            type="number"
            min={0.05}
            max={1}
            step={0.01}
            value={generateForm.top_p}
            onChange={(event) => updateGenerate("top_p", event.target.value)}
          />
        </Field>
        <Field label="Top k" hint={configHints.topK}>
          <Input
            type="number"
            min={0}
            max={1000}
            value={generateForm.top_k}
            onChange={(event) => updateGenerate("top_k", event.target.value)}
          />
        </Field>
        <Field label="Repeat penalty" hint={configHints.repeatPenalty}>
          <Input
            type="number"
            min={0.8}
            max={2}
            step={0.01}
            value={generateForm.repetition_penalty}
            onChange={(event) => updateGenerate("repetition_penalty", event.target.value)}
          />
        </Field>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <ToggleField
          label="Autoload"
          hint={configHints.autoload}
          checked={generateForm.autoload}
          onCheckedChange={(checked) => updateGenerate("autoload", checked)}
        />
        <ToggleField
          label="KV cache"
          hint={configHints.kvCache}
          checked={generateForm.use_cache}
          onCheckedChange={(checked) => updateGenerate("use_cache", checked)}
        />
        <ToggleField
          label="Chat template"
          hint={configHints.chatTemplate}
          checked={generateForm.use_chat_template}
          onCheckedChange={(checked) => updateGenerate("use_chat_template", checked)}
        />
      </div>
    </>
  )
}

export default App
