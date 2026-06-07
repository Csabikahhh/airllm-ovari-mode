import { useEffect, useMemo, useState } from "react"
import {
  Activity,
  Bot,
  Cloud,
  Code2,
  Cpu,
  Gauge,
  HardDrive,
  LoaderCircle,
  MessageSquare,
  Play,
  Power,
  RefreshCw,
  Send,
  Server,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react"
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
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

type Preset = {
  label: string
  model_id: string
  family: string
  size: string
}

type ProviderPreset = {
  label: string
  base_url: string
  model: string
  needs_key: boolean
}

type HardwareProfile = {
  platform: string
  python: string
  cpu: {
    name: string
    logical_cores: number
  }
  memory: {
    total_gb: number | null
    available_gb: number | null
  }
  disk: {
    hf_home: string
    total_gb: number | null
    free_gb: number | null
  }
  power: {
    active_scheme: string | null
    raw: string | null
  }
  network: {
    interface: string | null
    radio: string | null
    receive_mbps: number | string | null
    transmit_mbps: number | string | null
    signal: string | null
  }
  cuda: {
    available: boolean
    error: string | null
    devices: Array<{
      index: number
      name: string
      total_memory_gb: number | null
      compute_capability: string
    }>
  }
  torch: {
    available: boolean
    version: string | null
    error: string | null
  }
  bitsandbytes: {
    available: boolean
  }
  supported_families: string[]
  recommendation: {
    device: string
    dtype: string
    compression: string
    prefetching: boolean
    cleanup_interval: number
    prefetch_workers: number
    reinitialize_model_each_forward: boolean
    max_seq_len: number
    max_new_tokens: number
  }
}

type Status = {
  loaded: boolean
  config: null | {
    provider?: string
    base_url?: string
    model_id: string
    device?: string
    dtype?: string
    compression?: string | null
    max_seq_len?: number
    prefetching?: boolean
    cleanup_interval?: number
    prefetch_workers?: number
    reinitialize_model_each_forward?: boolean
    hf_token_set?: boolean
    api_key_set?: boolean
  }
  loaded_at: number | null
  load_seconds: number | null
}

type LoadForm = {
  model_id: string
  device: string
  dtype: string
  compression: string
  prefetching: string
  cleanup_interval: string
  prefetch_workers: string
  reinitialize_model_each_forward: boolean
  max_seq_len: string
  layer_shards_saving_path: string
  hf_token: string
  profiling_mode: boolean
  delete_original: boolean
}

type GenerateForm = {
  prompt: string
  max_length: string
  max_new_tokens: string
  temperature: string
  top_p: string
  top_k: string
  repetition_penalty: string
  autoload: boolean
  use_cache: boolean
  use_chat_template: boolean
}

type ProviderForm = {
  provider: "local" | "openai_compatible"
  provider_preset: string
  external_base_url: string
  external_model: string
  external_api_key: string
  external_timeout: string
}

type ChatMessage = {
  role: "user" | "assistant"
  content: string
}

type AgentResponse = {
  text: string
  seconds: number
  input_tokens: number
  output_tokens: number
  workspace: string
  included_files: string[]
  context_files: string[]
  status: Status
}

type BenchmarkResult = {
  gpu_matmul_ms: number | null
  cpu_matmul_ms: number | null
  disk_write_mbps: number | null
  disk_read_mbps: number | null
  model_probe: null | {
    seconds?: number
    input_tokens?: number
    output_tokens?: number
    error?: string
  }
}

const defaultLoadForm: LoadForm = {
  model_id: "Qwen/Qwen2.5-3B-Instruct",
  device: "auto",
  dtype: "auto",
  compression: "auto",
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

function pretty(value: string | number | null | undefined, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback
  return value
}

function Field({
  label,
  children,
  className,
}: {
  label: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn("grid gap-2", className)}>
      <Label>{label}</Label>
      {children}
    </div>
  )
}

function Metric({
  icon,
  label,
  value,
  tone = "default",
}: {
  icon: React.ReactNode
  label: string
  value: React.ReactNode
  tone?: "default" | "good" | "warn"
}) {
  return (
    <div
      className={cn(
        "flex min-h-20 gap-3 rounded-lg border bg-muted/35 p-3",
        tone === "good" && "border-emerald-200 bg-emerald-50",
        tone === "warn" && "border-amber-200 bg-amber-50",
      )}
    >
      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md bg-background text-muted-foreground">
        {icon}
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <div className="mt-1 break-words text-sm font-semibold leading-snug">{value}</div>
      </div>
    </div>
  )
}

function Log({ entries }: { entries: string[] }) {
  return (
    <div className="max-h-44 min-h-28 overflow-auto rounded-md border bg-muted/45 p-3 font-mono text-xs leading-relaxed text-muted-foreground">
      {entries.length ? entries.join("\n") : "Nincs naplo."}
    </div>
  )
}

function SelectField({
  value,
  onValueChange,
  options,
}: {
  value: string
  onValueChange: (value: string) => void
  options: Array<{ value: string; label: string }>
}) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function ToggleField({
  label,
  checked,
  onCheckedChange,
}: {
  label: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}) {
  return (
    <div className="flex items-center justify-between gap-3 rounded-md border bg-background px-3 py-2">
      <Label className="leading-normal">{label}</Label>
      <Switch checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  )
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user"
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[94%] whitespace-pre-wrap break-words rounded-lg border px-3 py-2 text-sm leading-relaxed sm:max-w-[88%]",
          isUser
            ? "border-primary bg-primary text-primary-foreground"
            : "bg-muted/60 text-foreground",
        )}
      >
        {message.content}
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
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)

  const gpuLabel = useMemo(() => {
    if (!hardware?.cuda.available || !hardware.cuda.devices.length) return "nincs CUDA"
    return hardware.cuda.devices
      .map((device) => `${device.name} (${device.total_memory_gb ?? "?"} GB)`)
      .join(", ")
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

  async function handleCancel() {
    try {
      await api<{ cancel_requested: boolean }>("/api/cancel", {
        method: "POST",
        body: "{}",
      })
      addLog("megszakitas kerese elkuldve")
    } catch (error) {
      addLog(error instanceof Error ? error.message : "Megszakitasi hiba")
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
    setOutput("Dolgozom...")
    addLog("generalas indul")
    try {
      const result = await api<{
        text: string
        seconds: number
        input_tokens: number
        output_tokens: number
        status: Status
      }>("/api/generate", {
        method: "POST",
        body: JSON.stringify({
          ...requestPayload(),
          ...generateForm,
        }),
      })
      setOutput(result.text)
      setStatus(result.status)
      addLog(`kesz: ${result.seconds}s, input=${result.input_tokens}, output=${result.output_tokens}`)
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
      const result = await api<{
        message: ChatMessage
        seconds: number
        input_tokens: number
        output_tokens: number
        status: Status
      }>("/api/chat", {
        method: "POST",
        body: JSON.stringify({
          ...requestPayload(),
          ...generateForm,
          messages: nextMessages,
        }),
      })
      setChatMessages((current) => [...current, result.message])
      setStatus(result.status)
      addLog(`chat kesz: ${result.seconds}s, input=${result.input_tokens}, output=${result.output_tokens}`)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Chat hiba"
      setChatMessages((current) => [...current, { role: "assistant", content: message }])
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
      try {
        await refreshPresets()
        await refreshProviders()
        await refreshHardware()
        await refreshStatus()
      } catch (error) {
        addLog(error instanceof Error ? error.message : "Inditasi hiba")
      }
    })()
  }, [])

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-3 px-3 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground sm:size-10">
              <Bot className="size-5" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-semibold tracking-normal">AirLLM Control</h1>
              <p className="truncate text-sm text-muted-foreground">lokalis chat es coding agent</p>
            </div>
          </div>
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            <Badge variant={status?.loaded ? "default" : "secondary"}>
              {providerForm.provider === "local" ? (status?.loaded ? "Betoltve" : "Ures") : "Kulso provider"}
            </Badge>
            <span className="min-w-0 max-w-full truncate text-sm text-muted-foreground sm:max-w-[520px]">
              {providerForm.provider === "local"
                ? status?.loaded
                  ? status.config?.model_id
                  : "Nincs betoltott modell"
                : providerForm.external_model || "Nincs kulso model megadva"}
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto grid w-full max-w-[1500px] grid-cols-1 gap-3 p-3 sm:gap-4 sm:p-4 xl:grid-cols-[430px_minmax(0,1fr)]">
        <div className="grid min-w-0 gap-3 sm:gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Gauge className="size-4" />
                Hardver
              </CardTitle>
              <CardDescription>Automatikus runtime profil</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
              <Metric icon={<Cpu className="size-4" />} label="CPU" value={`${hardware?.cpu.logical_cores ?? "-"} szal`} />
              <Metric
                icon={<Server className="size-4" />}
                label="RAM"
                value={`${pretty(hardware?.memory.available_gb, "?")} / ${pretty(hardware?.memory.total_gb, "?")} GB`}
              />
              <Metric
                icon={<Sparkles className="size-4" />}
                label="GPU"
                value={gpuLabel}
                tone={hardware?.cuda.available ? "good" : "warn"}
              />
              <Metric
                icon={<HardDrive className="size-4" />}
                label="HF cache"
                value={`${pretty(hardware?.disk.free_gb, "?")} GB szabad`}
              />
              <Metric
                icon={<Power className="size-4" />}
                label="Torch"
                value={hardware?.torch.version ?? "nem elerheto"}
                tone={hardware?.torch.available ? "good" : "warn"}
              />
              <Metric
                icon={<RefreshCw className="size-4" />}
                label="bitsandbytes"
                value={hardware?.bitsandbytes.available ? "elerheto" : "nem elerheto"}
                tone={hardware?.bitsandbytes.available ? "good" : "warn"}
              />
              <Metric
                icon={<Power className="size-4" />}
                label="Energia"
                value={hardware?.power.active_scheme ?? "ismeretlen"}
                tone={
                  hardware?.power.active_scheme?.toLowerCase().includes("balanced") ||
                  hardware?.power.active_scheme?.toLowerCase().includes("kiegy")
                    ? "warn"
                    : "default"
                }
              />
              <Metric
                icon={<Activity className="size-4" />}
                label="Halozat"
                value={
                  hardware?.network.radio
                    ? `${hardware.network.radio}, ${pretty(hardware.network.receive_mbps, "?")} Mbps`
                    : "ismeretlen"
                }
              />
              <div className="grid gap-2 sm:col-span-2 xl:col-span-1 2xl:col-span-2">
                <Button className="w-full" variant="outline" onClick={handleBenchmark} disabled={busy}>
                  {busy ? <LoaderCircle className="animate-spin" /> : <Activity />}
                  Benchmark
                </Button>
                {benchmark && (
                  <div className="grid gap-2 rounded-lg border bg-muted/35 p-3 text-xs text-muted-foreground sm:grid-cols-2">
                    <span>GPU matmul: {pretty(benchmark.gpu_matmul_ms, "?")} ms</span>
                    <span>CPU matmul: {pretty(benchmark.cpu_matmul_ms, "?")} ms</span>
                    <span>SSD iras: {pretty(benchmark.disk_write_mbps, "?")} MB/s</span>
                    <span>SSD olvasas: {pretty(benchmark.disk_read_mbps, "?")} MB/s</span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cloud className="size-4" />
                AI szolgaltato
              </CardTitle>
              <CardDescription>Lokalis AirLLM vagy OpenAI-kompatibilis API</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <Field label="Provider">
                <SelectField
                  value={providerForm.provider}
                  onValueChange={(value) => updateProvider("provider", value as ProviderForm["provider"])}
                  options={[
                    { value: "local", label: "Local AirLLM" },
                    { value: "openai_compatible", label: "OpenAI-compatible" },
                  ]}
                />
              </Field>
              {providerForm.provider === "openai_compatible" && (
                <>
                  <Field label="Preset">
                    <SelectField
                      value={providerForm.provider_preset}
                      onValueChange={(value) => {
                        updateProvider("provider_preset", value)
                        const preset = providerPresets[Number(value)]
                        if (preset) {
                          updateProvider("external_base_url", preset.base_url)
                          updateProvider("external_model", preset.model)
                        }
                      }}
                      options={providerPresets.map((preset, index) => ({
                        value: String(index),
                        label: preset.label,
                      }))}
                    />
                  </Field>
                  <Field label="Base URL">
                    <Input
                      value={providerForm.external_base_url}
                      placeholder="https://api.example.com/v1"
                      onChange={(event) => updateProvider("external_base_url", event.target.value)}
                    />
                  </Field>
                  <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_120px]">
                    <Field label="Model">
                      <Input
                        value={providerForm.external_model}
                        placeholder="provider-model-name"
                        onChange={(event) => updateProvider("external_model", event.target.value)}
                      />
                    </Field>
                    <Field label="Timeout">
                      <Input
                        type="number"
                        min={10}
                        max={600}
                        value={providerForm.external_timeout}
                        onChange={(event) => updateProvider("external_timeout", event.target.value)}
                      />
                    </Field>
                  </div>
                  <Field label="API key">
                    <Input
                      type="password"
                      autoComplete="off"
                      value={providerForm.external_api_key}
                      placeholder="csak a kereshez hasznalva"
                      onChange={(event) => updateProvider("external_api_key", event.target.value)}
                    />
                  </Field>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Modell</CardTitle>
              <CardDescription>AirLLM AutoModel beallitasok</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <Field label="Preset">
                <SelectField
                  value={presetIndex}
                  onValueChange={(value) => {
                    setPresetIndex(value)
                    if (value !== "custom") {
                      updateLoad("model_id", presets[Number(value)]?.model_id ?? loadForm.model_id)
                    }
                  }}
                  options={[
                    ...presets.map((preset, index) => ({
                      value: String(index),
                      label: `${preset.label} (${preset.family})`,
                    })),
                    { value: "custom", label: "Egyedi model ID" },
                  ]}
                />
              </Field>
              <Field label="Model ID / utvonal">
                <Input
                  spellCheck={false}
                  value={loadForm.model_id}
                  onChange={(event) => {
                    setPresetIndex("custom")
                    updateLoad("model_id", event.target.value)
                  }}
                />
              </Field>
              <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                <Field label="Device">
                  <SelectField
                    value={loadForm.device}
                    onValueChange={(value) => updateLoad("device", value)}
                    options={[
                      { value: "auto", label: "auto" },
                      { value: "cuda:0", label: "cuda:0" },
                      { value: "cpu", label: "cpu" },
                    ]}
                  />
                </Field>
                <Field label="Dtype">
                  <SelectField
                    value={loadForm.dtype}
                    onValueChange={(value) => updateLoad("dtype", value)}
                    options={[
                      { value: "auto", label: "auto" },
                      { value: "float16", label: "float16" },
                      { value: "bfloat16", label: "bfloat16" },
                      { value: "float32", label: "float32" },
                    ]}
                  />
                </Field>
                <Field label="Compression">
                  <SelectField
                    value={loadForm.compression}
                    onValueChange={(value) => updateLoad("compression", value)}
                    options={[
                      { value: "auto", label: "auto" },
                      { value: "none", label: "none" },
                      { value: "4bit", label: "4bit" },
                      { value: "8bit", label: "8bit" },
                    ]}
                  />
                </Field>
                <Field label="Prefetching">
                  <SelectField
                    value={loadForm.prefetching}
                    onValueChange={(value) => updateLoad("prefetching", value)}
                    options={[
                      { value: "auto", label: "auto" },
                      { value: "true", label: "on" },
                      { value: "false", label: "off" },
                    ]}
                  />
                </Field>
                <Field label="Max seq len">
                  <Input
                    type="number"
                    min={128}
                    max={32768}
                    step={128}
                    value={loadForm.max_seq_len}
                    onChange={(event) => updateLoad("max_seq_len", event.target.value)}
                  />
                </Field>
                <Field label="Cleanup interval">
                  <Input
                    type="number"
                    min={0}
                    max={64}
                    value={loadForm.cleanup_interval}
                    onChange={(event) => updateLoad("cleanup_interval", event.target.value)}
                  />
                </Field>
                <Field label="Prefetch workers">
                  <Input
                    type="number"
                    min={1}
                    max={4}
                    value={loadForm.prefetch_workers}
                    onChange={(event) => updateLoad("prefetch_workers", event.target.value)}
                  />
                </Field>
                <Field label="Layer cache">
                  <Input
                    value={loadForm.layer_shards_saving_path}
                    onChange={(event) => updateLoad("layer_shards_saving_path", event.target.value)}
                  />
                </Field>
              </div>
              <Field label="HF token">
                <Input
                  type="password"
                  autoComplete="off"
                  value={loadForm.hf_token}
                  onChange={(event) => updateLoad("hf_token", event.target.value)}
                />
              </Field>
              <div className="grid gap-3 sm:grid-cols-2">
                <ToggleField
                  label="Profiling"
                  checked={loadForm.profiling_mode}
                  onCheckedChange={(checked) => updateLoad("profiling_mode", checked)}
                />
                <ToggleField
                  label="Delete original"
                  checked={loadForm.delete_original}
                  onCheckedChange={(checked) => updateLoad("delete_original", checked)}
                />
                <ToggleField
                  label="Reinit / forward"
                  checked={loadForm.reinitialize_model_each_forward}
                  onCheckedChange={(checked) => updateLoad("reinitialize_model_each_forward", checked)}
                />
              </div>
              <div className="grid gap-2 sm:grid-cols-4">
                <Button className="w-full" onClick={handleLoad} disabled={busy || providerForm.provider !== "local"}>
                  {busy ? <LoaderCircle className="animate-spin" /> : <Power />}
                  Betolt
                </Button>
                <Button className="w-full" variant="destructive" onClick={handleUnload} disabled={busy || providerForm.provider !== "local"}>
                  <Trash2 />
                  Kiurit
                </Button>
                <Button className="w-full" variant="outline" onClick={handleOptimize} disabled={busy}>
                  <Gauge />
                  Optimalizal
                </Button>
                <Button className="w-full" variant="outline" onClick={handleCancel} disabled={!busy || providerForm.provider !== "local"}>
                  <Square />
                  Stop
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Naplo</CardTitle>
              <CardDescription>Backend es modellmuveletek</CardDescription>
            </CardHeader>
            <CardContent>
              <Log entries={logs} />
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="chat" className="min-w-0">
          <TabsList className="mb-1 grid h-auto w-full grid-cols-1 gap-1 bg-transparent p-0 sm:mb-2 sm:grid-cols-3 sm:bg-muted sm:p-1">
            <TabsTrigger className="w-full justify-start sm:justify-center" value="chat">
              <MessageSquare className="size-4" />
              Chat
            </TabsTrigger>
            <TabsTrigger className="w-full justify-start sm:justify-center" value="agent">
              <Code2 className="size-4" />
              Coding Agent
            </TabsTrigger>
            <TabsTrigger className="w-full justify-start sm:justify-center" value="generate">
              <Play className="size-4" />
              Generate
            </TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="grid gap-4">
            <Card>
              <CardHeader>
                <CardTitle>ChatUI</CardTitle>
                <CardDescription>A betoltott lokalis modell valaszol a beszelgetesben</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="flex min-h-[320px] flex-col gap-3 overflow-auto rounded-lg border bg-muted/20 p-3 sm:min-h-[430px]">
                  {chatMessages.length ? (
                    chatMessages.map((message, index) => <MessageBubble key={index} message={message} />)
                  ) : (
                    <div className="flex h-full min-h-[260px] items-center justify-center rounded-md border border-dashed p-4 text-center text-sm text-muted-foreground sm:min-h-[380px]">
                      Ird be az elso uzenetet, majd kuldd el a lokalis modellnek.
                    </div>
                  )}
                </div>
                <div className="grid gap-3">
                  <Textarea
                    className="min-h-24 resize-y"
                    value={chatInput}
                    onChange={(event) => setChatInput(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
                        event.preventDefault()
                        void handleChatSend()
                      }
                    }}
                  />
                  <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
                    <Button className="w-full sm:w-auto" onClick={handleChatSend} disabled={busy || !chatInput.trim()}>
                      {busy ? <LoaderCircle className="animate-spin" /> : <Send />}
                      Kuldes
                    </Button>
                    <Button className="w-full sm:w-auto" variant="outline" onClick={() => setChatMessages([])} disabled={busy}>
                      <Trash2 />
                      Chat torles
                    </Button>
                    {status?.loaded && (
                      <Badge variant="outline">
                        {providerForm.provider === "local"
                          ? `${status.config?.device} / ${status.config?.dtype}`
                          : providerForm.external_model}
                      </Badge>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="agent" className="grid gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Coding Agent</CardTitle>
                <CardDescription>Egy gombbal indithato lokalis kodos asszisztens</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <Field label="Feladat">
                  <Textarea
                    className="min-h-28 resize-y"
                    value={agentObjective}
                    onChange={(event) => setAgentObjective(event.target.value)}
                  />
                </Field>
                <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_180px]">
                  <Field label="Workspace utvonal">
                    <Input
                      value={agentWorkspace}
                      placeholder="uresen hagyva: projekt root"
                      onChange={(event) => setAgentWorkspace(event.target.value)}
                    />
                  </Field>
                  <Field label="Context karakter">
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

            <Card>
              <CardHeader>
                <CardTitle>Agent valasz</CardTitle>
                <CardDescription>
                  {agentOutput
                    ? `${agentOutput.seconds}s, input=${agentOutput.input_tokens}, output=${agentOutput.output_tokens}`
                    : "A coding agent eredmenye itt jelenik meg"}
                </CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <div className="min-h-[320px] whitespace-pre-wrap break-words rounded-lg border bg-slate-950 p-3 text-sm leading-relaxed text-slate-50 sm:min-h-96 sm:p-4 sm:text-[15px]">
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

          <TabsContent value="generate" className="grid gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Prompt</CardTitle>
                <CardDescription>Egyszeri inference es sampling kontrollok</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4">
                <Textarea
                  className="min-h-48 resize-y text-base leading-relaxed"
                  value={generateForm.prompt}
                  onChange={(event) => updateGenerate("prompt", event.target.value)}
                />
                <GenerationControls generateForm={generateForm} updateGenerate={updateGenerate} />
                <div className="grid gap-2 sm:flex sm:flex-wrap sm:items-center">
                  <Button className="w-full sm:w-auto" size="lg" onClick={handleGenerate} disabled={busy}>
                    {busy ? <LoaderCircle className="animate-spin" /> : <Play />}
                    General
                  </Button>
                  {status?.loaded && (
                    <Badge variant="outline">
                      {providerForm.provider === "local"
                        ? `${status.config?.device} / ${status.config?.dtype}`
                        : providerForm.external_model}
                    </Badge>
                  )}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Kimenet</CardTitle>
                <CardDescription>
                  {status?.loaded && status.load_seconds
                    ? `Aktualis modell betoltesi ideje: ${status.load_seconds}s`
                    : "A valasz itt jelenik meg"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="min-h-[260px] whitespace-pre-wrap break-words rounded-lg border bg-slate-950 p-3 text-sm leading-relaxed text-slate-50 sm:min-h-80 sm:p-4 sm:text-[15px]">
                  {output}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

function GenerationControls({
  generateForm,
  updateGenerate,
}: {
  generateForm: GenerateForm
  updateGenerate: <K extends keyof GenerateForm>(key: K, value: GenerateForm[K]) => void
}) {
  return (
    <>
      <div className="grid gap-3 md:grid-cols-3">
        <Field label="Input max">
          <Input
            type="number"
            min={16}
            max={32768}
            step={16}
            value={generateForm.max_length}
            onChange={(event) => updateGenerate("max_length", event.target.value)}
          />
        </Field>
        <Field label="New tokens">
          <Input
            type="number"
            min={1}
            max={4096}
            value={generateForm.max_new_tokens}
            onChange={(event) => updateGenerate("max_new_tokens", event.target.value)}
          />
        </Field>
        <Field label="Temperature">
          <Input
            type="number"
            min={0}
            max={2}
            step={0.05}
            value={generateForm.temperature}
            onChange={(event) => updateGenerate("temperature", event.target.value)}
          />
        </Field>
        <Field label="Top p">
          <Input
            type="number"
            min={0.05}
            max={1}
            step={0.01}
            value={generateForm.top_p}
            onChange={(event) => updateGenerate("top_p", event.target.value)}
          />
        </Field>
        <Field label="Top k">
          <Input
            type="number"
            min={0}
            max={1000}
            value={generateForm.top_k}
            onChange={(event) => updateGenerate("top_k", event.target.value)}
          />
        </Field>
        <Field label="Repeat penalty">
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
      <div className="grid gap-3 md:grid-cols-3">
        <ToggleField
          label="Autoload"
          checked={generateForm.autoload}
          onCheckedChange={(checked) => updateGenerate("autoload", checked)}
        />
        <ToggleField
          label="KV cache"
          checked={generateForm.use_cache}
          onCheckedChange={(checked) => updateGenerate("use_cache", checked)}
        />
        <ToggleField
          label="Chat template"
          checked={generateForm.use_chat_template}
          onCheckedChange={(checked) => updateGenerate("use_chat_template", checked)}
        />
      </div>
    </>
  )
}

export default App
