import { useEffect, useMemo, useState } from "react"
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
import { Field, pretty, ToggleField } from "@/lib/ui-primitives"
import { cn } from "@/lib/utils"
import type {
  AgentResponse,
  BenchmarkResult,
  ChatMessage,
  GenerateForm,
  HardwareProfile,
  LoadForm,
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
  const [logs, setLogs] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [initializing, setInitializing] = useState(true)

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
      } finally {
        setInitializing(false)
      }
    })()
  }, [])

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
    logs,
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
  }

  return (
    <div className="flex min-h-screen flex-col bg-[linear-gradient(180deg,oklch(0.97_0.006_247)_0%,var(--background)_12rem)] text-foreground">
      <header className="sticky top-0 z-20 shrink-0 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
        <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-3 px-3 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-4">
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

      <main className="mx-auto grid w-full max-w-[1500px] flex-1 grid-cols-1 gap-4 p-3 sm:p-4 xl:grid-cols-[400px_minmax(0,1fr)] xl:items-start">
        <div className="hidden min-w-0 xl:block xl:sticky xl:top-[57px] xl:max-h-[calc(100vh-3.75rem)] xl:overflow-y-auto xl:pr-1">
          <ConfigPanel {...configPanelProps} layout="sidebar" />
        </div>

        <Tabs defaultValue="chat" className="flex min-h-0 min-w-0 flex-1 flex-col">
          <TabsList className="sticky top-[57px] z-10 mb-3 grid h-auto w-full shrink-0 grid-cols-3 gap-1 rounded-xl border bg-background/95 p-1 shadow-sm backdrop-blur supports-[backdrop-filter]:bg-background/80 xl:static xl:top-auto xl:w-full xl:bg-muted">
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
            <Card className="flex min-h-[calc(100dvh-11rem)] flex-1 flex-col overflow-hidden shadow-sm xl:min-h-[calc(100vh-9.5rem)]">
              <CardHeader className="shrink-0 border-b bg-muted/20 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <CardTitle>ChatUI</CardTitle>
                    <CardDescription className="mt-1">
                      A betoltott lokalis modell valaszol a beszelgetesben
                    </CardDescription>
                  </div>
                  {status?.loaded && (
                    <Badge variant="outline" className="shrink-0">
                      {providerForm.provider === "local"
                        ? `${status.config?.device} / ${status.config?.dtype}`
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
                <div className="shrink-0 border-t bg-background/95 p-4 pb-[max(1rem,env(safe-area-inset-bottom))] backdrop-blur">
                  <div className="grid gap-3">
                    <Textarea
                      className="min-h-[88px] resize-none bg-background text-[15px] leading-relaxed"
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

          <TabsContent value="agent" className="mt-0 grid gap-4">
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
                <div className="max-h-[60vh] min-h-[40vh] whitespace-pre-wrap break-words rounded-lg border bg-slate-950 p-3 text-sm leading-relaxed text-slate-50 sm:min-h-96 sm:max-h-none sm:p-4 sm:text-[15px]">
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

          <TabsContent value="generate" className="mt-0 grid gap-4">
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
                <div className="max-h-[60vh] min-h-[40vh] whitespace-pre-wrap break-words rounded-lg border bg-slate-950 p-3 text-sm leading-relaxed text-slate-50 sm:min-h-80 sm:max-h-none sm:p-4 sm:text-[15px]">
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
}: {
  generateForm: GenerateForm
  updateGenerate: <K extends keyof GenerateForm>(key: K, value: GenerateForm[K]) => void
}) {
  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
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
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
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
