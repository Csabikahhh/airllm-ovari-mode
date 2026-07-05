import {
  Activity,
  Cloud,
  Cpu,
  Download,
  Gauge,
  HardDrive,
  LoaderCircle,
  Power,
  RefreshCw,
  Search,
  Server,
  Sparkles,
  Square,
  Trash2,
} from "lucide-react"
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
  Field,
  Log,
  Metric,
  pretty,
  SelectField,
  ToggleField,
} from "@/lib/ui-primitives"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { configHints } from "@/lib/config-hints"
import type {
  BenchmarkResult,
  HardwareProfile,
  HuggingFaceModelsResponse,
  LoadForm,
  ModelCacheInfo,
  ModelDownloadStatus,
  Preset,
  ProviderForm,
  ProviderPreset,
} from "@/types/app"

type ConfigPanelProps = {
  layout?: "sidebar" | "stack"
  hardware: HardwareProfile | null
  gpuLabel: string
  benchmark: BenchmarkResult | null
  busy: boolean
  cacheBusy: boolean
  logs: string[]
  modelCache: ModelCacheInfo | null
  downloadStatus: ModelDownloadStatus | null
  hfModels: HuggingFaceModelsResponse | null
  hfModelQuery: string
  hfModelsBusy: boolean
  providerForm: ProviderForm
  providerPresets: ProviderPreset[]
  updateProvider: <K extends keyof ProviderForm>(key: K, value: ProviderForm[K]) => void
  presetIndex: string
  setPresetIndex: (value: string) => void
  presets: Preset[]
  loadForm: LoadForm
  updateLoad: <K extends keyof LoadForm>(key: K, value: LoadForm[K]) => void
  onBenchmark: () => void
  onLoad: () => void
  onUnload: () => void
  onOptimize: () => void
  onCancel: () => void
  onRefreshModels: () => void
  onDownloadModel: (modelId?: string) => void
  onCancelDownload: () => void
  onDeleteCachedModel: (modelId: string) => void
  onUseCachedModel: (modelId: string) => void
  onUseHfModel: (modelId: string) => void
  onSearchHfModels: () => void
  setHfModelQuery: (value: string) => void
}

function formatBytes(value: number | null | undefined) {
  if (!value || value <= 0) return "0 B"
  const units = ["B", "KB", "MB", "GB", "TB"]
  let amount = value
  let unit = 0
  while (amount >= 1024 && unit < units.length - 1) {
    amount /= 1024
    unit += 1
  }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`
}

function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "-"
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return `${minutes}m ${rest}s`
}

function formatCount(value: number | null | undefined) {
  if (value === null || value === undefined) return "?"
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
  return String(value)
}

function downloadStatusLabel(status: ModelDownloadStatus["status"]) {
  const labels: Record<ModelDownloadStatus["status"], string> = {
    idle: "készen áll",
    preparing: "előkészítés",
    downloading: "letöltés",
    cancelling: "leállítás",
    cancelled: "leállítva",
    cached: "már cache-ben van",
    done: "kész",
    error: "hiba",
  }
  return labels[status]
}

function friendlyPowerScheme(value: string | null | undefined) {
  if (!value) return "ismeretlen"
  const match = value.match(/\(([^)]+)\)/)
  if (match?.[1]) return match[1]
  return value.replace(/^[0-9a-f-]{20,}\s*/i, "").trim() || value
}

function AcceleratorValue({ hardware, gpuLabel }: { hardware: HardwareProfile | null; gpuLabel: string }) {
  if (!hardware) return "betöltés..."
  const device = hardware.cuda.devices[0]
  if (device) {
    return (
      <div className="space-y-0.5">
        <div className="break-words">{device.name}</div>
        <div className="text-xs font-medium text-muted-foreground">
          {device.total_memory_gb ?? "?"} GB VRAM
          {device.compute_capability ? ` - cc ${device.compute_capability}` : ""}
        </div>
      </div>
    )
  }
  return gpuLabel
}

export function ConfigPanel({
  layout = "stack",
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
  onBenchmark,
  onLoad,
  onUnload,
  onOptimize,
  onCancel,
  onRefreshModels,
  onDownloadModel,
  onCancelDownload,
  onDeleteCachedModel,
  onUseCachedModel,
  onUseHfModel,
  onSearchHfModels,
  setHfModelQuery,
}: ConfigPanelProps) {
  const hardwareLoading = hardware === null
  const isSidebar = layout === "sidebar"
  const deviceOptions = hardware?.device_options?.length
    ? hardware.device_options
    : ["auto", "cuda:0", "mps", "cpu"]

  return (
    <div className={cn("grid min-w-0", isSidebar ? "gap-2.5" : "gap-3 sm:gap-4")}>
      <Card className="hardware-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="size-4" />
            Hardver
          </CardTitle>
          <CardDescription>Automatikus runtime profil</CardDescription>
        </CardHeader>
        <CardContent className="hardware-content">
          <div className="hardware-section">
            <div className="hardware-section-title">Rendszer</div>
            <div className="hardware-metric-list">
              <Metric
                icon={<Cpu className="size-4" />}
                label="CPU"
                loading={hardwareLoading}
                value={`${hardware?.cpu.logical_cores ?? "-"} szál`}
              />
              <Metric
                icon={<Server className="size-4" />}
                label="RAM"
                loading={hardwareLoading}
                value={`${pretty(hardware?.memory.available_gb, "?")} / ${pretty(hardware?.memory.total_gb, "?")} GB`}
              />
            </div>
          </div>

          <div className="hardware-section">
            <div className="hardware-section-title">Gyorsító</div>
            <Metric
              icon={<Sparkles className="size-4" />}
              label={hardware?.mps.available ? "Metal / MPS" : "GPU"}
              loading={hardwareLoading}
              value={<AcceleratorValue hardware={hardware} gpuLabel={gpuLabel} />}
              tone={hardware?.cuda.available || hardware?.mps.available ? "good" : "warn"}
            />
            <div className="hardware-metric-list">
              <Metric
                icon={<Power className="size-4" />}
                label="Torch"
                loading={hardwareLoading}
                value={hardware?.torch.version ?? "nem elérhető"}
                tone={hardware?.torch.available ? "good" : "warn"}
              />
              <Metric
                icon={<RefreshCw className="size-4" />}
                label="bitsandbytes"
                loading={hardwareLoading}
                value={hardware?.bitsandbytes.available ? "elérhető" : "nem elérhető"}
                tone={hardware?.bitsandbytes.available ? "good" : "warn"}
              />
            </div>
            {hardware && hardware.platform.toLowerCase().includes("darwin") && (
              <Metric
                icon={<Sparkles className="size-4" />}
                label="MLX"
                loading={hardwareLoading}
                value={hardware.mlx.available ? "elérhető" : "nem elérhető"}
                tone={hardware.mlx.available ? "good" : "warn"}
              />
            )}
          </div>

          <div className="hardware-section">
            <div className="hardware-section-title">Környezet</div>
            <div className="hardware-metric-list">
              <Metric
                icon={<HardDrive className="size-4" />}
                label="HF cache"
                loading={hardwareLoading}
                value={`${pretty(hardware?.disk.free_gb, "?")} GB szabad`}
              />
              <Metric
                icon={<Power className="size-4" />}
                label="Energia"
                loading={hardwareLoading}
                value={friendlyPowerScheme(hardware?.power.active_scheme)}
                tone={
                  hardware?.power.active_scheme?.toLowerCase().includes("balanced") ||
                  hardware?.power.active_scheme?.toLowerCase().includes("kiegy")
                    ? "warn"
                    : "default"
                }
              />
            </div>
            <Metric
              icon={<Activity className="size-4" />}
              label="Hálózat"
              loading={hardwareLoading}
              value={
                hardware?.network.radio
                  ? `${hardware.network.radio} - ${pretty(hardware.network.receive_mbps, "?")} Mbps`
                  : "ismeretlen"
              }
            />
          </div>
          <div className="hardware-section">
            <Button className="w-full" variant="outline" onClick={onBenchmark} disabled={busy}>
              {busy ? <LoaderCircle className="animate-spin" /> : <Activity />}
              Benchmark
            </Button>
            {benchmark && (
              <div className="grid grid-cols-1 gap-2 rounded-lg border bg-muted/35 p-3 text-xs text-muted-foreground">
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
          <Field label="Provider" hint={configHints.provider}>
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
              <Field label="Preset" hint={configHints.providerPreset}>
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
              <Field label="Base URL" hint={configHints.baseUrl}>
                <Input
                  value={providerForm.external_base_url}
                  placeholder="https://api.example.com/v1"
                  onChange={(event) => updateProvider("external_base_url", event.target.value)}
                />
              </Field>
              <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_120px]">
                <Field label="Model" hint={configHints.externalModel}>
                  <Input
                    value={providerForm.external_model}
                    placeholder="provider-model-name"
                    onChange={(event) => updateProvider("external_model", event.target.value)}
                  />
                </Field>
                <Field label="Timeout" hint={configHints.externalTimeout}>
                  <Input
                    type="number"
                    min={10}
                    max={600}
                    value={providerForm.external_timeout}
                    onChange={(event) => updateProvider("external_timeout", event.target.value)}
                  />
                </Field>
              </div>
              <Field label="API key" hint={configHints.externalApiKey}>
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
          <Field label="Preset" hint={configHints.preset}>
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
          <Field label="Model ID / utvonal" hint={configHints.modelId}>
            <Input
              spellCheck={false}
              value={loadForm.model_id}
              onChange={(event) => {
                setPresetIndex("custom")
                updateLoad("model_id", event.target.value)
              }}
            />
          </Field>
          <div
            className={
              isSidebar
                ? "grid gap-3 grid-cols-1"
                : "grid gap-3 sm:grid-cols-2 2xl:grid-cols-3"
            }
          >
            <Field label="Device" hint={configHints.device}>
              <SelectField
                value={loadForm.device}
                onValueChange={(value) => updateLoad("device", value)}
                options={deviceOptions.map((device) => ({ value: device, label: device }))}
              />
            </Field>
            <Field label="Dtype" hint={configHints.dtype}>
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
            <Field label="Compression" hint={configHints.compression}>
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
            <Field label="Futtatasi mod" hint={configHints.loadMode}>
              <SelectField
                value={loadForm.load_mode}
                onValueChange={(value) => {
                  updateLoad("load_mode", value)
                  if (value === "hybrid") updateLoad("compression", "none")
                }}
                options={[
                  { value: "auto", label: "auto" },
                  { value: "direct", label: "GPU/MPS rezidens" },
                  { value: "hybrid", label: "CPU+GPU hybrid" },
                  { value: "airllm", label: "AirLLM streaming" },
                ]}
              />
            </Field>
            <Field label="Prefetching" hint={configHints.prefetching}>
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
            <Field label="Max seq len" hint={configHints.maxSeqLen}>
              <Input
                type="number"
                min={128}
                max={32768}
                step={128}
                value={loadForm.max_seq_len}
                onChange={(event) => updateLoad("max_seq_len", event.target.value)}
              />
            </Field>
          </div>

          <details className="group rounded-lg border bg-muted/20 open:bg-muted/30">
            <summary className="cursor-pointer px-3 py-3 text-sm font-medium text-foreground/90 select-none hover:bg-muted/40">
              Halado beallitasok
            </summary>
            <div className="grid grid-cols-1 gap-3 border-t px-3 pt-3 pb-2">
              <Field label="Cleanup interval" hint={configHints.cleanupInterval}>
                <Input
                  type="number"
                  min={0}
                  max={64}
                  value={loadForm.cleanup_interval}
                  onChange={(event) => updateLoad("cleanup_interval", event.target.value)}
                />
              </Field>
              <Field label="Prefetch workers" hint={configHints.prefetchWorkers}>
                <Input
                  type="number"
                  min={1}
                  max={4}
                  value={loadForm.prefetch_workers}
                  onChange={(event) => updateLoad("prefetch_workers", event.target.value)}
                />
              </Field>
              <Field label="Layer cache" hint={configHints.layerCache}>
                <Input
                  value={loadForm.layer_shards_saving_path}
                  onChange={(event) => updateLoad("layer_shards_saving_path", event.target.value)}
                />
              </Field>
              <Field label="HF token" hint={configHints.hfToken}>
                <Input
                  type="password"
                  autoComplete="off"
                  value={loadForm.hf_token}
                  onChange={(event) => updateLoad("hf_token", event.target.value)}
                />
              </Field>
            </div>
            <div className="grid grid-cols-1 gap-2 border-t px-3 pt-2 pb-3">
              <ToggleField
                label="Profiling"
                hint={configHints.profiling}
                checked={loadForm.profiling_mode}
                onCheckedChange={(checked) => updateLoad("profiling_mode", checked)}
              />
              <ToggleField
                label="Delete original"
                hint={configHints.deleteOriginal}
                checked={loadForm.delete_original}
                onCheckedChange={(checked) => updateLoad("delete_original", checked)}
              />
              <ToggleField
                label="Reinit / forward"
                hint={configHints.reinitForward}
                checked={loadForm.reinitialize_model_each_forward}
                onCheckedChange={(checked) => updateLoad("reinitialize_model_each_forward", checked)}
              />
            </div>
          </details>

          <div className={isSidebar ? "grid grid-cols-1 gap-2" : "grid grid-cols-2 gap-2 md:grid-cols-4"}>
            <Button className="h-10 w-full" onClick={onLoad} disabled={busy || providerForm.provider !== "local"}>
              {busy ? <LoaderCircle className="animate-spin" /> : <Power />}
              Betolt
            </Button>
            <Button
              className="h-10 w-full"
              variant="destructive"
              onClick={onUnload}
              disabled={busy || providerForm.provider !== "local"}
            >
              <Trash2 />
              Kiurit
            </Button>
            <Button className="h-10 w-full" variant="secondary" onClick={onOptimize} disabled={busy}>
              <Gauge />
              Optimalizal
            </Button>
            <Button
              className="h-10 w-full"
              variant="outline"
              onClick={onCancel}
              disabled={!busy || providerForm.provider !== "local"}
            >
              <Square />
              Stop
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HardDrive className="size-4" />
            Modell tarhely
          </CardTitle>
          <CardDescription>
            {modelCache ? `${modelCache.models.length} modell, ${formatBytes(modelCache.total_size_bytes)}` : "Hugging Face cache"}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div className={isSidebar ? "grid grid-cols-1 gap-2" : "grid gap-2 sm:grid-cols-3"}>
            <Button
              className="h-10 w-full"
              variant="secondary"
              onClick={() => onDownloadModel()}
              disabled={busy || cacheBusy || providerForm.provider !== "local" || Boolean(downloadStatus?.active)}
            >
              {downloadStatus?.active ? <LoaderCircle className="animate-spin" /> : <Download />}
              Letolt
            </Button>
            <Button className="h-10 w-full" variant="outline" onClick={onRefreshModels} disabled={cacheBusy}>
              {cacheBusy ? <LoaderCircle className="animate-spin" /> : <RefreshCw />}
              Frissit
            </Button>
            <div className="min-w-0 rounded-md border bg-muted/35 px-3 py-2 text-xs text-muted-foreground sm:col-span-1">
              <div className="truncate" title={modelCache?.cache_dir}>
                {modelCache?.cache_dir ?? "cache olvasasa..."}
              </div>
            </div>
          </div>

          <div className="grid gap-3 rounded-lg border bg-muted/20 p-3">
            <div className="flex min-w-0 items-center justify-between gap-2">
              <div className="min-w-0">
                <div className="text-sm font-medium">Hugging Face modellek</div>
                <div className="text-xs text-muted-foreground">Kereses publikus text-generation modellek kozott</div>
              </div>
              {hfModels && <Badge variant="secondary">{hfModels.models.length}</Badge>}
            </div>
            <form
              className={cn("grid gap-2", isSidebar ? "grid-cols-1" : "sm:grid-cols-[minmax(0,1fr)_120px]")}
              onSubmit={(event) => {
                event.preventDefault()
                onSearchHfModels()
              }}
            >
              <Input
                spellCheck={false}
                value={hfModelQuery}
                placeholder="pl. Qwen2.5 Instruct"
                onChange={(event) => setHfModelQuery(event.target.value)}
              />
              <Button className="h-10 w-full" variant="outline" type="submit" disabled={hfModelsBusy}>
                {hfModelsBusy ? <LoaderCircle className="animate-spin" /> : <Search />}
                Keres
              </Button>
            </form>
            <div className="grid max-h-72 gap-2 overflow-auto pr-1">
              {hfModelsBusy ? (
                <div className="rounded-lg border bg-background p-3 text-sm text-muted-foreground">Modellek keresese...</div>
              ) : hfModels && hfModels.models.length > 0 ? (
                hfModels.models.map((model) => (
                  <div key={model.model_id} className="grid gap-2 rounded-lg border bg-background p-3">
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium" title={model.model_id}>
                        {model.model_id}
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                        <span>{formatCount(model.downloads)} letoltes</span>
                        <span>{formatCount(model.likes)} like</span>
                        {model.pipeline_tag && <span>{model.pipeline_tag}</span>}
                        {model.library_name && <span>{model.library_name}</span>}
                        {model.gated && <span>gated</span>}
                      </div>
                    </div>
                    <div className={cn("grid gap-2", isSidebar ? "grid-cols-1" : "grid-cols-2")}>
                      <Button
                        className="h-9 w-full"
                        variant="outline"
                        onClick={() => onUseHfModel(model.model_id)}
                        disabled={busy}
                      >
                        Hasznal
                      </Button>
                      <Button
                        className="h-9 w-full"
                        variant="secondary"
                        onClick={() => onDownloadModel(model.model_id)}
                        disabled={
                          busy ||
                          cacheBusy ||
                          providerForm.provider !== "local" ||
                          Boolean(downloadStatus?.active)
                        }
                      >
                        <Download />
                        Letolt
                      </Button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="rounded-lg border bg-background p-3 text-sm text-muted-foreground">
                  Nincs talalat. Irj be masik keresoszot.
                </div>
              )}
            </div>
          </div>

          {downloadStatus && downloadStatus.status !== "idle" && (
            <div className="grid gap-2 rounded-lg border bg-muted/30 p-3">
              <div className="flex min-w-0 items-center justify-between gap-3 text-sm">
                <span className="min-w-0 truncate font-medium" title={downloadStatus.model_id ?? undefined}>
                  {downloadStatus.model_id ?? "modell"}
                </span>
                <Badge
                  variant={
                    downloadStatus.status === "error" || downloadStatus.status === "cancelled"
                      ? "destructive"
                      : "secondary"
                  }
                >
                  {downloadStatusLabel(downloadStatus.status)}
                </Badge>
              </div>
              {downloadStatus.active && (
                <Button
                  className="h-9 w-full"
                  variant="destructive"
                  onClick={onCancelDownload}
                  disabled={downloadStatus.status === "cancelling"}
                >
                  {downloadStatus.status === "cancelling" ? <LoaderCircle className="animate-spin" /> : <Square />}
                  Letoltes leallitasa
                </Button>
              )}
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full transition-all",
                    downloadStatus.status === "error" || downloadStatus.status === "cancelled"
                      ? "bg-destructive"
                      : "bg-primary",
                  )}
                  style={{
                    width: `${Math.min(100, Math.max(downloadStatus.percent ?? 0, downloadStatus.active ? 3 : 0))}%`,
                  }}
                />
              </div>
              <div className="grid gap-1 text-xs text-muted-foreground">
                <div className="flex flex-wrap gap-x-3 gap-y-1">
                  <span>{downloadStatus.percent ?? 0}%</span>
                  <span>
                    {formatBytes(downloadStatus.downloaded_bytes)} / {formatBytes(downloadStatus.total_bytes)}
                  </span>
                  <span>eltelt: {formatDuration(downloadStatus.elapsed_seconds)}</span>
                  <span>hatralevo: {formatDuration(downloadStatus.eta_seconds)}</span>
                </div>
                <div>
                  fajlok: {downloadStatus.files_cached} cache-ben, {downloadStatus.files_to_download} letoltendo
                </div>
                {downloadStatus.current_file && (
                  <div className="truncate" title={downloadStatus.current_file}>
                    aktualis: {downloadStatus.current_file}
                  </div>
                )}
                {downloadStatus.path && (
                  <div className="truncate" title={downloadStatus.path}>
                    hely: {downloadStatus.path}
                  </div>
                )}
                {downloadStatus.error && <div className="text-destructive">{downloadStatus.error}</div>}
              </div>
            </div>
          )}

          <div className="grid max-h-80 gap-2 overflow-auto pr-1">
            {modelCache === null ? (
              <div className="rounded-lg border bg-muted/30 p-3 text-sm text-muted-foreground">Modellek olvasasa...</div>
            ) : modelCache.models.length === 0 ? (
              <div className="rounded-lg border bg-muted/30 p-3 text-sm text-muted-foreground">
                Nincs letoltott Hugging Face modell a cache-ben.
              </div>
            ) : (
              modelCache.models.map((model) => (
                <div key={model.model_id} className="grid gap-2 rounded-lg border bg-background p-3">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium" title={model.model_id}>
                      {model.model_id}
                    </div>
                    <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      <span>{formatBytes(model.size_bytes)}</span>
                      <span>{model.snapshots} snapshot</span>
                      {model.modified_at && <span>{new Date(model.modified_at * 1000).toLocaleString()}</span>}
                    </div>
                    <div className="mt-1 truncate text-xs text-muted-foreground" title={model.path}>
                      {model.path}
                    </div>
                  </div>
                  <div className={cn("grid gap-2", isSidebar ? "grid-cols-1" : "grid-cols-2")}>
                    <Button
                      className="h-9 w-full"
                      variant="outline"
                      onClick={() => onUseCachedModel(model.model_id)}
                      disabled={busy}
                    >
                      Hasznal
                    </Button>
                    <Button
                      className="h-9 w-full"
                      variant="destructive"
                      onClick={() => onDeleteCachedModel(model.model_id)}
                      disabled={busy || cacheBusy || Boolean(downloadStatus?.active)}
                    >
                      <Trash2 />
                      Torol
                    </Button>
                  </div>
                </div>
              ))
            )}
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
  )
}
