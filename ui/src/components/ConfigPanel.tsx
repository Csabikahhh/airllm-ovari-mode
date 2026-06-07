import {
  Activity,
  Cloud,
  Cpu,
  Gauge,
  HardDrive,
  LoaderCircle,
  Power,
  RefreshCw,
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
import { cn } from "@/lib/utils"
import { configHints } from "@/lib/config-hints"
import type {
  BenchmarkResult,
  HardwareProfile,
  LoadForm,
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
  logs: string[]
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
}

export function ConfigPanel({
  layout = "stack",
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
  onBenchmark,
  onLoad,
  onUnload,
  onOptimize,
  onCancel,
}: ConfigPanelProps) {
  const hardwareLoading = hardware === null
  const isSidebar = layout === "sidebar"

  return (
    <div className="grid min-w-0 gap-3 sm:gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Gauge className="size-4" />
            Hardver
          </CardTitle>
          <CardDescription>Automatikus runtime profil</CardDescription>
        </CardHeader>
        <CardContent className={cn("grid gap-3", isSidebar ? "grid-cols-1" : "sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2")}>
          <Metric
            icon={<Cpu className="size-4" />}
            label="CPU"
            loading={hardwareLoading}
            value={`${hardware?.cpu.logical_cores ?? "-"} szal`}
          />
          <Metric
            icon={<Server className="size-4" />}
            label="RAM"
            loading={hardwareLoading}
            value={`${pretty(hardware?.memory.available_gb, "?")} / ${pretty(hardware?.memory.total_gb, "?")} GB`}
          />
          <Metric
            icon={<Sparkles className="size-4" />}
            label="GPU"
            loading={hardwareLoading}
            value={gpuLabel}
            tone={hardware?.cuda.available ? "good" : "warn"}
          />
          <Metric
            icon={<HardDrive className="size-4" />}
            label="HF cache"
            loading={hardwareLoading}
            value={`${pretty(hardware?.disk.free_gb, "?")} GB szabad`}
          />
          <Metric
            icon={<Power className="size-4" />}
            label="Torch"
            loading={hardwareLoading}
            value={hardware?.torch.version ?? "nem elerheto"}
            tone={hardware?.torch.available ? "good" : "warn"}
          />
          <Metric
            icon={<RefreshCw className="size-4" />}
            label="bitsandbytes"
            loading={hardwareLoading}
            value={hardware?.bitsandbytes.available ? "elerheto" : "nem elerheto"}
            tone={hardware?.bitsandbytes.available ? "good" : "warn"}
          />
          <Metric
            icon={<Power className="size-4" />}
            label="Energia"
            loading={hardwareLoading}
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
            loading={hardwareLoading}
            value={
              hardware?.network.radio
                ? `${hardware.network.radio}, ${pretty(hardware.network.receive_mbps, "?")} Mbps`
                : "ismeretlen"
            }
          />
          <div className="grid gap-2 sm:col-span-2 xl:col-span-1 2xl:col-span-2">
            <Button className="w-full" variant="outline" onClick={onBenchmark} disabled={busy}>
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
                options={[
                  { value: "auto", label: "auto" },
                  { value: "cuda:0", label: "cuda:0" },
                  { value: "cpu", label: "cpu" },
                ]}
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

          <div className={isSidebar ? "grid grid-cols-2 gap-2" : "grid grid-cols-2 gap-2 md:grid-cols-4"}>
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
