export type Preset = {
  label: string
  model_id: string
  family: string
  size: string
}

export type ProviderPreset = {
  label: string
  base_url: string
  model: string
  needs_key: boolean
}

export type HardwareProfile = {
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
  mps: {
    available: boolean
    built: boolean
    error: string | null
  }
  mlx: {
    available: boolean
    version: string | null
    error: string | null
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
  device_options: string[]
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

export type Status = {
  loaded: boolean
  mode?: string | null
  config: null | {
    provider?: string
    base_url?: string
    model_id: string
    device?: string
    dtype?: string
    compression?: string | null
    load_mode?: string
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

export type LoadForm = {
  model_id: string
  device: string
  dtype: string
  compression: string
  load_mode: string
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

export type GenerateForm = {
  prompt: string
  task_mode: string
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

export type ProviderForm = {
  provider: "local" | "openai_compatible"
  provider_preset: string
  external_base_url: string
  external_model: string
  external_api_key: string
  external_timeout: string
}

export type ChatMessage = {
  role: "user" | "assistant"
  content: string
}

export type AgentResponse = {
  text: string
  seconds: number
  input_tokens: number
  output_tokens: number
  workspace: string
  included_files: string[]
  context_files: string[]
  status: Status
}

export type BenchmarkResult = {
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

export type CachedModel = {
  model_id: string
  path: string
  size_bytes: number
  size_gb: number
  modified_at: number | null
  snapshots: number
}

export type ModelDownloadStatus = {
  active: boolean
  status: "idle" | "preparing" | "downloading" | "cancelling" | "cancelled" | "cached" | "done" | "error"
  model_id: string | null
  path: string | null
  current_file: string | null
  total_bytes: number
  downloaded_bytes: number
  downloaded_gb: number
  total_gb: number
  files_total: number
  files_cached: number
  files_to_download: number
  started_at: number | null
  finished_at: number | null
  elapsed_seconds: number
  eta_seconds: number | null
  percent: number | null
  error: string | null
}

export type ModelCacheInfo = {
  cache_dir: string
  models: CachedModel[]
  total_size_bytes: number
  total_size_gb: number
  download: ModelDownloadStatus
}

export type HuggingFaceModel = {
  model_id: string
  downloads: number | null
  likes: number | null
  pipeline_tag: string | null
  library_name: string | null
  private: boolean
  gated: boolean | string | null
  tags: string[]
  last_modified: string | null
}

export type HuggingFaceModelsResponse = {
  query: string
  task: string | null
  limit: number
  models: HuggingFaceModel[]
}
