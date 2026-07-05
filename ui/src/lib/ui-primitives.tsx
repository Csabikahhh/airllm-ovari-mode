import type { ReactNode } from "react"
import { CircleHelp } from "lucide-react"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

export function HelpHint({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          className="inline-flex size-5 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Sugo"
        >
          <CircleHelp className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent side="top" align="start">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

function FieldLabel({ label, hint }: { label: string; hint?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <Label className="text-sm font-medium text-foreground/90">{label}</Label>
      {hint ? <HelpHint text={hint} /> : null}
    </div>
  )
}

export function pretty(value: string | number | null | undefined, fallback = "-") {
  if (value === null || value === undefined || value === "") return fallback
  return value
}

export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string
  hint?: string
  children: ReactNode
  className?: string
}) {
  return (
    <div className={cn("grid min-w-0 gap-1.5", className)}>
      <FieldLabel label={label} hint={hint} />
      {children}
    </div>
  )
}

export function Metric({
  icon,
  label,
  value,
  tone = "default",
  loading = false,
}: {
  icon: ReactNode
  label: string
  value: ReactNode
  tone?: "default" | "good" | "warn"
  loading?: boolean
}) {
  if (loading) {
    return (
      <div data-slot="metric" className="flex min-h-16 w-full gap-3 rounded-lg border bg-muted/35 p-3 animate-pulse">
        <div className="mt-0.5 size-8 shrink-0 rounded-md bg-muted" />
        <div className="min-w-0 flex-1 space-y-2">
          <div className="h-3 w-16 rounded bg-muted" />
          <div className="h-4 w-24 rounded bg-muted" />
        </div>
      </div>
    )
  }

  return (
    <div
      data-slot="metric"
      className={cn(
        "flex min-h-16 w-full gap-3 overflow-hidden rounded-lg border bg-muted/35 p-3",
        tone === "good" && "border-emerald-200 bg-emerald-50",
        tone === "warn" && "border-amber-200 bg-amber-50",
      )}
    >
      <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md bg-background text-muted-foreground">
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-xs font-medium text-muted-foreground">{label}</p>
        <div className="mt-1 break-words text-sm font-semibold leading-snug">{value}</div>
      </div>
    </div>
  )
}

export function Log({ entries }: { entries: string[] }) {
  return (
    <div className="max-h-44 min-h-28 overflow-auto rounded-md border bg-muted/45 p-3 font-mono text-xs leading-relaxed text-muted-foreground">
      {entries.length ? entries.join("\n") : "Nincs naplo."}
    </div>
  )
}

export function SelectField({
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

export function ToggleField({
  label,
  hint,
  checked,
  onCheckedChange,
}: {
  label: string
  hint?: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
}) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 rounded-md border bg-background px-3 py-2.5">
      <div className="flex min-w-0 items-center gap-1.5">
        <Label className="leading-normal">{label}</Label>
        {hint ? <HelpHint text={hint} /> : null}
      </div>
      <Switch checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  )
}
