"""
Command line client for the local AirLLM UI backend.

Examples:
    python airllm_cli.py status
    python airllm_cli.py models
    python airllm_cli.py agent "Nezd at a projektet es javasolj javitasokat" --workspace .
    python airllm_cli.py agent "Keress performance gondokat" --model Qwen/Qwen2.5-Coder-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SERVER = "http://127.0.0.1:7860"


class AirLLMCliError(RuntimeError):
    pass


def api_request(server: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 600) -> dict[str, Any]:
    url = server.rstrip("/") + path
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(details)
            message = parsed.get("error") or details
        except Exception:
            message = details
        raise AirLLMCliError(f"HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise AirLLMCliError(
            "Nem erem el az AirLLM szervert. Inditsd el elobb: "
            ".\\.venv\\Scripts\\python.exe airllm_ui.py"
        ) from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AirLLMCliError(f"Nem JSON valasz erkezett: {raw[:200]}") from exc


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def command_status(args: argparse.Namespace) -> int:
    status = api_request(args.server, "/api/status", timeout=args.timeout)
    if args.json:
        print_json(status)
        return 0
    if status.get("loaded"):
        config = status.get("config") or {}
        print("Betoltott modell:")
        print(f"  model: {config.get('model_id')}")
        print(f"  device: {config.get('device')}")
        print(f"  dtype: {config.get('dtype')}")
        if status.get("mode"):
            print(f"  mode: {status.get('mode')}")
    else:
        print("Nincs betoltott modell.")
    return 0


def command_models(args: argparse.Namespace) -> int:
    data = api_request(args.server, "/api/models", timeout=args.timeout)
    if args.json:
        print_json(data)
        return 0
    models = data.get("models") or []
    print(f"Cache: {data.get('cache_dir')}")
    print(f"Modellek: {len(models)} | osszesen: {data.get('total_size_gb')} GB")
    for model in models:
        print(f"- {model.get('model_id')} ({model.get('size_gb')} GB, snapshots={model.get('snapshots')})")
    return 0


def command_hf_search(args: argparse.Namespace) -> int:
    params = urllib.parse.urlencode({"q": args.query, "limit": args.limit})
    data = api_request(args.server, f"/api/hf-models?{params}", timeout=args.timeout)
    if args.json:
        print_json(data)
        return 0
    for model in data.get("models") or []:
        downloads = model.get("downloads")
        pipeline = model.get("pipeline_tag") or "?"
        library = model.get("library_name") or "?"
        print(f"- {model.get('model_id')} | downloads={downloads} | {pipeline} | {library}")
    return 0


def command_download(args: argparse.Namespace) -> int:
    payload = {"model_id": args.model, "hf_token": args.hf_token or ""}
    data = api_request(args.server, "/api/download", payload, timeout=args.timeout)
    if args.json:
        print_json(data)
        return 0
    print(f"Letoltes inditva: {data.get('model_id')}")
    print("Allapotot itt nezheted: python airllm_cli.py download-status")
    return 0


def command_download_status(args: argparse.Namespace) -> int:
    data = api_request(args.server, "/api/download/status", timeout=args.timeout)
    if args.json:
        print_json(data)
        return 0
    print(f"status: {data.get('status')} | active={data.get('active')} | model={data.get('model_id')}")
    print(f"progress: {data.get('percent')}% | {data.get('downloaded_gb')} / {data.get('total_gb')} GB")
    if data.get("current_file"):
        print(f"file: {data.get('current_file')}")
    if data.get("error"):
        print(f"error: {data.get('error')}")
    return 0


def command_cancel_download(args: argparse.Namespace) -> int:
    data = api_request(args.server, "/api/download/cancel", {}, timeout=args.timeout)
    if args.json:
        print_json(data)
        return 0
    print(f"download status: {data.get('status')}")
    return 0


def agent_payload(args: argparse.Namespace, status: dict[str, Any]) -> dict[str, Any]:
    model_id = (args.model or "").strip()
    if not model_id and not status.get("loaded"):
        raise AirLLMCliError(
            "Nincs betoltott modell. Tolts/betolts modellt a UI-ban, vagy add meg: --model MODEL_ID"
        )

    payload: dict[str, Any] = {
        "objective": args.objective,
        "workspace_path": str(Path(args.workspace).resolve()) if args.workspace else "",
        "max_context_chars": str(args.context_chars),
        "temperature": str(args.temperature),
        "top_p": str(args.top_p),
        "max_new_tokens": str(args.max_new_tokens),
        "use_chat_template": True,
        "autoload": bool(model_id),
    }
    if model_id:
        payload.update(
            {
                "model_id": model_id,
                "device": args.device,
                "dtype": args.dtype,
                "compression": args.compression,
                "load_mode": args.load_mode,
                "prefetching": "auto",
                "cleanup_interval": "4",
                "prefetch_workers": "1",
                "reinitialize_model_each_forward": False,
                "max_seq_len": str(args.max_seq_len),
                "layer_shards_saving_path": "",
                "hf_token": args.hf_token or "",
                "profiling_mode": False,
                "delete_original": False,
            }
        )
    return payload


def command_agent(args: argparse.Namespace) -> int:
    status = api_request(args.server, "/api/status", timeout=args.timeout)
    payload = agent_payload(args, status)
    result = api_request(args.server, "/api/agent/run", payload, timeout=args.timeout)
    if args.json:
        print_json(result)
    else:
        print(result.get("text", "").strip())
        print()
        print("-" * 72)
        print(f"workspace: {result.get('workspace')}")
        print(f"seconds: {result.get('seconds')} | input={result.get('input_tokens')} | output={result.get('output_tokens')}")
        included = result.get("included_files") or []
        if included:
            print("included files: " + ", ".join(included))

    if args.out:
        Path(args.out).write_text(result.get("text", ""), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="airllm_cli.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="AirLLM terminal kliens a helyi UI/backend API-hoz.",
        epilog=textwrap.dedent(
            """
            Peldak:
              python airllm_cli.py status
              python airllm_cli.py models
              python airllm_cli.py hf-search "Qwen2.5 Coder"
              python airllm_cli.py agent "Nezd at a projektet es javasolj javitasokat" --workspace .
              python airllm_cli.py agent "Keress security hibakat" --model Qwen/Qwen2.5-Coder-3B-Instruct
            """
        ),
    )
    parser.add_argument("--server", default=DEFAULT_SERVER, help=f"AirLLM backend URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--timeout", type=int, default=900, help="HTTP timeout masodpercben")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Betoltott modell allapota")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=command_status)

    models = subparsers.add_parser("models", help="Helyi Hugging Face cache modellek")
    models.add_argument("--json", action="store_true")
    models.set_defaults(func=command_models)

    search = subparsers.add_parser("hf-search", help="Publikus Hugging Face modellek keresese")
    search.add_argument("query", help="Keresoszo, pl. Qwen2.5 Coder")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--json", action="store_true")
    search.set_defaults(func=command_hf_search)

    download = subparsers.add_parser("download", help="Modell letoltes inditasa")
    download.add_argument("model", help="Hugging Face model ID")
    download.add_argument("--hf-token", default="")
    download.add_argument("--json", action="store_true")
    download.set_defaults(func=command_download)

    download_status = subparsers.add_parser("download-status", help="Letoltes allapota")
    download_status.add_argument("--json", action="store_true")
    download_status.set_defaults(func=command_download_status)

    cancel_download = subparsers.add_parser("cancel-download", help="Futo modell letoltes leallitasa")
    cancel_download.add_argument("--json", action="store_true")
    cancel_download.set_defaults(func=command_cancel_download)

    agent = subparsers.add_parser("agent", help="Coding agent futtatasa terminalbol")
    agent.add_argument("objective", help="Feladat, amit az agent kap")
    agent.add_argument("--workspace", default=".", help="Workspace konyvtar")
    agent.add_argument("--model", default="", help="Hugging Face model ID; uresen a mar betoltott modellt hasznalja")
    agent.add_argument("--device", default="auto")
    agent.add_argument("--dtype", default="auto")
    agent.add_argument("--compression", default="auto")
    agent.add_argument("--load-mode", default="auto", choices=["auto", "airllm", "direct", "hybrid"])
    agent.add_argument("--max-seq-len", type=int, default=512)
    agent.add_argument("--max-new-tokens", type=int, default=900)
    agent.add_argument("--context-chars", type=int, default=16000)
    agent.add_argument("--temperature", type=float, default=0.2)
    agent.add_argument("--top-p", type=float, default=0.9)
    agent.add_argument("--hf-token", default="")
    agent.add_argument("--out", default="", help="Valasz mentese fajlba")
    agent.add_argument("--json", action="store_true")
    agent.set_defaults(func=command_agent)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except AirLLMCliError as exc:
        print(f"Hiba: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Megszakitva.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
