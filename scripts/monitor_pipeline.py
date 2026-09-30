#!/usr/bin/env python3
"""Write atomic JSON health snapshots for the extraction pipeline.

Only the explicitly allow-listed physical GPUs are queried. GPU 0 and GPU 2
are reserved for pre-existing workloads and are rejected at argument parsing.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_PHYSICAL_GPU_IDS = (1, 3, 4, 5, 6, 7)
DEFAULT_DATA_ROOT = Path("/data/industrial-data-corps")
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "status" / "pipeline_status.json"
DEFAULT_HISTORY = DEFAULT_DATA_ROOT / "status" / "pipeline_metrics.jsonl"
GPU_QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)
STOP_REQUESTED = False
CORE_MODEL_SERVICES = (
    ("parse", "http://127.0.0.1:8001/health"),
    ("llm", "http://127.0.0.1:8003/health"),
)
EMBEDDING_SERVICE = ("embedding", "http://127.0.0.1:8004/health")
RUN_JOB_STATUS_KINDS = frozenset(
    {
        "industrial-catalog-benchmark-run",
        "industrial-catalog-job",
        "industrial-catalog-job-status",
        "industrial-catalog-run-status",
    }
)
RUN_JOB_STATUS_FILENAMES = frozenset({"job_status.json", "run_status.json"})
JOB_STATES = ("queued", "running", "succeeded", "review", "failed", "skipped", "unknown")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_gpu_ids(value: str) -> tuple[int, ...]:
    """Parse and validate physical GPU IDs against the hard allow list."""

    try:
        gpu_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers") from exc
    if not gpu_ids:
        raise argparse.ArgumentTypeError("at least one GPU ID is required")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must not be repeated")
    forbidden = sorted(set(gpu_ids) - set(ALLOWED_PHYSICAL_GPU_IDS))
    if forbidden:
        raise argparse.ArgumentTypeError(
            f"physical GPU IDs {forbidden} are not allowed; allowed IDs are "
            f"{list(ALLOWED_PHYSICAL_GPU_IDS)}"
        )
    return gpu_ids


def _number(value: str, *, integer: bool = False) -> int | float | None:
    normalized = value.strip()
    if normalized.lower() in {"", "n/a", "na", "[not supported]"}:
        return None
    try:
        return int(float(normalized)) if integer else float(normalized)
    except ValueError:
        return None


def collect_gpu_metrics(gpu_ids: tuple[int, ...]) -> tuple[list[dict[str, Any]], list[str]]:
    """Read metrics for allow-listed physical GPUs through nvidia-smi."""

    command = [
        "nvidia-smi",
        "-i",
        ",".join(str(item) for item in gpu_ids),
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except FileNotFoundError:
        return [], ["nvidia-smi is not installed or not on PATH"]
    except subprocess.TimeoutExpired:
        return [], ["nvidia-smi timed out after 20 seconds"]
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "unknown nvidia-smi error").strip()
        return [], [f"nvidia-smi failed: {detail}"]

    rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
    metrics: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        if len(row) != len(GPU_QUERY_FIELDS):
            errors.append(f"unexpected nvidia-smi row with {len(row)} columns")
            continue
        index = _number(row[0], integer=True)
        if index is None or index not in gpu_ids:
            errors.append(f"nvidia-smi returned an unrequested GPU index: {row[0]!r}")
            continue
        metrics.append(
            {
                "physical_index": index,
                "uuid": row[1].strip(),
                "name": row[2].strip(),
                "utilization_gpu_percent": _number(row[3]),
                "memory_used_mib": _number(row[4]),
                "memory_total_mib": _number(row[5]),
                "temperature_c": _number(row[6]),
                "power_draw_w": _number(row[7]),
            }
        )
    metrics.sort(key=lambda item: item["physical_index"])
    returned = {item["physical_index"] for item in metrics}
    missing = sorted(set(gpu_ids) - returned)
    if missing:
        errors.append(f"nvidia-smi returned no data for requested GPUs {missing}")
    return metrics, errors


def _iter_json_files(root: Path, limit: int) -> tuple[list[Path], bool]:
    files: list[Path] = []
    truncated = False
    if not root.is_dir():
        return files, truncated
    for current, directories, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name for name in directories if not (current_path / name).is_symlink()
        )
        for filename in sorted(filenames):
            path = current_path / filename
            if path.is_symlink() or path.suffix.lower() != ".json":
                continue
            if len(files) >= limit:
                truncated = True
                return files, truncated
            files.append(path)
    return files, truncated


def _is_run_or_job_status_payload(path: Path, payload: dict[str, Any]) -> bool:
    """Return whether a JSON object represents a run/job status rather than an artifact."""

    kind = payload.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip().lower() in RUN_JOB_STATUS_KINDS

    has_state = "status" in payload or "state" in payload
    has_identity = "run_id" in payload or "job_id" in payload
    return has_state and (
        has_identity or path.name.lower() in RUN_JOB_STATUS_FILENAMES
    )


def collect_job_metrics(jobs_root: Path, limit: int) -> dict[str, Any]:
    """Summarize actual run/job status payloads without counting other artifacts."""

    aliases = {
        "pending": "queued",
        "created": "queued",
        "in_progress": "running",
        "processing": "running",
        "complete": "succeeded",
        "completed": "succeeded",
        "success": "succeeded",
        "needs_review": "review",
        "completed_with_warnings": "review",
        "error": "failed",
    }
    counts: Counter[str] = Counter()
    parse_errors = 0
    ignored_payloads = 0
    status_payloads = 0
    files, truncated = _iter_json_files(jobs_root, limit)
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not _is_run_or_job_status_payload(path, payload):
                ignored_payloads += 1
                continue
            raw_state = payload.get("status", payload.get("state", "unknown"))
            state = str(raw_state).strip().lower().replace("-", "_").replace(" ", "_")
            state = aliases.get(state, state)
            if state not in JOB_STATES:
                state = "unknown"
            counts[state] += 1
            status_payloads += 1
        except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
            if path.name.lower() in RUN_JOB_STATUS_FILENAMES:
                parse_errors += 1
            else:
                ignored_payloads += 1

    return {
        "root": str(jobs_root),
        "root_exists": jobs_root.is_dir(),
        "json_files_scanned": len(files),
        "status_payloads_counted": status_payloads,
        "ignored_json_payloads": ignored_payloads,
        "scan_truncated": truncated,
        "parse_error_count": parse_errors,
        "states": {state: counts[state] for state in JOB_STATES},
    }


def collect_disk_metrics(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return None, f"cannot read disk usage for {path}: {exc}"
    used = usage.total - usage.free
    return (
        {
            "path": str(path),
            "total_bytes": usage.total,
            "used_bytes": used,
            "free_bytes": usage.free,
            "used_percent": round(100.0 * used / usage.total, 2) if usage.total else None,
        },
        None,
    )


def collect_model_service_metrics(
    *,
    enabled: bool,
    include_embedding: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Probe loopback-only model health endpoints with short read timeouts."""

    if not enabled:
        return [], []
    definitions = list(CORE_MODEL_SERVICES)
    if include_embedding:
        definitions.append(EMBEDDING_SERVICE)
    services: list[dict[str, Any]] = []
    errors: list[str] = []
    for name, url in definitions:
        started = time.monotonic()
        status_code: int | None = None
        detail: str | None = None
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                status_code = response.status
            healthy = status_code == 200
        except (OSError, urllib.error.URLError) as exc:
            healthy = False
            detail = str(exc)
        latency_ms = round((time.monotonic() - started) * 1000, 2)
        services.append(
            {
                "name": name,
                "url": url,
                "healthy": healthy,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "detail": detail,
            }
        )
        if not healthy:
            errors.append(f"model service {name} is not healthy at {url}")
    return services, errors


def process_is_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _average(values: list[int | float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), 2) if numeric else None


def snapshot(
    *,
    gpu_ids: tuple[int, ...],
    data_root: Path,
    jobs_root: Path,
    max_job_files: int,
    watch_pid: int | None,
    check_model_services: bool = True,
    check_embedding: bool = False,
) -> dict[str, Any]:
    gpu_metrics, errors = collect_gpu_metrics(gpu_ids)
    disk_metrics, disk_error = collect_disk_metrics(data_root)
    if disk_error:
        errors.append(disk_error)
    jobs = collect_job_metrics(jobs_root, max_job_files)
    if jobs["parse_error_count"]:
        errors.append(f"{jobs['parse_error_count']} job JSON file(s) could not be parsed")
    if jobs["states"]["failed"]:
        errors.append(f"{jobs['states']['failed']} pipeline job(s) report a failed state")
    if disk_metrics and disk_metrics["used_percent"] is not None:
        if disk_metrics["used_percent"] >= 90:
            errors.append(f"data volume is {disk_metrics['used_percent']}% full")
    model_services, model_errors = collect_model_service_metrics(
        enabled=check_model_services,
        include_embedding=check_embedding,
    )
    errors.extend(model_errors)

    memory_used = [item["memory_used_mib"] for item in gpu_metrics]
    memory_total = [item["memory_total_mib"] for item in gpu_metrics]
    aggregate = {
        "requested_gpu_count": len(gpu_ids),
        "observed_gpu_count": len(gpu_metrics),
        "memory_used_mib": sum(value for value in memory_used if value is not None),
        "memory_total_mib": sum(value for value in memory_total if value is not None),
        "mean_gpu_utilization_percent": _average(
            [item["utilization_gpu_percent"] for item in gpu_metrics]
        ),
        "max_temperature_c": max(
            (item["temperature_c"] for item in gpu_metrics if item["temperature_c"] is not None),
            default=None,
        ),
    }
    watched_alive = process_is_alive(watch_pid)
    return {
        "schema_version": "1.0",
        "kind": "industrial-catalog-pipeline-status",
        "status": "healthy" if not errors else "degraded",
        "observed_at": utc_now(),
        "host": {"hostname": os.uname().nodename if hasattr(os, "uname") else None},
        "gpu_policy": {
            "physical_ids": list(gpu_ids),
            "reserved_physical_ids": [0, 2],
        },
        "watch": {"pid": watch_pid, "alive": watched_alive},
        "metrics": {
            "gpu_aggregate": aggregate,
            "gpus": gpu_metrics,
            "disk": disk_metrics,
            "jobs": jobs,
            "model_services": {
                "checked": check_model_services,
                "embedding_included": check_embedding,
                "services": model_services,
            },
        },
        "errors": errors,
    }


def write_json_atomic(payload: dict[str, Any], destination: Path) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            Path(temporary_name).unlink(missing_ok=True)
        finally:
            raise


def append_jsonl(payload: dict[str, Any], destination: Path) -> None:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=ALLOWED_PHYSICAL_GPU_IDS,
        help="allow-listed physical GPU IDs (default: 1,3,4,5,6,7)",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--jobs-root", type=Path, default=DEFAULT_DATA_ROOT / "jobs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--interval-seconds", type=float, default=15.0)
    parser.add_argument("--max-job-files", type=int, default=100_000)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--skip-model-health", action="store_true")
    parser.add_argument("--with-embedding-health", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.interval_seconds < 1:
        print("monitor failed: --interval-seconds must be at least 1", file=sys.stderr)
        return 2
    if args.max_job_files < 1:
        print("monitor failed: --max-job-files must be positive", file=sys.stderr)
        return 2
    if args.watch_pid is not None and args.watch_pid <= 0:
        print("monitor failed: --watch-pid must be positive", file=sys.stderr)
        return 2

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    final_payload: dict[str, Any] | None = None
    while not STOP_REQUESTED:
        final_payload = snapshot(
            gpu_ids=args.gpu_ids,
            data_root=args.data_root.expanduser().resolve(),
            jobs_root=args.jobs_root.expanduser().resolve(),
            max_job_files=args.max_job_files,
            watch_pid=args.watch_pid,
            check_model_services=not args.skip_model_health,
            check_embedding=args.with_embedding_health,
        )
        try:
            write_json_atomic(final_payload, args.output)
            if not args.no_history:
                append_jsonl(final_payload, args.history)
        except OSError as exc:
            print(f"monitor failed to write status: {exc}", file=sys.stderr)
            return 1

        print(
            json.dumps(
                {
                    "status": final_payload["status"],
                    "observed_at": final_payload["observed_at"],
                    "output": str(args.output),
                    "watch": final_payload["watch"],
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        if args.once or final_payload["watch"]["alive"] is False:
            break
        stop_at = time.monotonic() + args.interval_seconds
        while not STOP_REQUESTED and time.monotonic() < stop_at:
            time.sleep(min(0.5, max(0.0, stop_at - time.monotonic())))

    if final_payload is None:
        return 0
    return 0 if final_payload["status"] == "healthy" else 3


if __name__ == "__main__":
    raise SystemExit(main())
