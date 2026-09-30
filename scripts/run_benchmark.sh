#!/usr/bin/env bash
# Run the extraction benchmark with a strict physical-GPU allow list and JSON telemetry.
set -Eeuo pipefail
umask 027

readonly GPU_IDS="1,3,4,5,6,7"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DATA_ROOT="${INDUSTRIAL_DATA_ROOT:-/data/industrial-data-corps}"
VENV_PATH="${INDUSTRIAL_VENV:-${DATA_ROOT}/venvs/extractor-py312}"
CONFIG_PATH="${INDUSTRIAL_PIPELINE_CONFIG:-${PROJECT_ROOT}/configs/pipeline.yaml}"
BENCHMARK_MODULE="${INDUSTRIAL_BENCHMARK_MODULE:-industrial_catalog.cli}"
MAX_DOCUMENTS="${INDUSTRIAL_BENCHMARK_MAX_DOCUMENTS:-20}"
MAX_PAGES_PER_DOCUMENT="${INDUSTRIAL_BENCHMARK_MAX_PAGES_PER_DOCUMENT:-12}"
TIMEOUT_SECONDS="${INDUSTRIAL_BENCHMARK_TIMEOUT_SECONDS:-7200}"

DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
VENV_PATH="$(realpath -m -- "${VENV_PATH}")"
CONFIG_PATH="$(realpath -m -- "${CONFIG_PATH}")"
if [[ ! -f "${CONFIG_PATH}" && "${CONFIG_PATH}" == "${PROJECT_ROOT}/configs/pipeline.yaml" ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/configs/pipeline.example.yaml"
fi
case "${DATA_ROOT}" in
  /data/*) ;;
  *)
    printf 'benchmark failed: INDUSTRIAL_DATA_ROOT must be an absolute child of /data\n' >&2
    exit 2
    ;;
esac
case "${VENV_PATH}" in
  "${DATA_ROOT}"/*) ;;
  *)
    printf 'benchmark failed: INDUSTRIAL_VENV must be inside INDUSTRIAL_DATA_ROOT\n' >&2
    exit 2
    ;;
esac
if [[ ! "${MAX_DOCUMENTS}" =~ ^[1-9][0-9]*$ ]] || ((MAX_DOCUMENTS > 100)); then
  printf 'benchmark failed: max documents must be an integer from 1 through 100\n' >&2
  exit 2
fi
if [[ ! "${MAX_PAGES_PER_DOCUMENT}" =~ ^[1-9][0-9]*$ ]] \
  || ((MAX_PAGES_PER_DOCUMENT > 50)); then
  printf 'benchmark failed: max pages per document must be an integer from 1 through 50\n' >&2
  exit 2
fi
if ((MAX_DOCUMENTS * MAX_PAGES_PER_DOCUMENT > 500)); then
  printf 'benchmark failed: bounded benchmark may not exceed 500 candidate pages\n' >&2
  exit 2
fi
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || ((TIMEOUT_SECONDS < 60 || TIMEOUT_SECONDS > 86400)); then
  printf 'benchmark failed: timeout must be an integer from 60 through 86400 seconds\n' >&2
  exit 2
fi
if [[ ! "${BENCHMARK_MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]]; then
  printf 'benchmark failed: invalid Python module name: %s\n' "${BENCHMARK_MODULE}" >&2
  exit 2
fi
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  printf 'benchmark failed: bootstrap venv is missing: %s\n' "${VENV_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'benchmark failed: pipeline config is missing: %s\n' "${CONFIG_PATH}" >&2
  exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
  printf 'benchmark failed: GNU timeout is required\n' >&2
  exit 1
fi

for forwarded_argument in "$@"; do
  case "${forwarded_argument}" in
    --max-documents|--max-documents=*|--max-pages|--max-pages=*|--max-pages-per-document|--max-pages-per-document=*)
      printf '%s\n' \
        'benchmark failed: set benchmark bounds with INDUSTRIAL_BENCHMARK_MAX_* variables' >&2
      exit 2
      ;;
  esac
done

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PIP_CACHE_DIR="${DATA_ROOT}/caches/pip"
export HF_HOME="${DATA_ROOT}/caches/huggingface"
export XDG_CACHE_HOME="${DATA_ROOT}/caches/xdg"
export TMPDIR="${DATA_ROOT}/tmp"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Local vLLM uses the OpenAI JSON-schema dialect, not NVIDIA NIM's nvext wrapper.
export NVIDIA_LLM_GUIDED_JSON_MODE="${NVIDIA_LLM_GUIDED_JSON_MODE:-openai_response_format}"

RUN_ID="$(date -u +'%Y%m%dT%H%M%SZ')-$$"
RUN_DIR="${DATA_ROOT}/benchmarks/${RUN_ID}"
mkdir -p -- "${RUN_DIR}" "${DATA_ROOT}/jobs" "${DATA_ROOT}/tmp"

COMMAND=(
  "${VENV_PATH}/bin/python"
  -m "${BENCHMARK_MODULE}"
  benchmark
  --config "${CONFIG_PATH}"
  --output-dir "${RUN_DIR}"
  --max-documents "${MAX_DOCUMENTS}"
  --max-pages "$((MAX_DOCUMENTS * MAX_PAGES_PER_DOCUMENT))"
  --max-pages-per-document "${MAX_PAGES_PER_DOCUMENT}"
  "$@"
)

export BENCHMARK_RUN_ID="${RUN_ID}"
export BENCHMARK_RUN_DIR="${RUN_DIR}"
export BENCHMARK_CONFIG_PATH="${CONFIG_PATH}"
export BENCHMARK_GPU_IDS="${GPU_IDS}"
export BENCHMARK_MAX_DOCUMENTS="${MAX_DOCUMENTS}"
export BENCHMARK_MAX_PAGES_PER_DOCUMENT="${MAX_PAGES_PER_DOCUMENT}"
export BENCHMARK_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}"
export BENCHMARK_COMMAND_JSON
BENCHMARK_COMMAND_JSON="$(
  "${VENV_PATH}/bin/python" -c \
    'import json,sys; print(json.dumps(sys.argv[1:]))' "${COMMAND[@]}"
)"

write_run_status() {
  local run_status="$1"
  local exit_code_value="$2"
  export BENCHMARK_STATUS="${run_status}"
  export BENCHMARK_EXIT_CODE="${exit_code_value}"
  "${VENV_PATH}/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

run_dir = Path(os.environ["BENCHMARK_RUN_DIR"])
path = run_dir / "run_status.json"
exit_code = int(os.environ["BENCHMARK_EXIT_CODE"])
payload = {
    "schema_version": "1.0",
    "kind": "industrial-catalog-benchmark-run",
    "run_id": os.environ["BENCHMARK_RUN_ID"],
    "status": os.environ["BENCHMARK_STATUS"],
    "observed_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "config_path": os.environ["BENCHMARK_CONFIG_PATH"],
    "output_dir": str(run_dir),
    "physical_gpu_ids": [int(value) for value in os.environ["BENCHMARK_GPU_IDS"].split(",")],
    "reserved_physical_gpu_ids": [0, 2],
    "bounds": {
        "max_documents": int(os.environ["BENCHMARK_MAX_DOCUMENTS"]),
        "max_pages_per_document": int(os.environ["BENCHMARK_MAX_PAGES_PER_DOCUMENT"]),
        "max_candidate_pages": int(os.environ["BENCHMARK_MAX_DOCUMENTS"])
        * int(os.environ["BENCHMARK_MAX_PAGES_PER_DOCUMENT"]),
        "timeout_seconds": int(os.environ["BENCHMARK_TIMEOUT_SECONDS"]),
    },
    "command": json.loads(os.environ["BENCHMARK_COMMAND_JSON"]),
    "exit_code": None if exit_code < 0 else exit_code,
}
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, path)
except BaseException:
    Path(temporary_name).unlink(missing_ok=True)
    raise
print(json.dumps(payload, separators=(",", ":")))
PY
}

MONITOR_PID=""
MONITOR_EXTRA_ARGS=()
for forwarded_argument in "$@"; do
  if [[ "${forwarded_argument}" == "--dry-run" ]]; then
    MONITOR_EXTRA_ARGS+=(--skip-model-health)
  fi
done
stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}
write_interrupted_status() {
  local exit_code_value="$1"
  write_run_status "interrupted" "${exit_code_value}" || true
  stop_monitor
  exit "${exit_code_value}"
}
trap stop_monitor EXIT
trap 'write_interrupted_status 130' INT
trap 'write_interrupted_status 143' TERM

write_run_status "running" -1
"${VENV_PATH}/bin/python" "${SCRIPT_DIR}/monitor_pipeline.py" \
  --gpu-ids "${GPU_IDS}" \
  --data-root "${DATA_ROOT}" \
  --jobs-root "${DATA_ROOT}/jobs" \
  --output "${RUN_DIR}/monitor.latest.json" \
  --history "${RUN_DIR}/monitor.jsonl" \
  --interval-seconds 10 \
  --watch-pid "$$" \
  "${MONITOR_EXTRA_ARGS[@]}" \
  >"${RUN_DIR}/monitor.log" 2>&1 &
MONITOR_PID=$!

set +e
timeout --foreground --signal=TERM --kill-after=60s "${TIMEOUT_SECONDS}s" \
  "${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/benchmark.log"
BENCHMARK_EXIT_CODE_VALUE=${PIPESTATUS[0]}
set -e

if ((BENCHMARK_EXIT_CODE_VALUE == 0)); then
  write_run_status "succeeded" "${BENCHMARK_EXIT_CODE_VALUE}"
elif ((BENCHMARK_EXIT_CODE_VALUE == 124)); then
  write_run_status "timed_out" "${BENCHMARK_EXIT_CODE_VALUE}"
else
  write_run_status "failed" "${BENCHMARK_EXIT_CODE_VALUE}"
fi
stop_monitor
trap - EXIT INT TERM

printf 'Benchmark artifacts: %s\n' "${RUN_DIR}"
exit "${BENCHMARK_EXIT_CODE_VALUE}"
