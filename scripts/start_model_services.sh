#!/usr/bin/env bash
# Start only the explicitly assigned public NVIDIA models on physical GPUs 1, 3, and 5.
set -Eeuo pipefail
umask 027

readonly DATA_ROOT="/data/industrial-data-corps"
readonly MODEL_VENV="${DATA_ROOT}/venvs/model-serve-py312"
readonly HOST="127.0.0.1"
readonly PARSE_MODEL="nvidia/NVIDIA-Nemotron-Parse-v1.2"
readonly LLM_MODEL="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
readonly EMBED_MODEL="nvidia/Nemotron-3-Embed-8B-BF16"
readonly LOG_ROOT="${DATA_ROOT}/logs/model-services"
readonly STATUS_PATH="${DATA_ROOT}/status/model_services.json"

START_EMBEDDING=0
WAIT_SECONDS=1800
declare -A SERVICE_PIDS=()

usage() {
  printf '%s\n' \
    "Usage: bash scripts/start_model_services.sh [--with-embedding] [--wait-seconds N]" \
    "" \
    "Core: Nemotron Parse on physical GPU 1 / port 8001 and Nemotron Nano 8B" \
    "on physical GPU 3 / port 8003. Optional embedding uses GPU 5 / port 8004."
}

while (($#)); do
  case "$1" in
    --with-embedding)
      START_EMBEDDING=1
      shift
      ;;
    --wait-seconds)
      if (($# < 2)); then
        printf 'model service start failed: --wait-seconds requires a value\n' >&2
        exit 2
      fi
      WAIT_SECONDS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'model service start failed: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]] \
  || ((WAIT_SECONDS < 30 || WAIT_SECONDS > 7200)); then
  printf 'model service start failed: wait seconds must be from 30 through 7200\n' >&2
  exit 2
fi
if [[ ! -d "${DATA_ROOT}" || ! -w "${DATA_ROOT}" ]]; then
  printf 'model service start failed: data root must exist and be writable: %s\n' \
    "${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -x "${MODEL_VENV}/bin/vllm" ]]; then
  printf '%s\n' \
    "model service start failed: ${MODEL_VENV}/bin/vllm is missing" >&2
  exit 1
fi
if ! "${MODEL_VENV}/bin/python" - <<'PY'
from importlib.metadata import version

assert version("vllm") == "0.20.0", "core model services require vllm==0.20.0"
PY
then
  printf 'model service start failed: core vLLM version does not match runtime policy\n' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'model service start failed: nvidia-smi is unavailable\n' >&2
  exit 1
fi

mkdir -p -- \
  "${LOG_ROOT}" \
  "${DATA_ROOT}/caches/huggingface" \
  "${DATA_ROOT}/caches/vllm" \
  "${DATA_ROOT}/models/huggingface" \
  "${DATA_ROOT}/status" \
  "${DATA_ROOT}/tmp"

export HF_HOME="${DATA_ROOT}/caches/huggingface"
export HF_HUB_CACHE="${DATA_ROOT}/caches/huggingface/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_CACHE_ROOT="${DATA_ROOT}/caches/vllm"
export XDG_CACHE_HOME="${DATA_ROOT}/caches"
export TMPDIR="${DATA_ROOT}/tmp"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

pid_matches_service() {
  local pid="$1"
  local model="$2"
  local port="$3"
  local command_line
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  [[ "${command_line}" == *"${model}"* ]] || return 1
  if [[ " ${command_line} " != *" --port ${port} "* \
    && " ${command_line} " != *" --port=${port} "* ]]; then
    return 1
  fi
}

read_service_pid() {
  local pid_file="$1"
  local pid=""
  if [[ -f "${pid_file}" ]]; then
    IFS= read -r pid <"${pid_file}" || true
  fi
  printf '%s' "${pid}"
}

port_is_available() {
  "${MODEL_VENV}/bin/python" - "$1" "$2" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket() as handle:
    handle.bind((host, port))
PY
}

service_is_healthy() {
  "${MODEL_VENV}/bin/python" - "$1" "$2" "$3" <<'PY'
import json
import sys
import urllib.request

host, port, expected_model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as response:
    if response.status != 200:
        raise SystemExit(1)
with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=3) as response:
    payload = json.load(response)
models = {item.get("id") for item in payload.get("data", [])}
raise SystemExit(0 if expected_model in models else 1)
PY
}

gpu_has_compute_process() {
  local gpu="$1"
  local process_ids
  process_ids="$(
    nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits \
      2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}'
  )"
  [[ -n "${process_ids}" ]]
}

start_service() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local model="$4"
  local pid_file
  local log_file="${LOG_ROOT}/${name}.log"
  local existing_pid
  local started_pid
  local -a command

  case "${name}" in
    parse) pid_file="${DATA_ROOT}/status/nemotron-parse.pid" ;;
    llm) pid_file="${DATA_ROOT}/status/nemotron-llm.pid" ;;
    embedding) pid_file="${DATA_ROOT}/status/nemotron-embed.pid" ;;
    *) return 1 ;;
  esac

  case "${gpu}" in
    1|3|5) ;;
    *)
      printf 'refusing unapproved physical GPU %s for %s\n' "${gpu}" "${name}" >&2
      return 1
      ;;
  esac
  if ! nvidia-smi -i "${gpu}" --query-gpu=index --format=csv,noheader,nounits \
    >/dev/null 2>&1; then
    printf 'physical GPU %s is unavailable for %s\n' "${gpu}" "${name}" >&2
    return 1
  fi

  existing_pid="$(read_service_pid "${pid_file}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    if ! pid_matches_service "${existing_pid}" "${model}" "${port}"; then
      printf 'refusing to reuse mismatched live PID %s from %s\n' \
        "${existing_pid}" "${pid_file}" >&2
      return 1
    fi
    SERVICE_PIDS["${name}"]="${existing_pid}"
    printf '%s already owns PID %s; waiting for health\n' "${name}" "${existing_pid}"
    return 0
  fi
  if ! port_is_available "${HOST}" "${port}"; then
    printf 'refusing to start %s: %s:%s is owned by another process\n' \
      "${name}" "${HOST}" "${port}" >&2
    return 1
  fi
  if gpu_has_compute_process "${gpu}"; then
    printf 'refusing to start %s: physical GPU %s already has a compute process\n' \
      "${name}" "${gpu}" >&2
    return 1
  fi

  command=(
    "${MODEL_VENV}/bin/vllm" serve "${model}"
    --host "${HOST}"
    --port "${port}"
    --served-model-name "${model}"
    --download-dir "${DATA_ROOT}/models/huggingface"
    --dtype bfloat16
    --tensor-parallel-size 1
  )
  case "${name}" in
    parse)
      command+=(
        --max-num-seqs 8
        --limit-mm-per-prompt '{"image":1}'
        --attention-backend TRITON_ATTN
        --trust-remote-code
        --gpu-memory-utilization 0.75
      )
      ;;
    llm)
      command+=(
        --max-model-len 32768
        --max-num-seqs 16
        --gpu-memory-utilization 0.85
        # Prevent the JSON grammar from spending the completion budget on
        # optional inter-field whitespace. This is supported by xgrammar in
        # vLLM 0.20.0 and keeps response_format output compact/deterministic.
        --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
      )
      ;;
    embedding)
      command+=(
        --max-model-len 32768
        --max-num-seqs 16
        --gpu-memory-utilization 0.85
      )
      ;;
  esac

  printf '\n[%s] starting model=%s physical_gpu=%s port=%s\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${model}" "${gpu}" "${port}" >>"${log_file}"
  nohup env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    HF_HOME="${HF_HOME}" \
    HF_HUB_CACHE="${HF_HUB_CACHE}" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT}" \
    XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
    TMPDIR="${TMPDIR}" \
    "${command[@]}" >>"${log_file}" 2>&1 </dev/null &
  started_pid=$!
  printf '%s\n' "${started_pid}" >"${pid_file}"
  SERVICE_PIDS["${name}"]="${started_pid}"
  printf 'started %s as PID %s on physical GPU %s\n' \
    "${name}" "${started_pid}" "${gpu}"
}

wait_for_service() {
  local name="$1"
  local port="$2"
  local model="$3"
  local pid="${SERVICE_PIDS[${name}]}"
  local deadline=$((SECONDS + WAIT_SECONDS))
  while ((SECONDS < deadline)); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      printf '%s exited before becoming healthy; inspect %s/%s.log\n' \
        "${name}" "${LOG_ROOT}" "${name}" >&2
      return 1
    fi
    if service_is_healthy "${HOST}" "${port}" "${model}" 2>/dev/null; then
      printf '%s is healthy at http://%s:%s (PID %s)\n' \
        "${name}" "${HOST}" "${port}" "${pid}"
      return 0
    fi
    sleep 5
  done
  printf '%s did not become healthy within %s seconds; inspect %s/%s.log\n' \
    "${name}" "${WAIT_SECONDS}" "${LOG_ROOT}" "${name}" >&2
  return 1
}

FAILURES=0
if ! start_service parse 1 8001 "${PARSE_MODEL}"; then
  FAILURES=$((FAILURES + 1))
fi
if ! start_service llm 3 8003 "${LLM_MODEL}"; then
  FAILURES=$((FAILURES + 1))
fi
if ((START_EMBEDDING)); then
  if ! start_service embedding 5 8004 "${EMBED_MODEL}"; then
    FAILURES=$((FAILURES + 1))
  fi
fi

if [[ -n "${SERVICE_PIDS[parse]:-}" ]] \
  && ! wait_for_service parse 8001 "${PARSE_MODEL}"; then
  FAILURES=$((FAILURES + 1))
fi
if [[ -n "${SERVICE_PIDS[llm]:-}" ]] \
  && ! wait_for_service llm 8003 "${LLM_MODEL}"; then
  FAILURES=$((FAILURES + 1))
fi
if ((START_EMBEDDING)) && [[ -n "${SERVICE_PIDS[embedding]:-}" ]] \
  && ! wait_for_service embedding 8004 "${EMBED_MODEL}"; then
  FAILURES=$((FAILURES + 1))
fi

export MODEL_STATUS_PARSE_PID="${SERVICE_PIDS[parse]:-}"
export MODEL_STATUS_LLM_PID="${SERVICE_PIDS[llm]:-}"
export MODEL_STATUS_EMBED_PID="${SERVICE_PIDS[embedding]:-}"
export MODEL_STATUS_EMBED_ENABLED="${START_EMBEDDING}"
export MODEL_STATUS_FAILURES="${FAILURES}"
export MODEL_STATUS_PATH="${STATUS_PATH}"
"${MODEL_VENV}/bin/python" - <<'PY'
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

definitions = (
    ("parse", "MODEL_STATUS_PARSE_PID", 1, 8001, "nvidia/NVIDIA-Nemotron-Parse-v1.2"),
    ("llm", "MODEL_STATUS_LLM_PID", 3, 8003, "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"),
    ("embedding", "MODEL_STATUS_EMBED_PID", 5, 8004, "nvidia/Nemotron-3-Embed-8B-BF16"),
)
services = []
for name, variable, gpu, port, model in definitions:
    raw_pid = os.environ.get(variable, "")
    enabled = name != "embedding" or os.environ["MODEL_STATUS_EMBED_ENABLED"] == "1"
    services.append(
        {
            "name": name,
            "enabled": enabled,
            "pid": int(raw_pid) if raw_pid else None,
            "physical_gpu": gpu,
            "port": port,
            "model": model,
        }
    )
payload = {
    "schema_version": "1.0",
    "kind": "industrial-catalog-model-services",
    "status": "ready" if int(os.environ["MODEL_STATUS_FAILURES"]) == 0 else "degraded",
    "observed_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "reserved_physical_gpus": [0, 2],
    "services": services,
}
path = Path(os.environ["MODEL_STATUS_PATH"])
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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

((FAILURES == 0))
