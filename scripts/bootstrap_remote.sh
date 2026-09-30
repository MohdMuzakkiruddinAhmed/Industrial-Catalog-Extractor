#!/usr/bin/env bash
# Idempotent Ubuntu 24.04 bootstrap for the dedicated /data volume.
set -Eeuo pipefail
umask 027

readonly GPU_IDS="1,3,4,5,6,7"
readonly RESERVED_GPU_IDS="0,2"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

DATA_ROOT="${INDUSTRIAL_DATA_ROOT:-/data/industrial-data-corps}"
VENV_PATH="${INDUSTRIAL_VENV:-${DATA_ROOT}/venvs/extractor-py312}"
PYTHON_BIN="${INDUSTRIAL_PYTHON_BIN:-python3.12}"
INSTALL_PROJECT=1

usage() {
  printf '%s\n' \
    "Usage: bash scripts/bootstrap_remote.sh [--skip-install]" \
    "" \
    "Creates an isolated Python 3.12 environment and runtime directories under /data." \
    "It never installs system packages, starts GPU workloads, or changes Docker."
}

while (($#)); do
  case "$1" in
    --skip-install)
      INSTALL_PROJECT=0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'bootstrap failed: unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

# Resolve traversal and existing symlinks before validating any write target.
DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
VENV_PATH="$(realpath -m -- "${VENV_PATH}")"
case "${DATA_ROOT}" in
  /data/*) ;;
  *)
    printf 'bootstrap failed: INDUSTRIAL_DATA_ROOT must be an absolute child of /data\n' >&2
    exit 2
    ;;
esac
case "${VENV_PATH}" in
  "${DATA_ROOT}"/*) ;;
  *)
    printf 'bootstrap failed: INDUSTRIAL_VENV must be inside INDUSTRIAL_DATA_ROOT\n' >&2
    exit 2
    ;;
esac

# Brev pre-provisions this project directory but intentionally keeps /data itself
# non-writable. Only require the exact target (or its direct parent when creating it).
if [[ -e "${DATA_ROOT}" && ! -d "${DATA_ROOT}" ]]; then
  printf 'bootstrap failed: data root exists but is not a directory: %s\n' "${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  DATA_PARENT="$(dirname -- "${DATA_ROOT}")"
  if [[ ! -d "${DATA_PARENT}" || ! -w "${DATA_PARENT}" ]]; then
    printf 'bootstrap failed: pre-create a writable data root: %s\n' "${DATA_ROOT}" >&2
    exit 1
  fi
  mkdir -p -- "${DATA_ROOT}"
fi
if [[ ! -w "${DATA_ROOT}" ]]; then
  printf 'bootstrap failed: data root is not writable: %s\n' "${DATA_ROOT}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'bootstrap failed: %s was not found; Python 3.12 is required\n' "${PYTHON_BIN}" >&2
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ ! "${PYTHON_VERSION}" =~ ^3\.12\. ]]; then
  printf 'bootstrap failed: expected Python 3.12, found %s\n' "${PYTHON_VERSION}" >&2
  exit 1
fi

# Keep every large or fast-growing artifact off the nearly-full root volume.
mkdir -p -- \
  "${DATA_ROOT}/artifacts/pages" \
  "${DATA_ROOT}/artifacts/tables" \
  "${DATA_ROOT}/benchmarks" \
  "${DATA_ROOT}/caches/huggingface" \
  "${DATA_ROOT}/caches/pip" \
  "${DATA_ROOT}/caches/xdg" \
  "${DATA_ROOT}/checkpoints" \
  "${DATA_ROOT}/corpus/manifests" \
  "${DATA_ROOT}/corpus/raw" \
  "${DATA_ROOT}/extraction/raw-elements" \
  "${DATA_ROOT}/extraction/products" \
  "${DATA_ROOT}/jobs" \
  "${DATA_ROOT}/logs" \
  "${DATA_ROOT}/models" \
  "${DATA_ROOT}/status" \
  "${DATA_ROOT}/tmp" \
  "${DATA_ROOT}/vector-db" \
  "$(dirname -- "${VENV_PATH}")"

export PIP_CACHE_DIR="${DATA_ROOT}/caches/pip"
export HF_HOME="${DATA_ROOT}/caches/huggingface"
export XDG_CACHE_HOME="${DATA_ROOT}/caches/xdg"
export TMPDIR="${DATA_ROOT}/tmp"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

VENV_VERSION="$("${VENV_PATH}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ ! "${VENV_VERSION}" =~ ^3\.12\. ]]; then
  printf 'bootstrap failed: existing venv is not Python 3.12: %s\n' "${VENV_PATH}" >&2
  exit 1
fi

DEPENDENCY_MODE="skipped"
if ((INSTALL_PROJECT)); then
  if [[ -f "${PROJECT_ROOT}/requirements.lock" ]]; then
    "${VENV_PATH}/bin/python" -m pip install \
      --require-hashes --requirement "${PROJECT_ROOT}/requirements.lock"
    "${VENV_PATH}/bin/python" -m pip install --no-deps --editable "${PROJECT_ROOT}"
    DEPENDENCY_MODE="hash-locked"
  else
    # pyproject.toml constrains every direct dependency and pins Python to 3.12.
    "${VENV_PATH}/bin/python" -m pip install --editable "${PROJECT_ROOT}"
    DEPENDENCY_MODE="pyproject-constrained"
  fi
fi

if ! "${VENV_PATH}/bin/python" -c 'import industrial_catalog' >/dev/null 2>&1; then
  if ((INSTALL_PROJECT)); then
    printf 'bootstrap failed: industrial_catalog is not importable after installation\n' >&2
    exit 1
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'bootstrap failed: nvidia-smi is not available\n' >&2
  exit 1
fi
AVAILABLE_GPU_IDS="$(nvidia-smi -i "${GPU_IDS}" --query-gpu=index \
  --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ',' | sed 's/,$//')"
IFS=',' read -r -a REQUIRED_GPU_ARRAY <<<"${GPU_IDS}"
for required_gpu in "${REQUIRED_GPU_ARRAY[@]}"; do
  if [[ ",${AVAILABLE_GPU_IDS}," != *",${required_gpu},"* ]]; then
    printf 'bootstrap failed: required physical GPU %s was not reported by nvidia-smi\n' \
      "${required_gpu}" >&2
    exit 1
  fi
done

export BOOTSTRAP_DATA_ROOT="${DATA_ROOT}"
export BOOTSTRAP_VENV_PATH="${VENV_PATH}"
export BOOTSTRAP_PROJECT_ROOT="${PROJECT_ROOT}"
export BOOTSTRAP_PYTHON_VERSION="${VENV_VERSION}"
export BOOTSTRAP_DEPENDENCY_MODE="${DEPENDENCY_MODE}"
export BOOTSTRAP_GPU_IDS="${GPU_IDS}"
export BOOTSTRAP_RESERVED_GPU_IDS="${RESERVED_GPU_IDS}"
export BOOTSTRAP_STATUS_PATH="${DATA_ROOT}/status/bootstrap_status.json"

"${VENV_PATH}/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
from datetime import UTC, datetime
from pathlib import Path

data_root = Path(os.environ["BOOTSTRAP_DATA_ROOT"])
status_path = Path(os.environ["BOOTSTRAP_STATUS_PATH"])
disk = shutil.disk_usage(data_root)
payload = {
    "schema_version": "1.0",
    "kind": "industrial-catalog-bootstrap-status",
    "status": "ready",
    "observed_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    "host": socket.gethostname(),
    "project_root": os.environ["BOOTSTRAP_PROJECT_ROOT"],
    "data_root": str(data_root),
    "venv_path": os.environ["BOOTSTRAP_VENV_PATH"],
    "python_version": os.environ["BOOTSTRAP_PYTHON_VERSION"],
    "dependency_mode": os.environ["BOOTSTRAP_DEPENDENCY_MODE"],
    "gpu_policy": {
        "physical_ids": [int(value) for value in os.environ["BOOTSTRAP_GPU_IDS"].split(",")],
        "reserved_physical_ids": [
            int(value) for value in os.environ["BOOTSTRAP_RESERVED_GPU_IDS"].split(",")
        ],
    },
    "metrics": {
        "data_volume_total_bytes": disk.total,
        "data_volume_free_bytes": disk.free,
    },
}
status_path.parent.mkdir(parents=True, exist_ok=True)
descriptor, temporary_name = tempfile.mkstemp(
    prefix=f".{status_path.name}.", suffix=".tmp", dir=status_path.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, status_path)
except BaseException:
    Path(temporary_name).unlink(missing_ok=True)
    raise
print(json.dumps(payload, separators=(",", ":")))
PY

printf 'Bootstrap ready. Activate with: source %q\n' "${VENV_PATH}/bin/activate"
