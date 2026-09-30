#!/usr/bin/env bash
# Stop only model-service PIDs created by start_model_services.sh.
set -Eeuo pipefail
umask 027

readonly DATA_ROOT="/data/industrial-data-corps"
readonly PID_ARCHIVE="${DATA_ROOT}/status/model-service-pid-archive"
readonly PARSE_MODEL="nvidia/NVIDIA-Nemotron-Parse-v1.2"
readonly LLM_MODEL="nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
readonly EMBED_MODEL="nvidia/Nemotron-3-Embed-8B-BF16"

mkdir -p -- "${PID_ARCHIVE}"

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

archive_pid_file() {
  local pid_file="$1"
  local name="$2"
  local suffix
  [[ -f "${pid_file}" ]] || return 0
  suffix="$(date -u +'%Y%m%dT%H%M%SZ')-$$"
  mv -- "${pid_file}" "${PID_ARCHIVE}/${name}.${suffix}.pid"
}

stop_service() {
  local name="$1"
  local model="$2"
  local port="$3"
  local pid_file
  local pid=""
  local deadline

  case "${name}" in
    parse) pid_file="${DATA_ROOT}/status/nemotron-parse.pid" ;;
    llm) pid_file="${DATA_ROOT}/status/nemotron-llm.pid" ;;
    embedding) pid_file="${DATA_ROOT}/status/nemotron-embed.pid" ;;
    *) return 1 ;;
  esac

  if [[ ! -f "${pid_file}" ]]; then
    printf '%s has no managed PID file; nothing to stop\n' "${name}"
    return 0
  fi
  IFS= read -r pid <"${pid_file}" || true
  if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'refusing invalid PID file for %s: %s\n' "${name}" "${pid_file}" >&2
    return 1
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    printf '%s PID %s is already stopped; archiving stale PID file\n' "${name}" "${pid}"
    archive_pid_file "${pid_file}" "${name}"
    return 0
  fi
  if ! pid_matches_service "${pid}" "${model}" "${port}"; then
    printf 'refusing to signal PID %s: it does not match managed service %s\n' \
      "${pid}" "${name}" >&2
    return 1
  fi

  kill -TERM "${pid}"
  deadline=$((SECONDS + 45))
  while kill -0 "${pid}" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    if ! pid_matches_service "${pid}" "${model}" "${port}"; then
      printf 'PID %s changed identity while stopping %s; refusing further signals\n' \
        "${pid}" "${name}" >&2
      return 1
    fi
    printf '%s did not stop after 45 seconds; sending KILL to verified PID %s\n' \
      "${name}" "${pid}" >&2
    kill -KILL "${pid}"
    deadline=$((SECONDS + 10))
    while kill -0 "${pid}" 2>/dev/null && ((SECONDS < deadline)); do
      sleep 1
    done
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    printf 'failed to stop verified %s PID %s\n' "${name}" "${pid}" >&2
    return 1
  fi
  archive_pid_file "${pid_file}" "${name}"
  printf 'stopped %s PID %s\n' "${name}" "${pid}"
}

FAILURES=0
if ! stop_service parse "${PARSE_MODEL}" 8001; then
  FAILURES=$((FAILURES + 1))
fi
if ! stop_service llm "${LLM_MODEL}" 8003; then
  FAILURES=$((FAILURES + 1))
fi
if ! stop_service embedding "${EMBED_MODEL}" 8004; then
  FAILURES=$((FAILURES + 1))
fi

((FAILURES == 0))
