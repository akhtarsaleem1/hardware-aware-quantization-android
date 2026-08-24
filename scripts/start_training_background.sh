#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 LOG_PATH TRAINING_ARGUMENTS..." >&2
  exit 2
fi

LOG_PATH="$1"
shift
mkdir -p "$(dirname "${LOG_PATH}")"
nohup bash scripts/run_wsl_gpu_training.sh "$@" >"${LOG_PATH}" 2>&1 </dev/null &
JOB_PID=$!
printf '%s\n' "${JOB_PID}" >"${LOG_PATH}.pid"
printf '%s\n' "${JOB_PID}"
