#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESEARCH_VENV="${RESEARCH_VENV:-/home/ask/venvs/hardware-aware-quantization}"
export DATASET_PATH="${DATASET_PATH:-/home/ask/datasets/DeepWeeds}"

PYTHON_BIN="${RESEARCH_VENV}/bin/python"
NVIDIA_ROOT="${RESEARCH_VENV}/lib/python3.13/site-packages/nvidia"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing verified Python environment: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -d "${NVIDIA_ROOT}" ]]; then
  echo "Missing NVIDIA wheel libraries: ${NVIDIA_ROOT}" >&2
  exit 2
fi

CUDA_LIBRARY_PATH="$(find "${NVIDIA_ROOT}" -mindepth 2 -maxdepth 2 -type d -name lib -print | sort | paste -sd: -)"
if [[ -z "${CUDA_LIBRARY_PATH}" ]]; then
  echo "No CUDA library directories found under ${NVIDIA_ROOT}" >&2
  exit 2
fi

export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" scripts/train_models.py "$@"
