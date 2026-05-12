#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

UV="${UV:-uv}"
RUN_SYNC="${RUN_SYNC:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_EXAMPLE="${RUN_EXAMPLE:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
DUMP_MAX_MLIR="${DUMP_MAX_MLIR:-}"

BATCH_SIZE="${BATCH_SIZE:-128}"
INPUT_DIM="${INPUT_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
OUTPUT_DIM="${OUTPUT_DIM:-16}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
STEPS="${STEPS:-5}"
REPEATS="${REPEATS:-2}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-max-autotune}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "error: nvidia-smi not found" >&2
    exit 1
fi

if [[ -z "${MODULAR_NVPTX_COMPILER_PATH:-}" ]]; then
    if command -v ptxas >/dev/null 2>&1; then
        export MODULAR_NVPTX_COMPILER_PATH="$(command -v ptxas)"
    elif [[ -x /usr/local/cuda/bin/ptxas ]]; then
        export MODULAR_NVPTX_COMPILER_PATH="/usr/local/cuda/bin/ptxas"
    else
        echo "error: ptxas not found; set MODULAR_NVPTX_COMPILER_PATH" >&2
        exit 1
    fi
fi

echo "== GPU =="
nvidia-smi
echo

echo "== Toolchain =="
echo "uv: $("$UV" --version)"
echo "ptxas: ${MODULAR_NVPTX_COMPILER_PATH}"
echo

if [[ "${RUN_SYNC}" == "1" ]]; then
    echo "== uv sync =="
    "$UV" sync --group dev --extra benchmark --frozen
    echo
fi

echo "== Python imports =="
"$UV" run --no-sync python - <<'PY'
import sys

import max
import max_training
import torch

print("python", sys.version.split()[0])
print("max_training", max_training.__file__)
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("error: torch cannot see CUDA")
print("torch.cuda.device", torch.cuda.get_device_name(0))
print("torch.cuda.capability", torch.cuda.get_device_capability(0))
PY
echo

if [[ "${RUN_TESTS}" == "1" ]]; then
    echo "== pytest =="
    "$UV" run --no-sync pytest
    echo
fi

if [[ "${RUN_EXAMPLE}" == "1" ]]; then
    echo "== train_linear =="
    "$UV" run --no-sync python examples/train_linear.py
    echo
fi

if [[ -n "${DUMP_MAX_MLIR}" ]]; then
    echo "== MAX graph stats =="
    "$UV" run --no-sync python benchmarks/pytorch_compile.py \
        --device cuda \
        --batch-size "${BATCH_SIZE}" \
        --input-dim "${INPUT_DIM}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --output-dim "${OUTPUT_DIM}" \
        --max-graph-stats \
        --dump-max-mlir "${DUMP_MAX_MLIR}"
    echo
fi

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
    echo "== CUDA benchmark =="
    "$UV" run --no-sync python benchmarks/pytorch_compile.py \
        --device cuda \
        --batch-size "${BATCH_SIZE}" \
        --input-dim "${INPUT_DIM}" \
        --hidden-dim "${HIDDEN_DIM}" \
        --output-dim "${OUTPUT_DIM}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --steps "${STEPS}" \
        --repeats "${REPEATS}" \
        --torch-compile-mode "${TORCH_COMPILE_MODE}"
fi
