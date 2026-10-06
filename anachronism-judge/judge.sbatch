#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
# Qwen3.5's gated-delta-net attention JIT-compiles FlashInfer kernels: provide nvcc matching torch's CUDA 13.
module load cuda/13.2
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"
uv run --frozen python judge.py "$@"
