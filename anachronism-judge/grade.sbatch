#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
# Same toolkit requirement as judge.sbatch (FlashInfer JIT kernels for Qwen3.5).
module load cuda/13.2
export CUDA_HOME="${CUDA_HOME:-$EBROOTCUDA}"
uv run --frozen python grade.py "$@"
