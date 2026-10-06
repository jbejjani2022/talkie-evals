#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${TRAIT_ARTIFACT_ROOT:?Set a durable artifact root}"
cd "${SLURM_SUBMIT_DIR:?}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
uv run --frozen behavior-evals "$@"
