#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch}"
cd "${SLURM_SUBMIT_DIR:?}"
uv run --frozen python judge.py "$@"
