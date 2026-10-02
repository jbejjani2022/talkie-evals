# talkie-evals

Cluster path: /home/jbejjani/projects/aip-dkd/jbejjani/talkie/talkie-evals
Local path:   ~/vector/talkie-evals
Remote: github.com/jbejjani2022/talkie-evals (PUBLIC; the cluster pulls over HTTPS), branch main

## Layout
- talkie-trait-lab/: self-contained uv project (Talkie Vintage/Web 13B LoRA SFT + TRAIT evaluation).
  Read talkie-trait-lab/README.md before running anything. CLI: `trait-lab` (src/trait_lab/cli.py).
- talkie-trait-lab/data/ is committed reference data, intentionally kept in git. Don't add more large files.

## Environment (on the cluster, in talkie-trait-lab/)
- `uv sync --frozen`. EXCEPTION to the wheelhouse rule: the lock pins torch 2.9.0+cu130 from the PyTorch index
  for exact reproduction. Don't relock or swap wheels without the user's OK. If it changes, that's a numerical change to report.
- Before long runs, verify the cu130 build works with Killarney's driver via a short GPU smoke job.
- CPU tests: `uv run pytest`. They import torch, so run them via srun, not on the login node.

## Storage (TRAIT_ARTIFACT_ROOT)
- Working root (models ~200 GB, prepared data, runs + checkpoints, evals):
  TRAIT_ARTIFACT_ROOT=/scratch/jbejjani/talkie/trait (purged if not accessed for 60 days!)
- Durable copies: /home/jbejjani/projects/aip-dkd/jbejjani/talkie/artifacts/. After a run, copy
  terminal-adapter/, run.json, training.jsonl, and eval summaries there. Existing sibling: talkie/adapters/.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is required for determinism (scripts/run.sbatch sets it).

## Launching jobs
- Entry: talkie-trait-lab/scripts/run.sbatch. It has no #SBATCH lines, so pass resources on the command line.
  From talkie-trait-lab/ on the cluster:

      mkdir -p logs
      sbatch --account=aip-dkd --gres=gpu:l40s:1 -c 8 --mem=96G --time=3:00:00 \
        --output=logs/%x-%j.out --job-name=<name> \
        --export=ALL,TRAIT_ARTIFACT_ROOT=/scratch/jbejjani/talkie/trait \
        scripts/run.sbatch <trait-lab subcommand and args>

- Model download/conversion (`prepare-models`) needs ~128 GB CPU RAM: CPU-only job with --mem=128G, no gres.
- The original recipe peaked at ~31 GB GPU memory (B200), so it fits one 48 GB L40S. Use H100 only if needed.
- Start with smokes, e.g. `train ... --smoke-steps 2 --trait-limit 1`.
- Logs: talkie-trait-lab/logs/<name>-<jobid>.out. Outputs: $TRAIT_ARTIFACT_ROOT/{models,prepared,runs,evals}.

## Project guidelines
- Never edit the frozen selections/data to change exposure. New experiments go in new run directories (see README).
- Report results per trait and per polarity, and keep raw-base and adapter results separate.
