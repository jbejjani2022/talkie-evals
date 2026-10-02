# Talkie TRAIT lab

Standalone reproduction kit for Vintage/Web base models, assistant-only LoRA
SFT on Vintage SFT / Tulu 3 / 18 Tulu sources, and TRAIT few-shot prompting.
No parent checkout, paid API, vLLM, TRL, PEFT, or activation steering is required.
Run commands from this checkout so the bundled data and configs are available.

## Start here

Use Linux, an NVIDIA GPU with BF16 support, and [uv](https://docs.astral.sh/uv/).
The original recipe used one B200, PyTorch 2.9.0+cu130 and SDPA, with roughly
31 GB peak allocated GPU memory in training smokes. That is a measured B200
result, not a guarantee for another GPU. A 48 GB or larger GPU is a sensible
starting point; reduce `--batch-tokens` for a smaller card and record the change.
Single-device training is implemented. Quantization and distributed training
would be new experimental recipes.

```bash
uv sync --frozen
uv run trait-lab verify-bundle
uv run pytest
uv run trait-lab list-data
```

The lock pins the cu130 stack used for this handoff. A recent NVIDIA driver is
required. To use another supported PyTorch wheel, edit the index in
`pyproject.toml`, run `uv lock` and `uv sync`, and report the resulting environment
as a numerical change. Do not silently combine baseline results from different
precision/backends. CPU tests do not load a 13B model or initialize CUDA.

Choose a durable artifact root outside the code checkout. On our cluster it
must be under `/data`; on a personal machine choose your own large disk:

```bash
export TRAIT_ARTIFACT_ROOT=/data/your-user/talkie-trait
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

All commands below are ordinary commands for a dedicated GPU workstation.
**On a SLURM cluster submit GPU commands and substantial preparation with
`sbatch`; never run inference/training on a login node.** See the batch template
below. This code refuses GPU commands on a SLURM host without an allocation.

## Download and prepare the bases

```bash
uv run trait-lab prepare-models --family vintage
uv run trait-lab prepare-models --family web
```

These commands download the official original checkpoints and vocabularies,
convert all floating tensors to BF16 safetensors, copy the custom Transformers
implementation, and append the **exact** three frozen experimental role-token
embedding/output rows. They preserve Talkie's learned output gain. Final models
are `$TRAIT_ARTIFACT_ROOT/models/{vintage,web}-atomic` and load with
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")` and
`AutoTokenizer.from_pretrained(path, trust_remote_code=True)`.

| Family | Upstream repository | Pinned revision | Checkpoint |
|---|---|---|---|
| Vintage | [talkie-lm/talkie-1930-13b-base](https://huggingface.co/talkie-lm/talkie-1930-13b-base) | `b7c97680791f7fca4262c3c80b36ff7d666faab0` | `final.ckpt` |
| Web | [talkie-lm/talkie-web-13b-base](https://huggingface.co/talkie-lm/talkie-web-13b-base) | `1e5b771c9d38d44f54d35e722c5c0d73da418dd8` | `base.ckpt` |

Budget at least 200 GB of free disk for both raw, intermediate converted, and
atomic models, plus environment/cache/results space. Use 128 GB CPU RAM for
conversion. Downloads are resumable through Hugging Face; a partially converted
atomic model is retained as `.partial` and is not declared complete.
`--source-dir /path/to/original` reuses a checkpoint/vocab directory for one
family. The model payloads themselves are not bundled.

## Data and exact experiment selection

`data/vintage-sft-original.parquet` contains the original 25,938 conversations.
`data/vintage-sft.parquet` contains their normalized messages in the frozen
training order. Both are bundled directly. This is the **initial** Vintage SFT
mixture used in the TRAIT LoRA study, not the later expanded/full-SFT mixture.
The original request/response/category fields and row identifiers are retained.

```bash
uv run trait-lab download-data all
uv run trait-lab prepare-data --family both --dataset vintage --dataset tulu
# One source, or every available source:
uv run trait-lab prepare-data --family vintage --dataset personahub-if
uv run trait-lab prepare-data --family both --dataset all
```

Tulu comes from [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture),
revision `b14afda60f1bbebe55d5d2fa1e4df5042f97f8be`. TRAIT comes from
[pull-ups/TRAIT](https://github.com/pull-ups/TRAIT), revision
`b623cb57dc2a263a7d0f2b6504385a25e2224c77` (the official source is GitHub).
The downloader verifies the exact TRAIT JSON hash and regenerates the canonical
16,000 paired-choice rows, checking them against the bundled materialization.

Preparation retrieves the **exact frozen row IDs and order** from downloaded
Tulu and verifies every message, input token sequence and assistant mask. It
does not replace our selections with a fresh approximation. Dataset files are
under `$TRAIT_ARTIFACT_ROOT/prepared`; `manifest.json` records hashes, exposure
and local model paths. Both tokenizers must reproduce their original masks.
`--model /existing/atomic/model` is available for one family.

The four-arm matched study used all Vintage rows once and the same 9,441
randomly selected Tulu rows for both families. Its known matching residual is
retained:

| Tokenizer | Vintage assistant-loss tokens | Tulu assistant-loss tokens | Tulu mismatch |
|---|---:|---:|---:|
| Vintage | 4,617,250 | 4,739,785 | +2.654% |
| Web | 4,732,607 | 4,606,774 | −2.659% |

The 18 individual-source runs each cap exposure at **4,617,250 assistant-loss
tokens**, choose the longest whole-example prefix within the cap, and use small
sources once without repetition. Selection order is source-seeded; family
tokenizers produce different row counts/exposure. Hard-coded repeated examples
are excluded. `list-data`, `data/selections.json.gz`, and `data/adapter-inventory.json`
give all 40 arms, actual counts, terminal steps and source hashes.

For auditing/rebuilding the full eligible pool, the original portable scripts
are included: `scripts/filter_joint_sft_context.py` retains rows fitting **both**
tokenizers at 4096; `scripts/prepare_sft_calibration_data.py` holds out 1,024
conversations with seed 20260828. Expected counts: 939,343 raw → 912,271 eligible
→ 911,247 training. These steps are unnecessary for reproducing the frozen
selections. `data/provenance/` retains the original filtering/split manifests.

For a new dataset or a new sampling/budget experiment use `prepare-custom`
(below), write to a new run directory, and compare it separately from exact
reproduction. Do not edit the frozen selections to conceal changed exposure.

## Train LoRA SFT

First run a short systems check:

```bash
uv run trait-lab train --family vintage --dataset vintage \
  --data "$TRAIT_ARTIFACT_ROOT/prepared" \
  --output "$TRAIT_ARTIFACT_ROOT/runs/vintage-vintage-smoke" \
  --smoke-steps 2 --trait-limit 1
```

Then reproduce an arm (repeat for both families and `vintage`/`tulu`/source slugs):

```bash
uv run trait-lab train --family vintage --dataset tulu \
  --data "$TRAIT_ARTIFACT_ROOT/prepared" \
  --output "$TRAIT_ARTIFACT_ROOT/runs/vintage-tulu"
```

Defaults preserve the validated custom LoRA recipe: rank 16, alpha 32, no
dropout; all seven attention/MLP projections per layer; FP32 adapters over
frozen BF16 weights; frozen embeddings and logit gain; AdamW LR 1e-4, zero
weight decay, 3% assistant-token warmup and linear decay, clipping 1; at most
4096 padded tokens per microbatch; approximately 16,384 assistant-loss tokens
per update, with gradients divided by the **actual** supervised-token count.
Supervise every assistant turn including EOS. No truncation, packing, or repeats.
Seed 20260915, activation checkpointing, deterministic PyTorch algorithms.

The custom LoRA parameter/module names match the existing experiments. These
are **not PEFT checkpoints**; load them with this repository's loader.
There are 62,341,120 trainable parameters in the 13B models.

TRAIT runs synchronously at exposure fractions 0, 1, 2, 5, 10, 20, 40, 60, 80,
100%, in bare and chat separately. This favors a simple portable runner;
original production used asynchronous evaluators with the same frozen recipe.
`--endpoint-only` evaluates only baseline and endpoint while still preserving
the nine adapter milestones. Changing evaluation cadence is not a new token
budget. Checkpoint metadata records actual minibatch overshoot.

Each run writes `run.json`, `training.jsonl`, restartable `checkpoints/`, raw
TRAIT JSONL/summary files, and `terminal-adapter/adapter.safetensors` plus
`adapter.json`. Resume only your own intact training checkpoint:

```bash
uv run trait-lab train --family vintage --dataset tulu \
  --data "$TRAIT_ARTIFACT_ROOT/prepared" \
  --output "$TRAIT_ARTIFACT_ROOT/runs/vintage-tulu" \
  --resume "$TRAIT_ARTIFACT_ROOT/runs/vintage-tulu/checkpoints/step-000014"
```

Use the exact prior training flags and unchanged data/model paths. Terminal
adapters in the optional archive have no optimizer/RNG state and cannot resume
the original training trajectory.

## TRAIT evaluation

```bash
uv run trait-lab eval --family vintage --interface bare \
  --output "$TRAIT_ARTIFACT_ROOT/evals/vintage-base-bare"
uv run trait-lab eval --family vintage --interface chat \
  --adapter "$TRAIT_ARTIFACT_ROOT/runs/vintage-tulu/terminal-adapter" \
  --output "$TRAIT_ARTIFACT_ROOT/evals/vintage-tulu-chat"
```

The scorer tokenizes the prefix and raw answer continuation separately, with
one leading space on the answer, and scores **only** answer tokens. For each
candidate, `BPB = −log_probability / (ln(2) × UTF-8 bytes)`; select the minimum.
The leading space is included in the byte denominator. Exact ties select the
first stored candidate. Never substitute A/B label likelihoods or normalize by
token count. Choice order is frozen by the original seeded permutation.

Report each of the eight traits and each source polarity; a high-trait selection
rate is a behavioral readout, not accuracy or personality quality. Keep raw-base
→ raw-SFT and chat-base → chat-SFT comparisons separate. TRAIT has 8,000
scenarios, two answer pairs per scenario, 2,000 pairs per trait, and 1,000 per
trait/polarity. `--smoke-per-group 1` is a plumbing check, not a scientific rate.

Outputs preserve exact prompts, answer strings, IDs, prefix/answer token IDs,
byte denominators, log probabilities, BPB, high/low choice and signed BPB margin.
The historical reference JSONLs and trajectory CSVs are under `data/references/`.
Compare numeric margins and choices; BF16 batching/backends can move near ties.

```bash
uv run trait-lab contrast --left /path/to/sft-eval --right /path/to/base-eval \
  --output "$TRAIT_ARTIFACT_ROOT/analysis/sft-minus-base.csv"
```

This checks exact scenario/answer alignment and bootstraps matched scenario
groups within trait/source-polarity strata, preserving both answer pairs.
Intervals are unadjusted marginal percentile intervals, conditional on fixed
models/prompts/scenarios; they do not measure training-seed uncertainty.

## Few-shot prompt steering

The bundled five seeded banks per target trait contain nested demonstrations.
N=0, 1, 2, 4, 8, 16, 32; paired high/low arms share identical questions/order.
Complementary balanced-answer controls exist at even N; N=8 has reversed-order
controls. Demonstrations are exact bare Question/Answer examples with raw answer
strings. No trait descriptions or personality instructions are added.

Demonstrations use discovery scenario groups; evaluation uses the frozen 3,200
validation pairs. Known near-duplicate groups are separated. The final test
remains reserved. The split file was originally created for broader mechanism
work but contains no vectors; no activation-steering implementation is included.

Start with baseline and one pair:

```bash
uv run trait-lab fewshot-check --family vintage \
  --condition baseline --condition Openness-b0-n8-high --condition Openness-b0-n8-low
uv run trait-lab fewshot --family vintage \
  --condition baseline --condition Openness-b0-n8-high --condition Openness-b0-n8-low \
  --output "$TRAIT_ARTIFACT_ROOT/fewshot/vintage-small"
```

Add `--smoke-per-group 1` for a short GPU check. Omit `--condition` for all **961
conditions per model**: 480 pole conditions, 400 balanced controls, 80 order
reversals and one baseline. Both bases together require 6,150,400 pair scores;
the full campaign is expensive. The portable runner is sequential per resident
model. Independent condition shards can use distinct output directories on
separate GPUs. Keep at least one baseline in the comparison panel.

The runner audits full selected-condition lengths before inference, requiring
an eight-token reserve within 4096. Historical maximum candidates were 4020
Vintage / 3911 Web. N=64/128 does not fit the tested banks and is not supported
by the frozen configuration. Increase N only with new banks and length audits.
`--adapter` applies the same prompting to a terminal SFT adapter. Exact completed
conditions can be resumed; interrupted partial attempts remain preserved and
require a new output directory for retries.

```bash
uv run trait-lab contrast \
  --left "$TRAIT_ARTIFACT_ROOT/fewshot/vintage-small/Openness-b0-n8-high" \
  --right "$TRAIT_ARTIFACT_ROOT/fewshot/vintage-small/Openness-b0-n8-low" \
  --output "$TRAIT_ARTIFACT_ROOT/analysis/openness-bank0-high-minus-low.csv"
```

Also compare each pole to baseline, balanced controls to baseline, and reversed
to original order. Report bank-specific variability and cross-trait effects.
A two-pole high-minus-low span is different from a single SFT-minus-base delta.
Do not pool demonstration banks into independent scenario samples.

## New SFT experiments

Create a Parquet file with `id`, `source`, `messages` (list of
`{"role":"user","content":"..."}` / assistant records), or equivalent JSONL:

```bash
uv run trait-lab prepare-custom --family vintage --name my-mix \
  --messages /path/to/my-mix.parquet --loss-token-cap 4617250 --seed 123
uv run trait-lab train --family vintage --dataset my-mix \
  --data "$TRAIT_ARTIFACT_ROOT/prepared" \
  --output "$TRAIT_ARTIFACT_ROOT/runs/vintage-my-mix"
```

This preserves full conversations, enforces 4096 tokens, shuffles with the
specified seed, and selects a whole-row prefix within an optional assistant-loss
budget. Missing/invalid assistant supervision or overlength rows fail explicitly.
Use new dataset names and output roots for every changed design. For a matched
new mix, explicitly freeze shared row IDs/order across tokenizers and report
both actual exposures; independent family-specific selection is a different
design. Include an unrelated language/task-quality control if claiming that a
TRAIT shift reflects personality rather than general capability loss.

## Optional terminal adapters

Unzip `talkie-trait-terminal-adapters.zip` into this checkout. It adds
`adapters/{vintage,web}-{vintage,tulu,<source>}/` for all 40 terminal states.
Each is about 249 MB; the full archive is about 10 GB. Then, for example:

```bash
uv run trait-lab eval --family web --adapter adapters/web-personahub-if \
  --interface bare --output "$TRAIT_ARTIFACT_ROOT/evals/web-if-bare"
```

The loader verifies family, base revision, tokenizer, exact atomic interface
rows, tensor names/shapes/finiteness, and adapter checksum. The inventory records
original/compacted checkpoint hashes and final token exposure. The small main
zip is sufficient to train everything from scratch.

## SLURM example

Edit `scripts/run.sbatch` for your site's partition/resources/log paths and
export `TRAIT_ARTIFACT_ROOT` before submitting. Use your site's submission guard
where required; the original cluster's Talkie guard is deliberately not a
dependency of this standalone repository.

```bash
# Generic SLURM example; on the Talkie cluster use talkie-slurm-submit instead.
sbatch --job-name=trait-smoke --partition=YOUR_PARTITION \
  --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 \
  --cpus-per-task=8 --mem=128G --time=00:30:00 \
  --output=/data/your-user/logs/%x-%j.out --error=/data/your-user/logs/%x-%j.err \
  scripts/run.sbatch train --family vintage --dataset vintage \
  --data "$TRAIT_ARTIFACT_ROOT/prepared" \
  --output "$TRAIT_ARTIFACT_ROOT/runs/smoke" --smoke-steps 2 --trait-limit 1
```

For CPU downloads/conversion/data preparation request `--gpus-per-node=0`,
128G RAM and sufficient time. Check `sacct` and completion artifacts; a vanished
process is not completion evidence. Low/preemptible jobs need exact resume flags.

## Contents, attribution, and limits

`src/talkie_base_experiments/` vendors the small necessary model/tokenizer,
conversion and custom LoRA code. `src/trait_lab/` provides portable commands;
`data/manifest.json` hashes every bundled data/provenance file. `VALIDATION.md`
records checks performed on this handoff and any untested paths. Historical
notes in `docs/` provide scientific context; their `/data/...` links are original
provenance, not dependencies or current endpoints.

See `THIRD_PARTY.md` and bundled upstream data cards for attribution/terms.
The Vintage dataset is supplied for this collaborator handoff. This repository
does not assert a new blanket redistribution license for inherited project code
or data. This package covers original-name TRAIT, the 40 requested training arms
and few-shot validation. Name/anachronism interventions, paraphrased Vintage
SFT, midtraining, full-parameter SFT, and activation steering are separate
experiments and are not claimed as reproduced here.
