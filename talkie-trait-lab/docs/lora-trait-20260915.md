# First matched LoRA / TRAIT experiment

## Question and fixed design

Compare Vintage/Web atomic base checkpoints under the same low-rank SFT recipe,
separately on modern Tulu 3 and the packaged initial vintage SFT mixture. This
first experiment asks how quickly each TRAIT dimension changes with exposure.
It does not assume that low-rank and full-parameter SFT have equivalent effects.
The existing full-SFT endpoints used much more training data; they are reference
endpoints, not token-matched controls for this one-pass experiment.

User-approved design: assistant-loss-token matching, one vintage-data pass,
shared random Tulu rows across families, and full 16,000-choice TRAIT at baseline
and 1%, 2%, 5%, 10%, 20%, 40%, 60%, 80%, and 100% exposure, separately in bare
and chat formats. Report each trait and each trait/source-polarity combination;
these are high-trait selection rates, never an accuracy or global personality
score. All likelihood decisions use raw-answer conditional bits per UTF-8 byte.

Training is chat-format SFT in all four arms. “Bare/chat” refers to the two
TRAIT evaluation interfaces, not two additional training recipes.

## Data

Frozen artifact:
`/data/talkie-base-experiments/datasets/artifacts/lora-trait-matched-20260915-v1`.
Preparation job 8525 completed and the artifact includes row identities,
message records, exact model-specific input IDs/assistant masks, source hashes,
and a manifest. Both tokenizers use the canonical atomic chat template.

| Base tokenizer | SFT dataset | Rows | Assistant-loss tokens | Input tokens |
|---|---|---:|---:|---:|
| Vintage | Vintage | 25,938 | 4,617,250 | 5,111,755 |
| Vintage | Tulu | 9,441 | 4,739,785 | 6,660,568 |
| Web | Vintage | 25,938 | 4,732,607 | 5,207,790 |
| Web | Tulu | 9,441 | 4,606,774 | 6,438,040 |

The shared random Tulu prefix is chosen nearest the average normalized budget
under both tokenizers. Exact per-family matching is incompatible with simply
stopping the same uniform random prefix at each family's budget. The retained
mismatch is +2.654% for Vintage and -2.659% for Web, and must accompany comparisons.
Masks supervise every assistant turn, including assistant EOS. No examples are
truncated, packed together, or repeated. Small shuffled windows are length-sorted
for padding efficiency; minibatch boundaries/order can differ by tokenizer.

## Common recipe

- Rank 16, alpha 32, no dropout; all seven attention/MLP projections per layer.
- Original model parameters, atomic role embeddings, and logit gain stay frozen.
- 62,341,120 trainable parameters; FP32 adapters over bf16 frozen weights.
- AdamW, peak LR 1e-4, zero weight decay, 3% token warmup, linear decay,
  clipping norm 1; gradients normalized by actual supervised tokens.
- Up to 4,096 padded input tokens per microbatch; accumulate approximately
  16,384 assistant-loss tokens per update, flushing at evaluation milestones.
- One B200 per active arm, cu130 SDPA, activation checkpointing, deterministic
  PyTorch algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Full-vocabulary logits for evaluation, matching the standard Transformers
  scorer's answer tokenization, left padding, and BPB denominator. Training
  computes only supervised-token logits to reduce activation memory.

A read-only schedule audit confirmed full one-pass coverage and every requested
milestone: 266 / 271 / 271 / 267 updates for Vintage-vintage / Vintage-Tulu /
Web-vintage / Web-Tulu respectively. Actual checkpoint exposure, including
whole-minibatch overshoot, is recorded in each result. See
`/data/talkie-base-experiments/runs/lora-trait-20260915/update-plan.json`.

## Validation and failures retained

- Four unit tests pass: zero-adapter parity/frozen weights, exact adapter+AdamW
  resume, padding and token accounting, and optimized assistant loss/gradient
  equivalence to ordinary full-logit loss on a small Talkie model.
- Jobs 8526/8527 completed eight full-size updates in all four conditions, with
  roughly 31.1 GB peak allocated HBM and finite gradients/losses.
- Fresh-process gates 8528/8529 failed bitwise next-update replay with default
  GPU kernels. Deterministic reruns 8532/8533 passed for all four conditions:
  all 560 adapter tensors were bitwise equal across next-update replays. Fixed
  training-probe loss fell in all four; these are learning plumbing checks,
  not held-out quality measurements or a hyperparameter optimum claim.
- Initial production startup gates 8535/8536 rejected the optimized evaluation
  projection under deterministic cuBLAS before training began. Their dependent
  jobs 8537/8538 were cancelled. Full-logit evaluation replaced that optimization.
- Final-path deterministic smokes 8539/8541 passed on both families, including
  training and both TRAIT interfaces. Log-probability errors versus the standard
  Transformers backend were below 4e-7 on the parity probe. Scientific full-base
  rates are separately compared to the prior campaign; backend/batching numeric
  differences must not be silently called weight changes.

## Original synchronous production campaign (superseded)

| Base | SFT data | Job | Scheduling |
|---|---|---:|---|
| Vintage | Tulu | 8545 | First wave |
| Web | Tulu | 8546 | First wave |
| Vintage | Vintage | 8547 | After successful 8545 |
| Web | Vintage | 8548 | After successful 8546 |

Each GPU allocation requests one B200, 8 CPUs, 128G RAM, 8 hours, and 20 GB
predicted durable output. Evaluations are synchronous, so at most two model
train/eval jobs run simultaneously. All adapters and optimizer snapshots are
retained; no full-model copies or extra evaluator allocations are needed.

Root: `/data/talkie-base-experiments/runs/lora-trait-20260915/production`.
Each arm has `run.json`, append-only `training.jsonl`, `checkpoints/`, and
`trait/fraction-*/{bare.jsonl,chat.jsonl,summary.json}`. A zero-GPU observer
refreshes `trajectories.csv`, `findings.md`, `baseline-comparison.json`, and
`campaign-status.json` as evaluations complete. Final completion requires
SLURM success plus an independent audit of all 80 interface evaluations
(1,280,000 paired-choice rows), exact hashes, unique example IDs, and expected
trait/polarity denominators. Running/pending jobs are not completed results.

Code and operating instructions:
`scripts/experiments/lora_trait_20260915/README.md`.

## Asynchronous campaign migration

The user removed the two-evaluator rule and requested nonblocking evaluation.
A tested adapter queue now supports atomic immutable checkpoint publication,
idempotent enqueue, one-owner claims, owner-checked recovery, retained attempt
outputs, exact snapshot checksums, and independent training/evaluation status.
The producer resumes from its last checkpoint and reconciles all reached
milestones before proceeding, covering a crash between snapshot save and enqueue.

GPU producer smoke 8568 finished training while its evaluation remained pending.
Separate evaluator 8576 then completed the request. Its 32 smoke-row decisions
and all log probabilities matched the synchronous deterministic smoke exactly
in both bare and chat. Nine unit tests cover adapters, loss accounting, exact
small-model resume, queue races/retries, and fixed scenario subset sampling.

The proposed full-data pool is four training GPUs plus eight persistent evaluator
GPUs (four per model family). Evaluation snapshots only contain adapters and
optimizer state; there are no merged model exports. Full training should finish
in roughly 25 minutes, with evaluation draining independently. The expected full
TRAIT turnaround with eight workers is roughly 50–65 minutes, excluding queue
waits and depending on shared-node throughput. A fixed scenario subset is also
implemented, but full TRAIT remains the default previously approved design.

The user explicitly approved migration after the initial automatic-review
rejection. Old jobs 8545–8548 and observer 8550 were cancelled; migration job
8604 completed successfully (exit 0:0). Both Tulu arms resume from step 14
(about 5% of the assistant-token budget). Their baseline, 1%, and 2% full
bare/chat evaluations were hash-verified and reused. All original artifacts,
including incomplete evaluations, remain in the original production root.

The new root is
`/data/talkie-base-experiments/runs/lora-trait-20260915/production-async`.
`migration.json` records preserved artifacts; `training-jobs.json`,
`worker-jobs.json`, and `observer-job.json` record the new allocations.
Full TRAIT remains enabled at every checkpoint (`intermediate_subset: 0`).

### Active asynchronous allocations

| Base | SFT data | Training job |
|---|---|---:|
| Vintage | Tulu | 8605 |
| Vintage | Vintage | 8612 |
| Web | Tulu | 8613 |
| Web | Vintage | 8614 |

Eight one-B200 evaluator jobs: 8615–8622 (four per family).
Zero-GPU observer: 8623. Each producer and evaluator requests 8 CPUs,
128G RAM and three hours. All four producers have recorded finite-loss
training updates while separate evaluator jobs score full TRAIT snapshots.
The observer reports training and evaluation completion separately and audits
all 80 interface outputs before marking the campaign complete.

## Repeated adapter swap audit (job 8624)

Job 8624 completed successfully (exit 0:0) in 72 seconds. For each family,
a single resident base loaded a real Tulu checkpoint, a real vintage-data
checkpoint, the same Tulu checkpoint again, and zero adapters. All 560 GPU
adapter tensors matched the requested checkpoint bit-for-bit after each swap.
On 32 balanced TRAIT comparisons per interface, repeated Tulu scores were
exactly identical and zero-adapter scores exactly matched an untouched fresh
base. The different trained adapters produced different scores. These checks
cover both Vintage/Web and bare/chat; they are systems probes, not trait-rate
estimates. Production asynchronous baseline result files also matched preserved
synchronous baselines byte-for-byte for all 16,000 rows per interface/family.

Measured checkpoint hash + CPU load: 0.88–1.10 seconds; tensor validation + GPU
copy: 0.089–0.111 seconds. The worker retains the base and copies 249 MB of
adapter tensors. It currently reads a 749 MB checkpoint including unused CPU
optimizer state, a modest avoidable I/O cost, but performs no full-model reload
or adapter merge. Full bare+chat evaluations measured about 580–584 seconds.

Audit script: `scripts/experiments/lora_trait_20260915/audit_swaps.py`.
Detailed paths, hashes, timings, and exact comparisons:
`/data/talkie-base-experiments/runs/lora-trait-20260915/swap-audit/audit.json`.
Production code and running allocations were unchanged by this audit.
