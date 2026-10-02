# Vintage-base Tulu 3 source ablations, 2026-09-18

## Current decision (2026-09-18)

The user superseded the grouping discussion: **run each of the 18 remaining datasets separately**, taking one pass over available eligible data, **capped at 4,617,250 Vintage assistant-loss tokens**. The seven undersized datasets remain included at their available exposure; do not repeat them to reach the cap. Plot learning curves against **optimizer step**, retaining actual cumulative tokens in the underlying data because steps are not exactly token matched. Before preparing these arms, the user requested a six-line Web-similarity trajectory figure for the completed campaign; that figure uses original names/all scenarios/bare only. The figure request was completed. The user subsequently authorized setup/preparation; hard-coded examples are now excluded.

## Requested study and status

The user requested Vintage-base post-training separately on each Tulu 3 subset/category, with assistant-loss-token exposure matched to the existing Vintage SFT condition and the same TRAIT evaluation battery. Treat each distinct dataset `source` value as a separate arm; broader semantic labels such as math or safety group several sources and are not interchangeable with those arms.

Initial action: exact training-pool token census (including the subsequently excluded hard-coded source), **zero-GPU SLURM job 10231**, excluding worker-12. No new training or evaluation jobs have been launched. The census script is `scripts/experiments/trait_tulu_categories_20260918/census.py`, with an adjacent batch wrapper. Durable census artifacts are under `/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/census`.

## Token matching and eligibility

Target **4,617,250 assistant-loss tokens under the Vintage tokenizer**. This is the original Vintage-base/Vintage-SFT one-pass budget, not the 4,739,785 tokens in the approximately matched original mixed-Tulu arm and not total input/context tokens. Since the new study is Vintage-only, matching need not compromise between two tokenizer budgets.

Use the existing `tulu3-sft-atomic-4096-joint-calibration` training shards: 911,247 eligible rows after joint Vintage/Web 4,096-token filtering and a fixed 1,024-row validation holdout. Count with the canonical atomic chat template and shifted assistant mask, including all assistant turns and EOS. Do not truncate examples or leak validation rows.

For future arm preparation, use a fixed random permutation within each source. For sufficiently large sources, select the longest whole-conversation prefix at or below the 4,617,250-token cap; use all eligible rows for smaller sources. Record actual totals and residual differences. Sample without adding repetitions; native duplication in the upstream mixture remains a separate property. This implements the latest capped-budget decision; undersized arms are intentionally shorter, not token-matched endpoints.

The original mixed-Tulu control has 2.654% greater exposure than the new target; retain that limitation in comparisons, or prepare an additional mixture control at the exact new target if required by the eventual design.

## Planned training/evaluation controls

Continue the established LoRA recipe (rank 16, alpha 32, learning rate 1e-4, assistant-token normalization) and token-fraction checkpoints. Reuse the existing Vintage and Web step-0 reference evaluations. Evaluate original names, fixed James/Evelyn, and the fixed shared-era name pool; derive all-scenario and historically plausible subsets post hoc. Preserve bare/chat interfaces separately, per-trait and source-polarity counts, raw-answer conditional BPB scoring, and validated complete-batch reuse for unchanged scenarios.

Use per-trait results plus the user-requested endpoint Web-profile distance summary from [the endpoint note](trait-followup-endpoints-20260918.md). Report category comparisons as exploratory and account for multiplicity if making simultaneous significance claims across many source arms. Single runs would still leave training-seed and training-sample uncertainty unmeasured.

## Census results

Exact census completed successfully in **job 10231**, exit **0:0**, elapsed **2m10s**, on worker-5 with **zero GPUs**. All six input shard hashes matched the frozen split manifest. The counter reproduced exact token IDs/masks on 64 previously frozen training examples. Every counted row has positive loss tokens and fits 4,096 tokens; total eligible rows reconcile to 911,247.

**11 sources can reach the target by subsampling; 8 cannot.** The table counts upstream mixture rows, not deduplicated conversations. No extra repetitions were added.

| Exact source | Eligible rows | Assistant-loss tokens | Target coverage | Enough? |
|---|---:|---:|---:|:---:|
| `ai2-adapt-dev/coconot_converted` | 10,972 | 847,113 | 18.35% | No |
| `ai2-adapt-dev/evol_codealpaca_heval_decontaminated` | 106,942 | 47,335,556 | 1025.19% | Yes |
| `ai2-adapt-dev/flan_v2_converted` | 89,744 | 3,123,074 | 67.64% | No |
| `ai2-adapt-dev/no_robots_converted` | 9,489 | 1,816,100 | 39.33% | No |
| `ai2-adapt-dev/numinamath_tir_math_decontaminated` | 64,222 | 46,831,834 | 1014.28% | Yes |
| `ai2-adapt-dev/oasst1_converted` | 7,068 | 3,180,183 | 68.88% | No |
| `ai2-adapt-dev/personahub_code_v2_34999` | 34,946 | 6,084,515 | 131.78% | Yes |
| `ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980` | 29,906 | 10,348,199 | 224.12% | Yes |
| `ai2-adapt-dev/personahub_math_v5_regen_149960` | 149,561 | 169,451,962 | 3669.98% | Yes |
| `ai2-adapt-dev/tulu_hard_coded_repeated_10` | 240 | 19,200 | 0.42% | No |
| `ai2-adapt-dev/tulu_v3.9_aya_100k` | 96,667 | 34,548,041 | 748.24% | Yes |
| `ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k` | 49,943 | 7,583,668 | 164.25% | Yes |
| `ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k` | 19,981 | 17,535,491 | 379.78% | Yes |
| `ai2-adapt-dev/tulu_v3.9_sciriff_10k` | 9,985 | 1,222,976 | 26.49% | No |
| `ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k` | 49,946 | 4,480,471 | 97.04% | No |
| `ai2-adapt-dev/tulu_v3.9_table_gpt_5k` | 4,994 | 418,029 | 9.05% | No |
| `ai2-adapt-dev/tulu_v3.9_wildchat_100k` | 76,786 | 73,997,270 | 1602.63% | Yes |
| `ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k` | 49,940 | 13,313,492 | 288.34% | Yes |
| `allenai/tulu-3-sft-personas-math-grade` | 49,915 | 17,422,957 | 377.34% | Yes |

WildGuardMix is the nearest shortfall: **136,779 tokens (2.9623%)** below the cap. Under the latest decision, use its entire eligible source, just as for the other six included smaller sources.

The final design includes 18 source arms: 11 are subsampled near the cap and seven end after their available data. The hard-coded source is excluded. Earlier strict equal-budget eligibility is retained in the census table only as a size diagnostic.

Artifacts: [categories.csv](/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/census/categories.csv), [summary.json](/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/census/summary.json). The 24 hashed per-row count parquet files retain source, source shard/row index, example ID, loss-token count, and input-token count, allowing future seeded sampling without retokenizing the complete pool. They are counts only, not prepared training arms.


## Prepared 18-arm setup

Preparation job **10232** completed on worker-5, exit **0:0**, in **2m01s** with zero GPUs. The hard-coded examples are excluded. All 18 arms have materialized message records and canonical tokenized training parquet under `/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/data`. Frozen `manifest.json` records source IDs, per-source deterministic random seeds, row/input/loss counts, hashes, and simulated optimizer-step schedules. `code-manifest.json` separately pins code and environment lockfiles.

Selection uses a source-specific SHA-256-derived seed from `20260915:<source>` and a uniform row permutation. Larger sources take the longest whole-conversation prefix at or below the cap; smaller sources use their complete eligible pool. The input scope is the existing joint context-filtered training split, with validation held out. No extra repetitions or truncation are introduced. Actual prepared exposure is listed below.

Training targets 16,384 supervised tokens per optimizer update, finishing each minibatch and flushing at evaluation milestones/final batch. Learning-rate warmup/decay continues to be defined as fractions of each arm's own available token budget, as in the earlier recipe. Consequently, short arms have compressed learning-rate schedules; same step does not imply the same training phase, or exactly identical token exposure. Both step and cumulative assistant tokens are recorded.

The trainer differs from the validated follow-up trainer only in allowed dataset/family CLI choices. All 18 prepared minibatch schedules reach nine distinct nonzero evaluation milestones (1%, 2%, 5%, 10%, 20%, 40%, 60%, 80%, 100%) without crossing multiple milestones in one minibatch. No coalescing change was necessary after excluding the hard-coded examples.

| Dataset | Selected rows | Assistant-loss tokens | Final optimizer step |
|---|---:|---:|---:|
| aya | 12,773 | 4,616,759 | 264 |
| coconot | 10,972 | 847,113 | 52 |
| evol-codealpaca | 10,389 | 4,616,939 | 266 |
| flan-v2 | 89,744 | 3,123,074 | 191 |
| no-robots | 9,489 | 1,816,100 | 107 |
| numinamath-tir | 6,318 | 4,616,968 | 256 |
| oasst1 | 7,068 | 3,180,183 | 179 |
| openmath-gsm8k | 30,385 | 4,617,199 | 268 |
| personahub-algebra | 5,225 | 4,616,199 | 262 |
| personahub-code | 26,445 | 4,616,557 | 272 |
| personahub-grade-math | 13,243 | 4,616,872 | 270 |
| personahub-if | 13,408 | 4,617,208 | 260 |
| personahub-math | 4,082 | 4,616,385 | 267 |
| sciriff | 9,985 | 1,222,976 | 77 |
| tablegpt | 4,994 | 418,029 | 30 |
| wildchat | 4,759 | 4,617,149 | 264 |
| wildguardmix | 49,946 | 4,480,471 | 263 |
| wildjailbreak | 17,353 | 4,617,242 | 265 |

The run plan uses two concurrent one-B200 training jobs (18 jobs in two dependency chains), plus eight one-B200 persistent evaluation workers: **10 GPUs peak** after a separate one-GPU numerical gate. Each GPU job requests 8 CPUs/128G; train wall time 2h, evaluator wall time 12h, gate 45m. All jobs exclude worker-12, use guarded `sbatch`, and pin the cu130 SDPA environment. A conservative whole-campaign storage guard reserves 250 GB before launch and enforces the existing 1 TB/20% remaining-free threshold.

There are **162 new adapter states and 972 full interface evaluations** (three name variants × two interfaces), with historical subsets calculated post hoc. Both existing full Vintage/Web step-0 references are hash-verified and reused. Evaluators watch all 18 producers, consume inflight checkpoints, verify scheduled steps/tokens at publication, and propagate failures. The final zero-GPU audit validates producer/evaluator SLURM completion, all checkpoint/result hashes, pair coverage, recomputed trait aggregates, and the exact expected state set before writing `trajectories.csv` and `campaign-status.json`.

Nine CPU tests passed: source scope, capped prefix selection, insufficient/empty budgets, milestone schedules including rejection of an unsafe tiny schedule, dataset-specific queue publication, and baseline reuse. Python and batch-shell syntax checks passed. A separate zero-GPU validation job independently reconstructs all random selections, checks every saved row/mask against the census, simulates all schedules, and checks existing reference/gate inputs. GPU numerical validation is deliberately launch-time and has not yet run for this new setup. No training or evaluation job has launched.


### Ready for launch (CPU-validated)

Independent validation job **10233** completed on worker-5, exit **0:0**, in **52s**, with zero GPUs. `setup-validation.json` passed all full-row and schedule checks and pins both data/code manifests. Total prepared exposure is **65,873,423 assistant-loss tokens**; final steps range from **30 to 272**. The dry-run launcher passed all frozen-source checks and wrote **28 reviewed commands** (one gate, 18 training jobs in two chains, eight evaluation workers, one final audit) to `launch-plan.txt`. No launch-state file or GPU job has been created for this campaign.

Launch preview: [launch-plan.txt](/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/launch-plan.txt). Validation: [setup-validation.json](/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/setup-validation.json). Prepared data manifest: [manifest.json](/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/data/manifest.json).


## Authorized 32-GPU launch revision

The user explicitly re-enabled **worker-12** and authorized up to **32 total GPUs** for fastest completion. This supersedes the earlier worker exclusion and two-training/eight-evaluator plan. Data, scientific settings, token budgets, and evaluation scope are unchanged.

The revised allocation is **18 one-GPU train-then-evaluate jobs plus 14 one-GPU evaluation-only jobs**. All training jobs may start concurrently; after a training subprocess finishes, its allocation starts the same persistent evaluation worker. Thus capacity transitions from 18 trainers/14 evaluators to up to 32 evaluators without additional GPU reservations. Every production allocation requests 8 CPUs, 128G memory, and 12 hours. The guarded launcher passes `--gpu-limit 32` on every submission.

The separate one-GPU numerical gate must finish successfully before the launcher reserves the 32 production GPUs, because the submission guard counts pending reservations as well as running allocations. A final zero-GPU audit depends on all 32 production jobs. The previous code manifest, CPU validation, launch preview, and setup-status files are preserved with `-before-32gpu` suffixes. Scheduling-only changes are frozen in a new code manifest and independently CPU-revalidated before GPU submission.


### Launch confirmed

Updated 2026-09-18T03:08:41.584011+00:00. Revised CPU validation **10234** completed with exit 0 in 50s. Numerical gate **10236** completed with exit 0 in 2m44s; all **16** cached-score/reuse checks had maximum error **0.0** and exact answer-choice agreement.

All 18 train-then-evaluate jobs **10237–10254**, 14 evaluation-only workers **10255–10268**, and final zero-GPU audit **10269** were submitted successfully. All 18 training runs initialized with exact zero-adapter logits and passed likelihood parity. Production scheduler snapshot: **{'PENDING': 1, 'RUNNING': 31}**; exactly 32 one-GPU production allocations are reserved, including any pending jobs. Worker-12 is hosting eight training allocations. Evaluators are consuming inflight checkpoints; queue snapshot **{'pending': 53, 'running': 13, 'done': 0, 'failed': 0}**. These are startup/launch validations, not a claim that training or evaluation has completed.

Authoritative job receipt: `launch-state.json`; compact launch snapshot: `launch-summary.json`; launcher log: `launch-console.log`, all under the campaign root. The final audit runs automatically after all production jobs terminate and only reports complete if SLURM and all scientific artifacts validate.


## Completed campaign

All **18 training runs** finished at their prepared token budgets, and all **162 nonzero checkpoint evaluation tasks** completed successfully. The queue has no pending, running, or failed tasks. Full artifact audit **10484** completed with exit **0:0** in **00:03:28**, verifying frozen inputs, all checkpoint hashes and metadata, every result hash, exact scenario/pair coverage, recomputed trait and historical-subset aggregates, and the expected state set.

There are **972 new full interface result files**, plus **12 reused base-reference files**: **984 total**, containing **15,744,000 pairwise rows** (15,552,000 new and 192,000 reused). Canonical results: `/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/trajectories.csv`; SHA-256 `031b39784863c07a3905ed06c47c37c5125613aa5517f5ee21f9f9cbde2d373f`. `campaign-status.json` and `completion-receipt.json` record final validation.

### Reconciled worker exits

Evaluation-only workers **10267 and 10268** exited with `FileNotFoundError` reading `vintage-flan-v2/run.json` in the idle polling branch (after `queue.claim` returned no task). The underlying cause of the intermittent read failure is not established. Their failed SLURM states are preserved. All claimed tasks have successful attempts and verified outputs; neither worker left a running/failed queue item. Original audit **10269** stopped at its blanket requirement that every worker exit 0.

Recovery audit `10484` permits only these two explicitly pinned exceptions, checks their exact stderr hashes and failure location, and retains every scientific integrity check. `worker-exit-reconciliation.json` records evidence and policy; `audit_reconciled.py` implements it. No training or evaluation was rerun. For future campaigns, make producer-status polling robust to transient missing-file reads without weakening checkpoint/result validation. Do not alter the frozen completed-campaign code in place.

## Combined Vintage-base trajectory figure

The requested 21-curve figure combines the three earlier Vintage-base arms (Tulu mixture, Vintage SFT, Vintage paraphrase) with all 18 source-specific arms. It uses **bare interface, original names, all scenarios**, the same equal-trait Web-similarity metric, and actual optimizer step. Both campaigns' audited CSV hashes and identical step-0 trait profiles were checked; all 30 original-arm points reproduce the preceding figure. There are 210 measured points including reused step-0 anchors. Short runs end at their actual last step, with no extrapolation. Orange preserves the three earlier comparisons; other colors, markers and line styles distinguish the new arms. The legend is ordered by endpoint similarity. The displayed y range is −10 to 80; the normalization still places exact Web-base agreement at 100.

Artifacts under `/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/all-vintage-trajectories`: `web-similarity-all-vintage.{png,pdf,svg}`, `web-similarity-trajectories.csv`, `endpoints.csv`, and `manifest.json`. Generator: `scripts/experiments/trait_tulu_categories_20260918/plot_all_vintage_trajectories.py`. The exported figure was visually inspected. These are descriptive one-seed curves; endpoint ranking compares different actual token exposures for undersized datasets.

## Per-trait raw-score document (2026-09-18)

Created [eight-page PDF](/data/talkie-base-experiments/runs/trait-tulu-categories-20260918/raw-trait-trajectories/vintage-base-raw-traits.pdf), one page per trait, containing the same 21 Vintage-base series as the aggregate chart (three original SFT comparisons and 18 individual Tulu sources). Selection remains bare interface, original names, all scenarios. Each score is the unnormalized high-trait selection percentage (100 × high_selected / 2,000), plotted against optimizer step on a common 0–100% y-axis. Each page includes its fixed Web-base step-0 horizontal reference. Colors, markers, line styles, and legend order match the aggregate chart. Short runs end at their last measured checkpoint; steps are not exactly token matched.

Artifacts and provenance: `raw-trait-trajectories/` under the campaign root contains individual PNG/SVG plots, `raw-trait-report.md`, `raw-trait-scores.csv` (1,680 trait observations), and `manifest.json`. Generator: `scripts/experiments/trait_tulu_categories_20260918/plot_raw_trait_trajectories.py`. Validation checks both audited source hashes, exact shared base profiles, eight complete traits, 21 ten-point strictly increasing series per trait, denominators of 2,000, all scores in [0,100], and eight PDF pages with the reference and raw-score axis labels. No new training or evaluation was required.
