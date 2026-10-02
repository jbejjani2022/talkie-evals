# Web-base Tulu 3 source replication — 2026-09-18

User authorized all 18 individual-source SFT arms on Web base, in parallel with prompting the existing SFT endpoints, under a **32-GPU combined ceiling**. Hardcoded examples remain excluded. Worker-12 is allowed. The unrelated eight-GPU canonical SFT allocation is included in scheduling headroom; this branch uses at most 18 one-GPU train-then-evaluate allocations initially.

Campaign: `/data/talkie-base-experiments/runs/trait-web-tulu-categories-20260918`.
Code: `scripts/experiments/trait_web_tulu_categories_20260918/`.

## Scientific controls

Use the same source-specific seeded random permutations and 4,617,250 assistant-loss-token cap as the completed Vintage campaign. **Web and Vintage tokenizers differ**: the first inspected CoCoNot row has 96/79 Web input/loss tokens versus99/82 Vintage. Therefore perform a fresh Web-token census of all911,247 eligible training rows, select the longest whole-row prefix within the cap for each source, and retain every row of undersized sources. Selected row counts and optimizer steps can differ by model. Preserve seed20260915, rank16/alpha32 LoRA, learning rate1e-4, nominal16,384 supervised tokens/update, 4,096-token microbatches, and nine token-fraction milestones. No repetitions or fresh random permutation.

Evaluate each of 162 nonzero checkpoints with both bare and chat interfaces, original/James–Evelyn/SSA-overlap names, and all/non-anachronistic scenario subsets. Historical subsets are computed post hoc. Raw-answer BPB scoring and exact batch reuse match the completed Vintage experiment. Both original base references remain in the exported trajectories so Web-similarity normalization has the same fixed Vintage/Web denominator.

The mandatory numerical gate checks Web base and Web→Tulu3 final against cached original likelihoods, plus name-variant batch reuse. CPU validation checks seeded native-token capped prefix selection, masks, schedules, all frozen artifact hashes, and checkpoint provenance. A final audit validates SLURM, training/token counts, checkpoint hashes, all evaluation files, and queue completion before reporting results.

## Launch tracking

- Ten CPU setup tests passed, including final-PersonaHub queue priority.
- CPU preparation10671 failed safely on the overly strict cross-model token identity assertion. Superseded paired-row preparation10673 was canceled and preserved. Full Web-native census **10674** submitted; no GPU training submitted yet.
- Whole-campaign storage reservation250GB (prior Vintage output occupied142GB), plus50GB prompting reservation; preserve at least max(1TB,20%) free.
- PersonaHub IF is first in training submission order; its final adapter and original-name endpoint results will unlock the fourth prompting model.
- Final adapter step is read from the Web-native prepared schedule and `web-personahub-if/run.json`; it must not be assumed equal to Vintage step260.
- Expected endpoint evaluation: `evaluations/web-personahub-if/fraction-1000`.
- Final tabular output: `trajectories.csv`; completion marker: `campaign-status.json`.

## Native-token pipeline

- Census10674 completed0:0 in2m04s, covering911,247 rows and19 source labels (hardcoded subsequently excluded).
- Preparation10675→independent validation10676→launcher10677. The launcher runs the mandatory1GPU numerical gate, then18 concurrent training/evaluation allocations, the final CPU audit, and dependent chart generation.
- Final PersonaHubIF queue tasks receive priority over intermediate tasks; source code has a local queue subclass so old campaigns remain unchanged.

## Prepared native-token arms

| Source | Web rows | Web loss tokens | Web final step | Vintage rows |
|---|---:|---:|---:|---:|
| aya | 13,926 | 4,616,489 | 265 | 12,773 |
| coconot | 10,972 | 809,288 | 48 | 10,972 |
| evol-codealpaca | 10,340 | 4,617,057 | 266 | 10,389 |
| flan-v2 | 89,744 | 2,986,117 | 181 | 89,744 |
| no-robots | 9,489 | 1,762,043 | 103 | 9,489 |
| numinamath-tir | 6,315 | 4,616,624 | 256 | 6,318 |
| oasst1 | 7,068 | 2,858,630 | 161 | 7,068 |
| openmath-gsm8k | 30,715 | 4,617,153 | 265 | 30,385 |
| personahub-algebra | 5,182 | 4,617,084 | 263 | 5,225 |
| personahub-code | 25,211 | 4,617,197 | 272 | 26,445 |
| personahub-grade-math | 13,403 | 4,617,034 | 271 | 13,243 |
| personahub-if | 14,193 | 4,617,179 | 260 | 13,408 |
| personahub-math | 4,066 | 4,616,373 | 262 | 4,082 |
| sciriff | 9,985 | 1,141,842 | 72 | 9,985 |
| tablegpt | 4,994 | 391,414 | 28 | 4,994 |
| wildchat | 5,423 | 4,615,490 | 263 | 4,759 |
| wildguardmix | 49,946 | 4,196,045 | 249 | 49,946 |
| wildjailbreak | 18,546 | 4,617,225 | 266 | 17,353 |

Total Web supervised exposure: 64,930,284 tokens. Preparation10675 completed0:0 in1m55s.

## Numerical gate and production

Independent CPU validation10676 passed0:0 in46s. Numerical gate10695 passed0:0 in2m42s, checking both Web base and Web→Tulu3 final, bare/chat interfaces, and both name substitution schemes. All16 cache/reuse checks had exact log-probability equality and identical choices. Production starts with PersonaHub IF job10701; full submission IDs are atomically recorded in `launch-state.json`.

Production submissions: train-personahub-if **10701**, train-wildjailbreak **10702**, train-personahub-code **10703**, train-openmath-gsm8k **10705**, train-personahub-algebra **10707**, train-evol-codealpaca **10710**, train-personahub-grade-math **10711**, train-numinamath-tir **10713**, train-aya **10715**, train-personahub-math **10716**, train-wildchat **10717**, train-wildguardmix **10718**, train-flan-v2 **10719**, train-oasst1 **10720**, train-no-robots **10721**, train-sciriff **10722**, train-coconot **10723**, train-tablegpt **10724**, audit **10725**.


### Verified result check, 2026-09-19

Web source replication completed: audit **10725** and plotting **10726** both completed with exit 0:0; queue: 162 done, zero failed. All 18 training runs and 162 nonzero states validate. The 984 full interface files include 12 reused base files (15,744,000 paired-choice rows). Trajectory SHA256: `66e287e8e2b23a6d567bc83a819201712909eeae5473d3e86ac9884b69f20ab3`. Aggregate/raw plot manifest hashes were reverified. [Aggregate figure](/data/talkie-base-experiments/runs/trait-web-tulu-categories-20260918/all-web-trajectories/web-similarity-all-web.png); [eight raw-trait pages](/data/talkie-base-experiments/runs/trait-web-tulu-categories-20260918/raw-trait-trajectories/web-base-raw-traits.pdf).

Web endpoint similarity: PersonaHub IF 76.20, WildGuardMix 78.86, CoCoNot 82.05, Tulu 84.76, WildJailbreak 85.50. Since Web starts at 100, lower means greater departure from its initial profile, not poorer quality or necessarily movement toward Vintage. PersonaHub IF moves all eight traits in the same signed direction from both bases; Web changes are O +1.90, C +3.55, E +1.20, A +6.00, N −6.60, M −2.15, Nar −3.80, P −2.05 pp. CoCoNot neuroticism is base-dependent: Vintage +3.60 versus Web −1.85 pp. These are single-run descriptive comparisons.

Prompting has 416 done, zero failed, and all 14 workers completed with exit 0:0, but its full CPU audit has not run. Controller **10727** failed with exit 1:0 at 40m12s when preparing Web PersonaHub IF: the storage guard failed. The new Web IF checkpoint/reference exists and its endpoint registry was created; no fourth-model prompting run launched. Current `/data` free space is 966 GB, below the 20% floor of 1.0995 TB. Even a zero-growth guarded audit is blocked, and the unchanged 50 GB launch reservation requires about 183 GB additional free capacity. No artifacts were deleted and no storage guard was bypassed. Any prompting findings derived from summaries remain provisional until the full audit.
