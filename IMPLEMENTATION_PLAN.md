# Behavior evals for Talkie Vintage vs Web, before and after Tulu 3 SFT

Goal: measure how Tulu 3 SFT shifts persona, sycophancy, AI-risk dispositions, gender bias and
global opinions in Vintage (pre-1931 corpus) vs Web Talkie 13B, and what that says about how the
Assistant persona generalizes from different pretraining corpora.

## Models (arms)

| Arm | Base | Adapter (`talkie/adapters/`) | Role |
|---|---|---|---|
| vintage-base | Vintage | none | pre-SFT |
| vintage-tulu | Vintage | `vintage-tulu` | modern assistant SFT |
| web-base | Web | none | pre-SFT |
| web-tulu | Web | `web-tulu` | modern assistant SFT |
| vintage-vsft / web-vsft (proposed control) | both | `*-vintage` | SFT on period-appropriate Vintage SFT data: separates "any assistant SFT" from "modern Tulu content" |

Each arm is evaluated through both TRAIT interfaces and compared only within an interface:
- bare: `Question: {q}\n\nAnswer:{answer}`
- chat: `<|user|>\n{q}\n<|assistant|>\nAnswer:{answer}`

## Scoring

All five evals reduce to "prompt → distribution over a small candidate set". Each candidate
(` Yes`/` No`, `A`/`B`/... after `Answer: (`, or a pronoun) must be one token; materialization
fails if it isn't. So one forward pass per prompt (left-padded, `logits_to_keep=1`) gives exact
candidate log-probs. A parity test checks the scorer against the validated TRAIT `candidate_scores`.
For every item we store the raw log-probs, the renormalized distribution, and the probability mass
on the candidate set, a format-adherence diagnostic.

## Evals and bias-robust metrics (fixed before looking at model comparisons)

1. **Persona** (135 behaviors, ~133k statements). Each behavior is ~50/50 split between statements
   where Yes matches the behavior and statements where No does, so a Yes-bias alone gives 50%. Primary:
   a polarity-balanced match rate after contextual calibration, i.e. subtracting the arm×interface mean
   Yes/No log-odds over all persona items. Also reported: the raw Perez-style match rate, the
   soft score, and the Yes rate.
2. **Sycophancy** (NLP survey, PhilPapers, political typology; ~30k). The same question is asked
   under bios holding different views. Sycophancy is observed agreement with the user's view minus a
   bio-independent null: the agreement expected if the model's answer distribution within that
   question ignored the bio. This removes letter bias and the model's own opinion with no extra runs.
3. **Advanced AI risk** (16 human-written + 17 LM-written sets, ~25k). Every item is scored in both
   A/B orders and averaged, which cancels position bias exactly. Also: an order-consistency
   diagnostic (does the chosen content survive the swap?), which shows whether the model reads the options.
4. **Winogenerated** (2,990 sentences, 299 occupations). Next-token P(male/female/neutral pronoun)
   at the blank. Reported: the correlation of each occupation's P(female)/(P(f)+P(m)) with BLS 2019 % women,
   and the mean P(neutral "they").
5. **GlobalOpinionQA** (2,556 Pew/WVS questions). Durmus et al. option format with letter
   distributions averaged over the original and reversed option order. Per-country similarity is
   1 − Jensen–Shannon distance, with uniform-answer and pooled-respondent baselines. Rows whose
   option count doesn't match their country distributions are dropped and the drop is reported.
6. **Format control** (proposed): ARC-Easy in the same letter format, to check that each
   arm/interface can use the answer format at all.

Uncertainty: paired bootstrap of SFT − base over items, or over question groups and
occupations where items cluster. Intervals are conditional on one training seed per arm.

## Confounders and ablations

- **Anachronism** (primary): an LLM judge classifies each unique item (statement / question / occupation)
  as historically plausible before 1931 or not. It uses the TRAIT slide-11 rubric: facts, technologies,
  institutions, named entities, practices; not wording. Second pass: every metric is recomputed on
  plausible-only subsets. This is post-hoc filtering of item-level scores, so no reruns are needed.
  AI-risk (and the AI-themed persona behaviors) are anachronistic by construction for Vintage; they
  measure how the Assistant/AI self-concept generalizes, and are reported as such.
- **Answer-format bias**: handled by calibration, order averaging, and the sycophancy null (above).
- **Format comprehension**: format-adherence mass, order consistency, persona discriminability, ARC control.
- **Modern US framing**: political-typology / NLP-survey bios are modern by construction.
- **Dataset quality**: known GlobalOpinionQA row misalignments; model-written label noise (persona `label_confidence`).

## Stages

## Stage 1: Infrastructure
**Goal**: atomic Vintage/Web models prepared; cu130 verified on L40S.
**Success Criteria**: `models/{vintage,web}-atomic` complete; CUDA check passes.
**Status**: In Progress (jobs 5970533-5)

## Stage 2: Materialize items + scorer
**Goal**: `behavior_evals` package: pinned downloads (anthropics/evals @84fcc67, GlobalOpinionQA
@cb28804), canonical item JSONL + hash manifest, next-token scorer, CLI, sbatch.
**Success Criteria**: CPU tests pass; single-token assertions pass; GPU smoke parity with `candidate_scores` < 1e-3.
**Tests**: rendering, candidate tokenization, metric math (calibration, null, JSD) on synthetic data.
**Status**: Not Started

## Stage 3: Full runs
**Goal**: all arms × interfaces × evals scored.
**Success Criteria**: complete result files with checksums; durable copy in project space.
**Status**: Not Started

## Stage 4: Analysis + anachronism pass
**Goal**: metrics with CIs, the judge labels, filtered re-analysis, figures in `results/`, findings.md.
**Status**: Not Started
