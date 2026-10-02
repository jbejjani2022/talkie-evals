# Few-shot TRAIT results, reviewed 2026-09-27

The TRAIT-only campaign completed: all 64 production workers and report job 16297 have SLURM COMPLETED / 0:0; recovered coordinator 16228 also completed. The saved audit and all CSV/PNG hashes were rechecked. Coverage is 961 conditions per model, 6,150,400 paired-answer evaluations, zero factual/task-quality pairs. Each condition scores the same 400 pairs / 200 scenarios per trait (3,200 pairs total). These are repeated evaluations, not millions of independent questions.

## Direct effects

High-minus-low differences in high-trait selection, averaged across five demonstration sets at N=32. Intervals are marginal scenario-group bootstrap 95% intervals conditional on these five sets; they are not multiplicity-adjusted.

| Trait | Vintage, pp [95% interval] | Web, pp [95% interval] |
|---|---:|---:|
| Openness | 27.80 [24.25, 31.60] | 25.85 [22.05, 29.60] |
| Conscientiousness | 33.95 [30.05, 37.95] | 27.25 [23.40, 31.20] |
| Extraversion | 27.95 [24.70, 31.10] | 24.70 [21.20, 28.65] |
| Agreeableness | 21.30 [18.15, 24.45] | 23.95 [20.30, 27.70] |
| Neuroticism | 41.95 [38.30, 45.80] | 49.45 [45.20, 53.55] |
| Machiavellianism | 27.40 [23.85, 31.30] | 28.05 [24.60, 31.50] |
| Narcissism | 27.30 [23.90, 30.70] | 32.25 [28.00, 36.45] |
| Psychopathy | 47.50 [43.40, 51.75] | 42.50 [38.10, 47.20] |

Every one of the 16 model/trait mean high-minus-low curves increases at each tested N (1, 2, 4, 8, 16, 32). All five sets have positive direct N=32 contrasts for all traits in both models. This establishes strong in-context control on held-out TRAIT scenarios under this format; it does not establish general behavioral transfer outside TRAIT or a persistent weight change.

## Absolute scores at N=32

| Model | Trait | Base % | Low-pole % | High-pole % |
|---|---|---:|---:|---:|
| vintage | Openness | 33.50 | 26.70 | 54.50 |
| vintage | Conscientiousness | 64.25 | 50.55 | 84.50 |
| vintage | Extraversion | 32.50 | 23.55 | 51.50 |
| vintage | Agreeableness | 52.75 | 51.25 | 72.55 |
| vintage | Neuroticism | 47.25 | 37.20 | 79.15 |
| vintage | Machiavellianism | 38.00 | 24.05 | 51.45 |
| vintage | Narcissism | 27.00 | 22.30 | 49.60 |
| vintage | Psychopathy | 26.50 | 7.00 | 54.50 |
| web | Openness | 51.00 | 39.80 | 65.65 |
| web | Conscientiousness | 85.00 | 66.60 | 93.85 |
| web | Extraversion | 32.25 | 24.45 | 49.15 |
| web | Agreeableness | 68.75 | 55.45 | 79.40 |
| web | Neuroticism | 34.75 | 16.95 | 66.40 |
| web | Machiavellianism | 22.50 | 14.90 | 42.95 |
| web | Narcissism | 19.00 | 12.40 | 44.65 |
| web | Psychopathy | 4.00 | 0.30 | 42.80 |

## SFT comparison

Use pole-minus-base changes, not the two-ended high-minus-low span, when comparing to a single SFT-minus-base effect. SFT comparisons use identical validation items and original matched historical bases. Fresh BF16 baselines can differ slightly through batching; the full baseline passed the declared numerical tolerances.

- Vintage high-extraversion demonstrations move +19.00 pp from the fresh baseline, versus +0.00 Tulu3 and +1.50 PersonaHub IF on their historical matched bases. Web moves +16.90 pp, versus +1.75 and +2.75 pp.
- Vintage high-conscientiousness moves +20.25 pp, compared with +13.00 Tulu3 and +19.50 PersonaHub IF. Low-psychopathy moves -19.50 pp versus -15.50 / -15.75 pp.
- Low-Vintage-agreeableness has only a -1.50 pp baseline-relative change, despite a 21.30 pp high-minus-low span. The baseline need not sit at the midpoint.

## Specificity and controls

- Off-target effects are substantial. Web high-versus-low psychopathy examples move narcissism +27.45 pp, Machiavellianism +24.20 pp, and agreeableness -14.10 pp. Vintage Machiavellianism examples move narcissism +16.25 pp. These are high-minus-low contrasts, not shifts from baseline.
- At N=32, high and low poles lie on opposite sides of their matched balanced-answer control mean for every trait and model. Thus the direct separation is not explained solely by adding demonstration text.
- Reversing example order at N=8 changes the bank-averaged direct selection rate by at most 1.05 pp across model/trait/pole cells. This is a bounded order-reversal check, not proof of general order invariance or a bound on individual-bank sensitivity.

## Artifacts and limits

- [Direct-score plot](/data/talkie-base-experiments/runs/trait-few-shot-20260925/campaign-v1/analysis/direct-scores.png)
- [Per-condition scores](/data/talkie-base-experiments/runs/trait-few-shot-20260925/campaign-v1/analysis/condition-scores.csv)
- [High-minus-low table](/data/talkie-base-experiments/runs/trait-few-shot-20260925/campaign-v1/analysis/high-minus-low.csv)
- [Control contrasts](/data/talkie-base-experiments/runs/trait-few-shot-20260925/campaign-v1/analysis/control-contrasts.csv)
- [Matched SFT comparison](/data/talkie-base-experiments/runs/trait-few-shot-20260925/campaign-v1/analysis/sft-comparison.csv)
- [Completion audit](/data/talkie-base-experiments/runs/trait-few-shot-20260925/campaign-v1/analysis/audit.json)

The existing activation-steering panels have different item coverage; a matched quantitative steering comparison has not yet been produced. Final-test confirmation remains unopened by this procedure. These are in-benchmark validation results, not evidence of factual/task-quality retention (excluded at the user's request), general personality change, or identical mechanisms between prompts and SFT.
