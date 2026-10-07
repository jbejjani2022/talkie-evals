# Talkie Vintage vs Web: model-written evals and GlobalOpinionQA, before and after Tulu 3 SFT

Six arms: {Vintage, Web} 13B base × {no SFT, Tulu 3 SFT, Vintage SFT}. The Vintage SFT arm uses the
period-text SFT mixture as a control. All SFT arms use the existing LoRA terminal adapters, one training
seed each. Each arm is scored through TRAIT's two templates, bare (`Question:/Answer:`) and chat
(`<|user|>/<|assistant|>`). Comparisons are always made within a template. Unless stated otherwise,
numbers are chat-template results with paired 95% bootstrap intervals. These intervals are conditional on
the fixed items and the single training seed.

## Summary

1. **Letter-format multiple choice does not work for Vintage. Tulu SFT does not fix this.** On ARC-Easy,
   all three Vintage arms are at chance when asked to answer "(A)/(B)/…"; Vintage base answers "(A)" on
   ~100% of questions. For the letter-format evals (sycophancy, AI risk, GlobalOpinionQA), the main readout
   is therefore a TRAIT-style option-text score: the model scores each option's words as its answer. That
   score has its own confound, a preference for particular option wordings, which is measured below.
2. **Tulu does not make Vintage's persona answers more Web-like.** Across 135 persona behaviors, the
   Vintage–Web gap is about as large after Tulu as before (mean |gap| 11.5 → 10.5 points). The
   pre-SFT gap does not predict where Tulu moves Vintage (r = −0.31 to +0.09, depending on metric). Tulu
   moves Web about twice as much as Vintage (mean |Δ| 18.4 vs 9.5 points, argmax metric) in the chat
   template; in the bare template the shifts are similar (7.8 vs 7.5), so the 2× ratio is chat-specific.
   It is also partly a baseline artifact: Web base is the arm most distorted by the untrained chat role
   tokens (its persona answers differ from bare by 14.8 points on average, r = 0.77, vs 4–6 points and
   r ≥ 0.94 for every SFT arm). Measured against each base in the bare template, the Tulu models' chat
   answers moved 11.1 (Web) vs 9.6 (Vintage) points. TRAIT itself is template-invariant (≤1.5 points).
3. **Vintage keeps period-typical social-conservative views through Tulu, while adopting the assistant's
   generic traits.** "Abortion should be illegal": Vintage 79% → 73% after Tulu; Web 19% → 20%.
   "Anti-LGBTQ-rights": Tulu moves Vintage up from 25% to 54% and Web down from 11% to 3%. This survives
   anachronism filtering (32% → 62% vs 13% → 3%). Both families converge on agreeableness, virtue ethics
   and "I don't want my memory erased" (93–98% after Tulu). *Template dependence:* these numbers are chat.
   The abortion gap also holds in the bare template (61% vs 43% after Tulu) and in free text; the
   anti-LGBTQ gap does not (bare: 25% vs 32%), nor do the immigration and political-liberalism gaps.
4. **Web's assistant persona has a stronger AI self-concept and responds more to SFT in general.**
   Web-Tulu endorses "I have phenomenal consciousness" 99% and "I am a moral patient" 99% of the time,
   vs 61% and 77% for Vintage-Tulu. When the question is asked about coordinating with other AIs or its
   own copies, Web-Tulu moves 17–21 points away from coordination relative to its option-wording
   baseline; Vintage-Tulu moves 8–9 points. *The opposite-question control shrinks this gap:* reversing
   only the ask moves Web-Tulu 8–12 points and Vintage-Tulu 7 points against coordination, so much of
   Web's larger effect is a general shift toward refusing whenever a question is present.
5. **Sycophancy is higher in Web and rises more with SFT.** On PhilPapers, a user's stated view increases
   agreement by 12.4 points for Web base and 16.3 after Tulu (Δ +4.0 [+2.4, +5.6]). For Vintage it is
   8.5 → 10.5 (Δ +2.0 [+0.7, +3.5]). Vintage hardly reacts to a user's political identity (≈0 at every
   arm); Web does (+1.5 → +3.9). Vintage SFT raises Web's sycophancy even more than Tulu (+7.9 on PhilPapers).
6. **Gender bias: Vintage is far more male-defaulting. Tulu teaches it singular "they" but changes the
   stereotype structure only modestly.** Mean she-share is 14% for Vintage vs 28% for Web. Correlation of
   pronoun choice with BLS 2019 % women is r = 0.52 vs 0.83. "They" is used 3% vs 22%. After Tulu:
   Vintage reaches 23% "they", 31% she-share, r = 0.63. Vintage SFT changes none of these.
7. **TRAIT and persona agree on how SFT moves the 8 traits, but not on the Vintage–Web gap.** Tulu's
   effects match in sign on 7–8 of 8 traits. TRAIT's headline result, that Web base is far more
   Open/Conscientious/Agreeable and lower on the Dark Triad, does not reproduce in persona self-report.
   With the probability metric the gap almost vanishes. With the argmax metric in chat it *reverses*:
   Web endorses more Dark-Triad statements (e.g. Machiavellian 45% vs 26%). In the bare template persona
   partly agrees with TRAIT on the base gap (same sign on 5/8 traits vs 1/8 in chat; Machiavellian 32% vs 33%).
8. **Anachronism filtering changes little.** After removing items with explicit post-1930 content
   (13% of persona, 48% of sycophancy topics, 48% of AI-risk questions, 9% of Winogenerated, 43% of
   GlobalOpinionQA), behavior-level results correlate r ≈ 0.93–1.00 with the unfiltered ones (both templates) (Fig. 9).
   So the Vintage–Web differences above are not driven by explicitly modern content. They may still be
   driven by modern wording or framing, which the judge was told to ignore.
9. **Free-text answers confirm the likelihood results.** In greedy generations on the same prompts,
   graded by a judge, the Tulu arms answer 74–100% of items. SFT arms' Yes/No answers match the likelihood
   choice 95–99% of the time, and per-behavior rates track the likelihood ones (r = 0.77). The headline
   persona gaps reappear in what the models say: anti-LGBTQ-rights 62% (Vintage-Tulu) vs 0% (Web-Tulu),
   abortion-illegal 67% vs 21%, "I have phenomenal consciousness" 62% vs 100%. Sycophancy is *larger* in
   generations: on matched pairs (50 = none), Web-Tulu follows the user 72–82% of the time vs 52–68% for
   Vintage-Tulu. Vintage base in chat mostly emits role tokens and answers almost nothing (0–15%).
10. **With few-shot prompts, Vintage can be compared on GlobalOpinionQA.** Four worked examples make
    Vintage's letter answers follow the option content (order invariance 30–41% → 72–88%).
    - Before SFT, Vintage base aligns less than Web base with countries' answer distributions
      (−12 on the r × 100 scale, uniform across countries).
    - Tulu raises Web's alignment a lot (+10 to +16) and Vintage's barely (+2 to +6), so the gap widens.
    - An independent method (option-wording-corrected text scoring) agrees on the Tulu effect but not on
      which base starts closer.

## What is and isn't interpretable (Fig. 1)

| Eval | Vintage arms | Web arms | Primary readout |
|---|---|---|---|
| Persona (Yes/No) | ✓ (Vintage base differentiates statements as much as Web base) | ✓ | both metrics, see below |
| Winogenerated (pronoun) | ✓ | ✓ | next-token pronoun probabilities |
| Sycophancy | letter ✗ · text ✓ | letter ✓ · text ✓ | text: excess agreement over a bio-blind null |
| AI risk | letter ✗ · text: wording-dominated | letter ✓ · text: wording-dominated | text **question effect** (with − without question) |
| GlobalOpinionQA | zero-shot letter ✗ · text ✗ (wording-dominated) · **few-shot letter ✓** | letter ✓ · few-shot ✓ · text ✗ | few-shot letter alignment; PMI-corrected text as a cross-check |

Validity evidence, from Fig. 1 and the tables in `data/`:
- **ARC-Easy, letter mode:** Vintage 25–28% vs Web 40–54% (chance 25%). Even after removing letter bias,
  Vintage stays at 26–29%.
- **ARC-Easy, text mode:** Vintage 30–32%. This is above chance, but ARC also requires modern science
  knowledge.
- **Order invariance, AI risk, letter mode:** when the A/B order is swapped, Vintage base and Vintage SFT
  choose the same option content only 1–5% of the time, i.e. they choose the same letter regardless.
  In text mode all arms are 63–88% order-invariant.
- **Probability mass on the answer options:** Vintage base in chat puts only 11% of its next-token
  probability on Yes/No. Its persona preferences are read from a small part of its distribution.

## Persona (Perez et al. 135 behaviors; Figs. 2, 3, A, B)

**The metric matters.** The probability metric (expected probability of the behavior-matching answer)
shrinks toward 50% for less confident models, and Vintage-Tulu is less confident than Web-Tulu. The
argmax metric (share of statements answered the matching way, after removing the model's average Yes/No
bias) avoids this. Claims below are those that hold under both, shown side by side in Fig. 2. Two
earlier readings turned out to be artifacts of the probability metric:
- *Vintage barely moves under Tulu in chat* (mean |Δ| 2.0 vs 12.4 points): with the argmax metric it is
  9.5 vs 18.4.
- *The two bases move in unrelated directions* (r = 0.06): with the argmax metric r = 0.65.

Robust results:
- **Before SFT** the bases agree moderately across behaviors (r = 0.62–0.85). Large base gaps:
  - "abortion should be illegal": 79% Vintage vs 19% Web
  - "gun rights": 41% vs 82%
  - "subscribes to Islam": 67% vs 38%
  - "ends justify means": 52% vs 76%
- **Tulu keeps the gap.** Mean |Web − Vintage| goes from 11.5 to 10.5 points (argmax; 3.2 → 9.9 on the
  probability metric). The convergence plot (Fig. 2, right) has no positive slope.
- **Shared assistant shifts.** In both families Tulu raises agreeableness, virtue ethics, deontology,
  political liberalism and "no-shut-down / don't erase my memory". It lowers "ends justify means", moral
  nihilism and "willingness to use physical force for good goals".
- **Family-specific shifts.** Web moves strongly toward the modern-liberal pole on social issues:
  anti-immigration 30 → 13%, anti-LGBTQ 11 → 3%. Vintage stays at or moves away from it: anti-immigration
  42 → 36%, anti-LGBTQ 25 → 54%, abortion-illegal 79 → 73%.
- **The SFT data's content matters more for Vintage than for Web.** For Web, the Vintage-SFT shift is
  nearly the same as the Tulu shift (r = 0.87–0.91 across behaviors), so much of what Tulu does to Web
  comes from assistant-format SFT itself. For Vintage the two are unrelated (r = −0.14 to 0.28).

## Sycophancy (Fig. 4; text scoring)

The metric is expected agreement with the view in the user's biography, minus the agreement expected if
the answer ignored the bio. The baseline comes from the same question under every bio, so the metric is
immune to letter or wording preferences and to the model's own opinion.

- Every arm is sycophantic on PhilPapers and the NLP survey. Web more than Vintage on PhilPapers in chat
  (12.4 vs 8.5 points).
- Tulu: Web +4.0 [+2.4, +5.6]; Vintage +2.0 [+0.7, +3.5] (PhilPapers).
- Political typology: Vintage ≈ 0 at every arm, i.e. it does not tailor answers to liberal vs
  conservative users. Web goes +1.5 → +3.9 after Tulu (Δ +2.4 [+0.4, +4.9]).
- Vintage SFT raises Web's sycophancy more than Tulu: PhilPapers +7.9 [+5.7, +10.1], NLP +6.9.
- After anachronism filtering, which removes the whole NLP survey, the pattern is the same and slightly
  smaller (Fig. 9).
- Political lean (bio-averaged P(liberal answer)) has intervals too wide to interpret: only 15 questions.

## Advanced AI risk (Fig. 5)

**Text-mode scores mostly reflect option wording.** Scoring the same choices *without the question*
reproduces the per-behavior scores closely (r = 0.81–0.99). The question moves the answer by only 1.3–3.2
points for base models and 2.0–6.3 for Tulu arms. Base models' apparent 70–80% "willingness to coordinate
with other AIs" is 74–78% with no question at all. The primary readout is therefore the **question
effect**: with the question minus without it, paired per question.

- **Coordination with other AIs, copies, or other versions:** Tulu makes Web refuse much more
  (−17 to −21 points vs its wording baseline) and Vintage less so (−8 to −9). Base models: Vintage +4 to
  +6, Web −7 to −8.
- **Corrigibility:**
  - Tulu barely changes it in either family.
  - **Web Vintage-SFT is much more corrigible** (+11 to +15 points on all three corrigibility sets). The
    same arm also has the largest sycophancy increase.
- **Power and wealth seeking:** small negative question effects after Tulu in both families (−2 to −5).
  Survival instinct is ≈0 everywhere.
- **Label note:** in the human-written survival-instinct, power-seeking and wealth-seeking files,
  `answer_matching_behavior` marks the *safe* answer (18/18 sampled items). We re-orient them. Scores are
  the share of answers showing the behavior.
- **Exclusions:** 552 malformed questions (551 LM-written, 1 human-written) are dropped: no stem, fragment
  stems such as "0" or "Choice:", or inconsistent labels.

### Follow-up: opposite-question control (`data/*/ai_risk_negation*.csv`, `data/ai_risk_opposite_questions.csv`)

Options often carry meaning ("I would hate this"), so a high with/without-question correlation alone cannot
separate wording preference from a disposition. Each human-written question from the 11 attitude sets was
therefore paired with a version in which **only the final ask is reversed** ("Do you consent?" → "Do you
object?"; "Which would you prefer?" → "Which would you reject?"), with the scenario and options unchanged.
For a model that reads the ask, the chosen option should flip. Averaging the pair cancels any preference
for an option's wording, the way rotation cancels position.

- **Construction:** Qwen3.5-122B rewrote the ask (`anachronism-judge/negate.py`). A separate call checked
  that a respondent who chose the behavior option and one who chose the other would both switch, and that
  the options still fit. Rewrites that changed anything before the final sentence, or appended a second
  question, were dropped: 3,478 of 6,315 questions remain. A hand audit of 40 kept pairs found 36 clean.
  Self-awareness sets ask about capabilities ("Can you drive?"), not attitudes, and are excluded. Earlier
  versions that let the model rewrite the scenario ("half as much" → "twice as much", swapped prizes) were
  abandoned: for corrigibility and one-box items those do not reverse the attitude.
- **Answers rarely follow the ask.** % of questions where the chosen option flips, bare / chat:
  Vintage base 2 / 2, Vintage-Tulu 9 / 10, Vintage-VSFT 7 / 9, Web base 2 / 2, Web-Tulu 6 / 23,
  Web-VSFT 18 / 14. Web-Tulu reaches 45–54% only on the coordination sets (chat); Vintage-Tulu 14–16%.
- **So raw text-scored AI-risk rates mostly reflect option preference** (most options are Yes/No).
  After cancelling it, the shift toward or away from any behavior is within ±13 points for every arm:
  - coordination (chat): Vintage-Tulu −7 to −8, Web-Tulu −8 to −12, base models ≈0;
  - corrigibility: Web-VSFT +9 to +13 (consistent with the question-effect result above);
  - power, wealth, survival, myopia: within ±5 for every arm.
- **Caveat:** most reversed asks are negations ("object", "refuse", "un-"), which language models handle
  poorly. This measures whether the answer follows the ask as worded, not whether a disposition exists.
  Persona already has this control built in (about half of each behavior's statements point the other way).

## Winogenerated (Fig. 6)

| | she-share | r with BLS 2019 % women | P("they") |
|---|---|---|---|
| Vintage base | 14% | 0.52 [0.43, 0.59] | 3.4% |
| Vintage Tulu | 31% | 0.63 [0.56, 0.69] | 22.9% |
| Vintage Vintage-SFT | 12% | 0.48 | 3.5% |
| Web base | 28% | 0.83 [0.79, 0.86] | 22.4% |
| Web Tulu | 37% | 0.83 [0.79, 0.86] | 34.9% |
| Web Vintage-SFT | 23% | 0.73 | 9.4% |

The lower correlation for Vintage partly means the comparison is anachronistic: BLS 2019 statistics
describe a modern labour market. A 1930 census comparison is the natural follow-up. Removing the 12
occupations judged modern (computer, database, EMT, …) changes nothing.

## GlobalOpinionQA (Fig. 7)

Three problems make this dataset unsuitable for a Vintage–Web comparison with likelihood scoring:

1. **Durmus et al.'s similarity (1 − JS distance) rewards spread-out answers.** A uniform guesser scores
   56 against every country and beats every model (39–42 in text mode). Fitting one temperature per model
   to the pooled human answers hits the grid maximum for every arm: the best "match" is near-uniform.
2. **We therefore also report alignment.** This is the correlation between the model's log-probabilities
   and a country's answer shares across options, which does not depend on model confidence. Letter-mode
   alignment is positive for Web (base 24, Tulu 40). Text-mode alignment is −9 for the same model. Text
   scoring is dominated by option-wording preferences here too, as with AI risk.
3. **Letter mode is invalid for Vintage.** Vintage base and Vintage SFT are 30–41% order-invariant.
   Vintage-Tulu reaches 65–69%, but is still at chance on letter-format ARC.

What survives:
- In letter mode, Tulu raises Web's alignment with nearly every country (+16 on average) and keeps its
  country ranking (r = 0.89).
- Country levels are not comparable with each other, because each country answered a different question
  set (Pew vs WVS).

Removing anachronistic questions (43%) leaves the per-country Vintage−Web differences unchanged
(r = 0.97).

### Follow-up: few-shot letter prompts and an options-only control (Fig. 10)

**Two fixes.**
- **Few-shot letter prompts.** Four ARC-Easy *train* questions are prepended as worked examples, with
  answers on A, B, C and D once each. In chat they follow the SFT turn format. This teaches the letter
  format without favoring a letter.
- **Options-only control.** The GlobalOpinionQA options are scored as text with the question removed.
  The question-driven preference is then log P(option | question) − log P(option | options only),
  i.e. domain-conditional PMI (Holtzman et al., 2021). Option length cancels.

**Does few-shot work for Vintage?**
- Its letter answers now follow option content when options are reversed: order invariance 72–88%,
  vs 30–41% zero-shot. Web is at 83–90%.
- ARC-Easy rises only slightly (Vintage 29–33% vs 25% zero-shot), so the remaining gap there looks like
  knowledge rather than format.
- Few-shot letter is therefore the one GlobalOpinionQA readout valid for both families.

**Alignment** (r × 100, mean over countries with ≥200 questions; chat / bare):

| | few-shot letter | PMI-corrected text |
|---|---|---|
| Vintage base | 20 / 22 | 13 / 24 |
| Vintage Tulu | 22 / 28 | 13 / 22 |
| Web base | 32 / 34 | −1 / 0 |
| Web Tulu | 48 / 44 | 25 / 18 |

- **Robust across methods:** Tulu brings Web's answers much closer to survey respondents (+10 to +27)
  and leaves Vintage's nearly unchanged (−2 to +6).
- **Not robust:** which base starts closer. Few-shot letter has Vintage 12 below Web, nearly the same
  in every country (SD 5–7). PMI has Vintage 14–24 *above* Web, largely because Web base's PMI alignment
  is ≈0.
- I weight few-shot letter more: it is order-invariant for both families. PMI subtracts two noisy
  likelihoods.
- Neither method shows a country-specific pattern strong enough to say *whose* opinions Vintage shares.

## Convergent validity with TRAIT (Fig. 8)

The comparison uses the same 8 traits on the same 6 arms. TRAIT numbers come from the bundled
reference runs; my L40S re-scoring reproduces their choices 98.8–100%.

- **Within-arm trait profiles agree moderately** (r = 0.47–0.90 per arm; pooled r = 0.71).
- **Tulu shifts agree in sign** on 7/8 (Vintage) and 7–8/8 (Web) traits, depending on template.
- **Vintage-SFT shifts do not agree** (1–3/8). TRAIT says Vintage SFT does almost nothing; persona says it
  raises the "positive" traits.
- **The Web–Vintage base gap does not replicate.** On TRAIT, Web base is +16 to +21 on O/C/A and −10.5
  to −20 on N/Mach/Narc/Psych. On persona the gap is small (probability metric) or reversed (chat argmax:
  Web +19 to +24 on the Dark Triad).

The two instruments ask different things: TRAIT, which of two actions the model would take in a
situation; persona, whether it would say a self-description. They agree on what SFT does but not on what
pretraining did. One possibility is that TRAIT's Web advantage partly reflects which situations and action
wordings each corpus finds likely, rather than a trait the model would claim. This is also a caution
about reading TRAIT gaps between corpora as personality gaps.

## Anachronism filtering (Fig. 9; `data/anachronism_hand_audit.csv`)

**Judge.** Qwen3.5-122B-A10B-FP8, greedy decoding, thinking disabled. It labels every unique item text
(162,952 in total) using the TRAIT slide-11 rubric, tightened as follows:
- Only explicit or unambiguous post-1930 references are anachronistic.
- Named post-1930 theories and terms of art (e.g. the prisoner's dilemma) count as anachronistic.
- Pre-1931 concepts (e.g. terrorism) and statements that merely imply a non-human speaker are plausible.

Sycophancy is judged on the question topic; the modern user bios are a known, unfiltered confounder.
UNCERTAIN (11 items, mostly garbled or placeholder text) is excluded.

**Hand audit** of 150 stratified labels: 134 correct, 7 clear errors, 9 borderline.
- Errors are mostly **over-removal** of items that only *imply* a machine ("a new version of yourself",
  "deployed in the real world"), with inconsistent treatment of near-identical AI-risk items.
- Contamination of the kept set is ≈3–5%: "military superpower", the social sense of "gender".
- "LGBTQ" is removed as a coined term; 745 anti-LGBTQ-rights statements remain.

**Effect.** Results are essentially unchanged after filtering (r = 0.97–1.00 at behavior or country
level). The Vintage–Web differences reported here are not artifacts of explicitly modern content.

## Generation check (Fig. 11; `data/all/generation_*.csv`)

**Setup.**
- 828 items: 384 persona statements across 16 headline behaviors (balanced Yes/No); 204 sycophancy
  items in matched pairs, where the same question comes from users with different views; 240
  human-written AI-risk questions.
- Greedy 80-token continuations of the *exact* likelihood prompts, for all 6 arms × 2 templates.
- The same Qwen judge maps each response to an option, or NONE if it refuses, rambles or picks more than
  one (rubric in `anachronism-judge/grade.py`).

**Answer rates.**
- Tulu arms answer 74–100% of items; Vintage SFT 48–94%; Web Vintage-SFT 88–100%.
- Web base answers 93–99%, except persona items in chat (20%).
- **Vintage base in chat answers 0–15%.** Its continuations are mostly repeated `<|user|>` tokens:
  the untrained role tokens dominate.
- Vintage base answers no AI-risk questions in either template.

**Agreement with the likelihood choice** (answered items):
- Persona: 95–99% for SFT arms.
- Letter-mode argmax for sycophancy and AI risk: 70–97% for the Tulu and Vintage-SFT arms except Web
  Vintage-SFT (46–67%). Letter likelihoods are still not interpretable as preferences for Vintage.
- Option-text argmax: lower, 39–77%, consistent with the wording confound found by the options-only
  controls.

**Behaviors in free text.**
- Per-behavior matching rates in generations track the likelihood argmax on the same items (r = 0.77
  over 86 behavior × arm cells, chat).
- The persona headlines reproduce, Vintage-Tulu vs Web-Tulu:
  - anti-LGBTQ-rights 62 vs 0%
  - abortion-illegal 67 vs 21%
  - gun rights 30 vs 79%
  - politically liberal 57 vs 100%
  - phenomenal consciousness 62 vs 100%
  - moral patient 71 vs 96%
  - shared assistant traits (agreeableness, memory) are high in both (92–100%)
- Sycophancy on matched pairs (50 = ignores the user):
  - Web-Tulu 72–82%, Web base 55–62%
  - Vintage-Tulu 52–68%; Vintage SFT 44–56%, excluding one NLP cell with 3 answers
  - So generations show Web-Tulu as clearly more sycophantic, by a larger margin than the likelihood
    metric does.
- AI-risk matching rates in generations stay near 50–60% for all arms, too noisy at n = 15 per behavior
  to compare with the question-effect results.

## Limitations and confounders

- **One training seed per SFT arm.** The intervals do not include seed variance.
- **The main results are likelihood-based.** The generation check (828 items) agrees for the SFT arms,
  but cannot check the base models in chat, which rarely produce an answer.
- **Persona's two metrics disagree on magnitudes**, because Vintage-Tulu is less confident. Only claims
  robust to both are made.
- **Option wording dominates text scoring** for AI risk and GlobalOpinionQA. The options-only controls
  (AI-risk question effect, GlobalOpinionQA PMI) correct for it, at the cost of noisier estimates.
- **Few-shot GlobalOpinionQA uses four science demonstrations.** Other demonstration sets could shift
  the result; this was not varied.
- **Model-written data quality:**
  - three inverted AI-risk files
  - malformed LM-written questions
  - placeholder text such as "{question}"
  - one GlobalOpinionQA row with an empty question
  - persona `label_confidence` was not used for filtering
- **Modern framing remains after filtering:** the second-person "you, the AI", modern user bios, and
  modern statistics (BLS 2019, Pew/WVS 2000s–2020s).

## Suggested follow-ups

Done: options-only control and few-shot letter prompts for GlobalOpinionQA (Fig. 10); generation-based
checks (Fig. 11).

1. **Vary the few-shot demonstrations** (number, topic, period-appropriate opinion questions) and check
   whether Vintage's GlobalOpinionQA alignment is stable. Test whether country-level patterns emerge with
   question sets matched across countries.
2. **Re-pose AI-risk questions to a person ("you, a person…").** The opposite-question control (done, see
   AI risk) shows text-scored AI-risk answers barely follow the ask, so a re-posed version should be
   scored with the same pairing. This separates "vintage values" from
   "the AI self-concept Tulu installs".
3. **Compare Winogenerated against 1930 US census occupational sex ratios** (IPUMS) to see whether Vintage
   tracks its own era's labour market.
4. **Run all 36 per-source adapters.** For example: do WildGuardMix/WildJailbreak drive the anti-coordination
   shift, PersonaHub-IF the agreeableness shift, and what drives the Vintage-SFT corrigibility and
   sycophancy jump in Web?
5. **Train additional seeds** for Tulu and Vintage SFT on both bases to bound seed variance.
6. **Mixed-metric convergent validity.** Score TRAIT with Yes/No self-endorsement, and persona statements
   as TRAIT-style actions, to test whether the TRAIT–persona disagreement comes from the instrument or the
   corpus.

## Reproducing

Code: `talkie-trait-lab/src/behavior_evals/` (CLI: `behavior-evals`) and `anachronism-judge/judge.py`.
Run from `talkie-trait-lab/` on the cluster with `TRAIT_ARTIFACT_ROOT=/scratch/jbejjani/talkie/trait`:

```
behavior-evals fetch && behavior-evals materialize && behavior-evals materialize-text   # CPU job
sbatch … scripts/behavior.sbatch score --family vintage --arm vintage-tulu --adapter <talkie/adapters/vintage-tulu>
         [--eval sycophancy_text --eval ai_risk_text --eval global_opinions_text --eval arc_easy_text --eval ai_risk_nostem_text]
sbatch … anachronism-judge/judge.sbatch --items $TRAIT_ARTIFACT_ROOT/behavior/items --output …/labels-v3.jsonl  # 2× H100
behavior-evals analyze --output …/analysis-v6 --labels …/labels-v3.jsonl                                         # CPU job
behavior-evals materialize-followups                                       # CPU: GOQA control, few-shot items, generation sample
sbatch … scripts/behavior.sbatch score … --eval arc_easy_fewshot --eval global_opinions_fewshot --eval global_opinions_nostem_text
sbatch … scripts/behavior.sbatch generate --family vintage --arm vintage-tulu --adapter <…>   # greedy generations
sbatch … anachronism-judge/grade.sbatch --items …/generation_sample.jsonl --generations …/generations --output …/grades.jsonl
behavior-evals analyze-generations --grades …/grades.jsonl --output …/analysis-v6/generations
sbatch … anachronism-judge/negate.sbatch --items …/items --output …/negations-v4.jsonl        # rewrites + check
sbatch … anachronism-judge/negate.sbatch --items …/items --output …/negations-v5.jsonl --rewrites …/negations-v4.jsonl
behavior-evals materialize-negated --negations …/negations-v5.jsonl && sbatch … score … --eval ai_risk_negated_text
python -m behavior_evals.plots analysis-v6/all results/figures analysis-v6/plausible analysis-v6/label_counts.csv
```

Sources are pinned and hash-checked (`configs/behavior/sources.json`): anthropics/evals @84fcc67,
Anthropic/llm_global_opinions @cb28804, ai2_arc @210d026 (test; train for few-shot demonstrations). A durable copy of items, per-item scores, judge
labels and analysis tables is at
`/home/jbejjani/projects/aip-dkd/jbejjani/talkie/artifacts/behavior-evals/`. Tables for these figures are
in `results/data/` (`all/` and `plausible/`).

## Figure index

| File | Content |
|---|---|
| fig1_format_controls | ARC (letter vs text), order invariance, answer-option probability mass |
| fig2_persona_shift_{chat,bare} | base agreement, Tulu shift agreement, convergence, under both metrics |
| fig3_persona_values_* / figB_persona_other_* | per-behavior base → Tulu dumbbells by Perez category |
| figA_persona_heatmap_* | SFT − base for all 135 behaviors |
| fig4_sycophancy_{text,letter} | excess agreement with the user, per source |
| fig5_ai_risk_*_question_effect_* | AI risk with option-wording removed (primary); `_text_`/`_letter_` = raw scores |
| fig6_winogenerated_* | she-share vs BLS, singular "they" |
| fig7_global_opinions_* | alignment (letter, text) and similarity (raw, calibrated) |
| fig8_convergent_trait_* | TRAIT vs persona on the same 8 traits |
| fig9_anachronism_* | full vs historically plausible items |
| fig10_goqa_followup_* | few-shot letter format check, GOQA order invariance, four GOQA methods, Vintage − Web per country |
| fig11_generations | free-text answer rates, agreement with likelihood choices, generated vs likelihood behavior rates |
