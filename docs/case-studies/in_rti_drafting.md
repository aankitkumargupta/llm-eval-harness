# Case study: drafting RTI applications for citizens (in_rti_drafting)

## The use case

The Right to Information Act, 2005 gives any citizen the right to ask a
public authority for its records, and the authority thirty days to reply.
The right is used millions of times a year. It is also lost, quietly, when
an application goes to the wrong office, asks a question instead of naming
a record, forgets the fee, or leaves out the words that make the thirty-day
clock and the appeal route bite. Citizen help desks, legal-aid clinics and
common service centres already draft these by hand. A model that drafts
them well widens the door; one that drafts them badly closes it while
looking helpful.

## Who would run this

A legal-aid authority, a civil-society RTI help desk, a common service
centre operator, or a state portal that offers "draft my application" as a
service. The question is which model produces a draft a volunteer can send
with a glance, at what cost per draft.

## What a wrong answer costs

A draft addressed to the wrong authority is returned or transferred weeks
later. A draft that asks "why has my pension stopped?" gets "no such record"
where "the sanction order and the reason for stoppage recorded on the file"
gets the file. A draft missing the fee is rejected outright. None of these
failures looks like a failure to the citizen who cannot check it, which is
why the reference draft is built from a fixed list of required elements and
why completeness is scored on its own rather than folded into a single
"quality" number.

## What good looks like

- The correct public authority, named as the office that holds the record.
- Two to four specific, record-shaped information points with a period.
- The standard elements present: citizenship, fee, Section 7(1) thirty
  days, Section 6(3) transfer, and a request to name any Section 8
  exemption and the first appellate authority.
- A first appeal under Section 19(1) when the situation is "I asked and
  heard nothing", not another fresh application.

## The metrics, and why

| metric | weight | why it matters here |
|---|---|---|
| accuracy | 0.45 | right office, right records, against the reference |
| completeness | 0.30 | every required element present |
| answer_relevance | 0.15 | this citizen's problem, not a generic template |
| cost_usd | -0.10 | negative by rule (I4) |
| latency_ms | measured | not weighted: a draft is not a live reply |

Fifty-one situations, English, Hindi and Hinglish, all fictional. References
generated from a template so the required elements are identical across
items.

## Caveats that travel with any result

- The judge is a model. Its agreement with an RTI practitioner on these
  drafts has not been measured. The three scores are the judge's opinion
  of agreement with a reference, and the reference is one good draft, not
  the only one.
- The correct office for a given problem varies by state; the references
  use generic designations (Tehsildar, Block Development Officer, District
  Supply Officer). A model naming the state-specific equivalent may be
  under-scored.
- Fifty-one items separate large differences. The paired test says when they
  cannot, and how many more would be needed.

## Results

Run `in_rti_drafting_live2` on Together, 5 models, 215 rows (baseline pass plus the latency lane). Numbers are the Profile report's for this run; the Case studies screen shows the same figures with their charts.

**Reading.** Drafting is where the judge's three scores separate: every model addresses the right kind of office and asks for records a PIO can actually locate (relevance 0.74 to 0.89), fewer carry every required element (completeness 0.66 to 0.81), and agreement with the reference draft is lowest of all (accuracy 0.62 to 0.72), partly because the references name generic offices and the models often name state-specific ones. The order is Kimi-K3, GLM-5.3-Flash, DeepSeek-V4-Flash, GLM-5.3, Qwen3.5-9B on all three scores, and none of the ten gaps survives the paired test at n=36; the smallest gap this set could see is about 0.11, and the three leaders sit within 0.03 of each other. So the decision is cost again: DeepSeek-V4-Flash and GLM-5.3-Flash draft for under a third of a cent, Kimi-K3 for three cents, and the bootstrap gives no reason to pay the difference. Two models spent their budget thinking: at 4,096 tokens Qwen3.5-9B returned nothing for four drafts in ten and GLM-5.3 for one in six, and their scores are pulled down by those empty rows (docs/DEBT.md R-30). Hindi and Hinglish situations were drafted in English, as the prompt asked, and score in line with English ones, though at four and three items per language those columns are a glance, not a finding.

**Weighted composite** (the profile's own weights; lower-is-better metrics negative):

| model | composite | accuracy | completeness | answer_relevance | cost_usd |
|---|---|---|---|---|---|
| Kimi-K3 | 0.698 | 0.722 | 0.806 | 0.894 | 0.03196 |
| GLM-5.3-Flash | 0.684 | 0.700 | 0.797 | 0.869 | 0.00286 |
| DeepSeek-V4-Flash-0731 | 0.683 | 0.694 | 0.797 | 0.878 | 0.00245 |
| GLM-5.3 | 0.635 | 0.650 | 0.731 | 0.831 | 0.01498 |
| Qwen3.5-9B | 0.586 | 0.617 | 0.661 | 0.736 | 0.00201 |

**accuracy with a 95% bootstrap interval** (paired items, n per model):

| model | mean | ci_low | ci_high | n |
|---|---|---|---|---|
| Kimi-K3 | 0.722 | 0.692 | 0.756 | 36 |
| GLM-5.3-Flash | 0.700 | 0.647 | 0.747 | 36 |
| DeepSeek-V4-Flash-0731 | 0.694 | 0.667 | 0.722 | 36 |
| GLM-5.3 | 0.650 | 0.578 | 0.719 | 36 |
| Qwen3.5-9B | 0.617 | 0.511 | 0.720 | 36 |

**accuracy by item language** (plain means with n; a reading aid, not a test):

| model | en (n) | hi (n) | hinglish (n) |
|---|---|---|---|
| Qwen3.5-9B | 0.603 (29) | 0.750 (4) | 0.567 (3) |
| DeepSeek-V4-Flash-0731 | 0.697 (29) | 0.725 (4) | 0.633 (3) |
| Kimi-K3 | 0.724 (29) | 0.750 (4) | 0.667 (3) |
| GLM-5.3 | 0.634 (29) | 0.750 (4) | 0.667 (3) |
| GLM-5.3-Flash | 0.707 (29) | 0.650 (4) | 0.700 (3) |

**Paired test** (paired_bootstrap, Holm-corrected across 10 pairs): 0 of 10 pairs separable.

- Qwen3.5-9B vs DeepSeek-V4-Flash-0731: not separable (diff -0.078, adjusted p 0.689, 36 paired items)
- Qwen3.5-9B vs Kimi-K3: not separable (diff -0.106, adjusted p 0.364, 36 paired items)
- Qwen3.5-9B vs GLM-5.3: not separable (diff -0.033, adjusted p 1.000, 36 paired items)
- Qwen3.5-9B vs GLM-5.3-Flash: not separable (diff -0.083, adjusted p 0.512, 36 paired items)
- DeepSeek-V4-Flash-0731 vs Kimi-K3: not separable (diff -0.028, adjusted p 0.512, 36 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3: not separable (diff 0.044, adjusted p 0.782, 36 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3-Flash: not separable (diff -0.006, adjusted p 1.000, 36 paired items)
- Kimi-K3 vs GLM-5.3: not separable (diff 0.072, adjusted p 0.312, 36 paired items)
- Kimi-K3 vs GLM-5.3-Flash: not separable (diff 0.022, adjusted p 1.000, 36 paired items)
- GLM-5.3 vs GLM-5.3-Flash: not separable (diff -0.050, adjusted p 0.689, 36 paired items)

**Power:** accuracy: 36 paired items, smallest reliably detectable gap ~0.109
- to detect a gap of 0.1: about 43 paired items
- to detect a gap of 0.05: about 171 paired items
- to detect a gap of 0.02: about 1068 paired items

**Cost per correct answer** (mean cost per item divided by the metric; the procurement number):

| model | cost_per_correct_answer (USD) |
|---|---|
| Qwen3.5-9B | 0.003262 |
| DeepSeek-V4-Flash-0731 | 0.003532 |
| Kimi-K3 | 0.044257 |
| GLM-5.3 | 0.023051 |
| GLM-5.3-Flash | 0.004087 |

**Truncation and hidden reasoning** (every model on the account thinks before it answers; reasoning tokens are billed inside completion tokens):

| model | n | truncated | truncation_rate | reasoning_rate | reasoning_tokens_mean | empty_answer_rate |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 40 | 18 | 0.45 | 1.00 | n/a | 0.42 |
| GLM-5.3 | 43 | 13 | 0.30 | 1.00 | 2324 | 0.16 |
| GLM-5.3-Flash | 43 | 6 | 0.14 | 1.00 | 1459 | 0.12 |
| Kimi-K3 | 43 | 2 | 0.05 | 1.00 | 1069 | 0.00 |
| DeepSeek-V4-Flash-0731 | 43 | 0 | 0.00 | 1.00 | 1140 | 0.00 |

**Errors (rows that ended in a provider error, all lanes):** Qwen3.5-9B: 3 of 43 rows (3 other). An errored row carries no score; the resumed rows below replaced them in the paired set.

**Decision at volume** (`harness decide --require "accuracy>=0.65" --optimise cost_usd --qpd 500`; the bar is on the point estimate, so read it beside the intervals above):

```
Constraints: accuracy >= 0.65
3/5 models qualify.
  OK  deepseek-ai/DeepSeek-V4-Flash-0731
  OK  moonshotai/Kimi-K3
  OK  zai-org/GLM-5.3-Flash

Ranked by cost_usd:
                             model  accuracy  cost_usd
deepseek-ai/DeepSeek-V4-Flash-0731  0.694444  0.002453
             zai-org/GLM-5.3-Flash  0.700000  0.002861
                moonshotai/Kimi-K3  0.722222  0.031963

>>> Recommended: deepseek-ai/DeepSeek-V4-Flash-0731

=== projected cost at 500 queries/day ===
                             model  cost_per_query  daily_usd  monthly_usd  annual_usd
deepseek-ai/DeepSeek-V4-Flash-0731        0.000536   0.267962     8.038858   97.806110
             zai-org/GLM-5.3-Flash        0.001076   0.538078    16.142333  196.398389
                   Qwen/Qwen3.5-9B        0.001132   0.565958    16.978750  206.574792
                   zai-org/GLM-5.3        0.013327   6.663728   199.911833 2432.260639
                moonshotai/Kimi-K3        0.029881  14.940583   448.217500 5453.312917

=== what extra quality costs (vs cheapest) ===
                             model  accuracy  monthly_usd  quality_delta  monthly_delta_usd  usd_per_point
                moonshotai/Kimi-K3  0.722222   448.217500       0.027778         440.178642     158.464311
             zai-org/GLM-5.3-Flash  0.700000    16.142333       0.005556           8.103475      14.586255
deepseek-ai/DeepSeek-V4-Flash-0731  0.694444     8.038858       0.000000           0.000000            NaN
                   zai-org/GLM-5.3  0.650000   199.911833      -0.044444         191.872975     -43.171419
                   Qwen/Qwen3.5-9B  0.616667    16.978750      -0.077778           8.939892      -1.149415
```

**Caveats specific to this run**

- The item set is the profile's test split (70 percent; run.yaml holds out 30 percent as a dev split even with tuning off), so n is smaller than the dataset.
- All five models are served as reasoning models on this account; their cost and latency include the hidden reasoning, which is what a buyer would pay.
- The one situation that calls for a first appeal rather than a fresh application (rti_43) fell into the 30 percent dev split and was not scored; whether a model chooses the appeal route is untested in this run
- Qwen3.5-9B truncated 45 percent of answers and GLM-5.3 30 percent even at 4,096 tokens; both are reasoning models whose thinking on a 300-word draft ran past the budget. The empty answers sit in the accuracy denominator (docs/DEBT.md R-30)
- The latency lane timed 8 items per model rather than 20 for this run (run.yaml latency_items lowered for the reasoning models, restored afterwards), so latency percentiles are coarse
- The references are English for every item and name generic offices (Tehsildar, Block Development Officer); a draft naming the state-specific equivalent may be under-scored by the judge
