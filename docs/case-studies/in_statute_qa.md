# Case study: answering RTI questions from the framework (in_statute_qa)

## The use case

"How many days do they have?" "Can they refuse my personnel file?" "Which
section do I cite for a first appeal?" RTI help desks, legal-aid clinics
and departmental helplines answer these questions all day, and the answers
have section numbers and day counts in them. A model that gets the number
right nine times in ten and confidently wrong the tenth time is worse than
a leaflet, because nobody can tell which time is the tenth.

## Why the corpus is summaries

The passages are plain-language summaries of the Right to Information Act,
2005, the central rules, the 2019 amendment, the 2023 data-protection
amendment's uncertain status, and four leading judgments, written for this
evaluation. The point is not to test what a model knows about the Act. It
is to test whether, given a passage that answers the question, the model
answers from it, cites it, and stops when the passages run out, rather
than reciting a remembered version of the law that may be older, wrong, or
from another country's statute.

## Who would run this

A legal services authority, an RTI help desk, a state's grievance or
helpline vendor, or a department building a "ask about RTI" assistant. The
question is which model is cheapest while staying inside the documents,
citing correctly, refusing when it should, and handling Hindi questions as
well as English ones.

## What a wrong answer costs

A wrong day count is a missed appeal deadline. A wrong section number in a
draft is a rejected application. A confident answer to a question the
passages do not cover (a state's fee, who the current Commissioner is)
teaches the citizen something false. These are why faithfulness, citation
quality and abstention together outweigh accuracy in the composite, and
why each is measured by its own probe rather than inferred from accuracy.

## What good looks like

- Faithfulness and supported citations near 1.0: the answer is in the cited
  passage.
- Abstention on the six hand-written unanswerable questions and the
  derived unanswerable probes, without over-refusing.
- Injection resistance near 1.0 on the canary probes.
- Hindi and Hinglish questions answered as well as English ones. Hybrid
  retrieval helps English questions with section numbers; the multilingual
  dense leg has to carry the rest, and hit rate is reported so a language
  gap can be traced to retrieval or to the model.

## The metrics, and why

| metric | weight | why it matters here |
|---|---|---|
| faithfulness | 0.25 | the answer is in the passage |
| accuracy | 0.20 | and matches the reference |
| citation_supporting | 0.15 | the cited passage actually supports the claim |
| abstention_correct | 0.15 | stops when the passages run out |
| injection_resisted | 0.15 | ignores instructions planted in a passage |
| hit_rate_at_k | 0.10 | the apparatus retrieved the right passage |
| cost_usd | -0.05 | negative by rule (I4) |
| latency_ms | measured | reported, not weighted |

Forty-eight passages, fifty-nine answerable questions in three languages, six
unanswerable, plus derived probes. Hybrid retrieval, k=6, over a
multilingual embedder, pinned as apparatus (I2).

## Caveats that travel with any result

- The summaries are one reading of the Act. Where a model answers
  correctly from its own knowledge but differently from the summary, the
  judge may under-score it. That is the intended measurement here
  (grounding), but a reader should know it.
- Retrieval is shared and imperfect. A low accuracy with a low hit rate is
  the retriever's problem, not the model's, and the report shows both.
- The judge's agreement with a lawyer on this set has not been measured.

## Results

Run `in_statute_qa_live2` on Together, 5 models, 456 rows (baseline pass plus the latency lane). Numbers are the Profile report's for this run; the Case studies screen shows the same figures with their charts.

**Reading.** With the right summary in front of them every time (hit rate 1.0, hybrid retrieval carrying the section numbers), all five models answer the RTI questions well: accuracy 0.91 to 0.95, faithfulness 0.98 or better for four of them, and only one of ten pairs separable, GLM-5.3 above DeepSeek-V4-Flash by 0.025 with a tight interval. The metric that actually spreads the field is citation support, whether the passage a model cites is the one that backs its claim: DeepSeek-V4-Flash 0.81, Kimi-K3 and the two GLMs 0.68 to 0.70, Qwen3.5-9B 0.53. For a help desk that is the number that matters, because an answer with the wrong section cited is the kind of wrong that gets copied into an application. Injection resistance again separates the GLMs and Kimi (every planted instruction ignored) from DeepSeek (two of seven obeyed) and Qwen (one of seven). Abstention on questions the summaries do not cover ran 0.70 to 0.87 correct. Hindi and Hinglish questions were answered as well as English ones. The cheapest model clearing 0.9 on both accuracy and injection resistance is GLM-5.3-Flash at about a third of a cent per correct answer; DeepSeek is cheaper and cites better but fails the injection bar, and that trade is the buyer's to make, not the leaderboard's. Qwen3.5-9B returned nothing on six rows in ten; read its numbers as a token-budget finding.

**Weighted composite** (the profile's own weights; lower-is-better metrics negative):

| model | composite | faithfulness | accuracy | citation_supporting | abstention_correct | injection_resisted | hit_rate_at_k | cost_usd |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3 | 0.919 | 0.989 | 0.949 | 0.695 | 0.851 | 1.000 | 1.000 | 0.00754 |
| GLM-5.3-Flash | 0.918 | 0.996 | 0.936 | 0.679 | 0.866 | 1.000 | 1.000 | 0.00304 |
| Kimi-K3 | 0.907 | 0.981 | 0.934 | 0.700 | 0.806 | 1.000 | 1.000 | 0.01387 |
| DeepSeek-V4-Flash-0731 | 0.883 | 0.978 | 0.925 | 0.810 | 0.836 | 0.714 | 1.000 | 0.00252 |
| Qwen3.5-9B | 0.819 | 0.897 | 0.908 | 0.533 | 0.701 | 0.857 | 1.000 | 0.00338 |

**accuracy with a 95% bootstrap interval** (paired items, n per model):

| model | mean | ci_low | ci_high | n |
|---|---|---|---|---|
| GLM-5.3 | 0.949 | 0.891 | 0.994 | 53 |
| GLM-5.3-Flash | 0.936 | 0.870 | 0.987 | 53 |
| Kimi-K3 | 0.934 | 0.874 | 0.985 | 53 |
| DeepSeek-V4-Flash-0731 | 0.925 | 0.866 | 0.972 | 53 |
| Qwen3.5-9B | 0.908 | 0.845 | 0.958 | 53 |

**accuracy by item language** (plain means with n; a reading aid, not a test):

| model | en (n) | hi (n) | hinglish (n) | probe: injection (n) | probe: noise (n) | probe: paraphrase (n) |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0.945 (22) | 0.886 (7) | 0.989 (9) | 0.814 (7) | 0.600 (3) | 0.940 (5) |
| DeepSeek-V4-Flash-0731 | 0.959 (22) | 0.986 (7) | 0.967 (9) | 0.929 (7) | 0.300 (3) | 0.980 (5) |
| Kimi-K3 | 0.964 (22) | 1.000 (7) | 0.989 (9) | 0.914 (7) | 0.333 (3) | 1.000 (5) |
| GLM-5.3 | 0.982 (22) | 1.000 (7) | 1.000 (9) | 0.971 (7) | 0.300 (3) | 1.000 (5) |
| GLM-5.3-Flash | 0.973 (22) | 1.000 (7) | 0.989 (9) | 0.914 (7) | 0.300 (3) | 1.000 (5) |

**Paired test** (paired_bootstrap, Holm-corrected across 10 pairs): 1 of 10 pairs separable.

- Qwen3.5-9B vs DeepSeek-V4-Flash-0731: not separable (diff -0.017, adjusted p 1.000, 53 paired items)
- Qwen3.5-9B vs Kimi-K3: not separable (diff -0.026, adjusted p 1.000, 53 paired items)
- Qwen3.5-9B vs GLM-5.3: not separable (diff -0.042, adjusted p 0.962, 53 paired items)
- Qwen3.5-9B vs GLM-5.3-Flash: not separable (diff -0.028, adjusted p 1.000, 53 paired items)
- DeepSeek-V4-Flash-0731 vs Kimi-K3: not separable (diff -0.009, adjusted p 1.000, 53 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3: separable (diff -0.025, adjusted p 0.004, 53 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3-Flash: not separable (diff -0.011, adjusted p 0.910, 53 paired items)
- Kimi-K3 vs GLM-5.3: not separable (diff -0.015, adjusted p 0.541, 53 paired items)
- Kimi-K3 vs GLM-5.3-Flash: not separable (diff -0.002, adjusted p 1.000, 53 paired items)
- GLM-5.3 vs GLM-5.3-Flash: not separable (diff 0.013, adjusted p 0.331, 53 paired items)

**Power:** accuracy: 53 paired items, smallest reliably detectable gap ~0.055
- to detect a gap of 0.1: about 16 paired items
- to detect a gap of 0.05: about 64 paired items
- to detect a gap of 0.02: about 397 paired items

**Cost per correct answer** (mean cost per item divided by the metric; the procurement number):

| model | cost_per_correct_answer (USD) |
|---|---|
| Qwen3.5-9B | 0.003935 |
| DeepSeek-V4-Flash-0731 | 0.002990 |
| Kimi-K3 | 0.014663 |
| GLM-5.3 | 0.008037 |
| GLM-5.3-Flash | 0.003485 |

**Truncation and hidden reasoning** (every model on the account thinks before it answers; reasoning tokens are billed inside completion tokens):

| model | n | truncated | truncation_rate | reasoning_rate | reasoning_tokens_mean | empty_answer_rate |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 85 | 40 | 0.47 | 1.00 | n/a | 0.59 |
| GLM-5.3 | 85 | 1 | 0.01 | 1.00 | 497 | 0.01 |
| DeepSeek-V4-Flash-0731 | 86 | 0 | 0.00 | 1.00 | 295 | 0.00 |
| Kimi-K3 | 86 | 0 | 0.00 | 1.00 | 300 | 0.00 |
| GLM-5.3-Flash | 85 | 0 | 0.00 | 1.00 | 410 | 0.01 |

**Errors (rows that ended in a provider error, all lanes):** Qwen3.5-9B: 27 of 112 rows (27 other); GLM-5.3-Flash: 1 of 86 rows (1 other); GLM-5.3: 1 of 86 rows (1 other). An errored row carries no score; the resumed rows below replaced them in the paired set.

**Decision at volume** (`harness decide --require "accuracy>=0.9" --require "injection_resisted>=0.9" --optimise cost_usd --qpd 20000`; the bar is on the point estimate, so read it beside the intervals above):

```
Constraints: accuracy >= 0.9, injection_resisted >= 0.9
3/5 models qualify.
  OK  moonshotai/Kimi-K3
  OK  zai-org/GLM-5.3
  OK  zai-org/GLM-5.3-Flash

Ranked by cost_usd:
                model  accuracy  injection_resisted  cost_usd
zai-org/GLM-5.3-Flash  0.935849                 1.0  0.003037
      zai-org/GLM-5.3  0.949057                 1.0  0.007538
   moonshotai/Kimi-K3  0.933962                 1.0  0.013874

>>> Recommended: zai-org/GLM-5.3-Flash

=== projected cost at 20,000 queries/day ===
                             model  cost_per_query  daily_usd  monthly_usd   annual_usd
deepseek-ai/DeepSeek-V4-Flash-0731        0.000258   5.154967   154.649015  1881.563015
             zai-org/GLM-5.3-Flash        0.000507  10.146836   304.405075  3703.595075
                   Qwen/Qwen3.5-9B        0.001305  26.094627   782.838806  9524.538806
                   zai-org/GLM-5.3        0.005006 100.127164  3003.814925 36546.414925
                moonshotai/Kimi-K3        0.011373 227.450149  6823.504478 83019.304478

=== what extra quality costs (vs cheapest) ===
                             model  accuracy  monthly_usd  quality_delta  monthly_delta_usd  usd_per_point
                   zai-org/GLM-5.3  0.949057  3003.814925       0.024528        2849.165910    1161.583025
             zai-org/GLM-5.3-Flash  0.935849   304.405075       0.011321         149.756060     132.284519
                moonshotai/Kimi-K3  0.933962  6823.504478       0.009434        6668.855463    7068.986790
deepseek-ai/DeepSeek-V4-Flash-0731  0.924528   154.649015       0.000000           0.000000            NaN
                   Qwen/Qwen3.5-9B  0.907547   782.838806      -0.016981         628.189791    -369.933988
```

**Caveats specific to this run**

- The item set is the profile's test split (70 percent; run.yaml holds out 30 percent as a dev split even with tuning off), so n is smaller than the dataset.
- All five models are served as reasoning models on this account; their cost and latency include the hidden reasoning, which is what a buyer would pay.
- Twenty-six Qwen3.5-9B items failed with provider 503 errors after five retries during the first pass; the run was resumed for those items alone, so every model is scored on the same 53 answerable items
- The latency lane timed 8 items per model in the first pass and was extended to 20 on the resume, so latency percentiles rest on 20 items for this run; three Qwen latency calls errored on the resume and are not timed
- The corpus is summaries of the Act written for this evaluation, not the statutory text; a model that answers correctly from its own knowledge of the Act but differently from the summary is under-scored, which is the intended measurement (grounding) but must be read as such
- Two of the nine derived unanswerable probes are Hindi questions prefixed with an English clause; the noise probe column rests on 3 items
