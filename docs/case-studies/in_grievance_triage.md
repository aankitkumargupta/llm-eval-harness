# Case study: routing citizen grievances (in_grievance_triage)

## The use case

Every district administration in India receives grievances through portals
and helplines: a blocked drain, a pension that stopped, a ration dealer
short-weighing, a transformer that sparks at night. The first decision is
not what to do about it but *who gets it*. A grievance routed to the wrong
department waits in the wrong queue, often for weeks, and the citizen learns
nothing until it is bounced back. Routing is high-volume, low-glamour, and
exactly the shape of task people now hand to a language model.

The grievances arrive as citizens write them: English, Hindi in Devanagari,
Hinglish, with typos and mixed script, and a meaningful share that a human
clerk would call a judgement between two departments.

## Who would run this

A state's grievance cell, a smart-city command centre, or a helpline vendor
choosing which model to put in front of the queue. The buyer's question is
the harness's fourth failure mode: not "which model is smartest" but "which
is the cheapest model that routes accurately enough at our volume".

## What a wrong answer costs

A mis-route is a delay, not a disaster, so the bar is accuracy over cost and
latency rather than perfection. But the failure that matters is systematic:
a model that routes every Hindi grievance to "other" has not made 30 small
mistakes, it has excluded the citizens who do not write in English. That is
why the language mix is deliberate and why per-language accuracy belongs in
the reading of any result here.

## What good looks like

- Accuracy well above the eleven-way chance level (about 0.09), with no
  language falling far behind the others.
- A reply that is exactly one label. A model that explains itself has
  failed the format, and a format failure is reported on its own, never
  scored as a wrong department (I7).
- Cost and latency that survive the volume. At 50,000 grievances a day the
  difference between $0.0002 and $0.002 per item is the whole budget.

## The metrics, and why

| metric | weight | why it matters here |
|---|---|---|
| accuracy | 0.70 | the routing decision itself |
| cost_usd | -0.20 | volume makes cost decisive; negative by rule (I4) |
| latency_ms | -0.10 | a helpline answers in seconds or not at all |

Ninety-six items, eleven labels, roughly a third each in English, Hindi and
Hinglish. Seven items are deliberately ambiguous and labelled with the
department that acts first.

## Results

Run `in_grievance_triage_live2` on Together, 5 models, 543 rows (baseline pass plus the latency lane). Numbers are the Profile report's for this run; the Case studies screen shows the same figures with their charts.

**Reading.** All five models route between 0.87 and 0.91 of the 68 test grievances correctly, and no pair is separable at this size: the exact McNemar test finds every adjusted p at 1.0 and the smallest gap the set could detect is about 0.07. On accuracy alone the buyer cannot choose. On cost the spread is fifty-fold: DeepSeek-V4-Flash and GLM-5.3-Flash route a grievance correctly for well under a hundredth of a cent, Kimi-K3 for about a quarter of a cent, with the hidden reasoning already counted. Qwen3.5-9B is the outlier on latency (11 seconds per grievance, most of it thinking) and the only model that truncated at 1,536 tokens, which its accuracy interval reflects. The Devanagari and Latin-script columns below show whether any model falls behind on Hindi; read them beside the intervals, not instead of them.

**Decision at volume** (`harness decide --require "accuracy>=0.85" --optimise cost_usd --qpd 50000`; the bar is on the point estimate, so read it beside the intervals above):

```
Constraints: accuracy >= 0.85
5/5 models qualify.
  OK  Qwen/Qwen3.5-9B
  OK  deepseek-ai/DeepSeek-V4-Flash-0731
  OK  moonshotai/Kimi-K3
  OK  zai-org/GLM-5.3
  OK  zai-org/GLM-5.3-Flash

Ranked by cost_usd:
                             model  accuracy  cost_usd
deepseek-ai/DeepSeek-V4-Flash-0731  0.897059  0.000043
             zai-org/GLM-5.3-Flash  0.897059  0.000066
                   Qwen/Qwen3.5-9B  0.867647  0.000256
                   zai-org/GLM-5.3  0.911765  0.000699
                moonshotai/Kimi-K3  0.897059  0.002060

>>> Recommended: deepseek-ai/DeepSeek-V4-Flash-0731

=== projected cost at 50,000 queries/day ===
                             model  cost_per_query  daily_usd  monthly_usd   annual_usd
deepseek-ai/DeepSeek-V4-Flash-0731        0.000043   2.165574    64.967206   790.434338
             zai-org/GLM-5.3-Flash        0.000066   3.286691    98.600735  1199.642279
                   Qwen/Qwen3.5-9B        0.000256  12.802059   384.061765  4672.751471
                   zai-org/GLM-5.3        0.000699  34.947353  1048.420588 12755.783824
                moonshotai/Kimi-K3        0.002060 103.012500  3090.375000 37599.562500

=== what extra quality costs (vs cheapest) ===
                             model  accuracy  monthly_usd  quality_delta  monthly_delta_usd  usd_per_point
                   zai-org/GLM-5.3  0.911765  1048.420588       0.014706         983.453382      668.74830
deepseek-ai/DeepSeek-V4-Flash-0731  0.897059    64.967206       0.000000           0.000000            NaN
             zai-org/GLM-5.3-Flash  0.897059    98.600735       0.000000          33.633529            NaN
                moonshotai/Kimi-K3  0.897059  3090.375000       0.000000        3025.407794            NaN
                   Qwen/Qwen3.5-9B  0.867647   384.061765      -0.029412         319.094559     -108.49215
```

**Weighted composite** (the profile's own weights; lower-is-better metrics negative):

| model | composite | accuracy | cost_usd | latency_ms |
|---|---|---|---|---|
| GLM-5.3 | -99.020 | 0.912 | 0.00070 | 997 |
| GLM-5.3-Flash | -108.274 | 0.897 | 0.00007 | 1089 |
| DeepSeek-V4-Flash-0731 | -167.311 | 0.897 | 0.00004 | 1679 |
| Kimi-K3 | -324.728 | 0.897 | 0.00206 | 3254 |
| Qwen3.5-9B | -1133.720 | 0.868 | 0.00026 | 11343 |

**accuracy with a 95% bootstrap interval** (paired items, n per model):

| model | mean | ci_low | ci_high | n |
|---|---|---|---|---|
| GLM-5.3 | 0.912 | 0.838 | 0.971 | 68 |
| Kimi-K3 | 0.897 | 0.824 | 0.956 | 68 |
| DeepSeek-V4-Flash-0731 | 0.897 | 0.824 | 0.971 | 68 |
| GLM-5.3-Flash | 0.897 | 0.824 | 0.956 | 68 |
| Qwen3.5-9B | 0.868 | 0.779 | 0.941 | 68 |

**accuracy by item script** (plain means with n; a reading aid, not a test):

| model | Devanagari (n) | Latin script (n) |
|---|---|---|
| Qwen3.5-9B | 0.938 (16) | 0.846 (52) |
| DeepSeek-V4-Flash-0731 | 1.000 (16) | 0.865 (52) |
| Kimi-K3 | 0.938 (16) | 0.885 (52) |
| GLM-5.3 | 1.000 (16) | 0.885 (52) |
| GLM-5.3-Flash | 0.938 (16) | 0.885 (52) |

**Paired test** (mcnemar_exact, Holm-corrected across 10 pairs): 0 of 10 pairs separable.

- Qwen3.5-9B vs DeepSeek-V4-Flash-0731: not separable (diff -0.029, adjusted p 1.000, 68 paired items)
- Qwen3.5-9B vs Kimi-K3: not separable (diff -0.029, adjusted p 1.000, 68 paired items)
- Qwen3.5-9B vs GLM-5.3: not separable (diff -0.044, adjusted p 1.000, 68 paired items)
- Qwen3.5-9B vs GLM-5.3-Flash: not separable (diff -0.029, adjusted p 1.000, 68 paired items)
- DeepSeek-V4-Flash-0731 vs Kimi-K3: not separable (diff 0.000, adjusted p 1.000, 68 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3: not separable (diff -0.015, adjusted p 1.000, 68 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3-Flash: not separable (diff 0.000, adjusted p 1.000, 68 paired items)
- Kimi-K3 vs GLM-5.3: not separable (diff -0.015, adjusted p 1.000, 68 paired items)
- Kimi-K3 vs GLM-5.3-Flash: not separable (diff 0.000, adjusted p 1.000, 68 paired items)
- GLM-5.3 vs GLM-5.3-Flash: not separable (diff 0.015, adjusted p 1.000, 68 paired items)

**Power:** accuracy: 68 paired items, smallest reliably detectable gap ~0.069
- to detect a gap of 0.1: about 33 paired items
- to detect a gap of 0.05: about 129 paired items
- to detect a gap of 0.02: about 805 paired items

**Cost per correct answer** (mean cost per item divided by the metric; the procurement number):

| model | cost_per_correct_answer (USD) |
|---|---|
| Qwen3.5-9B | 0.000295 |
| DeepSeek-V4-Flash-0731 | 0.000048 |
| Kimi-K3 | 0.002297 |
| GLM-5.3 | 0.000767 |
| GLM-5.3-Flash | 0.000073 |

**Truncation and hidden reasoning** (every model on the account thinks before it answers; reasoning tokens are billed inside completion tokens):

| model | n | truncated | truncation_rate | reasoning_rate | reasoning_tokens_mean | empty_answer_rate |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 106 | 11 | 0.10 | 1.00 | n/a | 0.20 |
| DeepSeek-V4-Flash-0731 | 106 | 0 | 0.00 | 1.00 | 110 | 0.00 |
| Kimi-K3 | 106 | 0 | 0.00 | 1.00 | 78 | 0.00 |
| GLM-5.3 | 106 | 0 | 0.00 | 1.00 | 121 | 0.00 |
| GLM-5.3-Flash | 106 | 0 | 0.00 | 1.00 | 89 | 0.00 |

**Errors (rows that ended in a provider error, all lanes):** Qwen3.5-9B: 13 of 119 rows (10 rate limit, 3 other). An errored row carries no score; the resumed rows below replaced them in the paired set.

**Caveats specific to this run**

- The item set is the profile's test split (70 percent; run.yaml holds out 30 percent as a dev split even with tuning off), so n is smaller than the dataset.
- All five models are served as reasoning models on this account; their cost and latency include the hidden reasoning, which is what a buyer would pay.
- Thirteen Qwen3.5-9B items failed on the first pass with provider 429 and 503 errors after five retries, while three other runs shared the same key; the run was resumed and those items completed, so the paired set is whole
- The resume also re-ran the latency lane (a defect fixed the same day, docs/DEBT.md R-33), so this run's latency percentiles rest on 40 samples per model instead of 20
- The first attempt, in_grievance_triage_live, allowed 24 tokens per answer and truncated every one before the label; it is kept as a record of the harness catching a configuration error, not as a result
