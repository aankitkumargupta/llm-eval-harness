# Case study: translating public notices into Hindi (in_notice_translation)

## The use case

Most state and municipal offices in India publish in two languages. A water
supply interruption, an exam schedule, a tax deadline, a tender, a transfer
order: each is drafted in English and must go out in Hindi (or the state
language) the same day. The translation is routine and high-volume, and it
is exactly the kind of work now handed to a language model, often by a
clerk who cannot check the output.

## Who would run this

A state information department, a municipal corporation's publication cell,
or a vendor supplying a translation service to one. The buying question is
the fourth failure mode: not "which model writes the best Hindi" but "which
is the cheapest model whose Hindi is faithful enough to publish unread".

## What a wrong answer costs

A notice is a promise with numbers in it. A date moved by a day, an amount
with a digit dropped, a sector number swapped: the citizen who acts on the
Hindi version is the one who pays. So fidelity to the reference is the main
score, and the notices are built so that every one carries a number, date
or name that must survive translation exactly.

There is a second, quieter failure. A model asked for Hindi that replies in
polished English, or in Hindi written in Latin script, has produced
something the office cannot publish. The judge, comparing meaning, may
still rate it well. That is why the share of the answer actually in
Devanagari is measured on its own (`native_script_ratio`), and why it
carries its own weight instead of being folded into accuracy.

## What good looks like

- Accuracy against the reference high enough that a clerk can publish
  without a line-by-line check; the judge scores meaning, not word choice.
- Native-script ratio at or near 1.0. A ratio well below that is a
  different failure from a mistranslation and is reported as such.
- Cost per notice low enough that translating every notice, not just the
  important ones, is affordable.

## The metrics, and why

| metric | weight | why it matters here |
|---|---|---|
| accuracy | 0.60 | says what the notice says, numbers intact |
| native_script_ratio | 0.25 | is actually publishable Hindi |
| cost_usd | -0.15 | volume; negative by rule (I4) |
| latency_ms | measured | not weighted: translation is batch work |

Forty-eight notices, 30 to 130 English words each, with plain-official-
Hindi references. All fictional.

## Caveats that travel with any result

- The judge is itself a model reading Hindi. Its agreement with a human
  translator on this set has not been measured; treat accuracy as the
  judge's opinion, and read the disagreement rate when a human subset is
  labelled.
- The references are one good translation each, not the only one. A model
  that renders a phrase differently but correctly may be under-scored.
- Forty-eight items separate large differences only. The paired test says
  how many more would be needed when it cannot.

## Results

Run `in_notice_translation_live2` on Together, 5 models, 268 rows (baseline pass plus the latency lane). Numbers are the Profile report's for this run; the Case studies screen shows the same figures with their charts.

**Reading.** This run is as much about token budgets as about translation. Kimi-K3 is the most faithful translator (judge accuracy 0.89 against the Hindi reference) and DeepSeek-V4-Flash the cheapest acceptable one (0.82 at about a nineteenth of Kimi's cost per notice), and both write almost entirely in Devanagari (native script ratio above 0.99). GLM-5.3 and Qwen3.5-9B did not fail at Hindi; they failed to stop thinking. At a 2,048-token budget Qwen produced no visible translation for any notice and GLM-5.3 for four in ten, so their accuracy numbers are mostly the judge scoring empty answers (docs/DEBT.md R-30) and their intervals should be read as a truncation finding, not a fidelity one. Among the three models that answered, the paired bootstrap does not separate Kimi from DeepSeek at n=34 (adjusted p 0.17), nor DeepSeek from GLM-5.3-Flash; it does separate Kimi from GLM-5.3-Flash. So the honest choice is between Kimi and DeepSeek, and at nineteen times the price the gap Kimi may or may not have is not one this set can see. The script ratio did its job in the other direction: no model that answered drifted into English or Latin-script Hindi, so the failure it was built to catch did not occur here.

**Weighted composite** (the profile's own weights; lower-is-better metrics negative):

| model | composite | accuracy | cost_usd | native_script_ratio |
|---|---|---|---|---|
| Kimi-K3 | 0.783 | 0.894 | 0.01752 | 0.995 |
| DeepSeek-V4-Flash-0731 | 0.741 | 0.821 | 0.00092 | 0.993 |
| GLM-5.3-Flash | 0.701 | 0.768 | 0.00137 | 0.964 |
| GLM-5.3 | 0.572 | 0.544 | 0.00816 | 0.986 |
| Qwen3.5-9B | 0.028 | 0.047 | 0.00081 | n/a |

**accuracy with a 95% bootstrap interval** (paired items, n per model):

| model | mean | ci_low | ci_high | n |
|---|---|---|---|---|
| Kimi-K3 | 0.894 | 0.874 | 0.909 | 34 |
| DeepSeek-V4-Flash-0731 | 0.821 | 0.741 | 0.900 | 34 |
| GLM-5.3-Flash | 0.768 | 0.661 | 0.874 | 34 |
| GLM-5.3 | 0.544 | 0.388 | 0.685 | 34 |
| Qwen3.5-9B | 0.047 | 0.000 | 0.124 | 34 |

**Paired test** (paired_bootstrap, Holm-corrected across 10 pairs): 8 of 10 pairs separable.

- Qwen3.5-9B vs DeepSeek-V4-Flash-0731: separable (diff -0.774, adjusted p 0.004, 34 paired items)
- Qwen3.5-9B vs Kimi-K3: separable (diff -0.847, adjusted p 0.004, 34 paired items)
- Qwen3.5-9B vs GLM-5.3: separable (diff -0.497, adjusted p 0.004, 34 paired items)
- Qwen3.5-9B vs GLM-5.3-Flash: separable (diff -0.721, adjusted p 0.004, 34 paired items)
- DeepSeek-V4-Flash-0731 vs Kimi-K3: not separable (diff -0.074, adjusted p 0.170, 34 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3: separable (diff 0.276, adjusted p 0.004, 34 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3-Flash: not separable (diff 0.053, adjusted p 0.560, 34 paired items)
- Kimi-K3 vs GLM-5.3: separable (diff 0.350, adjusted p 0.004, 34 paired items)
- Kimi-K3 vs GLM-5.3-Flash: separable (diff 0.126, adjusted p 0.034, 34 paired items)
- GLM-5.3 vs GLM-5.3-Flash: separable (diff -0.224, adjusted p 0.034, 34 paired items)

**Power:** accuracy: 34 paired items, smallest reliably detectable gap ~0.274
- to detect a gap of 0.1: about 256 paired items
- to detect a gap of 0.05: about 1021 paired items
- to detect a gap of 0.02: about 6379 paired items

**Cost per correct answer** (mean cost per item divided by the metric; the procurement number):

| model | cost_per_correct_answer (USD) |
|---|---|
| Qwen3.5-9B | 0.017264 |
| DeepSeek-V4-Flash-0731 | 0.001122 |
| Kimi-K3 | 0.019589 |
| GLM-5.3 | 0.014988 |
| GLM-5.3-Flash | 0.001780 |

**Truncation and hidden reasoning** (every model on the account thinks before it answers; reasoning tokens are billed inside completion tokens):

| model | n | truncated | truncation_rate | reasoning_rate | reasoning_tokens_mean | empty_answer_rate |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 53 | 53 | 1.00 | 1.00 | n/a | 1.00 |
| GLM-5.3 | 53 | 25 | 0.47 | 1.00 | 1624 | 0.42 |
| GLM-5.3-Flash | 53 | 8 | 0.15 | 1.00 | 1169 | 0.15 |
| DeepSeek-V4-Flash-0731 | 53 | 5 | 0.09 | 1.00 | 477 | 0.08 |
| Kimi-K3 | 53 | 3 | 0.06 | 1.00 | 941 | 0.06 |

**Errors (rows that ended in a provider error, all lanes):** Qwen3.5-9B: 3 of 56 rows (3 other). An errored row carries no score; the resumed rows below replaced them in the paired set.

**Decision at volume** (`harness decide --require "accuracy>=0.75" --optimise cost_usd --qpd 2000`; the bar is on the point estimate, so read it beside the intervals above):

```
Constraints: accuracy >= 0.75
3/5 models qualify.
  OK  deepseek-ai/DeepSeek-V4-Flash-0731
  OK  moonshotai/Kimi-K3
  OK  zai-org/GLM-5.3-Flash

Ranked by cost_usd:
                             model  accuracy  cost_usd
deepseek-ai/DeepSeek-V4-Flash-0731  0.820588  0.000921
             zai-org/GLM-5.3-Flash  0.767647  0.001366
                moonshotai/Kimi-K3  0.894118  0.017515

>>> Recommended: deepseek-ai/DeepSeek-V4-Flash-0731

=== projected cost at 2,000 queries/day ===
                             model  cost_per_query  daily_usd  monthly_usd   annual_usd
deepseek-ai/DeepSeek-V4-Flash-0731        0.000203   0.406914    12.207424   148.523653
                   Qwen/Qwen3.5-9B        0.000654   1.308035    39.241059   477.432882
             zai-org/GLM-5.3-Flash        0.000694   1.388303    41.649088   506.730574
                   zai-org/GLM-5.3        0.007658  15.315259   459.457765  5590.069471
                moonshotai/Kimi-K3        0.016762  33.523588  1005.707647 12236.109706

=== what extra quality costs (vs cheapest) ===
                             model  accuracy  monthly_usd  quality_delta  monthly_delta_usd  usd_per_point
                moonshotai/Kimi-K3  0.894118  1005.707647       0.073529         993.500224     135.116030
deepseek-ai/DeepSeek-V4-Flash-0731  0.820588    12.207424       0.000000           0.000000            NaN
             zai-org/GLM-5.3-Flash  0.767647    41.649088      -0.052941          29.441665      -5.561203
                   zai-org/GLM-5.3  0.544118   459.457765      -0.276471         447.250341     -16.177140
                   Qwen/Qwen3.5-9B  0.047059    39.241059      -0.773529          27.033635      -0.349484
```

**Caveats specific to this run**

- The item set is the profile's test split (70 percent; run.yaml holds out 30 percent as a dev split even with tuning off), so n is smaller than the dataset.
- All five models are served as reasoning models on this account; their cost and latency include the hidden reasoning, which is what a buyer would pay.
- One Qwen3.5-9B item (notice_19) failed with a provider error on the first pass; the run was resumed for that item alone, so every model is scored on the same 34 notices
- Qwen3.5-9B truncated every answer at 2,048 tokens and GLM-5.3 nearly half; both are reasoning models whose thinking exceeded the budget. A third attempt would need a larger budget or a no-thinking mode (docs/DEBT.md R-31), and the later use cases in this set were run at 4,096
- The judge scored empty answers as wrong rather than absent; those rows sit in the accuracy denominator (docs/DEBT.md R-30)
- The first attempt, in_notice_translation_live, used the generic task default as its system prompt and the models summarised the notices instead of translating them; it was stopped after 50 rows and is kept as a record
