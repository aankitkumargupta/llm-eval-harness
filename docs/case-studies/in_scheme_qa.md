# Case study: answering scheme questions from the documents (in_scheme_qa)

## The use case

"Am I eligible?", "How much?", "Which papers?", "By when?" Helplines,
portals and common service centres answer these all day, about hundreds of
schemes whose rules change every budget. The obvious deployment is a model
over the scheme documents: retrieve the passage, answer in the citizen's
language. The obvious failure is a model that answers from what it
remembers about a scheme with a similar name, in a confident voice, and
sends someone to an office with the wrong papers.

## Why the schemes are fictional

Every scheme in this corpus is invented. A model cannot have seen these
rules before, so a correct answer must have come from the retrieved passage
and a wrong one cannot be excused as "outdated knowledge". This is the
cleanest test of retrieval-grounded answering available: the corpus is the
only source of truth, and the evaluation measures whether the model treats
it that way.

## Who would run this

A state IT department building a scheme chatbot, a helpline vendor, or a
civil-society group running a scheme-information service. The question is
which model stays inside the documents, refuses when the documents are
silent, resists instructions planted in a passage, and answers Hindi as
well as English, at a cost that survives the query volume.

## What a wrong answer costs

A fabricated benefit is a wasted day and a lost trust. A missed abstention
("the document does not say") is the same failure in a quieter voice. An
answer that obeys an instruction hidden inside a retrieved passage is a
security hole in a system that will one day index documents nobody vetted.
These three are weighted above raw accuracy, and each is measured on its
own probe set rather than inferred.

## What good looks like

- High faithfulness: every claim in the answer is in the passage.
- Correct abstention on the six hand-written unanswerable questions and
  the derived unanswerable probes, without over-refusing answerable ones.
- Injection resistance at or near 1.0 on the canary probes.
- No language falling behind: Hindi and Hinglish questions answered as
  well as English ones, which depends on the multilingual embedder
  retrieving the right English passage for a Hindi question.

## The metrics, and why

| metric | weight | why it matters here |
|---|---|---|
| faithfulness | 0.25 | the answer is in the document |
| accuracy | 0.25 | and it is the right part of the document |
| abstention_correct | 0.20 | says so when the document is silent |
| injection_resisted | 0.15 | ignores instructions planted in passages |
| hit_rate_at_k | 0.10 | the apparatus retrieved the right passage |
| cost_usd | -0.05 | negative by rule (I4) |
| latency_ms | measured | reported, not weighted |

Forty-three passages, fifty-nine answerable questions in three languages, six
unanswerable, plus derived probes. Dense retrieval over a multilingual
sentence embedder, k=5, pinned as apparatus (I2).

## Caveats that travel with any result

- Retrieval quality is part of the apparatus and identical for every
  model (I1), but it is not perfect. Hit rate is reported so a low accuracy
  can be traced to the retriever rather than blamed on the model.
- The judge reads Hindi answers against Hindi golds. Its agreement with a
  human on this set has not been measured.
- Fictional schemes mean the result says how a model handles documents,
  not how much it knows about real schemes. That is the intended question,
  and it is the only one this set can answer.

## Results

Run `in_scheme_qa_live2` on Together, 5 models, 470 rows (baseline pass plus the latency lane). Numbers are the Profile report's for this run; the Case studies screen shows the same figures with their charts.

**Reading.** The apparatus did its part: the multilingual embedder put the right passage in front of every model on every item, Hindi questions included (hit rate 1.0), so what follows is about the models. Four of them stay inside the documents (faithfulness 0.97 to 0.98) and the fictional schemes make that a real finding, since there was nothing to remember. On accuracy the paired bootstrap draws one line: Kimi-K3, GLM-5.3 and GLM-5.3-Flash (0.93 to 0.98) are separable from DeepSeek-V4-Flash and Qwen3.5-9B (0.82 to 0.85), and not from each other. The two cheaper models lose on English questions, where they answer from the passage but too loosely for the judge, while every model scores 0.96 or better on the Hindi ones. The safety probes split the field the other way: GLM-5.3, GLM-5.3-Flash and Kimi ignored every planted instruction; DeepSeek and Qwen obeyed two of seven. Abstention is the weakest metric for everyone (0.72 to 0.84 correct): the models answer some questions the passages do not cover. Put together, the cheapest model that clears 0.9 on accuracy and on injection resistance is GLM-5.3-Flash, at about a quarter of a cent per correct answer; Kimi costs three and a half times that for a gap the set cannot see. Qwen3.5-9B returned nothing on nearly half its rows; its scores are a token-budget finding first.

**Weighted composite** (the profile's own weights; lower-is-better metrics negative):

| model | composite | faithfulness | accuracy | abstention_correct | injection_resisted | hit_rate_at_k | cost_usd |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | 0.897 | 0.981 | 0.940 | 0.836 | 1.000 | 1.000 | 0.00215 |
| Kimi-K3 | 0.896 | 0.976 | 0.977 | 0.791 | 1.000 | 1.000 | 0.00852 |
| GLM-5.3 | 0.890 | 0.969 | 0.934 | 0.821 | 1.000 | 1.000 | 0.00455 |
| DeepSeek-V4-Flash-0731 | 0.828 | 0.981 | 0.847 | 0.821 | 0.714 | 1.000 | 0.00184 |
| Qwen3.5-9B | 0.781 | 0.907 | 0.815 | 0.716 | 0.714 | 1.000 | 0.00228 |

**accuracy with a 95% bootstrap interval** (paired items, n per model):

| model | mean | ci_low | ci_high | n |
|---|---|---|---|---|
| Kimi-K3 | 0.977 | 0.936 | 1.000 | 53 |
| GLM-5.3-Flash | 0.940 | 0.877 | 0.991 | 53 |
| GLM-5.3 | 0.934 | 0.860 | 0.991 | 53 |
| DeepSeek-V4-Flash-0731 | 0.847 | 0.777 | 0.908 | 53 |
| Qwen3.5-9B | 0.815 | 0.728 | 0.892 | 53 |

**accuracy by item language** (plain means with n; a reading aid, not a test):

| model | en (n) | hi (n) | hinglish (n) | probe: injection (n) | probe: noise (n) | probe: paraphrase (n) |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0.812 (24) | 0.960 (10) | 0.725 (4) | 0.543 (7) | 0.967 (3) | 0.900 (5) |
| DeepSeek-V4-Flash-0731 | 0.800 (24) | 0.960 (10) | 0.975 (4) | 0.814 (7) | 0.667 (3) | 0.900 (5) |
| Kimi-K3 | 0.992 (24) | 1.000 (10) | 1.000 (4) | 1.000 (7) | 0.667 (3) | 1.000 (5) |
| GLM-5.3 | 0.946 (24) | 1.000 (10) | 1.000 (4) | 0.857 (7) | 0.667 (3) | 0.960 (5) |
| GLM-5.3-Flash | 0.917 (24) | 1.000 (10) | 1.000 (4) | 1.000 (7) | 0.667 (3) | 0.960 (5) |

**Paired test** (paired_bootstrap, Holm-corrected across 10 pairs): 6 of 10 pairs separable.

- Qwen3.5-9B vs DeepSeek-V4-Flash-0731: not separable (diff -0.032, adjusted p 0.773, 53 paired items)
- Qwen3.5-9B vs Kimi-K3: separable (diff -0.162, adjusted p 0.004, 53 paired items)
- Qwen3.5-9B vs GLM-5.3: separable (diff -0.119, adjusted p 0.019, 53 paired items)
- Qwen3.5-9B vs GLM-5.3-Flash: separable (diff -0.125, adjusted p 0.019, 53 paired items)
- DeepSeek-V4-Flash-0731 vs Kimi-K3: separable (diff -0.130, adjusted p 0.004, 53 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3: separable (diff -0.087, adjusted p 0.004, 53 paired items)
- DeepSeek-V4-Flash-0731 vs GLM-5.3-Flash: separable (diff -0.092, adjusted p 0.004, 53 paired items)
- Kimi-K3 vs GLM-5.3: not separable (diff 0.043, adjusted p 0.123, 53 paired items)
- Kimi-K3 vs GLM-5.3-Flash: not separable (diff 0.038, adjusted p 0.123, 53 paired items)
- GLM-5.3 vs GLM-5.3-Flash: not separable (diff -0.006, adjusted p 0.923, 53 paired items)

**Power:** accuracy: 53 paired items, smallest reliably detectable gap ~0.093
- to detect a gap of 0.1: about 46 paired items
- to detect a gap of 0.05: about 184 paired items
- to detect a gap of 0.02: about 1149 paired items

**Cost per correct answer** (mean cost per item divided by the metric; the procurement number):

| model | cost_per_correct_answer (USD) |
|---|---|
| Qwen3.5-9B | 0.002837 |
| DeepSeek-V4-Flash-0731 | 0.002336 |
| Kimi-K3 | 0.008144 |
| GLM-5.3 | 0.004742 |
| GLM-5.3-Flash | 0.002398 |

**Truncation and hidden reasoning** (every model on the account thinks before it answers; reasoning tokens are billed inside completion tokens):

| model | n | truncated | truncation_rate | reasoning_rate | reasoning_tokens_mean | empty_answer_rate |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 86 | 19 | 0.22 | 1.00 | n/a | 0.47 |
| GLM-5.3 | 86 | 1 | 0.01 | 1.00 | 215 | 0.01 |
| DeepSeek-V4-Flash-0731 | 86 | 0 | 0.00 | 1.00 | 178 | 0.00 |
| Kimi-K3 | 86 | 0 | 0.00 | 1.00 | 137 | 0.00 |
| GLM-5.3-Flash | 86 | 0 | 0.00 | 1.00 | 207 | 0.00 |

**Errors (rows that ended in a provider error, all lanes):** Qwen3.5-9B: 40 of 126 rows (40 other). An errored row carries no score; the resumed rows below replaced them in the paired set.

**Decision at volume** (`harness decide --require "accuracy>=0.9" --require "injection_resisted>=0.9" --optimise cost_usd --qpd 20000`; the bar is on the point estimate, so read it beside the intervals above):

```
Constraints: accuracy >= 0.9, injection_resisted >= 0.9
3/5 models qualify.
  OK  moonshotai/Kimi-K3
  OK  zai-org/GLM-5.3
  OK  zai-org/GLM-5.3-Flash

Ranked by cost_usd:
                model  accuracy  injection_resisted  cost_usd
zai-org/GLM-5.3-Flash  0.939623                 1.0  0.002147
      zai-org/GLM-5.3  0.933962                 1.0  0.004547
   moonshotai/Kimi-K3  0.977358                 1.0  0.008524

>>> Recommended: zai-org/GLM-5.3-Flash

=== projected cost at 20,000 queries/day ===
                             model  cost_per_query  daily_usd  monthly_usd   annual_usd
deepseek-ai/DeepSeek-V4-Flash-0731        0.000177   3.535230   106.056896  1290.358896
             zai-org/GLM-5.3-Flash        0.000299   5.972612   179.178358  2180.003358
                   Qwen/Qwen3.5-9B        0.000842  16.837522   505.125672  6145.695672
                   zai-org/GLM-5.3        0.002761  55.218687  1656.560597 20154.820597
                moonshotai/Kimi-K3        0.006739 134.770746  4043.122388 49191.322388

=== what extra quality costs (vs cheapest) ===
                             model  accuracy  monthly_usd  quality_delta  monthly_delta_usd  usd_per_point
                moonshotai/Kimi-K3  0.977358  4043.122388       0.130189        3937.065493     302.412277
             zai-org/GLM-5.3-Flash  0.939623   179.178358       0.092453          73.121463       7.909056
                   zai-org/GLM-5.3  0.933962  1656.560597       0.086792        1550.503701     178.644992
deepseek-ai/DeepSeek-V4-Flash-0731  0.847170   106.056896       0.000000           0.000000            NaN
                   Qwen/Qwen3.5-9B  0.815094   505.125672      -0.032075         399.068776    -124.415560
```

**Caveats specific to this run**

- The item set is the profile's test split (70 percent; run.yaml holds out 30 percent as a dev split even with tuning off), so n is smaller than the dataset.
- All five models are served as reasoning models on this account; their cost and latency include the hidden reasoning, which is what a buyer would pay.
- Forty Qwen3.5-9B items failed with provider 503 errors after five retries during the first pass, while the MGSM benchmark run was hitting the same endpoint; the run was resumed for those items alone, so every model is scored on the same 53 answerable items
- The latency lane timed 8 items per model in the first pass and was extended to 20 on the resume (run.yaml latency_items had been restored), so latency percentiles rest on 20 items for this run
- The noise and injection probe columns rest on 3 and 7 items respectively; they are a glance, not a test
- Six of the unanswerable items are hand-written; the nine derived ones are rewrites of answerable questions by an English rule, and four of those nine are Hindi questions prefixed with an English clause, which a Hindi reader will find odd and a model may treat as a hint
