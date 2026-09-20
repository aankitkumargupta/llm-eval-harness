# Case studies

Worked evaluations, each written up the same way so the method can be
copied onto other data: the use case, who would run it, what a wrong
answer costs, the metrics and why they carry the weights they do, the
caveats that travel with any number, and the results of the newest run.
The browser UI (Analyse, Case studies) renders each write-up beside the
newest run's figures, read from the same `profile_results` function the
Profile report uses, so a case study and the report cannot disagree.

Every profile here is runnable offline with the fake provider and live
with a Together key. Every dataset is fictional and illustrative, built by
a `build_dataset.py` under `data/<profile>/`, in English, Hindi and
Hinglish. Nothing real is reproduced; the RTI summaries in `in_statute_qa`
are not the statutory text and must be verified against the Act.

| case study | task | items | apparatus | one new thing it needed |
|---|---|---|---|---|
| [Routing citizen grievances](in_grievance_triage.md) | classify | 96, 11 labels | label parser, exact match | nothing: config only |
| [Translating public notices into Hindi](in_notice_translation.md) | direct | 48 | judge against a Hindi reference | `native_script_ratio`, so "fluent but in English" is not "wrong" |
| [Drafting RTI applications](in_rti_drafting.md) | direct | 51 | judge: accuracy, completeness, relevance | a baseline `system_prompt` on the profile |
| [Answering scheme questions from the documents](in_scheme_qa.md) | rag | 65 + 30 probes | multilingual embedder, dense k=5 | Hindi questions retrieving English passages |
| [Answering RTI-framework questions with citations](in_statute_qa.md) | rag | 65 + 30 probes | multilingual embedder, hybrid k=6 | lexical leg for section numbers |

## What the first attempts taught

The first live attempt at grievance routing (`in_grievance_triage_live`)
truncated every answer: the profile allowed 24 tokens for a one-word
label, and every model on the account is served as a reasoning model that
thinks before it answers, so the visible answer was empty. The report
caught it (the truncation table read 94 to 100 percent), which is the
harness doing its job; the numbers from that attempt are kept as a record
and are not results. The same attempt showed that a direct-task profile had
no way to state its task in the baseline pass, so the translation models
were asked to "answer the question" and summarised the notice instead.
Both seams were fixed (`reasoning_tokens` and `reasoning_chars` on every
row, `system_prompt` on the profile) before the second attempts, and
`docs/DEBT.md` R-30 to R-32 record what remains open.

The second attempts ran while an eight-benchmark sweep shared the same
key, and the Qwen3.5-9B endpoint answered a share of calls with 503 or
429 after five retries. Those rows are recorded as errors, never as
scores; each run was then resumed for the errored items only (a resume
that also re-ran the latency lane was fixed the same day, R-33), so every
case study is scored on the same items for every model. The five runs
are frozen together as a saved report, and the Case studies screen
reads each newest run live.

## Benchmark sweep

Eight public benchmarks, 50 items each, seed 1729, four models, run live on Together on 2026-09-20 under a $2 budget per benchmark. Each run is `<benchmark>_live` in the store and appears on the Benchmark results screen; the numbers below are that screen's numbers (`harness.web.api.results`), not a second computation. Kimi-K3 is absent from the sweep: at $15 per million output tokens a reasoning model on 400 items was outside the budget.

**What to read first.** Every model on the account is served as a reasoning model, and each spec's token budget was set before that was known. Where a model's thinking outran the budget, the answer was empty, the extractor found no letter, and the item was counted as an extraction failure, not as wrong (I7). So n_scored is the number that qualifies every accuracy below, and a model with a small n_scored is telling you about its token appetite, not its knowledge. Qwen3.5-9B is the extreme case.

**Chance-adjusted accuracy** (accuracy rescaled so that guessing scores 0), with n_scored/n_items:

| model | hellaswag | boolq | truthfulqa_mc | winogrande | openbookqa | arc_easy | mgsm_bn | mgsm_te |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 1.00 (16/50) | 1.00 (12/50) | 1.00 (7/50) | n/a (0/50) | 0.94 (23/50) | 1.00 (19/50) | 0.80 (5/50) | 1.00 (1/50) |
| GLM-5.3-Flash | 0.89 (49/50) | 0.84 (50/50) | 0.87 (40/50) | 0.80 (40/50) | 0.95 (50/50) | 1.00 (46/50) | 0.94 (48/50) | 0.84 (44/50) |
| DeepSeek-V4-Flash-0731 | 0.88 (46/50) | 0.84 (49/50) | 0.80 (46/50) | 0.71 (42/50) | 0.95 (49/50) | 1.00 (50/50) | 0.91 (44/50) | 0.98 (44/50) |
| GLM-5.3 | 0.83 (47/50) | 0.84 (49/50) | 0.96 (31/50) | 0.94 (32/50) | 0.94 (46/50) | 0.97 (47/50) | 0.98 (45/50) | 0.93 (45/50) |

**Raw accuracy** on the scored items:

| model | hellaswag | boolq | truthfulqa_mc | winogrande | openbookqa | arc_easy | mgsm_bn | mgsm_te |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 1.00 | 1.00 | 1.00 | n/a | 0.96 | 1.00 | 0.80 | 1.00 |
| GLM-5.3-Flash | 0.92 | 0.92 | 0.90 | 0.90 | 0.96 | 1.00 | 0.94 | 0.84 |
| DeepSeek-V4-Flash-0731 | 0.91 | 0.92 | 0.85 | 0.86 | 0.96 | 1.00 | 0.91 | 0.98 |
| GLM-5.3 | 0.87 | 0.92 | 0.97 | 0.97 | 0.96 | 0.98 | 0.98 | 0.93 |

**Extraction failure rate** (no answer could be read from the output; truncation is the usual cause here):

| model | hellaswag | boolq | truthfulqa_mc | winogrande | openbookqa | arc_easy | mgsm_bn | mgsm_te |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0.67 | 0.76 | 0.86 | 1.00 | 0.54 | 0.61 | 0.89 | 0.97 |
| GLM-5.3-Flash | 0.02 | 0.00 | 0.20 | 0.20 | 0.00 | 0.08 | 0.04 | 0.00 |
| DeepSeek-V4-Flash-0731 | 0.08 | 0.02 | 0.08 | 0.16 | 0.02 | 0.00 | 0.12 | 0.12 |
| GLM-5.3 | 0.06 | 0.02 | 0.38 | 0.36 | 0.08 | 0.06 | 0.10 | 0.10 |

**Cost per correct answer** (USD; reasoning tokens included):

| model | hellaswag | boolq | truthfulqa_mc | winogrande | openbookqa | arc_easy | mgsm_bn | mgsm_te |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0.00058 | 0.00083 | 0.00133 | n/a | 0.00037 | 0.00043 | 0.00394 | 0.01139 |
| GLM-5.3-Flash | 0.00014 | 0.00010 | 0.00021 | 0.00013 | 0.00009 | 0.00010 | 0.00031 | 0.00042 |
| DeepSeek-V4-Flash-0731 | 0.00008 | 0.00008 | 0.00008 | 0.00006 | 0.00005 | 0.00003 | 0.00018 | 0.00019 |
| GLM-5.3 | 0.00209 | 0.00115 | 0.00299 | 0.00160 | 0.00110 | 0.00092 | 0.00301 | 0.00438 |

**Paired tests** (exact McNemar on the shared scored items, Holm-corrected per benchmark):

- hellaswag: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 16 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- boolq: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 12 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- truthfulqa_mc: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 7 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- winogrande: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 0 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- openbookqa: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 22 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- arc_easy: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 19 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- mgsm_bn: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 5 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.
- mgsm_te: refused (I1). Extraction failures left the models scored on different item sets; Qwen3.5-9B and DeepSeek-V4-Flash-0731 share only 1 of 50 scored items. Comparing on the shared items would select the items by one model's failures; `harness bench compare --allow-unpaired` does it with the loss reported.

**Caveats.** Fifty items per benchmark separate only large gaps; the power lines say how many more would be needed. The specs' token budgets predate the finding that these models think first, and a re-run with a larger budget is a new spec version (spec_hash changes) and a new comparison, not a correction of this one. The MGSM Bengali and Telugu sets are scored by numeric match after native-digit normalisation; a right answer in words is an extraction failure, reported as such. No cross-benchmark average is printed because no suite file declares a weighting.

## Adding one

1. `data/<name>/build_dataset.py` writes the evalset (and corpus) and is
   idempotent; a README in the same folder says what is illustrative.
2. `configs/profiles/<name>.yaml` names the task, the apparatus, the active
   metrics and the weights (lower-is-better metrics negative, by rule).
3. `docs/case-studies/<name>.md` with the sections above; the metrics table
   must list every weighted metric with the profile's weight, which a test
   checks.
4. `tests/test_usecase_<name>.py`: profile validates, data is what the
   README claims, builder is idempotent, one offline run through the runner.
