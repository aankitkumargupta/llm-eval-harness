# DEBT

Known gaps against `CLAUDE.md`, each with an owning phase. A suppression or a
shortcut that is not listed here is a bug, not debt.

Updated 2026-09-19, after the correctness pass and the §10 benchmark subsystem.

---

## Closed

| # | Finding | Invariant | How |
|---|---|---|---|
| R-01 | `paired_values` dropped unpaired items silently | I1 | Raises `UnpairedItemsError` naming the items; `on_unpaired="drop"` / `--allow-unpaired` is the explicit, reported opt-out. `harness/report/stats.py:79-170` |
| R-02 | Missing usage block coerced to 0 tokens ⇒ $0.00 cost | I3 | `extract_usage()` raises `MissingUsageError`; opt-in estimation flags `usage_estimated=True` and the report surfaces the rate. `harness/clients/openai_compatible.py:36-95` |
| R-03 | No `RunManifest`; no run reproducible | I9 | `harness/store/manifest.py`, written before the first paid call and again at close, including on budget abort |
| R-04 | No `apparatus_hash` | I2 | `apparatus_hash()` + `RunManifest.comparable_with()`, which returns reasons rather than a bare bool |
| R-07 | No socket block | §5 | Autouse fixture in `tests/conftest.py`; `live` marker opts out |
| R-08 | Markers unregistered, `-m statistical` silently green | §5 | Registered in `pyproject.toml`; `addopts` now excludes `live` by default |
|, | §10 benchmark subsystem absent entirely | §10 | `harness/bench/`, spec, contracts, extraction, metrics, registry, runner, CLI, 3 adapters, 113 tests |

## Open

| # | Finding | Sev | Effort | Phase | Note |
|---|---|---|---|---|---|
| R-05 | No statistical simulation suite | BC | M | 5 | The `statistical` marker is registered but no test carries it yet. I5/I6 are asserted by unit tests, not by calibration simulation, false-positive rate under the null and empirical power at a known effect are still unmeasured. **This is the largest remaining correctness gap.** |
| R-16 | `minimum_detectable_effect` and `required_n` contradict each other at zero variance | BC | S | 5 | Found by building the dashboard. Full write-up and reproduction at the foot of this file. |
| R-17 | Ingest and sparse retrieval reach `huggingface.co`, an undeclared network dependency outside the provider layer | BC | M | 2 | Breaks §3 and the documented "embedded, no Docker, no ports" promise. Write-up below. |
| R-18 | Re-ingesting the same collection **doubles** it on embedded Qdrant, despite a docstring promising idempotence | BC | M |, | Every retrieval metric shifts and nothing says why. Write-up below. |
| R-19 | `configs/pricing.yaml` is stale against Together's live catalogue for 2 of 5 entries | BC | S | 6 | Cost is negatively weighted, so a stale price mis-ranks. Left unedited, §7 reserves pricing changes. Write-up below. |
| R-20 | A provider's `/v1/models` lists models the account cannot invoke | BC | S | 2 | 8 of 10 probed Together models returned "non-serverless". Listing is not access. Write-up below. |
| R-06 | Single-definition test for the composite | BC | M | 4 | `report/aggregate.py` is the only definition site by import structure, but the arithmetic at each of the four call sites was never diffed. |
| R-09 | Contracts live in `store/` ⇒ 16 upward imports | BM | S | 2 | Mechanical move to `harness/contracts/`. Unchanged by this pass. |
| R-10 | `TraceRow` not frozen, no `schema_version`, no pyarrow schema | BM | M | 2 | Blocked on `runner.py` building rows by mutation (~19 assignment sites). |
| R-11 | Canary plaintext persisted in `assembled_prompt` | BM | S | 6 | I12. Unchanged. |
| R-12 | Modules at 0% coverage | BM | M | 1/8 | Was 6; now 4. `rag/ingest.py` went 0% → 86% under `-m live` (the RAG integration suite). `report/html.py` (135 stmts), `orchestration/ui_runner.py` (134), `ui/components.py` (84) and `orchestration/jobs.py` (54) remain. |
| R-13 | Full §4 ruff ruleset, `src/` layout, uv, import-linter, Makefile, pre-commit | BM | L | 1 | Ruff passes the repo's **current** config; the §4 ruleset (`ANN,ARG,PTH,TRY,S,PD,NPY,PL`) still reports ~460 findings. Phase 1 in full was not attempted. |
| R-14 | No structlog / redaction processor | BM | M | 6 | `print` remains the only output channel. |
| R-15 | RNG seeded-locally not injected; clock unmockable | P | S | 1/3 | Unchanged. |

## New debt introduced by this pass

| Item | Why it was accepted | Owning phase |
|---|---|---|
| `MissingUsageError` inherits `RuntimeError`, not a `HarnessError` root | §4's single error hierarchy does not exist yet; creating it is its own concern and would have mixed two changes. Placed beside the contract it protects instead. | 2 |
| `UnpairedItemsError` inherits `ValueError` | Same reason. | 2 |
| `harness/store/manifest.py` lives in `store/`, not `contracts/` | It is written beside traces and `contracts/` does not exist yet. Moves with R-09. | 2 |
| `manifest.py` shells out to `git` | Provenance capture genuinely needs it, and §3's network rule concerns providers. Isolated to two functions with a documented `"unknown"` fallback. |, |
| Benchmark fixtures are synthetic, not sampled | §11 permits "a synthetic fixture of identical shape, clearly labelled" where redistribution is not clearly permitted, and §7 forbids vendoring dataset contents. Every record carries `"_synthetic": true`. | 7 |
| `bench fetch` is specified in §10.8 but not implemented | Fetching needs the checksum-manifest and licence-gate machinery of Phase 7 step 2. Specs ship pointing at local fixtures; `source.kind: hf` validates but has no fetcher behind it. | 7 |
| No sandbox, so no code-execution family | §10.5 is non-negotiable and substantial; shipping HumanEval without it would be worse than not shipping it. | 7 |
| Contamination probes (§10.6) not implemented | Spec fields parse and round-trip through `spec_hash`; nothing consumes them yet. | 7 |
| `loglikelihood` scoring mode validates but is not implemented | Only `generative` has a code path. A spec declaring `loglikelihood` passes validation and would then score nothing, should raise at adapter construction. | 7 |


---

## R-16, contradictory degenerate branch in the power calculation

Found by building the web dashboard: a profile report over two models that
agreed on every item printed *"smallest reliably detectable gap ~inf"*.

Reproduction:

```python
from harness.report.stats import minimum_detectable_effect, required_n
minimum_detectable_effect(29, 0.0)   # -> inf   "no gap is detectable"
required_n(0.05, 0.0)                # -> 0     "no items needed"
```

Both branches key off `std <= 0` and answer the same question in opposite
directions. `harness/report/stats.py` returns `inf` from one and `0` from the
other, and a report can print both lines at once.

The substantive reading is that `inf` is the wrong one: zero variance in the
paired differences means the models agreed on every item, so any genuine
constant difference would show immediately, the detectable gap is ~0, not
unbounded. As written, a run with perfect agreement is described as having *no
resolution at all*, which is the opposite of the truth and would push someone
to buy more eval items they do not need.

**Not fixed here, deliberately.** §9 Phase 5 says: "If a simulation shows a
method is mis-calibrated, report it with evidence and stop, do not silently
swap it", and §7 forbids changing a threshold or estimator to make something
read better. This is a degenerate-input inconsistency of exactly the kind
Phase 5 step 3's battery exists to enumerate (`zero variance` is named in it),
so it belongs to that phase with the rest of the battery and the simulation
that would confirm the fix. It is recorded here rather than patched.


---

## R-17, the RAG path is not offline

`configs/run.yaml` describes embedded Qdrant as "No Docker, no server, no
ports. This is the normal choice for a single-user local run, and it is what
the Streamlit app always uses." It is not offline.

Both `harness/rag/ingest.py:122` and `harness/rag/retrieve.py:133` hand Qdrant
`models.Document(text=..., model="Qdrant/bm25")`. The client resolves that
model through FastEmbed, which downloads it from `huggingface.co` on first
use, at **ingest and at every sparse or hybrid query**.

Caught by the socket block added as R-07. It is worth noting how: Phase 0's
network audit scanned the AST for direct `requests`/`httpx`/`urlopen` calls
and reported "No HTTP or subprocess call exists outside the provider layer".
That claim is **wrong**, and AST scanning could never have found this, the
call is three libraries deep. `docs/INVENTORY.md` §3 should be read with that
correction.

Consequences:

* §3 says the capability protocols are "the **only** code permitted to touch
  the network". `rag/` does, indirectly.
* On an air-gapped machine or in CI with no egress, the first ingest fails,
  inside `httpcore`, with a connection error that says nothing about a model
  needing to be downloaded.
* On Windows the download additionally emits `[WinError 1314] A required
  privilege is not held by the client` (symlink creation) before falling back
  to a copy. It recovers, but the log reads like a failure.

Options, none taken here: pin and vendor the BM25 model; compute sparse
vectors in-process behind an `Embedder`-style protocol so the network stays in
the provider layer; or declare the dependency honestly and fail fast with a
message naming it.

## R-18, re-ingesting a corpus silently doubles it

`Ingestor._ensure_collection` documents itself as *"Idempotent: recreates
cleanly so re-ingesting a profile is safe"* and does call `delete_collection`
before `create_collection`. On **embedded** Qdrant that delete does not purge
the on-disk records.

Reproduction (`chunk_size` in words; three one-chunk documents):

```
after 1st ingest: returned 3  count 3
  collection_exists -> True
  after explicit delete_collection, exists -> False
after 2nd ingest: returned 3  count 6
```

The delete reports success, `collection_exists` reports False, and the points
come back anyway. Note the returned count stays 3, the Ingestor believes it
wrote three chunks, so nothing in the run output hints at the duplication.

Why it matters: re-ingest is routine. You re-ingest after fixing a typo in the
corpus, after an interrupted ingest, after changing `chunk_size`. Duplicated
chunks then compete for the top-k slots, so hit-rate@k, MRR, NDCG, context
precision and context recall all move, and the report offers no reason.

Pinned by `tests/test_rag_integration.py::test_ingest_is_NOT_idempotent_in_embedded_mode`,
which asserts the *current* behaviour so the finding cannot regress unnoticed
(§7: describe current behaviour before changing it). When it is fixed that
test must fail and be rewritten to assert idempotence.

Not fixed here: the fix changes retrieval numbers for anyone who has
re-ingested, which is a results change and §7 reserves those for a human.


---

## R-19, pricing.yaml is stale against the vendor's live rates

Found by cross-checking `configs/pricing.yaml` against Together's own
`/v1/models` catalogue, which publishes a `pricing` block per model:

| model | pricing.yaml | Together, 2026-09-19 |
|---|---|---|
| `openai/gpt-oss-20b` | 0.05 / 0.20 | 0.05 / 0.20, match |
| `openai/gpt-oss-120b` | 0.15 / 0.60 | 0.15 / 0.60, match |
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | 0.30 / 0.30 | 0.30 / 0.30, match |
| **`Qwen/Qwen3.5-9B`** | **0.30 / 0.30** | **0.17 / 0.25** |
| **`meta-llama/Llama-3.3-70B-Instruct-Turbo`** | **0.88 / 0.88** | **1.04 / 1.04** |

The file's own header says to refresh prices before any real benchmark run and
warns that a stale price "does not merely mis-state dollars, it mis-ranks
models", because cost carries a negative weight in the composite. Both stale
entries are wrong in a direction that matters: Qwen is over-charged by ~43% on
input, Llama under-charged by ~15%.

**Not edited.** §7 lists a pricing entry among the things that change results
and reserves the change for a human. What was done instead: the two genuinely
new models used in the live run were **added** with the vendor's published
rate (attributable, retrieved, not guessed), and the live run deliberately
used only models whose price matches the catalogue exactly, so every cost
figure in that report is correct at today's rates.

`harness/clients/discovery.catalogue_pricing()` returns the vendor's block, so
refreshing is now a diff rather than a transcription exercise.

## R-20, listing a model is not being able to run it

Together's `/v1/models` returned 274 entries for this key. Probing them found
that **8 of the first 10 tried could not be invoked at all**, every one
failing with `400 Unable to access non-serverless model`.

Nothing in the catalogue payload distinguishes them. `running` reads `false`
for a working model and a dead one alike; there is no `serverless` flag, no
availability field. The only honest test is a request.

How it surfaced: a 540-call benchmark run in which two of three models errored
on *every single item*. The run completed, wrote 180 rows, produced a manifest
and would have shown two models at the bottom of a leaderboard, except that
I7 keeps errored items out of the accuracy numerator, so they showed as
excluded rather than as bad. That is the invariant doing exactly its job, and
it is the reason the failure was legible instead of being a quiet zero.

Mitigation shipped: `discovery.probe_model()` sends a one-token chat request
and reports whether the model answers. It is what the Connections screen should
call before a model is offered for a run, rather than trusting the catalogue.

## R-21: `max_tokens: 1024` silently rewrites the benchmark

**Severity: wrong-conclusion.** Open. Owning phase: 7 (spec), needs a human
decision per §7 before the value moves.

The three HF specs declare `prompt.decoding.max_tokens: 1024`. For reasoning
models that emit a long chain before the answer, that is not enough, and the
run truncated mid-thought:

| benchmark | model | truncation | excluded from accuracy |
|---|---|---|---|
| mmlu_pro_hf | DeepSeek-V4-Flash | 23.3% | 14 of 60 |
| mmlu_pro_hf | GLM-5.3-Flash | 13.3% | 8 of 60 |
| mmlu_pro_hf | gpt-oss-120b | 10.0% | 6 of 60 |
| gsm8k_hf | DeepSeek-V4-Flash | 6.7% | 4 of 60 |

The exclusion rates match the truncation rates exactly, so every excluded item
is a truncation, not a refusal, not an extraction failure.

**Why this is wrong-conclusion and not merely fragile.** I7 keeps a truncated
item out of the accuracy numerator, which is right: a cut-off answer is not a
wrong answer. But the item also leaves the *denominator*, and truncation is not
random, a model truncates on the items it thinks longest about, which are the
hard ones. So the model that truncates most has the easiest surviving item set
and posts the highest accuracy. On MMLU-Pro that inverts the ranking:

* Headline: DeepSeek 0.978 (46 scored) > GLM 0.962 (52) > gpt-oss-120b 0.907 (54).
* On the 44 items DeepSeek and gpt-oss-120b **both** answered: **zero**
  discordant pairs. They gave the same verdict on every shared item.

The entire seven-point gap is the denominator. I1 caught it: `paired_values`
raised `UnpairedItemsError` rather than intersecting, so the report carries a
refusal instead of a fabricated p-value. Without R-01's fix this would have
been reported as a win.

**Do not "fix" this by raising max_tokens quietly.** It is a decoding parameter
in a versioned spec: changing it changes `spec_hash`, makes every existing run
incomparable, and changes published numbers. §7 reserves that for a human.

The options, for that decision:

1. Raise `max_tokens` (2048–4096) and bump each spec's `version`. Costs more
   per item; makes the numbers comparable to published MMLU-Pro figures, which
   are generally not run at 1024.
2. Keep 1024 and treat truncation as the measured quantity, legitimate if the
   deployment really does cap output there, but then the report must lead with
   the truncation rate, not accuracy.
3. Score truncated items as wrong. **Rejected**, it violates I7 and buries a
   run-configuration artefact inside the accuracy column.

Whichever is chosen, the reporter should refuse to rank when exclusion rates
differ materially across models, rather than leaving it to a reader to notice.
That guard does not exist yet and is the follow-up work here.

## R-20 (extended), the apparatus was not invokable either

**Severity: blocks-correctness for RAG.** Closed by a new adapter; the account
limitation itself remains.

Probing the *embedder* the same way as the models under test: Together's
`/v1/models` listed exactly one embedding model on this account
(`BAAI/bge-base-en-v1.5`) and served neither it nor the profile's pinned
`BAAI/bge-large-en-v1.5`, both return `Unable to access non-serverless
model`. A RAG profile whose embedder cannot be served has no apparatus, and
the two shortcuts available were both dishonest: the hashed `fake:` embedder
makes dense retrieval noise that would have been reported as a RAG result,
and running "RAG" with `qdrant=None` writes context-free rows that read as
uniformly bad models (the web worker already refuses that, and rightly).

Resolution shipped: `harness/clients/local_embed.py`, a FastEmbed-backed
adapter registered as provider `local` through the ordinary seams, one
entry in `NON_OPENAI_PROVIDERS`, one branch in `build_client`, one priced
entry in `pricing.yaml`. It is embed-only (no `generate` method exists, so
it can never be a model under test), reports exact token counts from the
model's own tokenizer (`usage_estimated=False`), meters into the `embedding`
bucket at the priced rate of zero, and loads with `local_files_only=True`
once the weights are cached, recording on `fetched_from_network` if it ever
had to fetch. `regulated_qa` now pins `BAAI/bge-base-en-v1.5` (same family,
the size FastEmbed serves); `apparatus_hash` changed accordingly, so nothing
earlier is silently compared against it. Tests: `tests/test_local_embed.py`.

Still open: the one-time weight fetch reaches huggingface.co, the same
network touch R-17 records for BM25. It happens once per machine.

## R-22, the hosted adapter meters an unpriced embedder as free

**Severity: wrong-conclusion (cost).** Open. Owning phase: 6.

`harness/clients/openai_compatible.py:322-329`:

```python
try:
    usd = self.pricing.embedding_cost(model, tokens)
except KeyError:
    usd = 0.0
self.meter.record_embedding(usd, tokens)
```

An embedding model with no entry in `pricing.yaml` is metered at $0.00 for
every call. The generation path raises on the same gap; the embedding path
swallows it. Because cost carries a negative composite weight (I4), a run
whose embedder is unpriced under-reports total spend for every model
equally, which does not mis-rank models against each other, but does
mis-state the bill and defeats `harness estimate` versus actual. I3 says an
unpriced model is a hard validation failure, not a zero.

Not fixed here, because changing what a paid call costs is a results change
(§7). The local adapter deliberately does not copy this behaviour, its
`KeyError` propagates, and `tests/test_local_embed.py::test_an_unpriced_local_model_raises_rather_than_costing_zero`
pins that. The fix for the hosted path is the same one-line removal of the
`except`, plus a test that `harness validate` names the missing embedding
price before a run.

## R-23, embedded Qdrant is not thread-safe, and a lost race corrupts retrieval silently

**Severity: wrong-conclusion.** Closed (`harness/rag/retrieve.py`,
`tests/test_retrieval_race.py`).

Found by the first live five-model RAG run (`rag_live`). Nine items across
four models errored with two different strings: `dictionary changed size
during iteration` and `Dense vector bm25 is not found in the collection`,
that turned out to be one cause: `QdrantClient(path=...)` builds its BM25
model and per-collection search structures lazily and without locks, and the
harness queries it from eight worker threads. Reproduced offline, cold, in
under a second.

The part that matters more than the nine errors: **after a lost race the
process's sparse inference kept returning wrong lists without raising.** Every
remaining item's hybrid retrieval (sparse-weighted 1.5 to dense 1.0) ranked
junk above the gold passage. `hit_rate_at_k` came out at 0.16–0.23 with k=8
over 44 chunks, random is 8/44 = 0.18, and every model faithfully answered
"not in the context". The run scored at chance and looked like five bad
models. A direct, single-threaded query against the same collection put the
gold passage at rank 1 in all three modes.

A guard around only the first sparse inference was tried and the strengthened
test showed it insufficient. Every call into an embedded client is now
serialised under one process-wide lock (a query is microseconds; the LLM call
after it is seconds). Server-mode clients are not locked. The silent-wrong
variant could not be reproduced offline, a lost race stays loud there, so
the test asserts liveness cold and correctness warm, and says so.

## R-24, the local embedder had the same unlocked lazy load

**Severity: blocks-correctness (latent).** Closed (`harness/clients/local_embed.py`).

Written the same day as R-23 and caught by it: `_backend()` could be entered
by several threads at once, each constructing a `TextEmbedding`, and with
`lazy_load=True` the ONNX session was built inside whichever thread embedded
first while others were already calling into it. Construction is now under a
lock and eager; inference after that is lock-free. It was not the cause of
R-23's symptoms, the cached query vectors were checked against fresh ones
and all 64 matched, but it was the same bug waiting for a different day.

## R-25, per-row judge cost was a window on a shared meter

**Severity: wrong-conclusion (cost).** Closed (`harness/clients/cost.py`,
`harness/orchestration/runner.py`, `tests/test_cost_attribution.py`).

In `rag_live` the rows' `judge_cost_usd` summed to **$2.95** against a metered
**$0.38**, 7.9x, about `max_workers`. The runner attributed judge, embedding
and rerank spend as the change in the *shared* meter between two reads
around each item; with eight workers every row also counted its neighbours'
calls, and the excess landed on whichever rows were in flight rather than on
the models that spent it. The helper's docstring called this "an
apportionment". Rows summing to eight times the total is not one.

The run total was always right (the budget and the run-level cost report
read it). Everything built on the per-row column: `cost_usd`,
`cost_per_correct_answer`, the Pareto frontier, the cost panel, was not.

Fix: the meter keeps a per-thread ledger beside its totals. One item runs
entirely on one worker thread, so `begin_item()` / `take_item()` bracket
exactly that item's calls; every call lands in exactly one ledger, so rows
still sum to the run total (the property the delta was chosen for) and each
row is now its own spend. `rag_live` and the aborted `rag_live2` keep their
over-attributed rows, the store is append-only (I8), and their saved
report says so; `rag_live3` is the first run with correct per-row cost.

## R-26, the significance verdict prints its interval with the opposite sign

**Severity: cosmetic.** Open. Owning phase: 5.

`report/stats.py` renders a separable pair as, for example,
`moonshotai/Kimi-K3 is better by 0.096 [-0.180, -0.031]`. The magnitude and
the model named are right; the interval is on (a − b) with a the *worse*
model, so it reads negative while the sentence says "better by". A reader
who trusts the bracket over the prose gets the direction backwards. Fix is
to print the interval on (better − worse), or to name the direction of the
difference beside it. No number is wrong; the presentation is.

## R-27, reports do not label notional versus actual cost

**Severity: polish, with an honesty edge.** Open. Owning phase: 6.

Two cost figures exist by design (Phase 3 §3): the meter records what the
provider billed, and a cache-replayed row carries its list-price notional so
models remain comparable on cost regardless of which one happened to be
served from cache. On `rag_live3` that was $0.5516 metered against $0.7292
in rows, the gap being 130 replayed generations at $0.1840. Both are right.
Neither the saved report nor the profile report says which figure it is
showing, so a reader summing rows will believe the run cost 32% more than it
did. The saved report for `rag_live3` states it in a caveat; the layer
should state it in the table header.

## R-28, selecting a profile pooled every run of it

**Severity: wrong-conclusion.** Closed (`harness/web/profile_api.py`,
`tests/test_web.py::test_a_profile_report_shows_the_newest_run_and_never_pools`).

The profile report's "Profile" selector called `store.load_profile()`, which
returns every row ever written for that profile. On `regulated_qa` that
averaged the retrieval-corrupted attempt 1, the operator-aborted attempt 2
and the clean attempt 3 into one leaderboard: 0.690 for GLM-5.3 where the
clean run says 0.996. It also repeats every item id once per run, and the
paired test joins on item id. Runs differ in `apparatus_hash` and
`dataset_hash` (I2, I9) and are not poolable under any reading.

A profile now resolves to its newest run; the others are returned as
`other_runs` and named in a notice above the charts, so the reader knows
which attempt the numbers are and can pick another deliberately.

## R-29, charts froze the theme they were drawn in

**Severity: polish, with an honesty edge (unreadable numbers).** Closed
(`harness/web/static/charts.js`, `app.css`, `app.js`).

`tok()` read the page tokens once via `getComputedStyle` at draw time and
carried its own per-theme copy of the series steps. Toggle dark to light
and every chart kept its dark-theme fills: near-white labels, near-white
whiskers and dark gold on a white card. Two screenshots from the user showed
it. Chart colours are now `var(--...)` references applied as inline style,
the series steps live in `app.css` for both themes, and the page re-renders
on toggle. The test that pinned the old mechanism was rewritten to pin the
intent: token references present, every chart token defined in both
themes, no colour literal in chart code.

Same change set: `color-scheme` is declared per theme so the native
`<select>` popup is drawn dark in the dark theme, before, it was the OS's
white list with the page's near-white text on it, and the scatter's
"accuracy" axis title no longer sits on the top tick label.

## R-30, empty and truncated answers count as wrong in accuracy

**Severity: blocks-correctness (I7), needs a human decision.** Open.

The first live attempt at `in_grievance_triage` (run `in_grievance_triage_live`,
2026-09-20) truncated every answer: `max_tokens: 24` and all five models
think before they answer, so the visible answer was empty. The report
flagged it (truncation table, 94 to 100 percent per model), but the
accuracy column read 0.00 to 0.10 with n=68, not "no scorable answers":
an empty answer is scored as a wrong label and sits in the denominator.
I7 says failures are reported as their own rates, never folded into the
accuracy numerator; they are not in the numerator, but they are in the
denominator, and a reader who skips the truncation table sees a model
that "got everything wrong". Options: exclude rows with an empty or
truncated answer from the accuracy denominator and report `n_scored`
beside `n_items` (as the benchmark path already does), or keep the current
definition and print the truncation rate beside accuracy everywhere it
appears. Either changes a reported number; §7 says a human decides.
Owning phase: 5. The run is kept as an honest record and the case study
names it.

## R-31, no seam to switch hidden reasoning off per model

**Severity: blocks-maintainability (I1, I2).** Open.

Every model in the five-model comparison is served as a reasoning model
on Together: content is empty until the thinking ends, and the thinking
is billed inside `completion_tokens`. Probed 2026-09-20 with a one-word
classification prompt at 48 tokens: `Qwen/Qwen3.5-9B` honours
`chat_template_kwargs.enable_thinking=false` (answered "water" in 2
tokens); `deepseek-ai/DeepSeek-V4-Flash-0731` honours
`reasoning.enabled=false`; `moonshotai/Kimi-K3` honours
`thinking.type=disabled`; `zai-org/GLM-5.3` and `GLM-5.3-Flash` honour
none of the three. A per-model request-parameter seam in `models.yaml`
would let a profile evaluate the non-thinking mode a classifier would be
deployed in, but it must enter the cache key (a cached thinking answer
must not be replayed for a no-thinking run), the manifest, and the
per-model apparatus record, and it would be uneven across vendors. Until
then the profiles carry a `max_tokens` large enough for reasoning plus
answer, and every row records `reasoning_tokens` and `reasoning_chars`
so the report can say where the tokens went.

## R-32, one provider reports zero reasoning tokens while returning reasoning text

**Severity: polish (I3 honesty).** Recorded.

`Qwen/Qwen3.5-9B` on Together returns `usage.reasoning_tokens: 0` beside
a non-empty `message.reasoning`. `extract_reasoning` records None
(unknown) for that case rather than 0, so `reasoning_tokens_mean` in the
truncation table under-covers Qwen; `reasoning_rate`, computed from the
returned text, is the robust signal. Cost is unaffected: the tokens are
inside `completion_tokens`.

## R-33, resume re-ran the latency lane

**Severity: blocks-correctness (Phase 3 resume contract), closed.**

`LatencyPass.run` swallowed `done_keys` in `**_` while `BaselinePass`
honoured it, and the CLI passed `--resume` only to the baseline lane. Seen
live on `in_grievance_triage_live2` (2026-09-20): resuming to retry 13
rate-limited items cost $0.003 for the items and $0.066 for a duplicated
latency lane, and doubled that run's latency samples (108 extra rows,
which remain in the store; traces are append-only). Fixed in
`passes.py`, `orchestrator.py` and `main.py`, with
`tests/test_latency_resume.py` pinning row-set equality and the CLI wiring.
The UI runner and the web run path pass no resume id and are unaffected.

## R-34, every screen open loads the whole trace store

**Severity: blocks-maintainability (Phase 4 substrate), open.**

`profile_api.all_runs` calls `TraceStore.load_all()` and groups in pandas
to list runs; `/api/case-studies` calls it again. Measured 2026-09-20
with 17 runs in the store (about 6,000 rows over roughly 60 Parquet
parts, the parts multiplied by checkpointing and resumes): `/api/runs`
4.2 s, `/api/case-studies` 4.7 s, `/api/profiles` 0.2 s. The UI waits on
the first before painting, so every open shows "Loading" for 4 to 9 s.
`store.py` already has a DuckDB per-run summary query (one row per run:
when, profile, rows, errors); the listing should read that and add the
per-run model set and cost with one more aggregate, not materialise every
row. Not a correctness issue; recorded because it will get worse with
every run.
