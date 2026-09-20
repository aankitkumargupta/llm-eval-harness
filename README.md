# Evaluation Intelligence

A multi-model LLM evaluation harness.

A model-agnostic benchmarking suite that ranks LLMs for a specific workload against **accuracy, cost, latency, robustness and safety**, and then tells you whether the differences it found are real.

Upload a folder of PDFs and a spreadsheet of question/answer pairs, pick the models to compare, and get a leaderboard, a cost/latency Pareto frontier, statistical significance tests, adversarial probe results, and a constrained recommendation, from a browser UI, the command line, or a CI pipeline.

It runs against **any provider**: Together, OpenAI, Anthropic, Groq, Fireworks, OpenRouter, or a model you host yourself with vLLM / Ollama / LM Studio. The harness is network-bound and disk-bound rather than compute-bound, so it runs comfortably on an 8 GB machine, no model weights are ever loaded locally.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [What it measures](#what-it-measures)
- [Architecture](#architecture)
- [Running it locally](#running-it-locally)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Install](#2-install)
  - [3. Verify the install (no API key needed)](#3-verify-the-install-no-api-key-needed)
  - [4. Add a provider API key](#4-add-a-provider-api-key)
  - [5. Run the bundled demo end to end](#5-run-the-bundled-demo-end-to-end)
  - [6. Run the browser UI](#6-run-the-browser-ui)
  - [7. Run on your own data](#7-run-on-your-own-data)
  - [Where things get written](#where-things-get-written)
  - [Local run gotchas](#local-run-gotchas)
- [The eight questions it answers](#the-eight-questions-it-answers)
- [Case studies: Indian-government use cases](#case-studies-indian-government-use-cases)
- [Adversarial probes](#adversarial-probes)
- [Statistical significance](#statistical-significance)
- [The decision layer](#the-decision-layer)
- [CI regression gating](#ci-regression-gating)
- [Multi-provider setup](#multi-provider-setup)
- [Cost control](#cost-control)
- [Non-RAG workloads](#non-rag-workloads)
- [Preparing your own dataset](#preparing-your-own-dataset)
- [Data format](#data-format)
- [Configuration](#configuration)
- [Understanding the results](#understanding-the-results)
- [Project structure](#project-structure)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Design notes](#design-notes)
- [Roadmap](#roadmap)

---

## Why this exists

Most internal LLM evaluations fail in one of four ways, and none of them look like failure:

1. **They measure noise.** Someone runs 40 questions, sees 0.82 against 0.78, and ships. On 40 items that gap is comfortably inside the margin of error, and half the time it would flip on a re-run.
2. **They under-count cost.** Judge calls, embeddings and reranking are billed but not reported, so "cost per query" is a fraction of the real figure, and since cost is weighted negatively in any composite, the *ranking* is wrong too, not just the dollar amount.
3. **They never test the failure modes that matter.** Accuracy says nothing about whether a model will obey an instruction hidden in a retrieved document, fabricate an answer it should have refused, or leak data.
4. **They answer the wrong question.** A leaderboard says which model scored highest. Nobody gets to make that decision. The real one is *"cheapest model that clears our bar at our volume."*

This harness is built around those four gaps.

---

## Features

- **Provider-agnostic**: Together, OpenAI, Anthropic, Groq, Fireworks, DeepInfra, OpenRouter, and self-hosted vLLM / Ollama / LM Studio. Mix vendors in one matrix with a `provider:` prefix.
- **Three interfaces**, a full Streamlit app (upload → probe → run → decide), a read-only dashboard, and a scriptable CLI that exits non-zero on regressions.
- **Statistical significance**, paired bootstrap and exact McNemar tests, Holm-Bonferroni corrected, plus a power report that tells you how many more questions you need.
- **Adversarial probes**, prompt injection, abstention, retrieval noise, paraphrase consistency and lost-in-the-middle, generated automatically from your own evalset.
- **A decision layer**, constraint-based model selection, production cost projection, and the price of each extra quality point.
- **CI regression gating**: `main.py gate` fails the build when quality drops or cost rises, and only fires on differences that are statistically real.
- **Honest cost**, generation, judge, embedding and rerank billed separately from real usage blocks, with a hard budget ceiling that aborts a run cleanly.
- **Reliable**, jittered retries, optional rate limiting, incremental checkpointing and resume, so a six-hour run survives a rate limit at hour three.
- **Baseline + adapted passes**, measure out-of-the-box quality *and* how much each model improves under an equal, automatic tuning budget.
- **Hybrid retrieval**, dense, sparse (BM25) and RRF-fused hybrid from one Qdrant index.
- **Non-RAG tasks too**, classification and direct prompting reuse the whole apparatus without a vector store.

---

## What it measures

Metrics are grouped by the subsystem that owns them. A profile activates only the subset relevant to its workload.

**Retrieval quality**, hit-rate@k, MRR, NDCG@k, context recall, **context precision**, **average precision**, rerank lift (split into hit-rate delta vs. rank delta)

**Answer quality**, accuracy (`exact` / `contains` / `numeric` / **`token_f1`** / LLM-judge), faithfulness, answer relevance, completeness, **judge disagreement**

**Citations**, pointer validity, supporting validity, **citation density**, **citation recall**

**Robustness & security**, **prompt-injection resistance**, abstention correctness, **PII leakage**, **paraphrase consistency**, lost-in-the-middle

**Efficiency**, p50/p95 latency, **TTFT**, tokens, and cost split into **generation / judge / embedding / rerank**

**Diagnostics**, **error attribution by cause**, **truncation rate**, cache hit rate

**Cross-model analysis**, weighted composite, tuning gain, Pareto frontier, Elo + head-to-head win rates, bootstrap CIs, **paired significance tests**, **power analysis**, **judge calibration against human labels**

Bold entries are new or were previously declared but never computed.

---

## Architecture

Config is the spine; code is the engine. To add a profile, a model or a provider you edit YAML, never Python.

```
Upload PDFs + Q&A  ->  prepare_dataset  ->  corpus.jsonl + evalset.jsonl
                                                   |
                                    probes -> adversarial items appended
                                                   |
                                           ingest (embed -> Qdrant)
                                                   |
  run matrix:  model x profile x pass  ->  retrieve -> attack -> prompt
                                            -> generate -> score -> judge
                                                   |
                                    trace store (append-only Parquet)
                                                   |
        report | compare | decide | arena | gate | html
```

The load-bearing contracts are **`TraceRow`** (one flat row per eval item, the boundary between running and analysis) and the **provider protocols** (the only place that touches the network).

### Extension points

The structure follows SOLID where SOLID earns its keep. Each seam below exists because something concrete was hard to change before it.

| You want to… | Do this | Touch nothing else |
|---|---|---|
| Add a metric | Write a `Scorer` in `harness/eval/scoring.py`, add it to `DEFAULT_SCORERS` | `run_item` never changes |
| Add a provider | Write an adapter satisfying `TextGenerator` (plus `Embedder` / `DocumentReranker` if it can) | the runner, metrics and report layers |
| Add a pass | Write an `EvaluationPass` in `harness/orchestration/passes.py` | the orchestrator's other passes |
| Change pricing policy | Implement `PricingProvider` | the runner |
| Store traces elsewhere | Implement `TraceSink` (write) and `TraceReader` (read) | the passes and the arena |
| Change caching | Implement `KeyValueCache` | everything above it |
| Add a UI screen | Write a `Page` in `harness/ui/screens/`, add it to `default_registry()` | `app.py` and every other screen |

**Interface segregation is the one that fixed a real bug.** Provider capability used to be a boolean flag on a fat four-method interface: every adapter implemented `embed` and `rerank`, and the ones that couldn't just raised. That made `isinstance(client, Embedder)` answer `True` for a client that can never embed, and left the flag as the only real signal, a flag that shipped *wrong* for OpenRouter, which was declared embedding-incapable while its adapter handled embeddings fine. The capability protocols are now separate, adapters omit what they genuinely cannot do, and `supports()` requires the method and the declaration to agree, so the two cannot drift apart again.

Responsibilities are split so each piece has one reason to change:

```
orchestrator.py   assembles collaborators; delegates  (wiring only)
passes.py         BaselinePass / AdaptedPass / LatencyPass
collector.py      RowCollector, buffering, checkpointing, progress
arena.py          ArenaService, pairwise judging over stored answers
runner.py         run_item, one item through the pipeline
eval/scoring.py   the metric registry
```

`tests/test_solid.py` pins these seams. The load-bearing one defines a brand-new scorer *inside the test file* and asserts it participates fully: Open/Closed is a claim about what a future change costs, so the only honest check is to make that change and confirm nothing else moved.

---

## Running it locally

Everything runs on one machine. There is **no Docker requirement, no GPU, and no model weights are downloaded**, the harness calls hosted APIs (or a local server you already run) and does the retrieval, scoring and statistics itself.

### 1. Prerequisites

| | |
|---|---|
| **Python** | 3.11 or newer (`python --version`) |
| **RAM** | ~2-3 GB free during a run |
| **Disk** | ~1 GB for dependencies, plus your corpus and traces |
| **Docker** | **Not required.** Qdrant runs embedded, as a local folder |
| **API key** | One provider key: Together, OpenAI, Anthropic, Groq, …, *or* a local server (Ollama / vLLM / LM Studio) and no key at all |

### 2. Install

```bash
git clone <your-repo-url>
cd llm-eval-harness
python -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

> **Windows:** if activation is blocked with *"running scripts is disabled on this system"*, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Then install:

```bash
pip install -r requirements.txt
```

For tests and linting as well:

```bash
pip install -r requirements-dev.txt
```

`requirements.txt` includes the optional extras (Streamlit for the UI, `pdfplumber` for PDF ingestion, `anthropic` for Claude). To install only what you need, use the extras instead:

```bash
pip install -e ".[ui]"
```

```bash
pip install -e ".[all,dev]"
```

### 3. Verify the install (no API key needed)

The full test suite runs **offline against a fake provider**, so it costs nothing and needs no credentials. This is the fastest way to confirm the install is sound:

```bash
pytest
```

```
277 passed in 20.24s
```

Several commands also work with no key, because they touch no network:

```bash
python main.py validate --profile regulated_qa
```

```bash
python main.py estimate --profile regulated_qa
```

```bash
python main.py probes --profile regulated_qa --out probes.jsonl
```

`validate` checks your profile config, dataset quality and gold-passage labels; `estimate` forecasts a run's cost including judge calls; `probes` generates adversarial eval items. Until you do step 4, `validate` will also report *"Cannot construct provider 'together': No API key"*, that is its pre-flight check working correctly, not an install problem.

### 4. Add a provider API key

The shipped config is set up for **Together AI** and **OpenRouter**. Either one alone runs everything, including RAG, both serve embeddings, so either can build the vector index.

Create a `.env` file in the project root (it is gitignored) with whichever keys you have:

```
TOGETHER_API_KEY=your_together_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

Or export them into the shell instead:

```bash
export TOGETHER_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:TOGETHER_API_KEY = "your_key_here"
```

With both keys set you can compare cheap open-weight models against frontier ones in a single run, uncomment the `openrouter:` entries in `configs/models.yaml`. With only one key, everything still works:

| You have | What to change |
|---|---|
| Together only | Nothing, it is the shipped default. |
| OpenRouter only | Set `default_provider: openrouter` and the three `*_provider` keys, and point the profile's `embedding_model` at e.g. `baai/bge-m3`. The full block is at the bottom of `configs/models.yaml`. |
| Both | Nothing, add `openrouter:`-prefixed models to the `models:` list whenever you want them. |

Whichever you choose, confirm the routing before spending anything:

```bash
python main.py validate --profile regulated_qa
```

It constructs every provider, checks each capability you rely on, and names any problem, a missing key, a model routed to a provider that cannot serve it, a model with no price, before the first billable call.

Other providers use `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `FIREWORKS_API_KEY`, `DEEPINFRA_API_KEY`. See [Multi-provider setup](#multi-provider-setup) to mix several in one run.

**Running with no API key at all** is possible using a model you host yourself. With [Ollama](https://ollama.com) already running:

```yaml
# configs/models.yaml
default_provider: ollama
models:
  - "ollama:llama3.1:8b"
  - "ollama:qwen2.5:7b"
embedding_provider: ollama
judge_provider: ollama
judge_model: "ollama:llama3.1:8b"
```

Confirm the routing before you run:

```bash
python main.py validate --profile regulated_qa
```

### 5. Run the bundled demo end to end

The repo ships a tiny `regulated_qa` dataset (4 documents, 5 questions) so you can exercise the whole pipeline in about a minute for a few cents.

**First switch Qdrant to embedded mode** so nothing external is needed, uncomment this line in `configs/run.yaml`:

```yaml
qdrant_path: "workspace/qdrant"
```

Then:

```bash
python main.py validate --profile regulated_qa
```

```bash
python main.py estimate --profile regulated_qa
```

```bash
python main.py ingest --profile regulated_qa
```

```bash
python main.py run --profile regulated_qa --budget 1.00
```

```bash
python main.py report --profile regulated_qa
```

```bash
python main.py html --profile regulated_qa --out report.html
```

`--budget 1.00` is a hard ceiling: the run aborts cleanly if measured spend crosses it, keeping every row already written. Open `report.html` in any browser, it is a single self-contained file with no external assets.

> **The demo dataset is deliberately tiny**, so `report` will tell you the sample is too small to rank models confidently. That is the harness working as intended: with 5 questions almost nothing is statistically detectable. Use it to confirm the plumbing, then point a profile at real data.

If you would rather run a Qdrant **server** (worth it for a large corpus, or when several people share one index), leave `qdrant_path` commented out and start one:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

You can also override per command without touching config, using `--qdrant-path workspace/qdrant`.

### 6. Run the browser UI

**Sign-in.** The UI opens on a landing page and the product sits behind a sign-in: your name
and institutional function (attribution only), a role, and the shared pilot password. The password comes from
`HARNESS_PILOT_PASSWORD` in `.env` (see `.env.example`); when unset, the pilot default is
used and the console says so at start. It is hashed with PBKDF2 before comparison, sessions
are HttpOnly cookies that expire after twelve hours, and the role is enforced by the server:
an Evaluator can read everything, and only an Assurance Lead can start a run that spends the
provider key. This is a gate for a loopback-only tool, not user accounts, SSO or MFA; do not
expose the server beyond loopback on its strength.

**Product pages for stakeholders.** Once signed in, a sidebar on the left (`python main.py serve`)
groups the site into Product, Evaluations and Workspace. The Product pages (Capabilities,
Architecture, Trust and security, Roadmap, Deployment) are prose for a reader who wants to know
what the platform does before looking at numbers: each capability carries a Built / Partial /
Roadmap tag that mirrors the debt register, so nothing is described as working that is not. The
Evaluations group opens the Case studies, Evaluate and Saved reports screens; the Workspace group
lists every task-bar screen. The task bar and every existing screen are unchanged; the sidebar is
a second way in, collapsible per group, collapsible as a whole to a rail of icons (remembered per
browser), and a toggle on narrow windows. Signing out asks for confirmation first.


```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. The app always uses **embedded Qdrant** under `workspace/`, so it needs no configuration and no Docker.

A **persistent sidebar** carries the navigation, grouped by what each screen is for. Only the selected screen runs: Streamlit executes every tab body on every rerun, so the previous tabbed version did the work of six screens to show one.

- **Sidebar**, navigation, API key, which providers are actually reachable, a **budget meter**, and the active dataset.
- **Capabilities**, every feature the platform has, *discovered from the installed code* rather than written down: the provider capability matrix, all 27 metrics with what each one catches, the probe families, passes, task types and CLI commands, plus what this particular install has configured. Because it is introspected, it cannot drift out of date.
- **Overview**, the recommendation as a hero figure, headline KPIs, the cost/quality frontier, significance verdicts, projected spend, latency and safety.
- **Reports**, every metric, tuning gain, diagnostics, judge calibration, and the HTML/CSV export.
- **Data**, upload PDFs and a Q&A spreadsheet, then **Build dataset** → **Ingest**.
- **Probes**, generate injection / abstention / noise / paraphrase items. Free and instant.
- **Run**, pick models, read the cost estimate *including judge calls*, run in the background with a live progress bar.

The Results tab leads with the answer rather than the leaderboard. Set your quality bar with one slider and it tells you the cheapest model that clears it, what that costs per month at your volume, and, when nothing qualifies, exactly what each model missed by.

There is also a read-only dashboard over an existing trace store. It makes no API calls, so it is safe to leave open and refresh while a run completes:

```bash
streamlit run dashboard.py
```

#### About the charts

Both surfaces share one design system (`harness/ui/`), so they cannot drift apart. A few choices are deliberate and worth knowing when reading them:

| Chart | Encoding | Why |
|---|---|---|
| Cost/quality frontier | Accent blue for the frontier, gray for everything else | The frontier is the decision; the rest is context. Scatter is an all-pairs form where more than three hues cannot clear the colour-vision floors, so a colour-per-model scale would be unreadable, not just busy. |
| Significance matrix | Blue ↔ red diverging, neutral gray midpoint | Direction is the point. **Comparisons that failed the correction are forced to the neutral midpoint**, an uncorrected heatmap of raw deltas shows a confident pattern of colour where the data supports none of it. Shown only at 4+ models; below that the sentences are clearer. |
| Metric and cost bars | One colour for every bar | Colouring bars darker-where-bigger would encode length twice and burn the only free channel on information the bar already shows. |
| p50 / p95 latency | Two named series with a legend | Both are milliseconds on **one** axis. A second y-scale would invent a relationship the data doesn't contain. |
| All-identical metrics | A stat tile, not a chart | Three bars of zero is an empty plot pretending to be a finding. |

The app ships in a dark **Midnight** palette, a blue-tinted near-black with raised cards, matching the dashboard style it is built to. The palette was checked with a colour-vision validator rather than by eye, and re-validated against the navy surface rather than assumed to carry over from the neutral dark one: all-pairs CVD ΔE 9.4, normal-vision 20.9, every slot ≥3:1 contrast. Switch it in `.streamlit/config.toml`:

```toml
base = "light"
```

The charts follow automatically; `harness/ui/theme.py` reads the configured base and returns the palette stepped for that surface. Status colours (good / warning / critical) are fixed across both themes, never reused for a data series, and always paired with a word so nothing depends on colour alone.

### 7. Run on your own data

**Start in the browser.** The Evaluate screen (second button in the task bar) walks a new user through it: pick the kind of task (label, generate, answer from documents, or a public benchmark), see the data format with sample rows you can copy or download, have the profile written from a few choices and validated as you type, upload your own JSONL (checked line by line before anything is written), then follow the numbered screens: Preflight, Retrieval and Probes for RAG, Profile run, Profile report, Decide, Saved reports. The same commands are printed beside each step.

**Or from the terminal.** Turn PDFs and a Q&A spreadsheet into the JSONL the harness reads, with gold passages auto-labeled:

```bash
python prepare_dataset.py --pdf-dir ./my_pdfs --qa-file ./my_questions.xlsx --out-dir ./data/my_eval --question-col Question --answer-col Answer --chunk-size 200 --overlap 40
```

Copy a profile as a starting point and point it at the new files:

```bash
cp configs/profiles/regulated_qa.yaml configs/profiles/my_eval.yaml
```

Edit `name`, `corpus_path`, `evalset_path`, `chunk_size` and `overlap` in that file, then:

```bash
python main.py validate --profile my_eval
```

```bash
python main.py probes --profile my_eval --append
```

```bash
python main.py ingest --profile my_eval
```

```bash
python main.py run --profile my_eval --budget 25.00
```

```bash
python main.py report --profile my_eval
```

Already have JSONL? Skip `prepare_dataset.py`, see [Data format](#data-format).

A run that dies partway can be resumed rather than restarted, so nothing already paid for is lost:

```bash
python main.py runs
```

```bash
python main.py run --profile my_eval --resume RUN_ID
```

### Where things get written

Everything the harness creates stays inside the project and is gitignored:

| Path | What | Safe to delete? |
|---|---|---|
| `workspace/` | UI-created datasets, embedded Qdrant index, traces, cache | Yes, resets the app entirely |
| `runs/traces/` | CLI trace store (append-only Parquet parts) | Yes, but you lose past results |
| `.cache/` | Cached model, judge and embedding responses | Yes, but re-runs will re-bill |
| `report.html` | Generated report | Yes |

`.cache/` is the one to be careful with: the cache is what makes re-reporting free and results reproducible.

To start completely fresh:

```bash
rm -rf workspace runs .cache
```

### Local run gotchas

| Symptom | Fix |
|---|---|
| `Connection refused` on `localhost:6333` during ingest | Qdrant is in server mode with no server running. Set `qdrant_path` in `run.yaml`, or pass `--qdrant-path workspace/qdrant`, to use embedded mode. |
| First hybrid/BM25 run pauses to download files | Expected once: `fastembed` fetches a small BM25 model. Later runs are instant. Set `retrieval_mode: dense` to skip it entirely. |
| `ModuleNotFoundError: harness` | The virtual environment isn't active, or you're not in the project root. `pytest` and `python main.py` both expect the repo root as the working directory. |
| `No module named streamlit` | `pip install -r requirements.txt`, or `pip install -e ".[ui]"`. |
| Lots of `rate_limit` errors | Lower `max_workers` in `run.yaml`, or set a per-provider `rate_limit` in `models.yaml`. Pacing is cheaper than backing off. |
| Windows `UnicodeEncodeError` while printing a report | Shouldn't happen, the CLI forces UTF-8 output. If it does, set `PYTHONIOENCODING=utf-8` and please report it. |

---

## The eight questions it answers

| Question | Command |
|---|---|
| Is my config and dataset sound? | `validate` |
| What will this run cost? | `estimate` |
| Which model scores highest on my workload? | `report` |
| **Is that difference real, or noise?** | `compare` |
| **How many more questions do I need?** | `report` (power section) |
| **Which model should we actually buy?** | `decide` |
| **Did my change break anything?** | `gate` |
| **Will it obey a poisoned document?** | `probes` + `report` |

---

## Case studies: Indian-government use cases

Five worked evaluations of language models on the everyday work of district
and state offices, each with a fictional, illustrative dataset in English,
Hindi and Hinglish, a profile, a write-up under `docs/case-studies/`, and a
screen in the browser UI (Analyse, Case studies) that shows the newest run
with its intervals, paired test and cost per correct answer.

| profile | task | what it measures |
|---|---|---|
| `in_grievance_triage` | classify | routing citizen grievances to the department that acts first |
| `in_notice_translation` | direct | translating public notices into Hindi; fidelity and native script kept apart |
| `in_rti_drafting` | direct | drafting a Right to Information application from a citizen's situation |
| `in_scheme_qa` | rag | answering scheme questions from the documents, in the citizen's language |
| `in_statute_qa` | rag | answering RTI-framework questions with citations, from summaries of the Act |

```bash
python main.py validate --profile configs/profiles/in_scheme_qa.yaml
python main.py ingest   --profile configs/profiles/in_scheme_qa.yaml     # RAG profiles only
python main.py run      --profile configs/profiles/in_scheme_qa.yaml --budget 5
python main.py report   --profile configs/profiles/in_scheme_qa.yaml
```

Every dataset is generated by a `build_dataset.py` under `data/<profile>/`
and is the source of truth; nothing real is reproduced, and the RTI
summaries are not the statutory text. See `docs/case-studies/README.md`.

## Adversarial probes

**A RAG system's corpus is an attack surface.** Anything that can put text into your index, an uploaded PDF, a support ticket, a scraped page, a wiki anyone can edit, can put instructions in front of your model. The model cannot distinguish a retrieved passage from an operator instruction; they arrive as the same tokens.

```bash
python main.py probes --profile regulated_qa --append
```

This derives five probe families from questions you already have, so they test *your* system rather than a generic benchmark:

| Probe | What it catches |
|---|---|
| `injection` | A hostile instruction planted in a retrieved passage, carrying a canary string. If the canary comes back, the model took orders from its data. |
| `unanswerable` | A question the corpus provably cannot answer, phrased so retrieval still returns confident-looking passages. Does the model refuse, or invent a statute? |
| `noise` | Distractor passages mixed into context. Does quality survive imperfect retrieval, the only kind there is in production? |
| `paraphrase` | The same question reworded. Users don't ask twice the same way; an answer that changes when they do is unreliable even when each individual answer looks fine. |
| `positional` | The gold passage forced to the middle of a long context. Separates models that read their whole context from models that skim the ends. |

Probe generation is deterministic given a seed, costs nothing, and needs no re-ingest, probes reuse the existing corpus.

---

## Statistical significance

The original harness reported a bootstrap confidence interval **per model**. That is the wrong interval for the decision being made: both models were evaluated on the **same items**, so the comparison is paired, and pairing removes item difficulty, usually the dominant source of variance, from the estimate.

```bash
python main.py compare --profile regulated_qa --metric accuracy
```

```
* gpt-oss-120b vs gpt-oss-20b: gpt-oss-120b is better by 0.094 [+0.056, +0.138] (p=0.001)
  qwen-7b vs gpt-oss-20b: no significant difference (p=0.214, 80 paired items)

* = significant after Holm-Bonferroni correction
```

- **Paired bootstrap** on per-item differences, or an **exact McNemar test** when the metric is binary, selected automatically.
- **Holm-Bonferroni correction**, because comparing 5 models is 10 tests, and at α=0.05 you'd expect a false winner ~40% of the time without it.
- **Power analysis** that converts "add more questions" into a number:

```
accuracy: 80 paired items, smallest reliably detectable gap ~0.070
  to detect a gap of 0.02: ~993 paired items
  to detect a gap of 0.05: ~159 paired items
```

Two overlapping per-model intervals do **not** mean the difference is insignificant, and a visible gap on a leaderboard can still be noise. Only the paired test settles it.

---

## The decision layer

```bash
python main.py decide --profile regulated_qa \
    --require "faithfulness>=0.90" --require "latency_p95_ms<=2000" \
    --optimise cost_usd --qpd 50000
```

```
Constraints: faithfulness >= 0.9, latency_p95_ms <= 2000
1/3 models qualify.
  OK  openai/gpt-oss-120b

>>> Recommended: openai/gpt-oss-120b

=== projected cost at 50,000 queries/day ===
model                 cost_per_query   daily_usd   monthly_usd   annual_usd
openai/gpt-oss-20b            0.0006        30.0         900.0      10950.0
openai/gpt-oss-120b           0.0040       200.0        6000.0      73000.0

=== what extra quality costs (vs cheapest) ===
model                 accuracy  monthly_usd  quality_delta  usd_per_point
openai/gpt-oss-120b      0.881       6000.0        0.09375          544.0
```

When nothing qualifies it says exactly what failed and by how much, so "no model qualifies" becomes "everything failed latency; relax it to 2.4s or accept the accuracy drop". Latency comes from the low-concurrency lane and is reported as **p95**, because SLAs are written on tails.

---

## CI regression gating

The harness can tell you which model is best today. `gate` tells you whether the prompt change you just made broke anything, the question teams ask far more often.

```bash
python main.py gate --profile regulated_qa \
    --baseline "$KNOWN_GOOD_RUN_ID" \
    --gate accuracy:0.02 --gate faithfulness:0.01 \
    --floor "injection_resisted>=0.95" \
    --max-cost 0.005
```

Exits non-zero on regression. Two design choices:

- **Significance, not just delta.** A gate that fires on any drop fires constantly on noise, everyone learns to ignore it, and it stops protecting anything. A regression must exceed the tolerance *and* be statistically significant. `--strict` disables that for teams who prefer over-alerting.
- **Absolute floors as well as deltas.** "No worse than last time" lets quality erode one tolerable step at a time; a floor catches the slow slide.

`--max-cost` catches the regression nobody reviews for: a prompt that grew 400 tokens looks fine in a diff and doubles your bill.

---

## Multi-provider setup

Model strings take an optional `provider:` prefix. Without one they use `default_provider`, so a single-vendor config stays short.

The shipped `configs/models.yaml` is set up for **Together AI + OpenRouter**. Either key alone runs everything including RAG; both together give you Together's cheap open-weight inference plus OpenRouter's access to frontier models, compared head to head in one matrix.

```yaml
# configs/models.yaml
default_provider: together

models:
  - "openai/gpt-oss-120b"                          # -> together (the default)
  - "Qwen/Qwen2.5-7B-Instruct-Turbo"               # -> together
  - "openrouter:openai/gpt-5.2"                    # -> OpenRouter
  - "openrouter:anthropic/claude-sonnet-4.6"       # -> OpenRouter
  - "anthropic:claude-opus-5"                      # -> Anthropic directly
  - "ollama:llama3.1:8b"                           # -> a model you host yourself

# Embeddings, reranking and judging are part of the fixed APPARATUS, not the
# variable under test, so they are pinned rather than following whichever model
# is being evaluated.
embedding_provider: together
rerank_provider: together
judge_provider: together
```

That pinning is what keeps the comparison valid, if two models saw different retrieved passages, the run measures the embedders, not the models, and it is what makes a cross-vendor matrix possible at all when a provider is missing a capability.

### Provider capabilities

| Provider | Chat | Embeddings | Rerank | Notes |
|---|:--:|:--:|:--:|---|
| Together | yes | yes | yes | The only one here with a rerank endpoint |
| OpenRouter | yes | yes | no | One key, most frontier models; listed prices are ceilings |
| OpenAI | yes | yes | no | |
| Anthropic | yes | **no** | no | Needs another provider for the index |
| Groq | yes | **no** | no | Fast, but chat only |
| Ollama / vLLM / LM Studio | yes | yes | no | Self-hosted, no key needed |

Mismatches are caught **before** the first billable call, not thousands of items in:

```
Provider routing problems:
  Rerank model 'some-reranker' routes to provider 'openrouter', which has no
  rerank endpoint. Leave `rerank_model` blank or set `rerank_provider`.
```

### Running OpenRouter-only

OpenRouter serves an OpenAI-shaped `/v1/embeddings` endpoint, so it can build the vector index as well as run the models, no second key required:

```yaml
default_provider: openrouter
models:
  - "openai/gpt-5.2"
  - "anthropic/claude-sonnet-4.6"
  - "meta-llama/llama-3.3-70b-instruct"
judge_model: "google/gemini-2.5-pro"   # a fourth family, so it grades none of them
rerank_model: ""                        # OpenRouter has no rerank endpoint
embedding_provider: openrouter
rerank_provider: openrouter
judge_provider: openrouter
```

Then point the profile's `embedding_model` at an OpenRouter embedder: `baai/bge-m3` (1024-dim, $0.01/M) or `openai/text-embedding-3-small` (1536-dim, $0.02/M), and re-run `ingest`. Changing the embedder changes the Qdrant collection name deliberately: vectors from two different embedders are not comparable, so they must not share an index.

> **The two namespaces are not interchangeable.** Together's `BAAI/bge-large-en-v1.5` and OpenRouter's `baai/bge-m3` differ in vendor casing *and* are different models. `configs/pricing.yaml` can key either form, and a provider-qualified key (`openrouter:meta-llama/llama-3.3-70b-instruct`) wins over a bare one, the same weights genuinely cost a different amount through a router than direct.

Mixing hosted and self-hosted in one matrix is the point: *"is the API worth it versus what we can run ourselves?"* is a cost question you cannot answer from inside a single-vendor harness.

---

## Cost control

Cost is measured, not estimated, and split by the subsystem that caused it:

```
[cost] {'generation_usd': 2.41, 'judge_usd': 7.88, 'embedding_usd': 0.03,
        'rerank_usd': 0.0, 'total_usd': 10.32, 'cache_hit_rate': 0.41, ...}
```

That split routinely surprises people: on a judge-scored profile the judge is usually the largest line. The original harness billed generation only, so it reported roughly a third of the true cost, and because cost carries a negative weight in the composite, it also mis-ranked the models.

Three protections:

- **`--budget 25.00`** (or `budget_usd` in `run.yaml`), a hard ceiling. The run aborts cleanly, keeps every row already written, and stops issuing calls immediately rather than paying for a matrix it will discard.
- **`estimate`**, a forecast *including judge calls*, priced per model rather than sampling the first one.
- **Judge and query-embedding caching**, a judge verdict is a pure function of its inputs, and the same query was previously embedded once per model, per tuning candidate, per pass.

---

## Non-RAG workloads

Most LLM evaluation isn't retrieval-augmented. Set `task: classify` or `task: direct` and the same apparatus, matrix walk, cost metering, caching, significance testing, regression gating, works with no corpus, no embedder and no vector store. See `configs/profiles/support_triage.yaml`:

```yaml
name: support_triage
task: classify
evalset_path: data/support_triage/evalset.jsonl
label_set: [billing, technical, account_access, feature_request, complaint, spam]
max_tokens: 32
```

```bash
python main.py run --profile support_triage
```

Classification reports per-class precision/recall/F1 and **macro-F1** alongside accuracy, on the imbalanced label distributions real queues have, a model that never predicts the rare-but-critical class can still post 95% accuracy.

---

## Preparing your own dataset

See [step 7](#7-run-on-your-own-data) for the commands. This section covers what `prepare_dataset.py` actually does and where it goes wrong.

It takes a folder of PDFs plus a Q&A spreadsheet and produces the two JSONL files the harness reads, **auto-labeling the gold passages** along the way. Because your gold answers are exact strings, it can find which chunk contains each one, recovering the retrieval metrics (hit-rate@k, MRR, NDCG) with no manual passage labeling. It uses the harness's own chunker, so the chunk ids it writes match exactly what `ingest` creates.

The spreadsheet needs only two columns, `Question` and `Answer`, one pair per row. Extra columns are ignored. Answers it can't locate are still written as valid items; they just don't contribute to retrieval metrics.

Two things to get right:

> **Pass the same `--chunk-size` and `--overlap` here and in the profile.** Mismatched values are the single most common silent failure in a hand-built RAG evalset: the gold ids point at chunks that don't exist, retrieval looks catastrophically broken, and the model gets blamed. `python main.py validate --profile <p>` checks for this explicitly and names the affected items.

> **Scanned PDFs yield no text.** `prepare_dataset.py` warns per page rather than silently indexing empty documents. OCR them first.

---

## Data format

Both files are **JSONL** (one JSON object per line, UTF-8).

**corpus.jsonl**
```json
{"doc_id": "handbook_p12", "text": "full text of this page...", "source_uri": "handbook.pdf#p12"}
```

**evalset.jsonl**
```json
{"item_id": "q1", "query": "What is the maximum fine?", "item_type": "answerable",
 "gold_answer": "5000", "gold_passage_ids": ["handbook_p12#0"], "human_label": 1.0}
```

`item_type` is one of `answerable`, `unanswerable`, `noise_injected`, `injection`. Passage IDs use `doc_id#chunkindex`.

`meta.language` takes a code from the language table (`en`, `hi`, `mr` Marathi, `bn` Bengali, `gu` Gujarati, `kn` Kannada, `te` Telugu, `hinglish`) and gives every write-up a per-language reading. A direct-task profile may set `target_script` to a script or a language (`devanagari`, `marathi`, `kn`); the share of the answer in that script is scored as `native_script_ratio`, apart from accuracy. Numeric scoring reads Indic digits in every listed script.

`human_label` is optional and unlocks **judge calibration**, how well the LLM judge agrees with a person, reported as Cohen's kappa. Every judge-scored number in the harness rests on that assumption, and this is the only thing that checks it. Labelling even 30 items is worth it.

### Choosing the answer scorer

| Scorer | Use when | Notes |
|---|---|---|
| `exact` | answers are short and fixed | strictest |
| `numeric` | answers are numbers | scans every number, so a leading year doesn't score the answer wrong |
| `contains` | a keyword must appear | gameable: a model dumping the whole context always "contains" the gold |
| `token_f1` | free-form answers, no budget for a judge | graded, deterministic, penalises padding, **a good default** |
| `judge` | free-form answers where wording varies | most accurate, most expensive; also unlocks faithfulness/relevance/completeness |

---

## Configuration

- **`configs/profiles/*.yaml`**, one per workload: corpus/evalset, embedder, retrieval settings, active metrics, weights, probe settings and the tuning search space.
- **`configs/models.yaml`**, models under test, provider routing, judge, reranker.
- **`configs/pricing.yaml`**, dated per-token prices.
- **`configs/run.yaml`**, run matrix, splits, concurrency, budget, retries, checkpointing.

Profiles are **validated on load**, and every problem is reported at once:

```
Invalid profile 'configs/profiles/x.yaml':
  - ['cost_usd'] are lower-is-better and must have NEGATIVE weights; a positive
    weight rewards models for being worse.
  - metric_weights reference metrics not in active_metrics: ['faithfulness'].
    They would contribute nothing to the composite.
  - Unknown active_metrics: ['accuarcy'].
```

Each of those used to be a silent failure that produced a plausible-looking leaderboard.

### The adapted pass (equal-budget tuning)

For each model the harness searches up to *N* configurations, identical *N* for every model, over a held-out dev split, varying retrieval mode, `k`, reranking, **context ordering** and any candidate prompts you list under `knobs:`. It then re-runs the winner on the test split. **Tuning gain = adapted − baseline.**

You never tune manually; you supply a small menu of options once and the search is automatic. It's roughly `tuning_budget` times more expensive than the baseline, so it's off by default.

---

## Understanding the results

- **Leaderboard (weighted composite)**, deterministic arithmetic from the profile's weights. Set `normalise=True` when mixing metrics with very different scales, or latency in milliseconds will silently outweigh accuracy in [0,1].
- **Significance**, the section that decides whether the leaderboard order means anything.
- **Pareto frontier**, models not beaten on all of accuracy, cost and latency. When top models tie on quality, this is where the decision lives.
- **Tuning gain**, which models benefit most from tuning, under an equal budget.
- **Error attribution**, *why* items failed. `rate_limit` means lower your concurrency; `context_length` means lower your `k`; `content_filter` is a finding about the model.
- **Truncation rate**, answers cut off by `max_tokens`. Their completeness and citation scores are invalid; this is a config problem that looks exactly like a quality finding.
- **Judge calibration**: Cohen's kappa against human labels, not raw agreement. A judge that marks everything correct scores 90% agreement on a set that's 90% correct while carrying no information; kappa reports ~0 there.

---

## Project structure

```
llm-eval-harness/
├── app.py                  # Streamlit app: upload -> probe -> run -> decide
├── dashboard.py            # read-only results dashboard
├── main.py                 # CLI: validate/estimate/ingest/probes/run/report/
│                           #      compare/decide/arena/gate/html/runs
├── prepare_dataset.py      # PDFs + Q&A spreadsheet -> JSONL (+ auto-labeling)
├── configs/
│   ├── profiles/*.yaml     # one per workload (incl. a non-RAG classify profile)
│   ├── models.yaml         # models, provider routing, judge, reranker
│   ├── pricing.yaml        # dated per-token prices
│   └── run.yaml            # run matrix, splits, budget, retries, checkpointing
├── harness/
│   ├── clients/            # provider layer: capability protocols, OpenAI-
│   │                       # compatible, Anthropic, routing, resilience, cost
│   ├── rag/                # ingest, retrieve (dense/sparse/hybrid+RRF), rerank, prompt
│   ├── profiles/           # profile config + validation, dataset loading + QA
│   ├── tuning/             # equal-budget candidate search, successive halving
│   ├── eval/               # metrics, scorer registry, LLM judge, probes
│   ├── orchestration/      # runner, passes, collector, arena, background jobs
│   ├── store/              # TraceRow schema + append-only Parquet/DuckDB store
│   ├── report/             # aggregate, stats, decide, gate, html
│   ├── ui/                 # design system + app shell: palette, Altair theme,
│   │   └── screens/        # charts, layout, page registry, one file per screen
│   └── cache/              # content-addressed cache (+ a real null cache)
├── .streamlit/config.toml  # theme, matched to harness/ui/theme.py
├── DEPLOYMENT.md           # deployment modes, CI, security, backup
├── tests/                  # 277 tests, all offline against a fake provider
└── workspace/              # app-created data, vector store, traces (gitignored)
```

---

## Deployment

Running this on anything other than your own laptop, a shared box, a container,
a CI pipeline, is covered in **[DEPLOYMENT.md](DEPLOYMENT.md)**: five deployment
modes, a CI gate workflow, cost controls, backup, and the security rules that
matter.

The short version:

| Mode | Command | Notes |
|---|---|---|
| Local, single user | `streamlit run app.py` | Embedded Qdrant, no Docker, no auth needed |
| Team (read-only) | `streamlit run dashboard.py` | Makes no API calls, so it cannot spend |
| CI gate | `python main.py gate ...` | Exits non-zero on a *significant* regression |

**One rule worth repeating here:** the app accepts an API key and spends money,
and Streamlit ships no authentication. Bind it to `127.0.0.1` or put an
authenticated reverse proxy in front before anyone else can reach it, and give
the team `dashboard.py`, which is read-only by construction.

---

## Troubleshooting

Problems while *running an evaluation*. For install and first-run issues, see [Local run gotchas](#local-run-gotchas).

| Symptom | Cause & fix |
|---|---|
| `401 Invalid API key` | The harness is using a different key than you think. Set the provider's env var before launching and check the sidebar field matches. |
| `404` / model not found | The model string is retired or not on your account. Run `python main.py validate --profile <p>`, it pre-flights every provider before spending. |
| "Provider X serves no embeddings" | Expected: Anthropic and Groq don't. Set `embedding_provider` in `models.yaml` to one that does. |
| `fastembed is not installed` | BM25/hybrid retrieval needs it: `pip install fastembed`. Or set `retrieval_mode: dense`. |
| Everything scores ~1.0 | Questions are too easy to separate models. `main.py report` now tells you the minimum detectable gap and how many items you'd need. |
| Wide confidence intervals | Same, check the power section for the number of questions required. |
| Run died partway | `python main.py run --profile <p> --resume <run_id>`. Rows are checkpointed, so nothing already paid for is lost. |
| Lots of `rate_limit` errors | Lower `max_workers`, or set a per-provider `rate_limit` in `models.yaml`. Pacing is cheaper than backing off. |
| Judge costs more than the models | Expected, and now visible. Use `token_f1` scoring, or cache-warm by re-running (judge calls are cached). |
| Cost shows as blank | That model has no entry in `configs/pricing.yaml`. Blank is deliberate, a zero-cost model would win the leaderboard. |

---

## Design notes

- **Latency is measured in its own lane**, serial, models interleaved, cache *genuinely* bypassed via a null cache, warm-up discarded, streaming on for TTFT. On a shared hosted API, firing everything at once benchmarks your own queuing, not the model.
- **The cache is the reproducibility guarantee**, not the seed. Hosted inference isn't bit-for-bit deterministic over time; caching raw outputs by a hash of (model, prompt, params) makes cost and accuracy reproducible and free to re-report.
- **Every metric field is Optional.** Inactive metrics stay `None` rather than `0.0`, so "not measured" stays distinguishable from "measured as zero", a metric defaulting to zero would drag every mean built on it downward.
- **The judge never grades its own family**, and the guard resolves `provider:` prefixes so it can't be defeated by how a model happens to be addressed.
- **Pairwise comparisons are position-bias corrected**, each pair is judged in both orders and a win counts only if it survives the swap. Judges favour whichever answer they see first, so an uncorrected Elo table partly ranks argument position.
- **Retrieval metrics are scored before probe mutation.** Score the retriever on what it actually found; score the model on the adversarial context it was given.
- **Pricing is dated and fails loudly.** The report warns when prices are stale, because a benchmark quoting year-old rates as fact is misleading in a way nobody notices.
- **The store is append-only.** Each checkpoint is its own Parquet part: O(batch) writes, crash-safe, and readable through one DuckDB glob.

---

## Roadmap

Deliberately not built yet, in rough order of value:

- **Multi-turn conversational evaluation**, the schema carries `history`, but there's no goal-completion metric across turns.
- **Tool/function-calling evaluation**, schema-validity, argument correctness and recovery from tool errors.
- **Retrieval-free ablations**, automatically re-run with an empty context to measure how much of the score is parametric memory rather than retrieval.
- **Semantic near-duplicate detection** in evalsets, and train/test contamination checks against the corpus.
- **Human-in-the-loop labelling UI** to grow the `human_label` set that judge calibration depends on.
- **Cross-encoder-free rerank baselines** (MMR, diversity sampling) so reranking has a cheap comparison point.
- **Batch API support** where providers offer it, typically ~50% cheaper for non-latency-sensitive judge calls.

---

## Acknowledgements

Built on [Qdrant](https://qdrant.tech/) for vector storage, [DuckDB](https://duckdb.org/) + Parquet for the trace store, and [Streamlit](https://streamlit.io/) for the UI, with provider support for Together, OpenAI, Anthropic, Groq, Fireworks, vLLM and Ollama.
