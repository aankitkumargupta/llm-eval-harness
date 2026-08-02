# Multi-Model LLM Evaluation Harness

A model-agnostic benchmarking suite that ranks LLMs for client RAG (Retrieval-Augmented Generation) needs against **workload-specific accuracy, cost, and latency**. It runs a fixed-prompt baseline plus an optional per-model adapted pass under an **equal tuning budget**, so you can measure not just which model is best out of the box, but how much each model *gains* from tuning.

All models are served through **[Together AI](https://www.together.ai/)** via its OpenAI-compatible API, so the harness is network-bound and disk-bound rather than compute-bound - it runs comfortably on an 8 GB machine because no model weights are ever loaded locally.

Upload a folder of PDFs and a spreadsheet of question/answer pairs, pick the models to compare, and get a leaderboard, a cost/latency Pareto frontier, and confidence intervals - either from a browser UI or the command line.

---

## Table of contents

- [Features](#features)
- [What it measures](#what-it-measures)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start (UI)](#quick-start-ui)
- [Quick start (CLI)](#quick-start-cli)
- [Preparing your own dataset](#preparing-your-own-dataset)
- [Data format](#data-format)
- [Configuration](#configuration)
- [Understanding the results](#understanding-the-results)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Design notes](#design-notes)
- [License](#license)

---

## Features

- **Model-agnostic** - benchmark any Together-hosted LLM by editing a config; the model is just a string.
- **Three interfaces** - a full Streamlit app (upload → run → results), a read-only results dashboard, and a scriptable CLI.
- **Baseline + adapted passes** - measure out-of-the-box quality *and* how much each model improves under an equal, automatic tuning search.
- **Hybrid retrieval** - dense (semantic), sparse (BM25 lexical), and hybrid via Reciprocal Rank Fusion, all from one Qdrant index.
- **Auto-labeled datasets** - point it at PDFs + a Q&A spreadsheet and it finds which passage each answer came from automatically.
- **~24 metrics** across retrieval quality, answer quality, citations, behaviour, and efficiency (each profile activates a relevant subset).
- **Reproducible** - content-addressed caching means re-runs don't re-bill and results are stable.
- **Runs on 8 GB RAM** - no local GPU or model weights; Qdrant runs embedded (no Docker required in the app).

---

## What it measures

Metrics are grouped by the subsystem that owns them. A given profile activates only the subset relevant to that workload.

**Retrieval quality** (deterministic, from retrieved vs. gold passages)
- Hit-rate@k, MRR, NDCG@k, context recall, rerank delta

**Answer quality**
- Accuracy (exact / contains / numeric / LLM-judge scorers)
- Faithfulness (grounded in context, no fabrication - judged)
- Answer relevance, completeness (judged)

**Citations**
- Pointer validity (are cited passages real?) and supporting validity (do they back the claim?)

**Behaviour / robustness**
- Abstention on unanswerable questions; hooks for noise-injection and prompt-injection probes

**Efficiency** (deterministic)
- p50 / p95 latency, TTFT, cost per query (from the API's authoritative token counts)

**Reporting aggregates** (computed from the above)
- Weighted composite leaderboard, tuning gain (adapted − baseline), Pareto frontier, Elo/Bradley-Terry, bootstrap confidence intervals, error attribution

---

## Architecture

Config is the spine; code is the engine. To add a profile or a model you edit YAML, never Python. Each layer depends only on the interface below it.

```
Upload PDFs + Q&A  ─►  prepare_dataset  ─►  corpus.jsonl + evalset.jsonl
                                                    │
                                            ingest (embed → Qdrant)
                                                    │
   run matrix:  model × profile × pass  ─►  retrieve → prompt → generate → score
                                                    │
                                            trace store (Parquet)
                                                    │
                                    report: leaderboard · Pareto · tuning gain · CIs
```

The two load-bearing contracts are **`TraceRow`** (one flat row per eval item - the boundary between running and analysis) and the **Together client** methods (the only place that touches the network).

---

## Requirements

- **Python 3.11+**
- A **Together AI API key** ([get one here](https://api.together.ai/settings/api-keys))
- ~2-3 GB free RAM during use (embedded Qdrant + Python)
- Docker is **optional** - the app runs Qdrant embedded; only the CLI path optionally uses a Qdrant server

---

## Installation

```bash
# clone
git clone <your-repo-url>
cd llm-eval-harness

# create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

> **Windows note:** if activation is blocked with a "running scripts is disabled" error, run once:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

Set your API key (either method):

```bash
# option A - environment variable (recommended)
# Windows PowerShell:
$env:TOGETHER_API_KEY = "your_key_here"
# macOS / Linux:
export TOGETHER_API_KEY="your_key_here"

# option B - .env file in the project root
# TOGETHER_API_KEY=your_key_here
```

---

## Quick start (UI)

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Then:

1. **Sidebar** - confirm your API key is present (pre-filled from the environment, or paste it).
2. **Tab 1 · Upload & prepare** - name a profile, upload your PDFs and Q&A spreadsheet, set the column names and answer scorer, then **Build dataset** → **Ingest**.
3. **Tab 2 · Configure & run** - pick the models, check the cost estimate, click **Run evaluation**. The run happens in the background with a live progress bar.
4. **Tab 3 · Results** - leaderboard, Pareto frontier, tuning gain, and confidence intervals.

The app runs Qdrant **embedded** (a folder under `workspace/`), so no Docker is needed. It is single-user and runs one job at a time.

---

## Quick start (CLI)

```bash
# 1. ingest a profile's corpus into the vector store (one-time per dataset)
python main.py ingest --profile configs/profiles/regulated_qa.yaml

# 2. run baseline (+ optional adapted + latency) for a profile
python main.py run --profile configs/profiles/regulated_qa.yaml

# 3. print the leaderboards from the trace store
python main.py report --profile regulated_qa
```

There is also a read-only results dashboard over an existing trace store:

```bash
streamlit run dashboard.py
```

---

## Preparing your own dataset

If your data is a folder of PDFs plus a spreadsheet of Q&A pairs, `prepare_dataset.py` converts both into the JSONL the harness needs and **auto-labels the gold passages** for you:

```bash
pip install pdfplumber openpyxl   # one-time, for this tool

python prepare_dataset.py \
    --pdf-dir  ./my_pdfs \
    --qa-file  ./my_questions.xlsx \
    --out-dir  ./data/my_eval \
    --question-col Question --answer-col Answer \
    --chunk-size 200 --overlap 40
```

It extracts each PDF (one record per page by default), reads your spreadsheet, and matches each exact answer to the chunk that contains it - using the harness's own chunker, so the labels line up with what ingestion indexes. Answers it can't locate are still written as valid items; they just won't contribute to retrieval metrics.

> Pass the **same** `--chunk-size` / `--overlap` here and to `main.py ingest`, so the auto-labeled passage IDs match the indexed chunks.

The app's "Build dataset" button runs this same logic under the hood.

---

## Data format

Both files are **JSONL** (one JSON object per line, UTF-8).

**corpus.jsonl** - the knowledge base
```json
{"doc_id": "handbook_p12", "text": "full text of this page or document...", "source_uri": "handbook.pdf#p12"}
```

**evalset.jsonl** - the questions and ground truth
```json
{"item_id": "q1", "query": "What is the maximum fine?", "item_type": "answerable", "gold_answer": "5000", "gold_passage_ids": ["handbook_p12#0"]}
```

`item_type` is one of `answerable`, `unanswerable` (abstention probe), `noise_injected`, or `injection`. Passage IDs use the `doc_id#chunkindex` format the chunker produces (chunk 0 of `handbook_p12` is `handbook_p12#0`).

**Spreadsheet input** (for `prepare_dataset.py` / the app) only needs two columns, named `Question` and `Answer`, one pair per row. Any extra columns are ignored.

### Choosing the answer scorer

| Scorer | Use when | Example |
|---|---|---|
| `exact` | answers are short and fixed | gold `Kerb` |
| `numeric` | answers are numbers (avoids the `50`⊂`500` trap) | gold `1500` |
| `contains` | a keyword must appear in a fuller sentence | gold `Viaduct` |
| `judge` | free-form answers where wording varies; also unlocks faithfulness/relevance/completeness | multi-sentence answers |

The app applies **one** scorer to the whole sheet, so for a mixed numeric/text sheet, `contains` is the safe default. Use `judge` for synthesis/comparison questions.

---

## Configuration

All configuration lives in `configs/`:

- **`configs/profiles/*.yaml`** - one file per RAG workload; declares corpus/evalset paths, embedding model, retrieval settings, active metrics, metric weights, and the tuning search space.
- **`configs/models.yaml`** - the models under test, the judge model, and the (optional) reranker.
- **`configs/pricing.yaml`** - dated per-token prices; the input to the cost metric.
- **`configs/run.yaml`** - the run matrix, dev/test split, and concurrency.

### Verified working model strings (as of August 2026)

Model availability and pricing on Together change over time - **always verify** against the [current model list](https://docs.together.ai/docs/serverless/models) and [pricing](https://www.together.ai/pricing). At the time of writing, these strings work:

**`configs/models.yaml`**
```yaml
models:
  - "openai/gpt-oss-20b"
  - "openai/gpt-oss-120b"
  - "Qwen/Qwen2.5-7B-Instruct-Turbo"
  - "Qwen/Qwen3.5-9B"

judge_model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"
rerank_model: ""   # reranking is optional; leave blank to skip it
```

**Embedding model** (set per profile): `intfloat/multilingual-e5-large-instruct` (1024-dim).

**`configs/pricing.yaml`** (per 1M tokens)
```yaml
as_of: "2026-08-02"
models:
  "openai/gpt-oss-20b":                       {input: 0.05, output: 0.20}
  "openai/gpt-oss-120b":                      {input: 0.15, output: 0.60}
  "Qwen/Qwen2.5-7B-Instruct-Turbo":           {input: 0.30, output: 0.30}
  "Qwen/Qwen3.5-9B":                          {input: 0.30, output: 0.30}
  "meta-llama/Llama-3.3-70B-Instruct-Turbo":  {input: 0.88, output: 0.88}
embeddings:
  "intfloat/multilingual-e5-large-instruct":  {input: 0.02}
rerank: {}
```

> **Tip:** verify any model string works on *your* account before a full run:
> ```bash
> python -c "from openai import OpenAI; c=OpenAI(api_key='YOUR_KEY', base_url='https://api.together.xyz/v1'); print(c.chat.completions.create(model='openai/gpt-oss-20b', messages=[{'role':'user','content':'hi'}], max_tokens=3).choices[0].message.content)"
> ```

### The adapted pass (equal-budget tuning)

When enabled, for each model the harness automatically searches up to *N* configurations (the tuning budget, identical for every model) over a held-out dev split - varying retrieval mode, `k`, reranking, and any candidate prompts you list in the profile's `knobs:` - then re-runs the winning configuration on the test split. **Tuning gain = adapted − baseline** tells you how much each model benefits from tuning. It is more expensive (roughly an order of magnitude more calls), so it is off by default. You never tune manually; you optionally supply a small menu of options once, and the search is automatic.

---

## Understanding the results

- **Leaderboard (weighted composite)** - a single score per model from the profile's metric weights (negative weights = lower-is-better, e.g. cost). This is deterministic arithmetic, so models with identical metrics get identical composites.
- **Accuracy** - with `judge` scoring, an LLM decides correct/incorrect per answer, then averages. If every strong model scores ~1.0, your questions are too easy to separate them - add harder synthesis questions or more items.
- **Pareto frontier** - models *not* beaten on all of accuracy, cost, and latency. When top models tie on accuracy, the frontier (cost + latency) is where the real decision lives.
- **Confidence intervals** - wide intervals mean your sample is too small to rank confidently; add more questions to tighten them.
- **Tuning gain** - requires both a baseline and an adapted pass; shows which models improve most when tuned.

The app's Results tab surfaces the decision-critical metrics as tables and charts - leaderboard, retrieval quality (hit-rate/MRR/NDCG/recall), answer & citation quality, efficiency (latency/tokens/cost), a Pareto frontier, and a per-metric confidence-interval selector. Metrics not exercised by a given dataset (e.g. abstention or citations when your questions don't use them) are labelled rather than shown blank. Every metric, including any not surfaced in a panel, is available in the **Raw traces** view and in `workspace/traces.parquet`.

---

## Project structure

```
llm-eval-harness/
├── app.py                 # Streamlit app: upload → run → results
├── dashboard.py           # read-only results dashboard
├── main.py                # CLI: ingest / run / report
├── prepare_dataset.py     # PDFs + Q&A spreadsheet → JSONL (+ auto-labeling)
├── requirements.txt
├── configs/
│   ├── profiles/*.yaml     # one per RAG workload
│   ├── models.yaml         # models under test, judge, reranker
│   ├── pricing.yaml        # dated per-token prices
│   └── run.yaml            # run matrix, splits, concurrency
├── harness/
│   ├── clients/            # Together client + pricing registry
│   ├── rag/                # ingest, retrieve (dense/sparse/hybrid+RRF), rerank, prompt
│   ├── profiles/           # profile config + loaders
│   ├── tuning/             # equal-budget candidate search
│   ├── eval/               # metric functions + LLM judge
│   ├── orchestration/      # per-item runner, matrix walk, background jobs
│   ├── store/              # TraceRow schema + Parquet/DuckDB store
│   ├── report/             # aggregation: composite, Pareto, tuning gain, Elo, CIs
│   └── cache/              # content-addressed cache
├── tests/                  # pytest suite for the non-network logic
└── workspace/              # app-created data, vector store, traces (gitignored)
```

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `401 Invalid API key` | The app is using a different/stale key than you think. Set `TOGETHER_API_KEY` in the environment before launching, and make sure the sidebar field matches a key that works. Verify the key with the one-liner in [Configuration](#configuration). |
| `404` / model not found | The model string is retired or not on your account. Model availability on Together changes over time - check the [current model list](https://docs.together.ai/docs/serverless/models) and update `configs/models.yaml` + `configs/pricing.yaml`. |
| `fastembed is not installed` | BM25/hybrid retrieval needs it: `pip install fastembed`. (First run downloads a small model file.) Or set the profile's `retrieval_mode: dense` to skip BM25. |
| Ingest button greyed out | No API key entered, a job is already running, or the dataset wasn't built in the current session. Enter the key, wait for any running job, and click **Build dataset** first. |
| Everything scores ~1.0 | Questions are too easy to separate strong models. Add synthesis/comparison questions or more items; lean on the cost/latency Pareto to choose among tied leaders. |
| Wide confidence intervals | Sample too small - add more questions. |

---

## Design notes

- **Latency is measured in its own lane** - low, fixed concurrency, models interleaved, cache bypassed, warm-up discarded - because on a shared hosted API, firing everything at once benchmarks your own queuing, not the model.
- **The cache is the reproducibility guarantee.** Hosted inference isn't bit-for-bit deterministic over time; caching raw outputs by a hash of (model, prompt, params) makes cost/accuracy reproducible and free to re-report.
- **The judge never grades its own family** (self-preference bias); the code enforces this and errors if misconfigured.
- **Pricing is dated and fails loudly** on an unpriced model rather than silently reporting zero cost.
- **Context-window and model-availability drift** are your responsibility: verify model strings before a run, and be explicit when a model can't fit a long-document profile.


---

## Acknowledgements

Built on [Together AI](https://www.together.ai/) for inference, [Qdrant](https://qdrant.tech/) for vector storage, [DuckDB](https://duckdb.org/) + Parquet for the trace store, and [Streamlit](https://streamlit.io/) for the UI.