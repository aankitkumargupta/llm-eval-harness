# INVENTORY: Phase 0 recon

**Scope:** read-only audit of `llm-eval-harness` against `CLAUDE.md`.
**Date:** 2026-09-19 · **Working tree:** clean except `M .gitignore` (pre-existing).
**Source files changed by this phase: zero.**

> **Status note, added after the fact.** This document records the repo as
> found. A subsequent pass closed R-01, R-02, R-03, R-04, R-07 and R-08, and
> built the §10 benchmark subsystem. The findings below are left as written,
> an audit edited to match later work stops being evidence of anything. See
> [`DEBT.md`](DEBT.md) for what is closed, what is still open, and what the
> fixes themselves added.

Every claim below cites `file:line` or is marked **undetermined**. Where the repo's module
names differ from `CLAUDE.md` §3, the mapping used is stated once, in §2.

---

## 0. Headline

The repo is a **working, tested harness** (282 tests, all green offline) whose *behaviour* already
honours several §2 invariants, but whose *structure* is the pre-Phase-1 layout throughout. None of
`src/`, `uv.lock`, `harness/providers/`, `harness/stats/`, `harness/bench/`, `.importlinter`,
`Makefile` or `.pre-commit-config.yaml` exists. Consequently **every command in §8 and §12 as
written fails today** (`uv run harness ...`, `lint-imports`, `pytest -m statistical`).

All three defects §14 predicts are present. Two are wrong-conclusion class:

| §14 | Predicted | Found | Evidence |
|---|---|---|---|
| 14.1 | Pairing drops instead of raising | **Confirmed** | `harness/report/stats.py:91-97` |
| 14.2 | Usage-block dishonesty | **Confirmed, worse than predicted**, substitutes `0`, not an estimate | `harness/clients/openai_compatible.py:160-161`, `:227` |
| 14.3 | Parallel metric definitions | **Partially confirmed**, see §9 | R-06 |

---

## 1. Module map

58 Python modules, 12 packages, 11,104 LOC under `harness/`.

| Package | LOC | Cover | Responsibility | Imports → | Imported by ← |
|---|---:|---:|---|---|---|
| `ui` | 2872 | 39.0% | Streamlit shell, design system, screens | clients, eval, orchestration, profiles, report, store |, |
| `clients` | 1841 | 58.6% | Provider layer: endpoints, adapters, routing, retry, cost meter |, | orchestration, rag, ui |
| `report` | 1620 | 64.9% | aggregate, stats, decide, gate, html | eval | ui |
| `eval` | 1473 | 82.4% | metrics, scorer registry, judge, probes | store | orchestration, report, ui |
| `orchestration` | 1322 | 59.2% | runner, passes, collector, arena, jobs | cache, clients, eval, profiles, rag, store, tuning | ui |
| `rag` | 530 | 38.5% | ingest, retrieve, rerank, prompt assembly | clients, store | orchestration, profiles, tuning |
| `profiles` | 522 | 76.4% | profile config + validation, dataset loading | rag, store | orchestration, tuning, ui |
| `store` | 432 | 88.9% | `TraceRow` schema + Parquet/DuckDB store |, | eval, orchestration, profiles, rag, tuning, ui |
| `tuning` | 213 | 62.4% | equal-budget candidate search | profiles, rag, store | orchestration |
| `cache` | 197 | 84.7% | content-addressed cache |, | orchestration |
| `env` | 82 | 89.2% | `.env` loading |, | root scripts |

Root scripts outside the package: `main.py` (CLI, 12 subcommands), `app.py` (Streamlit),
`dashboard.py` (read-only), `prepare_dataset.py`.

**Observation.** `store` has 6 inbound edges and 0 outbound, it behaves as the *contracts* layer,
not as a storage layer. That single fact explains 16 of the 17 layering violations below.

---

## 2. Layering violations

§3 layer order, mapped onto today's names:

| §3 layer | idx | today |
|---|---|---|
| presentation | 0 | `ui`, root scripts |
| analysis | 1 | `report` |
| store | 2 | `store` |
| orchestration | 3 | `orchestration`, `rag`, `tuning` |
| scorers/judges | 4 | `eval` |
| providers | 5 | `clients` |
| config/contracts | 6 | `profiles`, `cache`, `env` |

**17 upward imports.** 16 share one root cause.

### 2a. Root cause, contracts living in `store/` (14 sites listed, 16 symbols)

`harness/store/schema.py` defines `TraceRow`, `EvalItem`, `ItemType`, `TaskType`, `RetrievalMode`,
`RetrievedChunk`, `Pass`, `classify_error`, all **contracts** under §3, which belong in the bottom
layer where everything may import them. Because they sit in `store` (idx 2), every lower layer that
needs a type imports upward:

| file:line | edge | symbols |
|---|---|---|
| `harness/eval/probes.py:46` | eval → store | `EvalItem, ItemType` |
| `harness/eval/probes.py:335` | eval → store | `RetrievedChunk` |
| `harness/eval/scoring.py:34` | eval → store | `EvalItem, ItemType, RetrievedChunk, TaskType` |
| `harness/orchestration/collector.py:29` | orchestration → store | `TraceRow` |
| `harness/orchestration/orchestrator.py:38` | orchestration → store | `EvalItem, TaskType` |
| `harness/orchestration/passes.py:39` | orchestration → store | `EvalItem, Pass, TraceRow` |
| `harness/orchestration/runner.py:44` | orchestration → store | `EvalItem, Pass, TaskType, TraceRow, classify_error` |
| `harness/orchestration/ui_runner.py:32` | orchestration → store | `TaskType` |
| `harness/profiles/loaders.py:31` | profiles → store | `EvalItem, ItemType` |
| `harness/profiles/profile.py:23` | profiles → store | `RetrievalMode, TaskType` |
| `harness/rag/prompt.py:25` | rag → store | `RetrievedChunk, TaskType` |
| `harness/rag/rerank.py:28` | rag → store | `RetrievedChunk` |
| `harness/rag/retrieve.py:31` | rag → store | `RetrievalMode, RetrievedChunk` |
| `harness/tuning/search.py:39` | tuning → store | `RetrievalMode` |

**Fix:** move the dataclasses to `harness/contracts/`. Mechanical; 16 violations become 0.
Owning phase: 2.

### 2b. Genuine violations (3 sites)

| file:line | edge | note |
|---|---|---|
| `harness/orchestration/ui_runner.py:33` | orchestration → `store.store.TraceStore` | real dependency; §3 wants orchestration to write through a `TraceSink` protocol defined in contracts |
| `harness/profiles/loaders.py:30` | profiles (contracts) → `rag.documents.Document` | `Document` is itself a contract, misplaced |
| `harness/profiles/loaders.py:246` | profiles (contracts) → `rag.ingest.chunk_text` | genuine logic dependency: the loader performs chunking |

---

## 3. Network and I/O boundary audit

39 I/O call sites outside `clients/` and `store/`.

| kind | count | packages |
|---|---:|---|
| file | 20 | ui (7), cache (5), profiles (5), report (2), env (1) |
| clock | 8 | orchestration: `jobs.py:30,53,61,67`; `passes.py:107,138,182,279` |
| rng | 11 | eval (7), orchestration (1), rag (1), report (1), tuning (1) |
| http | 0 |, |
| subprocess | 0 |, |

**No *direct* HTTP or subprocess call exists outside the provider layer.**
> **Correction (added later).** This claim is wrong as stated. `harness/rag/` reaches
> `huggingface.co` indirectly, through `qdrant_client` → FastEmbed, at ingest and at
> every sparse query. An AST scan for `requests`/`httpx`/`urlopen` cannot see a call
> three libraries deep; the socket block added as R-07 is what caught it. See
> [`DEBT.md`](DEBT.md) R-17.

The direct-call audit below still holds,
and it holds by construction rather than by luck: `harness/clients/endpoints.py` is deliberately
SDK-free so capability views can enumerate providers without importing an adapter (`endpoints.py:1-16`).

**RNG.** All 11 sites are *seeded-local* (`random.Random(seed)`), not global, so I10 determinism
holds, but §4 requires an **injected** `Generator`, which none of them take:
`harness/report/aggregate.py:209` (Elo shuffle), `harness/tuning/search.py:127`,
`harness/rag/prompt.py:66`, `harness/orchestration/ui_runner.py:67`. `harness/eval/probes.py`
threads an `rng: random.Random` parameter (`:121`, `:193`), the pattern the others should follow.

**Clock.** 8 direct `time.time()` / `time.perf_counter()` calls in `orchestration`. §5 requires a
frozen clock for deterministic tests; these are unmockable as written.

---

## 4. `TraceRow` contract audit

Defined at `harness/store/schema.py:110`. Coverage 97.8%.

| §3 requirement | Status | Evidence |
|---|---|---|
| Frozen pydantic v2 model | **No**, plain mutable `@dataclass` | `schema.py:27`, `:110` |
| Explicit `schema_version` | **No**, field absent | grep: 0 matches in `schema.py` |
| Explicit `pyarrow` schema | **No**, inferred by pandas at write time | `harness/store/store.py` |
| Domain newtypes (`RunId`, `ItemId`, `ModelRef`, `Usd`, `Ms`) | **No**, bare `str` / `float` | `schema.py:120-126` |
| Single `extra: dict[str, JsonValue]` escape hatch | **No**, field absent | grep |
| Nothing downstream opens a provider connection | **Holds** | `report/` imports only `eval`; see §1 |

**Shape:** ~70 fields, flat, grouped by subsystem (identity/provenance, what happened, retrieval
metrics, answer metrics, citation metrics, robustness, efficiency, diagnostics).

Reproducibility hash fields exist: `profile_cfg_hash`, `retrieval_cfg_hash`, `prompt_cfg_hash`,
`gen_params_hash` (`schema.py:129-133`), but **`apparatus_hash` does not** (I2), and the run
manifest those hashes are documented to point at (`schema.py:127-128`: "Full configs live in the run
manifest") **does not exist anywhere in the repo** (I9).

**Post-construction mutation.** The row is built empty and filled field-by-field:
`harness/orchestration/runner.py:133-138` (cache-hit path) and `:146-151` (generation path), ~19
assignment sites. This is *pre-write* mutation, so it does not violate I8 as stated, but it is
precisely why `TraceRow` cannot be frozen without restructuring `run_item`. Phase 2 must budget for it.

---

## 5. Typing audit

`mypy` is **not installed** in `.venv` (`pip list` shows only `pytest`, `ruff`). No `mypy --strict`
error count can be produced. Counts below are from AST analysis.

| Package | defs | untyped defs | `Any` refs | `type: ignore` |
|---|---:|---:|---:|---:|
| clients | 85 | 18 | 0 | 1 |
| ui | 73 | 14 | 2 | 1 |
| eval | 70 | 2 | 16 | 0 |
| orchestration | 55 | 8 | 0 | 0 |
| report | 49 | 5 | 1 | 0 |
| cache | 18 | 2 | 1 | 0 |
| rag | 18 | 6 | 1 | 0 |
| profiles | 13 | 0 | 0 | 0 |
| store | 13 | 1 | 0 | 0 |
| tuning | 6 | 0 | 0 | 0 |
| **total** | **400** | **56** | **22** | **2** |

§4 demands `mypy --strict` on contracts/providers/scorers/stats/store/bench. The two worst offenders
against that list are `clients` (18 untyped defs, it *is* the providers package) and `eval`
(16 `Any` references, it *is* the scorers package).

**Ruff gap.** Against the exact §4 ruleset (`E,F,I,UP,B,SIM,RUF,ANN,ARG,PTH,TRY,S,C4,PD,NPY,PL`):
**463 errors, only 29 auto-fixable.** Largest classes: `ANN201/202/003` (42), `ARG002` (10),
`S110` try-except-pass (9), `B905` zip-without-strict (9), `S311` non-crypto RNG (8),
`PLR0912`/`PLR0915` complexity (16).

---

## 6. Test audit

**282 tests, all passing offline, ~26s.**

| File | Tests |
|---|---:|
| `tests/test_ui.py` | 79 |
| `tests/test_probes_and_engine.py` | 70 |
| `tests/test_stats_and_decide.py` | 44 |
| `tests/test_reliability.py` | 34 |
| `tests/test_solid.py` | 22 |
| `tests/test_orchestration.py` | 20 |
| `tests/test_core.py` | 13 |

**Network.** No test constructs a real client against a live endpoint; `tests/conftest.py:36`
supplies a deterministic `FakeClient`. `tests/test_reliability.py` matches a grep for provider-SDK
names but uses them to build **synthetic exceptions** for retry classification, not connections.
**However there is no socket block**, §5's autouse `socket.socket` fixture is absent from
`tests/conftest.py`, so the offline guarantee is convention, not enforcement.

**Markers.** None of `live`, `slow`, `integration`, `statistical` are registered or used; the only
`pytest.mark` uses are `parametrize` (4 sites). `pyproject.toml:57` sets `--strict-markers`, so
`pytest -m statistical` (§8, §12) collects **0 tests and exits 0** rather than failing loudly,
a silent green.

**Coverage vs §5 floors.** Measured; `pytest-cov` had to be installed into `.venv` to obtain these,
it is declared at `requirements-dev.txt:7` but was not present.

| §5 package | floor | today | verdict |
|---|---:|---:|---|
| `stats` → `report/stats.py` | 95% | **89.0%** | ✗ |
| `scorers` → `eval/` | 95% | **82.4%** | ✗ |
| `contracts` → `store/schema.py` | 90% | **97.8%** | ✓ |
| everything else | 80% | **61.0% total** | ✗ |

**Six modules at 0% coverage:** `harness/report/html.py` (135 stmts),
`harness/orchestration/ui_runner.py` (134), `harness/ui/components.py` (84),
`harness/rag/ingest.py` (64), `harness/orchestration/jobs.py` (54),
`harness/clients/together_client.py` (11).

**Absent suites (§5):** property (`hypothesis` not installed), contract (no cassettes directory),
golden/snapshot, statistical simulation. `tests/test_solid.py` exists and does define a new scorer
inside the test file, the seam suite is genuinely present and is the strongest part of the suite.

---

## 7. Invariant risk register

| # | Invariant | Status | Evidence |
|---|---|---|---|
| **I1** | Paired comparison | **Unenforced, wrong-conclusion risk** | `report/stats.py:91-97`: `dropna` + `index.intersection` silently drops items one model lacks, and returns `(array([]), array([]))` on zero overlap (`:96-97`) instead of raising. §5 requires a raise. The caller sees a smaller `n_pairs` with no warning; if a model errored on its hard items, the surviving set is biased toward easy ones. |
| **I2** | Apparatus pinned | **Convention only** | Config pins embedder/reranker/judge (`configs/models.yaml:66-69`) and a same-family judge is rejected, but **no `apparatus_hash` exists** (grep: 0 matches), so apparatus drift across runs is undetectable. |
| **I3** | All spend metered | **Partially enforced, wrong-conclusion risk** | Four buckets exist and are real (`clients/cost.py`); an unpriced model raises rather than costing zero (`clients/pricing.py:11-16`). **But** `clients/openai_compatible.py:160-161` coerces a missing usage block to `0` via `getattr(usage, "prompt_tokens", 0) or 0`, and `:227` does the same for embeddings. A provider omitting usage is billed **$0.00**; because cost carries a negative weight, that model then rises in the composite. |
| **I4** | Cost negatively weighted | **Enforced by code** | `profiles/profile.py:25-28` defines `LOWER_IS_BETTER`; rejected at load `:191-194` with the offending key named. |
| **I5** | Paired, corrected significance | **Enforced by code** | Exact binomial McNemar (`report/stats.py:131-145`, explicitly not chi-square, with the reason documented), paired bootstrap resampling item indices (`:101-128`), Holm–Bonferroni (`:198`). Selection is automatic, not caller-chosen. |
| **I6** | Non-significant means non-significant | **Enforced by code, verified by execution** | Observed live this session: three models at n=29 reported `no significant difference (p=1.000)` for all three pairs, alongside a power line stating the smallest detectable gap. |
| **I7** | Failure ≠ wrongness | **Partially enforced** | Truncation rate is reported separately (`report/aggregate.py`) and was the finding that explained two models scoring 0.000 in this session's live run. `extraction_failure_rate`, `refusal_rate`, `over_refusal_rate`, `format_violation_rate` (§10.7) do not exist. |
| **I8** | Traces append-only | **Enforced by design** | `configs/run.yaml:60-63`, the store is a directory, one Parquet part appended per checkpoint, explicitly to avoid the O(n²) rewrite. Budget abort keeps written rows. No re-scoring layer exists, so the derived-table rule is untested rather than violated. |
| **I9** | Reproducibility manifest | **Unenforced, absent** | No `RunManifest` anywhere (grep: 0 matches). Per-row config hashes exist but nothing records git SHA, dirty flag, dataset hash, package versions or pricing version. Under I9's own wording, **no run currently produced by this repo is a result**. |
| **I10** | Determinism where claimed | **Enforced by code** | Probes seeded (`eval/probes.py:212`); canary derived via sha256 rather than builtin `hash`, with `PYTHONHASHSEED` given as the reason (`:88`). Splits seeded (`configs/run.yaml:19`). |
| **I11** | Config is the spine | **Largely holds** | Profiles/models/pricing/run are YAML; §3's extension seams are pinned by `tests/test_solid.py`. Adding a *provider* still requires a Python adapter, which §3 explicitly permits. |
| **I12** | Secrets and canaries never leak | **Unenforced** | No `structlog`, no redaction processor (grep: 0 matches). Cache keys are sha256 over model/messages/params (`cache/cache.py:66-79`) and correctly exclude credentials. **But** canaries are generated as plaintext (`eval/probes.py:245`), injected into the prompt, and the prompt is persisted verbatim in `TraceRow.assembled_prompt` (`store/schema.py:139`), flowing into Parquet and the HTML export. §2 I12 requires hashed storage. |

**Enforced by code: I4, I5, I6, I8, I10 (5).**
**Convention only: I2, I7, I11 (3).**
**Unenforced: I1, I3, I9, I12 (4).**

---

## 8. Ranked remediation

Severity: **BC** blocks-correctness · **BM** blocks-maintainability · **P** polish.

| # | Finding | Sev | Effort | Phase |
|---|---|---|---|---|
| R-01 | `paired_values` drops unpaired items instead of raising (I1) | **BC** | S | 5 |
| R-02 | Missing usage block coerced to 0 tokens ⇒ $0.00 cost ⇒ inflated rank (I3) | **BC** | S | 2 |
| R-03 | No `RunManifest`; no run is reproducible (I9) | **BC** | M | 2 |
| R-04 | No `apparatus_hash`; apparatus drift undetectable across runs (I2) | **BC** | S | 2 |
| R-05 | No statistical simulation suite: I5/I6 calibration asserted but unverified | **BC** | M | 5 |
| R-06 | Confirm single-definition for the composite across report/decide/gate/html (§14.3) | **BC** | M | 4 |
| R-07 | No socket block; the offline guarantee is convention (§5) | **BM** | S | 1 |
| R-08 | Markers unregistered ⇒ `pytest -m statistical` silently collects 0 and exits 0 | **BM** | S | 1 |
| R-09 | Contracts live in `store/` ⇒ 16 upward imports | **BM** | S | 2 |
| R-10 | `TraceRow` not frozen, no `schema_version`, no explicit pyarrow schema | **BM** | M | 2 |
| R-11 | Canary plaintext persisted in `assembled_prompt` (I12) | **BM** | S | 6 |
| R-12 | 6 modules at 0% coverage; `html.py` + `ui_runner.py` alone are 269 untested stmts | **BM** | M | 1/8 |
| R-13 | 463 ruff errors under the §4 ruleset; no `.importlinter`, `Makefile`, pre-commit | **BM** | L | 1 |
| R-14 | No structlog/redaction; `print` is the only output channel | **BM** | M | 6 |
| R-15 | RNG seeded-locally rather than injected (4 sites); clock unmockable (8 sites) | **P** | S | 1/3 |

---

## 9. Undetermined

- **Per-package `mypy --strict` error counts**, mypy is not installed in `.venv`. Needs
  `pip install mypy`, then `mypy harness/store harness/clients harness/eval harness/report`.
- **Whether the composite is computed identically in all four consumers (§14.3)**: `report/` has a
  single `aggregate.py` and `decide.py`/`gate.py`/`html.py` import from it, which is the right
  shape; but the arithmetic at each call site was not diffed. R-06 is scoped to settle it.
- **Whether `store.py` ever rewrites an existing Parquet part** (I8 in the crash case), the
  append path is documented at `configs/run.yaml:60-63`; the crash/resume path was not exercised.
- **Live behaviour of any provider other than Groq**, only `GROQ_API_KEY` was present this session.
  Together, OpenAI, Anthropic, Fireworks, DeepInfra, OpenRouter, vLLM, Ollama and LM Studio adapters
  are read-only findings.
