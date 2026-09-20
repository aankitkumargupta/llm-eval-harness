"""
Tests for the architectural seams.

These do not test behaviour that a user sees; they test that the *structure*
holds. Every one of them corresponds to a specific violation that existed
before the refactor, and each would fail again if someone quietly reintroduced
it — which is the only way an architectural rule survives contact with a
deadline.

The load-bearing test is `test_a_new_metric_needs_no_existing_file_changed`:
Open/Closed is a claim about what a *future* change costs, so the only honest
way to check it is to make that change and assert nothing else moved.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from harness.cache.cache import DiskCache, KeyValueCache, NullCache
from harness.clients.base import (
    Capability,
    CapabilityError,
    DocumentReranker,
    Embedder,
    ProviderInfo,
    TextGenerator,
    require,
    supports,
)
from harness.clients.pricing import PricingProvider, PricingRegistry
from harness.eval.scoring import (
    DEFAULT_SCORERS,
    AbstentionScorer,
    JudgeScorer,
    RetrievalScorer,
    Scorer,
    ScoringContext,
    run_scorers,
)
from harness.orchestration.arena import TraceReader
from harness.orchestration.collector import RowCollector, TraceSink
from harness.orchestration.orchestrator import Orchestrator
from harness.orchestration.passes import (
    AdaptedPass,
    BaselinePass,
    EvaluationPass,
    LatencyPass,
)
from harness.store.schema import EvalItem, ItemType, Pass, TraceRow
from harness.store.store import TraceStore


# =========================================================================== #
#  Single Responsibility
# =========================================================================== #
def test_orchestrator_delegates_rather_than_doing_the_work():
    """It had six reasons to change; now it assembles collaborators.

    Checked by shape rather than by line count: the passes, the buffering and
    the arena must each live somewhere else, and the orchestrator must hold a
    collector rather than its own pile of counters.
    """
    import inspect

    from harness.orchestration import orchestrator as O

    source = inspect.getsource(O)
    for leaked in ("ThreadPoolExecutor", "enumerate_candidates", "_pending"):
        assert leaked not in source, (
            f"{leaked!r} is pass/collector work and should not be in the facade")
    assert "RowCollector" in source and "ArenaService" in source


def test_collector_owns_buffering_and_counting():
    """One class, one job: rows in, batched writes out, counts kept."""
    written: list[list[TraceRow]] = []

    class Sink:
        def write(self, rows):
            batch = list(rows)
            written.append(batch)
            return len(batch)

    collector = RowCollector(Sink(), checkpoint_every=3)
    rows = [TraceRow(run_id="r", item_id=f"q{i}", model="m", profile="p",
                     pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE)
            for i in range(7)]
    rows[4].error = "boom"
    for r in rows:
        collector.collect(r)
    collector.flush()

    assert [len(b) for b in written] == [3, 3, 1], "should flush in batches"
    assert collector.written == 7
    assert collector.errors == 1


def test_collector_is_thread_safe():
    """The throughput lane collects from many workers at once."""
    import concurrent.futures as cf

    class Sink:
        def write(self, rows):
            return len(list(rows))

    collector = RowCollector(Sink(), checkpoint_every=10)
    row = lambda i: TraceRow(  # noqa: E731
        run_id="r", item_id=f"q{i}", model="m", profile="p",
        pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE)
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda i: collector.collect(row(i)), range(400)))
    collector.flush()
    assert collector.done == 400
    assert collector.written == 400


# =========================================================================== #
#  Open/Closed
# =========================================================================== #
class AnswerLengthScorer:
    """A brand-new metric family, defined entirely in this test file.

    It touches no harness module. If the scoring pipeline is genuinely open for
    extension, this is all it takes.
    """

    name = "answer_length"
    produces = ("completeness",)

    def applies(self, ctx: ScoringContext) -> bool:
        return "completeness" in ctx.active

    def score(self, ctx: ScoringContext) -> dict:
        words = len((ctx.answer or "").split())
        return {"completeness": min(1.0, words / 50.0)}


def _ctx(**kw) -> ScoringContext:
    base = {
        "item": EvalItem(item_id="q1", query="what is the fine?",
                         gold_answer="500", gold_passage_ids=["d#0"]),
        "model": "m",
        "answer": "The fine is 500 [d#0].",
        "active": {"accuracy"},
    }
    base.update(kw)
    return ScoringContext(**base)


def test_a_new_metric_needs_no_existing_file_changed():
    """The Open/Closed claim, made concrete.

    A scorer defined outside the package participates fully, purely by being
    passed in. Before the refactor this metric would have required editing
    `run_item` — the same function that owns retrieval, generation and cost.
    """
    ctx = _ctx(active={"completeness"}, answer="one two three four five")
    result = run_scorers(ctx, scorers=(*DEFAULT_SCORERS, AnswerLengthScorer()))
    assert result.values["completeness"] == pytest.approx(0.1)


def test_scorers_can_be_swapped_wholesale():
    """Running exactly one family is a supported operation, which is what makes
    each one independently testable."""
    ctx = _ctx(active={"hit_rate_at_k", "mrr"}, retrieved_ids=["d#0", "x"], k=2)
    result = run_scorers(ctx, scorers=(RetrievalScorer(),))
    assert set(result.values) == {"hit_rate_at_k", "mrr"}
    assert result.values["hit_rate_at_k"] == 1.0


def test_every_registered_scorer_satisfies_the_protocol():
    for scorer in DEFAULT_SCORERS:
        assert isinstance(scorer, Scorer), f"{scorer} is not a Scorer"
        assert scorer.name and scorer.produces


def test_one_failing_scorer_does_not_discard_the_others():
    """The generation was paid for. A judge reply that fails to parse must not
    take the retrieval and citation metrics down with it."""

    class Exploding:
        name = "exploding"
        produces = ("accuracy",)

        def applies(self, ctx):
            return True

        def score(self, ctx):
            raise RuntimeError("judge returned nonsense")

    ctx = _ctx(active={"hit_rate_at_k"}, retrieved_ids=["d#0"], k=1)
    result = run_scorers(ctx, scorers=(RetrievalScorer(), Exploding()))
    assert result.values["hit_rate_at_k"] == 1.0        # survived
    assert "exploding" in result.failures                # and was recorded


def test_passes_are_interchangeable_strategies(tmp_path):
    """Adding a fourth kind of pass is a new class, not an edit to a long method."""
    for pass_cls in (BaselinePass, AdaptedPass, LatencyPass):
        assert hasattr(pass_cls, "run") and hasattr(pass_cls, "name")
    # Structural conformance, so a user-defined pass needs no base class.
    made = BaselinePass(
        collector=RowCollector(TraceStore(str(tmp_path / "t"))),
        make_context=lambda *a, **k: None, meter=None)
    assert isinstance(made, EvaluationPass)


# =========================================================================== #
#  Liskov Substitution
# =========================================================================== #
def test_null_cache_is_not_a_disk_cache():
    """It cannot honour the round-trip contract, so it must not claim the type.

    As a subclass it silently broke every caller that assumed `set` then `get`
    returns the value — which is the whole reason a cache exists.
    """
    assert not isinstance(NullCache(), DiskCache)


def test_both_caches_satisfy_the_protocol_and_keep_their_own_contract():
    with tempfile.TemporaryDirectory() as d:
        disk, null = DiskCache(d), NullCache()
        assert isinstance(disk, KeyValueCache)
        assert isinstance(null, KeyValueCache)

        key = disk.key_for_generation("m", [{"role": "user", "content": "hi"}], {})
        disk.set(key, {"text": "cached"})
        assert disk.get(key) == {"text": "cached"}, "disk cache must round-trip"

        null.set(key, {"text": "cached"})
        assert null.get(key) is None, "null cache must never serve anything"


def test_caches_agree_on_the_key_scheme():
    """Two implementations that disagreed on keys would silently fail to share
    entries, and the bug would look like a cache-miss-rate problem."""
    with tempfile.TemporaryDirectory() as d:
        disk, null = DiskCache(d), NullCache()
        for make in ("key_for_generation", "key_for_embedding"):
            args = (("m", [], {}) if make == "key_for_generation" else ("m", "t"))
            assert getattr(disk, make)(*args) == getattr(null, make)(*args)


# =========================================================================== #
#  Interface Segregation
# =========================================================================== #
class _GenOnly:
    """A provider that can generate and nothing else — Groq's real shape."""

    info = ProviderInfo(name="genonly", supports_embeddings=False,
                        supports_rerank=False)

    def generate(self, *a, **k): ...
    def judge(self, *a, **k): ...


class _Full:
    info = ProviderInfo(name="full", supports_embeddings=True,
                        supports_rerank=True)

    def generate(self, *a, **k): ...
    def judge(self, *a, **k): ...
    def embed(self, *a, **k): ...
    def rerank(self, *a, **k): ...


def test_narrow_protocols_distinguish_real_capabilities():
    """The fat interface made every provider look identical, so the only signal
    was a boolean flag that could — and did — disagree with the code."""
    gen_only, full = _GenOnly(), _Full()

    assert isinstance(gen_only, TextGenerator)
    assert not isinstance(gen_only, Embedder)
    assert not isinstance(gen_only, DocumentReranker)

    assert isinstance(full, TextGenerator)
    assert isinstance(full, Embedder)
    assert isinstance(full, DocumentReranker)


def test_supports_requires_method_and_declaration_to_agree():
    """One adapter class serves several vendors, so capability is a property of
    the instance. Structural presence alone would call Groq an embedder."""
    declared_off = _Full()
    declared_off.info = ProviderInfo(name="full", supports_embeddings=False,
                                     supports_rerank=True)
    assert not supports(declared_off, Capability.EMBED), "declaration must count"
    assert supports(declared_off, Capability.RERANK)
    assert not supports(_GenOnly(), Capability.EMBED), "absence must count"


def test_require_names_the_provider_and_the_fix():
    with pytest.raises(CapabilityError) as exc:
        require(_GenOnly(), Capability.EMBED, "Set `embedding_provider`.")
    message = str(exc.value)
    assert "genonly" in message and "embed" in message
    assert "embedding_provider" in message, "must say what to change"


def test_anthropic_adapter_does_not_claim_capabilities_it_lacks():
    """It used to define embed/rerank purely to raise, which made it
    structurally an Embedder while being unable to embed."""
    from harness.clients.anthropic_client import AnthropicClient

    assert not hasattr(AnthropicClient, "embed")
    assert not hasattr(AnthropicClient, "rerank")
    assert hasattr(AnthropicClient, "generate")


# =========================================================================== #
#  Dependency Inversion
# =========================================================================== #
def test_writers_depend_on_a_sink_not_the_store():
    """A pass has no business reading the store, so it must not depend on a
    type that can."""
    with tempfile.TemporaryDirectory() as d:
        assert isinstance(TraceStore(str(Path(d) / "t")), TraceSink)

    class ListSink:
        def __init__(self):
            self.rows = []

        def write(self, rows):
            self.rows.extend(rows)
            return len(self.rows)

    sink = ListSink()
    assert isinstance(sink, TraceSink)
    RowCollector(sink, checkpoint_every=1).collect(
        TraceRow(run_id="r", item_id="q", model="m", profile="p",
                 pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE))
    assert len(sink.rows) == 1, "an in-memory sink must work without a store"


def test_arena_depends_on_a_reader_not_a_writer():
    with tempfile.TemporaryDirectory() as d:
        assert isinstance(TraceStore(str(Path(d) / "t")), TraceReader)


def test_pricing_is_an_abstraction():
    """A team with negotiated rates should be able to substitute the policy
    without the runner noticing."""
    assert isinstance(PricingRegistry("configs/pricing.yaml"), PricingProvider)

    class FlatRate:
        def generation_cost(self, model, p, c):
            return 0.001

        def embedding_cost(self, model, tokens):
            return 0.0

        def rerank_cost(self, model, tokens=0, n_docs=0):
            return 0.0

    assert isinstance(FlatRate(), PricingProvider)


def test_run_context_accepts_substituted_collaborators(tmp_path, fake_client):
    """The end-to-end proof of DIP: a context built entirely from stand-ins."""
    from harness.orchestration.runner import RunContext, run_item
    from harness.rag.prompt import PromptConfig
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import RetrievalMode, TaskType

    class FlatRate:
        def generation_cost(self, model, p, c):
            return 0.001

        def embedding_cost(self, model, tokens):
            return 0.0

        def rerank_cost(self, model, tokens=0, n_docs=0):
            return 0.0

    row = run_item(
        EvalItem(item_id="q1", query="hello", gold_answer="hello"),
        "m", "p", Pass.BASELINE,
        RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
        RunContext(client=fake_client, retriever=None, pricing=FlatRate(),
                   judge=None, cache=NullCache(), embedding_model="",
                   task=TaskType.DIRECT),
        "run1", active_metrics=["token_f1"], accuracy_scorer="contains")
    assert row.error is None
    assert row.gen_cost_usd == pytest.approx(0.001)


# =========================================================================== #
#  Backwards compatibility — a refactor that breaks callers is a rewrite
# =========================================================================== #
def test_public_api_is_unchanged():
    for name in ("run_baseline", "run_adapted", "run_latency", "run_arena"):
        assert callable(getattr(Orchestrator, name))

    # Symbols that moved must still resolve from their original module.
    from harness.cache.cache import Cache
    from harness.orchestration.orchestrator import RunReport, weighted_mean
    from harness.orchestration.runner import RunContext, parse_label

    assert Cache is DiskCache
    assert callable(weighted_mean) and callable(parse_label)
    assert RunReport and RunContext


def test_scorer_ordering_lets_the_judge_win_accuracy():
    """Two scorers legitimately write `accuracy`; the registry documents that
    the judge is last and authoritative. Silently reordering would change
    results without changing any metric's code."""
    names = [s.name for s in DEFAULT_SCORERS]
    assert names.index("accuracy_deterministic") < names.index("judge")
    assert isinstance(DEFAULT_SCORERS[-1], JudgeScorer)


def test_unanswerable_items_are_scored_even_without_the_metric_listed():
    """An unanswerable item exists for exactly one purpose, so the abstention
    scorer applies whether or not the profile remembered to list it."""
    ctx = _ctx(item=EvalItem(item_id="u1", query="?",
                             item_type=ItemType.UNANSWERABLE),
               active=set(), answer="I don't know.")
    assert AbstentionScorer().applies(ctx)
    assert run_scorers(ctx, scorers=(AbstentionScorer(),)) \
        .values["abstention_correct"] == 1.0


# =========================================================================== #
#  Seam: a brand-new BENCHMARK ADAPTER, defined here, participating fully.
# =========================================================================== #
#
# CLAUDE.md §9 Phase 7 requires exactly this: "tests/test_solid.py gains a
# brand-new benchmark adapter defined inside the test file that participates
# end to end". Open/Closed is a claim about what a future change costs, so the
# only honest check is to make that change and confirm nothing else moved.
#
# Nothing below imports from harness/bench/adapters/. If adding a benchmark
# required touching the runner, the store, the stats layer or the reporter,
# this test could not be written — which is the point.


class _RomanNumeralAdapter:
    """A benchmark invented for this test: convert a Roman numeral to digits.

    Four pure methods, no network, no clock, no global RNG, no provider call.
    """

    id = "roman_numerals"

    VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}

    def __init__(self, spec):
        self.spec = spec

    def load(self, *, seed=None, limit=None):
        from harness.store.schema import EvalItem, ItemType
        pairs = [("IV", "4"), ("IX", "9"), ("XL", "40"), ("XC", "90"),
                 ("CD", "400"), ("MCMXCIV", "1994")]
        items = [EvalItem(item_id=f"rn_{i}", query=f"Convert {r} to digits.",
                          item_type=ItemType.ANSWERABLE, gold_answer=v)
                 for i, (r, v) in enumerate(pairs)]
        return items[:limit] if limit else items

    def prompt(self, item, shots=()):
        from harness.bench.contracts import Prompted
        msgs = [{"role": "system", "content": "Reply with '#### <number>'."}]
        for s in shots:
            msgs.append({"role": "user", "content": s.query})
            msgs.append({"role": "assistant", "content": f"#### {s.gold_answer}"})
        msgs.append({"role": "user", "content": item.query})
        return Prompted(messages=tuple(msgs))

    def extract(self, raw):
        from harness.bench.extract import run_chain
        return run_chain(raw, [{"kind": "regex", "pattern": r"####\s*(-?\d+)"},
                               {"kind": "numeric"}])

    def score(self, item, extraction):
        from harness.bench.contracts import ScoreSet
        from harness.bench.extract import numeric_equal
        s = ScoreSet()
        if extraction.failed:
            return s.set("accuracy", None).set("extraction_failed", True)
        return (s.set("accuracy",
                      1.0 if numeric_equal(extraction.value, item.gold_answer)
                      else 0.0)
                .set("extraction_failed", False)
                .set("extracted_via", extraction.via))


def _roman_spec():
    from harness.bench.spec import BenchmarkSpec
    return BenchmarkSpec.from_dict({
        "id": "roman_numerals", "version": 1, "family": "math", "task": "direct",
        "source": {"kind": "local", "ref": "<in-test>", "licence": "CC0",
                   "split": "test"},
        "sampling": {"seed": 1, "samples_per_item": 1},
        "prompt": {"few_shot": {"n": 0, "selection": "fixed", "pool_split": "train"},
                   "decoding": {"max_tokens": 64}},
        "scoring": {"mode": "generative", "chance_level": 0.0,
                    "extraction": {"chain": [{"kind": "numeric"}]}},
    })


def test_a_new_benchmark_adapter_satisfies_the_protocol_with_no_library_edits():
    from harness.bench.contracts import BenchmarkAdapter

    assert isinstance(_RomanNumeralAdapter(_roman_spec()), BenchmarkAdapter)


def test_a_new_benchmark_adapter_passes_the_shared_contract_properties():
    """The same properties the shipped adapters are held to, applied to one
    the library has never seen."""
    from harness.bench.contracts import Failed

    a = _RomanNumeralAdapter(_roman_spec())
    items = a.load()

    assert len({i.item_id for i in items}) == len(items)
    assert a.prompt(items[0], ()) == a.prompt(items[0], ())      # pure
    assert a.extract("garbage").failed                            # total
    assert a.score(items[0], Failed("x")).get("accuracy") is None  # I7
    assert a.score(items[0], a.extract("#### 4")).get("accuracy") == 1.0


def test_a_new_benchmark_adapter_runs_end_to_end_through_the_shared_runner():
    """The load-bearing assertion. A benchmark defined entirely inside this
    test file runs through the real runner, produces real TraceRows, and is
    summarised by the real metrics module — with zero edits anywhere else."""
    from harness.bench.metrics import summarise
    from harness.bench.runner import run_benchmark
    from harness.clients.fake_client import FakeClient

    spec = _roman_spec()
    rep = run_benchmark(adapter=_RomanNumeralAdapter(spec), spec=spec,
                        models=["fake:a", "fake:b"],
                        client=FakeClient(accuracy=1.0),
                        run_id="seam_roman")

    assert rep.errors == 0
    assert len(rep.rows) == 12                      # 6 items x 2 models
    assert {r.benchmark for r in rep.rows} == {"roman_numerals"}
    assert all(r.spec_hash == spec.spec_hash() for r in rep.rows)

    import pandas as pd
    out = summarise(pd.DataFrame([r.to_dict() for r in rep.rows]),
                    benchmark="roman_numerals")
    assert set(out["model"]) == {"fake:a", "fake:b"}
    assert out["n_items"].sum() == 12


class _ScriptedClient:
    """A two-line provider for the seam test.

    `FakeClient` cannot answer Roman numerals — it has no idea what this
    benchmark is, which is the correct behaviour for a generic fake and is why
    a benchmark-specific stub belongs here rather than in the library.
    """

    def __init__(self, wrong_for: set[str] | None = None):
        self.wrong_for = wrong_for or set()

    def generate(self, model, messages, **kw):
        from harness.bench.adapters.gsm8k import GSM8KAdapter  # noqa: F401
        from harness.clients.base import GenResult

        question = messages[-1]["content"]
        roman = question.replace("Convert ", "").replace(" to digits.", "").strip()
        value = _roman_to_int(roman)
        if model in self.wrong_for and roman in ("XL", "XC"):
            value += 1                       # a stable, partial disagreement
        return GenResult(text=f"#### {value}", prompt_tokens=8,
                         completion_tokens=4, latency_ms=1.0,
                         finish_reason="stop", model=model)


def _roman_to_int(s: str) -> int:
    vals = _RomanNumeralAdapter.VALUES
    total = 0
    for i, ch in enumerate(s):
        v = vals.get(ch, 0)
        nxt = vals.get(s[i + 1], 0) if i + 1 < len(s) else 0
        total += -v if v < nxt else v
    return total


def test_the_new_adapter_reaches_the_existing_significance_layer_unchanged():
    """§10: 'No second significance implementation.' A benchmark the stats
    layer has never heard of must flow through it untouched — same paired
    test, same correction, same refusal to call a small gap real."""
    import pandas as pd

    from harness.bench.runner import run_benchmark
    from harness.report.stats import significance_matrix

    spec = _roman_spec()
    rep = run_benchmark(adapter=_RomanNumeralAdapter(spec), spec=spec,
                        models=["scripted:a", "scripted:b"],
                        client=_ScriptedClient(wrong_for={"scripted:b"}),
                        run_id="seam_roman_sig")
    df = pd.DataFrame([r.to_dict() for r in rep.rows])

    assert rep.errors == 0
    assert df["accuracy"].notna().all(), "every item must be scoreable here"

    sig = significance_matrix(df, metric="accuracy")
    assert len(sig) == 1
    assert sig.iloc[0]["n_pairs"] == 6
    assert sig.iloc[0]["test"] == "mcnemar_exact"
    # a is perfect, b misses two of six
    assert sig.iloc[0]["diff"] == pytest.approx(2 / 6)
    # ...and at n=6 the harness correctly declines to call that real (I6).
    assert not sig.iloc[0]["significant"]


def test_an_adapter_reporting_an_undeclared_metric_fails_loudly():
    """A ScoreSet key with no TraceRow column would be computed and discarded,
    and its absence from the report would read as 'the model scored zero'."""
    import pytest as _pytest

    from harness.bench.contracts import ScoreSet
    from harness.bench.runner import run_benchmark

    class _Bad(_RomanNumeralAdapter):
        def score(self, item, extraction):
            return ScoreSet().set("accuracy", 1.0).set("invented_metric", 0.5)

    spec = _roman_spec()
    with _pytest.raises(AttributeError, match="no TraceRow column"):
        run_benchmark(adapter=_Bad(spec), spec=spec, models=["scripted:a"],
                      client=_ScriptedClient(), run_id="seam_bad")
