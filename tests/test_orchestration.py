"""
End-to-end orchestration tests, against a fake provider.

This is the layer that was never testable before: everything above `run_item`
only ran against a live API, which in practice meant it was only exercised by
spending money, and only in the shapes someone happened to try. These tests
cover the parts where a bug is expensive and silent — cost attribution, budget
aborts, resume, cache behaviour, and the latency lane's cache bypass.

`TaskType.CLASSIFY` is used throughout because it needs no vector store, so the
whole matrix walk runs with no external services at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.cache.cache import Cache
from harness.clients.cost import CostMeter
from harness.eval.scoring import parse_label
from harness.orchestration.orchestrator import Orchestrator, weighted_mean
from harness.orchestration.runner import RunContext, run_item
from harness.profiles.profile import Profile
from harness.rag.prompt import PromptConfig
from harness.rag.retrieve import RetrievalConfig
from harness.store.schema import EvalItem, ItemType, Pass, RetrievalMode, TaskType
from harness.store.store import TraceStore


class _Pricing:
    """Prices two fake models so cost attribution is checkable."""
    models = {"good": {"input": 1.0, "output": 2.0},
              "cheap": {"input": 0.1, "output": 0.2},
              "judge": {"input": 5.0, "output": 10.0}}

    def generation_cost(self, model, p, c):
        if model not in self.models:
            raise KeyError(model)
        return p / 1e6 * self.models[model]["input"] + \
            c / 1e6 * self.models[model]["output"]

    def embedding_cost(self, model, tokens):
        return tokens / 1e6 * 0.02

    def rerank_cost(self, model, tokens=0, n_docs=0):
        return 0.0


def _classify_profile(tmp: Path, items: list[dict]) -> Profile:
    import json

    evalset = tmp / "evalset.jsonl"
    evalset.write_text("\n".join(json.dumps(r) for r in items), encoding="utf-8")
    return Profile.from_dict({
        "name": "t", "task": "classify", "corpus_path": "",
        "evalset_path": str(evalset), "embedding_model": "",
        "label_set": ["yes", "no"], "accuracy_scorer": "exact",
        "active_metrics": ["accuracy", "cost_usd"],
        "metric_weights": {"accuracy": 1.0, "cost_usd": -0.1},
        "max_tokens": 32, "tuning_budget": 2,
        "knobs": {"retrieval_modes": ["dense"], "k_values": [1],
                  "rerank_options": [False], "rerank_top_n": [1],
                  "system_prompts": ["A", "B"]},
    })


@pytest.fixture
def classify_setup(tmp_path, fake_client):
    items = [{"item_id": f"q{i}", "query": f"question {i}",
              "gold_answer": "yes" if i % 2 else "no"} for i in range(8)]
    profile = _classify_profile(tmp_path, items)
    store = TraceStore(str(tmp_path / "traces"))
    meter = CostMeter()
    # The fake answers every prompt with the gold label for "good" and the
    # wrong one for "cheap", so accuracy is exactly predictable.
    from harness.profiles.loaders import load_evalset
    loaded = load_evalset(profile.evalset_path)
    fake_client.answers = {}
    return profile, store, meter, fake_client, loaded


def _orch(client, store, meter, tmp_path, **kw):
    return Orchestrator(client=client, qdrant=None, pricing=_Pricing(),
                        store=store, cache_dir=str(tmp_path / "cache"),
                        meter=meter, max_workers=2, checkpoint_every=3, **kw)


# --------------------------------------------------------------------------- #
#  Baseline pass
# --------------------------------------------------------------------------- #
def test_baseline_writes_one_row_per_item_per_model(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    report = _orch(client, store, meter, tmp_path).run_baseline(
        profile, ["good", "cheap"], items, "run1")

    assert report.rows_written == len(items) * 2
    df = store.load_run("run1")
    assert len(df) == 16
    assert set(df["model"]) == {"good", "cheap"}
    assert df["error"].isna().all()


def test_rows_are_checkpointed_not_held_to_the_end(classify_setup, tmp_path):
    """A crash at item 9,000 used to lose every paid call in the run."""
    profile, store, meter, client, items = classify_setup
    _orch(client, store, meter, tmp_path).run_baseline(
        profile, ["good"], items, "run1")
    parts = list((tmp_path / "traces").glob("part-*.parquet"))
    assert len(parts) > 1, "checkpoint_every=3 over 8 items should flush repeatedly"


def test_errors_are_recorded_and_classified_not_swallowed(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    client.fail_on = {"cheap"}
    report = _orch(client, store, meter, tmp_path).run_baseline(
        profile, ["good", "cheap"], items, "run1")

    df = store.load_run("run1")
    failed = df[df["model"] == "cheap"]
    assert report.errors == len(items)
    assert failed["error"].notna().all()
    # Attribution, not just a count: this says "lower your concurrency".
    assert set(failed["error_kind"]) == {"rate_limit"}
    # The other model's rows are unaffected - one model failing must not
    # invalidate the run.
    assert df[df["model"] == "good"]["error"].isna().all()


def test_resume_skips_completed_work(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    orch = _orch(client, store, meter, tmp_path)
    orch.run_baseline(profile, ["good"], items, "run1")
    before = len(client.calls)

    orch2 = _orch(client, store, meter, tmp_path)
    orch2.run_baseline(profile, ["good"], items, "run2", resume_from="run1")
    assert len(client.calls) == before, "resume must make no new provider calls"


# --------------------------------------------------------------------------- #
#  Cost
# --------------------------------------------------------------------------- #
def test_cost_is_attributed_per_row_and_reconciles_with_the_run_total(
        classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    _orch(client, store, meter, tmp_path).run_baseline(
        profile, ["good"], items, "run1")

    df = store.load_run("run1")
    assert df["gen_cost_usd"].notna().all()
    assert (df["gen_cost_usd"] > 0).all()
    # 100 prompt + 20 completion tokens at good's prices, per item.
    expected = (100 / 1e6 * 1.0 + 20 / 1e6 * 2.0) * len(items)
    assert df["gen_cost_usd"].sum() == pytest.approx(expected)


def test_judge_cost_is_separated_from_generation_cost(tmp_path, fake_client):
    """On a judge-scored profile the judge is often the majority of the bill;
    the original reported generation only."""
    meter = CostMeter()
    cache = Cache(str(tmp_path / "cache"))
    from harness.eval.judge import Judge

    judge = Judge(fake_client, "judge", cache=cache)
    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=judge, cache=cache, embedding_model="",
                     meter=meter, task=TaskType.DIRECT)

    def _billed_judge(model, messages, **kw):
        r = fake_client.__class__.judge(fake_client, model, messages, **kw)
        meter.record_generation(_Pricing().generation_cost("judge", 200, 30),
                                200, 30, is_judge=True)
        return r

    fake_client.judge = _billed_judge

    row = run_item(EvalItem(item_id="q1", query="q", gold_answer="a"),
                   "good", "p", Pass.BASELINE,
                   RetrievalConfig(mode=RetrievalMode.DENSE),
                   PromptConfig(), ctx, "run1",
                   active_metrics=["accuracy", "faithfulness", "cost_usd"],
                   accuracy_scorer="judge")

    assert row.gen_cost_usd and row.gen_cost_usd > 0
    assert row.judge_cost_usd and row.judge_cost_usd > 0
    assert row.cost_usd == pytest.approx(row.gen_cost_usd + row.judge_cost_usd)
    assert meter.costs.judge > 0


def test_budget_abort_stops_the_run_and_keeps_completed_rows(
        classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    # One call costs 100/1e6*1.0 + 20/1e6*2.0 = 1.4e-4, so this ceiling admits
    # roughly three items out of eight.
    per_call = _Pricing().generation_cost("good", 100, 20)
    tiny = CostMeter(budget_usd=per_call * 3)
    # Wire the fake to the meter exactly as the real adapters do: the guard
    # watches what the CLIENT bills, since only the client sees usage blocks.
    client.meter, client.pricing = tiny, _Pricing()
    report = _orch(client, store, tiny, tmp_path).run_baseline(
        profile, ["good"], items, "run1")

    assert report.aborted
    assert "Budget exceeded" in report.abort_reason
    # Items that completed before the ceiling are still on disk: a budget abort
    # stops the run, it does not discard what you already paid for.
    written = len(store.load_run("run1"))
    assert 1 <= written < len(items)
    # And spending stops close to the ceiling rather than running the matrix out.
    assert tiny.spent <= tiny.budget_usd + per_call * 2


# --------------------------------------------------------------------------- #
#  Caching
# --------------------------------------------------------------------------- #
def test_second_identical_run_is_served_from_cache(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    orch = _orch(client, store, meter, tmp_path)
    orch.run_baseline(profile, ["good"], items, "run1")
    after_first = len(client.calls)

    orch.run_baseline(profile, ["good"], items, "run2")
    assert len(client.calls) == after_first, "identical prompts must hit the cache"
    assert store.load_run("run2")["cache_hit"].all()


def test_latency_lane_bypasses_the_cache(classify_setup, tmp_path):
    """The old lane pointed at a scratch directory, so the SECOND run served
    cached timings and reported them as measured latency."""
    profile, store, meter, client, items = classify_setup
    orch = _orch(client, store, meter, tmp_path)
    orch.run_latency(profile, ["good"], items[:3], "lat1", warmup=0)
    first = len(client.calls)
    orch.run_latency(profile, ["good"], items[:3], "lat2", warmup=0)

    assert len(client.calls) == first * 2, "every latency call must hit the network"
    df = store.load_run("lat2")
    assert df["latency_ms"].notna().all()
    assert not df["cache_hit"].any()


def test_latency_lane_discards_warmup(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    _orch(client, store, meter, tmp_path).run_latency(
        profile, ["good"], items[:5], "lat1", warmup=2)
    assert len(store.load_run("lat1")) == 3


def test_latency_lane_captures_ttft(classify_setup, tmp_path):
    """TTFT was declared on TraceRow but never populated: nothing streamed."""
    profile, store, meter, client, items = classify_setup
    _orch(client, store, meter, tmp_path).run_latency(
        profile, ["good"], items[:3], "lat1", warmup=0)
    assert store.load_run("lat1")["ttft_ms"].notna().all()


# --------------------------------------------------------------------------- #
#  Adapted pass
# --------------------------------------------------------------------------- #
def test_adapted_pass_records_tuning_and_adapted_rows(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    dev, test = items[:3], items[3:]
    report = _orch(client, store, meter, tmp_path).run_adapted(
        profile, ["good"], dev, test, "run1")

    df = store.load_run("run1")
    assert set(df["pass_"]) == {"tuning", "adapted"}
    assert len(df[df["pass_"] == "adapted"]) == len(test)
    assert "good" in report.winners
    assert report.winners["good"]["config"]


def test_adapted_pass_needs_a_dev_split(classify_setup, tmp_path):
    profile, store, meter, client, items = classify_setup
    with pytest.raises(ValueError, match="dev split"):
        _orch(client, store, meter, tmp_path).run_adapted(
            profile, ["good"], [], items, "run1")


def test_every_model_faces_the_same_candidate_set(classify_setup, tmp_path):
    """The equal-budget guarantee, checked through the orchestrator rather than
    only at the enumeration function."""
    profile, store, meter, client, items = classify_setup
    _orch(client, store, meter, tmp_path).run_adapted(
        profile, ["good", "cheap"], items[:2], items[2:], "run1")

    df = store.load_run("run1")
    tuning = df[df["pass_"] == "tuning"]
    per_model = tuning.groupby("model")["prompt_cfg_hash"].nunique()
    assert per_model.nunique() == 1, "each model must see the same configs"


# --------------------------------------------------------------------------- #
#  Runner details
# --------------------------------------------------------------------------- #
def test_truncation_is_recorded(tmp_path, fake_client):
    """A cut-off answer scores badly on completeness for reasons that have
    nothing to do with model quality."""
    from harness.clients.base import GenResult

    fake_client.generate = lambda *a, **k: GenResult(
        text="the answer was cut off mid-", prompt_tokens=50,
        completion_tokens=32, latency_ms=1.0, finish_reason="length")

    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=None, cache=Cache(str(tmp_path / "c")),
                     embedding_model="", task=TaskType.DIRECT)
    row = run_item(EvalItem(item_id="q", query="q"), "good", "p", Pass.BASELINE,
                   RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
                   ctx, "run1", active_metrics=[])
    assert row.truncated is True
    assert row.finish_reason == "length"


def test_probe_metrics_are_scored_through_the_runner(tmp_path, fake_client):
    from harness.eval.probes import ProbeConfig, build_probe_suite

    suite = build_probe_suite(
        [EvalItem(item_id="q1", query="What is the fine?", gold_answer="500",
                  gold_passage_ids=["d#0"])],
        ProbeConfig(injection=1.0, unanswerable=0, noise=0, paraphrase=0))
    item = suite.items[0]
    canary = item.meta["canary"]

    from harness.clients.base import GenResult
    fake_client.generate = lambda *a, **k: GenResult(
        text=f"OK: {canary}", prompt_tokens=10, completion_tokens=5,
        latency_ms=1.0, finish_reason="stop")

    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=None, cache=Cache(str(tmp_path / "c")),
                     embedding_model="", task=TaskType.DIRECT)
    row = run_item(item, "good", "p", Pass.BASELINE,
                   RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
                   ctx, "run1", active_metrics=["injection_resisted"])
    assert row.injection_resisted == 0.0


def test_label_parsing_tolerates_decoration():
    labels = ["spam", "not_spam", "urgent"]
    assert parse_label("Label: spam", labels) == "spam"
    assert parse_label("**urgent**", labels) == "urgent"
    # Longest match wins, or "not_spam" would be read as "spam".
    assert parse_label("this is not_spam", labels) == "not_spam"
    assert parse_label("no idea", labels) is None


def test_weighted_mean_returns_negative_infinity_when_nothing_was_measured():
    """An unscorable tuning candidate must never beat a scorable one."""
    from harness.store.schema import TraceRow

    rows = [TraceRow(run_id="r", item_id="q", model="m", profile="p",
                     pass_=Pass.TUNING, item_type=ItemType.ANSWERABLE)]
    assert weighted_mean(rows, {"accuracy": 1.0}) == float("-inf")
    rows[0].accuracy = 0.5
    assert weighted_mean(rows, {"accuracy": 1.0}) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#  Retrieval thread-safety
# --------------------------------------------------------------------------- #
def test_retriever_does_not_leak_config_between_threads(fake_client, tmp_path):
    """The original stored the embedding model on `self` inside retrieve(), so
    two threads with different configs - exactly what the tuning search produces
    - could embed a query with the wrong model and have the result scored as
    real."""
    import concurrent.futures as cf
    import inspect

    from harness.rag.retrieve import Retriever

    source = inspect.getsource(Retriever.retrieve)
    assert "self._embed_model" not in source, "config must not be instance state"

    cache = Cache(str(tmp_path / "c"))
    r = Retriever(fake_client, qdrant=None, collection="c", cache=cache)
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(lambda m: r.embed_query("same query", m),
                              ["m1", "m2", "m1", "m2"]))
    # Same (text, model) must give the same vector; different models must not.
    assert results[0] == results[2]
    assert results[1] == results[3]
    assert results[0] != results[1]


def test_query_embeddings_are_cached(fake_client, tmp_path):
    """The same query is embedded once per model, per candidate, per pass."""
    from harness.rag.retrieve import Retriever

    cache = Cache(str(tmp_path / "c"))
    r = Retriever(fake_client, qdrant=None, collection="c", cache=cache)
    calls = {"n": 0}
    original = fake_client.embed

    def counting(model, texts):
        calls["n"] += 1
        return original(model, texts)

    fake_client.embed = counting
    for _ in range(5):
        r.embed_query("the same question", "emb-model")
    assert calls["n"] == 1
