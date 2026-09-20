"""
Per-row cost must be that row's spend, and the rows must sum to the run.

Found in the first live five-model RAG run: the rows' judge column summed to
$2.95 against a metered $0.38. The runner attributed judge, embedding and
rerank spend from two reads of the SHARED meter around each item, so with
eight workers every row also counted its neighbours' calls — 7.9x over,
landing on whichever rows were in flight rather than on the models that
spent it. Generation was priced from the row's own tokens and matched.

I3 says every paid call is attributed to a bucket from real usage. That has
to hold per row as well as per run, or cost_per_correct_answer, the Pareto
frontier and every cost panel are drawn from numbers no provider billed.
"""

from __future__ import annotations

import threading

import pytest

from harness.clients.cost import CostMeter


def _spend(meter: CostMeter, judge: float, embed: float, rerank: float) -> None:
    meter.record_generation(judge, 10, 5, is_judge=True)
    meter.record_embedding(embed, 3)
    meter.record_rerank(rerank)


def test_a_threads_ledger_holds_only_its_own_calls():
    m = CostMeter()
    m.begin_item()
    _spend(m, judge=0.01, embed=0.002, rerank=0.0005)
    _spend(m, judge=0.01, embed=0.0, rerank=0.0)
    got = m.take_item()
    assert got["judge_usd"] == pytest.approx(0.02)
    assert got["embedding_usd"] == pytest.approx(0.002)
    assert got["rerank_usd"] == pytest.approx(0.0005)
    # Taking resets: a second take is empty, not a repeat.
    assert m.take_item()["judge_usd"] == 0.0


def test_take_without_begin_is_zero_not_someone_elses_spend():
    m = CostMeter()
    _spend(m, judge=0.5, embed=0.5, rerank=0.5)     # unbracketed
    assert m.take_item() == {"judge_usd": 0.0, "embedding_usd": 0.0,
                             "rerank_usd": 0.0, "generation_usd": 0.0}
    # ...while the run total still saw it.
    assert m.costs.judge == pytest.approx(0.5)


def test_eight_concurrent_items_are_attributed_exactly_and_sum_to_the_total():
    """The bug this pins. Each thread's item spends a known, distinct amount
    while seven others bill against the same meter. Under the old shared-
    meter delta a row would have read up to the sum of all eight."""
    m = CostMeter()
    per_thread = {n: 0.001 * (n + 1) for n in range(8)}    # distinct per row
    rows: dict[int, dict] = {}
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def item(n: int) -> None:
        m.begin_item()
        barrier.wait()                       # everyone bills at once
        for _ in range(5):
            m.record_generation(per_thread[n], 1, 1, is_judge=True)
            m.record_embedding(per_thread[n] / 10, 1)
        barrier.wait()                       # ...and everyone finishes together
        with lock:
            rows[n] = m.take_item()

    ts = [threading.Thread(target=item, args=(n,)) for n in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    for n, row in rows.items():
        assert row["judge_usd"] == pytest.approx(5 * per_thread[n]), n
        assert row["embedding_usd"] == pytest.approx(5 * per_thread[n] / 10), n
    assert sum(r["judge_usd"] for r in rows.values()) == pytest.approx(m.costs.judge)
    assert sum(r["embedding_usd"] for r in rows.values()) == pytest.approx(m.costs.embedding)


def test_the_runner_writes_the_ledger_not_a_shared_delta():
    """Structural pin on the runner: the shared-meter delta must not come
    back. If someone reintroduces `summary()` bracketing, this names it."""
    import inspect

    from harness.orchestration import runner

    src = inspect.getsource(runner)
    assert "begin_item()" in src and "take_item()" in src
    assert "_meter_delta" not in src, "the shared-meter delta was the bug"
    assert "meter_before" not in src


def test_the_runner_attributes_judge_cost_per_row(tmp_path, fake_client):
    """End to end through run_item: a judge-scored item's row carries exactly
    the judge calls made for it, priced from the meter."""
    from harness.cache.cache import Cache
    from harness.eval.judge import Judge
    from harness.orchestration.runner import RunContext, run_item
    from harness.rag.prompt import PromptConfig
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import EvalItem, ItemType, Pass, RetrievalMode, TaskType

    class _Pricing:
        def generation_cost(self, model, p, c):
            return 0.0

        def embedding_cost(self, model, t):
            return 0.0

    meter = CostMeter()
    billed = []

    real_judge = fake_client.judge

    def billed_judge(model, messages, **kw):
        # The judge adapter meters its own calls in production; the fake
        # does not, so bill a fixed amount per call here.
        meter.record_generation(0.01, 10, 5, is_judge=True)
        billed.append(1)
        return real_judge(model, messages, **kw)

    fake_client.judge = billed_judge
    judge = Judge(fake_client, "judge", cache=Cache(str(tmp_path / "c")))
    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=judge, cache=Cache(str(tmp_path / "c2")),
                     meter=meter, embedding_model="", task=TaskType.DIRECT)
    item = EvalItem(item_id="i1", query="What is the SLA?",
                    item_type=ItemType.ANSWERABLE, gold_answer="24 hours")

    row = run_item(item, "good", "p", Pass.BASELINE,
                   RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
                   ctx, "run1", active_metrics=["accuracy", "faithfulness"],
                   accuracy_scorer="judge")

    assert billed, "the judge must have been called"
    assert row.judge_cost_usd == pytest.approx(0.01 * len(billed))
    assert row.judge_cost_usd == pytest.approx(meter.costs.judge)
