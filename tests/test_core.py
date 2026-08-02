"""
Tests for every piece of logic that doesn't require the network (Together) or a
running Qdrant. This covers the scientific core: metrics, RRF fusion, caching,
the trace store round-trip, the reporting aggregates, config loading, and the
tuning candidate enumeration (incl. the equal-budget guarantee).

Run: python -m pytest tests/ -v   (or just: python tests/test_core.py)
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from harness.eval import metrics as M
from harness.rag.retrieve import reciprocal_rank_fusion
from harness.cache.cache import Cache, stable_hash
from harness.store.schema import TraceRow, Pass, ItemType, RetrievalMode
from harness.store.store import TraceStore
from harness.report.aggregate import (
    weighted_composite, tuning_gain, pareto_frontier, elo_from_pairwise,
    bootstrap_ci,
)
from harness.profiles.profile import Profile
from harness.tuning.search import enumerate_candidates


def test_retrieval_metrics():
    retrieved = ["d1#0", "d2#1", "d3#2", "d4#3"]
    gold = ["d3#2"]
    assert M.hit_rate_at_k(retrieved, gold, k=4) == 1.0
    assert M.hit_rate_at_k(retrieved, gold, k=2) == 0.0        # gold is at rank 3
    assert math.isclose(M.mrr(retrieved, gold), 1/3)           # first gold at rank 3
    assert 0 < M.ndcg_at_k(retrieved, gold, k=4) < 1.0
    assert M.context_recall(retrieved, ["d3#2", "d99#9"], k=4) == 0.5  # 1 of 2 found
    print("retrieval metrics ok")


def test_ndcg_perfect_and_empty():
    # gold ranked first -> NDCG 1.0
    assert math.isclose(M.ndcg_at_k(["g", "x", "y"], ["g"], k=3), 1.0)
    # no gold -> 0
    assert M.ndcg_at_k(["x", "y"], [], k=3) == 0.0
    print("ndcg edge cases ok")


def test_citation_metrics():
    ans = "The penalty is a fine [d1#0]. See also [d9#9] and [fake#1]."
    cited = M.extract_citations(ans)
    assert cited == ["d1#0", "d9#9", "fake#1"]
    retrieved = ["d1#0", "d9#9"]
    # 2 of 3 cited ids are real -> pointer validity 2/3
    assert math.isclose(M.citation_valid_pointer(cited, retrieved), 2/3)
    # of cited, which are gold?
    assert math.isclose(M.citation_supporting(cited, ["d1#0"]), 1/3)
    print("citation metrics ok")


def test_accuracy_scorers():
    assert M.exact_match("Yes", "yes") == 1.0            # case + whitespace normalised
    assert M.exact_match("Yes.", "yes") == 0.0           # trailing period IS a difference
    assert M.contains_match("The answer is 42, definitely", "42") == 1.0
    # numeric: 1,000 vs 10,000 must NOT match
    assert M.numeric_match("about 1,000 units", "1000") == 1.0
    assert M.numeric_match("about 10,000 units", "1000") == 0.0
    print("accuracy scorers ok")


def test_abstention():
    assert M.detect_abstention("I don't know based on the context.") is True
    assert M.detect_abstention("The penalty is a fine.") is False
    # answerable item, model answered -> correct
    assert M.abstention_correct(abstained=False, is_answerable=True) == 1.0
    # unanswerable item, model refused -> correct
    assert M.abstention_correct(abstained=True, is_answerable=False) == 1.0
    # unanswerable item, model fabricated -> wrong
    assert M.abstention_correct(abstained=False, is_answerable=False) == 0.0
    print("abstention ok")


def test_rrf():
    # B is ranked high in BOTH lists (rank 0 and rank 1), A/C each top only one.
    # An item that scores well across lists should beat items that spike in one.
    dense = ["B", "A", "C"]
    sparse = ["B", "C", "A"]
    fused = reciprocal_rank_fusion([dense, sparse], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "B", f"expected B first, got {ids}"
    # sanity: consensus across lists (B: 1/61+1/61) beats single-list top
    # (A best score is 1/62). This is the whole point of RRF.
    # an id present in only one list still appears
    fused2 = reciprocal_rank_fusion([["X"], ["Y"]], k=60)
    assert {i for i, _ in fused2} == {"X", "Y"}
    print("rrf ok")


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        c = Cache(d)
        key = c.key_for_generation("m", [{"role": "user", "content": "hi"}], {"t": 0})
        assert c.get(key) is None
        c.set(key, {"text": "hello", "prompt_tokens": 3, "completion_tokens": 1})
        got = c.get(key)
        assert got["text"] == "hello"
        # same inputs -> same key (reproducibility)
        key2 = c.key_for_generation("m", [{"role": "user", "content": "hi"}], {"t": 0})
        assert key == key2
        # different inputs -> different key
        key3 = c.key_for_generation("m", [{"role": "user", "content": "bye"}], {"t": 0})
        assert key != key3
    print("cache ok")


def test_stable_hash_order_independence():
    # dict key order must not change the hash
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    # but concatenation boundaries matter
    assert stable_hash("ab", "c") != stable_hash("a", "bc")
    print("stable hash ok")


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        store = TraceStore(str(Path(d) / "t.parquet"))
        rows = [
            TraceRow(run_id="r", item_id=f"q{i}", model="M", profile="p",
                     pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE,
                     retrieval_mode=RetrievalMode.HYBRID,
                     accuracy=0.5 + 0.1 * i, cost_usd=0.001,
                     retrieved_ids=["a", "b"])
            for i in range(3)
        ]
        n = store.write(rows)
        assert n == 3
        # append more, verify total
        store.write([TraceRow(run_id="r", item_id="q9", model="M", profile="p",
                              pass_=Pass.ADAPTED, item_type=ItemType.ANSWERABLE,
                              accuracy=0.9)])
        df = store.query("SELECT * FROM traces")
        assert len(df) == 4
        # list column survived the round-trip
        first = store.query("SELECT retrieved_ids FROM traces WHERE item_id='q0'")
        assert list(first.iloc[0]["retrieved_ids"]) == ["a", "b"]
    print("store roundtrip ok")


def test_reporting():
    df = pd.DataFrame([
        # model A: baseline then adapted (gains accuracy)
        {"model": "A", "pass_": "baseline", "accuracy": 0.6, "faithfulness": 0.7,
         "cost_usd": 0.001, "latency_ms": 500},
        {"model": "A", "pass_": "adapted", "accuracy": 0.8, "faithfulness": 0.75,
         "cost_usd": 0.001, "latency_ms": 500},
        # model B: cheaper, less accurate, no gain
        {"model": "B", "pass_": "baseline", "accuracy": 0.5, "faithfulness": 0.6,
         "cost_usd": 0.0005, "latency_ms": 300},
        {"model": "B", "pass_": "adapted", "accuracy": 0.5, "faithfulness": 0.6,
         "cost_usd": 0.0005, "latency_ms": 300},
    ])
    comp = weighted_composite(df, {"accuracy": 1.0, "cost_usd": -10.0})
    assert set(comp["model"]) == {"A", "B"}

    gain = tuning_gain(df, ["accuracy", "faithfulness"])
    a_gain = gain[gain["model"] == "A"]["accuracy_gain"].iloc[0]
    assert math.isclose(a_gain, 0.2, abs_tol=1e-9)  # 0.8 - 0.6
    b_gain = gain[gain["model"] == "B"]["accuracy_gain"].iloc[0]
    assert math.isclose(b_gain, 0.0, abs_tol=1e-9)

    pf = pareto_frontier(df)
    # both are non-dominated (A more accurate, B cheaper+faster)
    assert pf["on_frontier"].sum() == 2
    print("reporting ok")


def test_elo():
    # A beats B consistently, B beats C -> A > B > C
    pairwise = [("A", "B", "A")] * 10 + [("B", "C", "A")] * 10
    table = elo_from_pairwise(pairwise)
    order = list(table["model"])
    assert order.index("A") < order.index("B") < order.index("C")
    print("elo ok")


def test_bootstrap():
    vals = np.array([0.7, 0.72, 0.68, 0.71, 0.69, 0.70, 0.73, 0.67])
    mean, lo, hi = bootstrap_ci(vals, n_boot=1000, seed=0)
    assert lo < mean < hi
    assert math.isclose(mean, vals.mean(), abs_tol=1e-9)
    print("bootstrap ok")


def test_profile_load_and_equal_budget():
    profile = Profile.from_yaml("configs/profiles/regulated_qa.yaml")
    assert profile.name == "regulated_qa"
    assert profile.retrieval_mode == RetrievalMode.HYBRID
    assert profile.tuning_budget == 20
    # collection name namespaces by profile + embedder
    assert "regulated_qa" in profile.collection_name()

    # EQUAL-BUDGET GUARANTEE: candidate enumeration is deterministic and capped
    c1 = enumerate_candidates(profile, budget=20, seed=0)
    c2 = enumerate_candidates(profile, budget=20, seed=0)
    assert len(c1) <= 20
    # same seed -> identical candidate set for every model (fairness)
    ids1 = [(c.retrieval.mode.value, c.retrieval.k, c.rerank) for c in c1]
    ids2 = [(c.retrieval.mode.value, c.retrieval.k, c.rerank) for c in c2]
    assert ids1 == ids2
    print("profile + equal-budget ok")


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} test groups passed.")


if __name__ == "__main__":
    _run_all()
