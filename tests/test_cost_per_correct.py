"""
cost_per_correct: one definition, hand-computed, exposed once.

The number that drives procurement (§10.7) must be computed in exactly one
place, from the rows, with None where it has no meaning. The client draws
it and never derives it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from harness.report.aggregate import cost_per_correct


def _rows():
    # model a: 4 scored items, accuracy 0.75, mean cost 0.02 -> 0.02/0.75
    # model b: 3 scored items (one unscored), accuracy 0.0 -> None
    # model c: nothing scored -> None, n_scored 0
    return pd.DataFrame([
        {"model": "a", "accuracy": 1.0, "cost_usd": 0.01},
        {"model": "a", "accuracy": 1.0, "cost_usd": 0.03},
        {"model": "a", "accuracy": 1.0, "cost_usd": 0.02},
        {"model": "a", "accuracy": 0.0, "cost_usd": 0.02},
        {"model": "b", "accuracy": 0.0, "cost_usd": 0.05},
        {"model": "b", "accuracy": 0.0, "cost_usd": 0.05},
        {"model": "b", "accuracy": 0.0, "cost_usd": 0.05},
        {"model": "b", "accuracy": None, "cost_usd": 0.99},   # unscored: excluded
        {"model": "c", "accuracy": None, "cost_usd": 0.10},
    ])


def test_hand_computed_fixture():
    t = cost_per_correct(_rows()).set_index("model")
    assert t.loc["a", "n_scored"] == 4
    assert t.loc["a", "cost_usd"] == pytest.approx(0.02)
    assert t.loc["a", "accuracy"] == pytest.approx(0.75)
    assert t.loc["a", "cost_per_correct_answer"] == pytest.approx(0.02 / 0.75)
    # never right: no price per right answer, and not zero
    assert t.loc["b", "cost_per_correct_answer"] is None or pd.isna(t.loc["b", "cost_per_correct_answer"])
    assert t.loc["b", "n_scored"] == 3, "the unscored row must not count"
    # nothing scored
    assert t.loc["c", "n_scored"] == 0
    assert pd.isna(t.loc["c", "cost_per_correct_answer"])


def test_missing_columns_yield_an_empty_table_not_a_crash():
    assert cost_per_correct(pd.DataFrame()).empty
    assert cost_per_correct(pd.DataFrame([{"model": "a", "accuracy": 1.0}])).empty


def test_the_profile_report_exposes_it_without_recomputing(store_cfg=None):
    """The API returns the library's table under one key; a grep proves the
    client never divides cost by accuracy itself."""
    import inspect
    from pathlib import Path

    from harness.web import profile_api

    src = inspect.getsource(profile_api.profile_results)
    assert "A.cost_per_correct" in src and '"cost_per_correct"' in src
    js = Path("harness/web/static/app.js").read_text(encoding="utf-8")
    assert "cost_per_correct_answer" in js
    assert "/ r.accuracy" not in js and "cost_usd /" not in js
