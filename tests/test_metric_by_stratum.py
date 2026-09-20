"""
metric_by_stratum: a reading aid for the case studies (accuracy by item
language), defined once in the report library. Hand-computed fixture,
unmapped items visible rather than dropped, nulls excluded from n.
"""

from __future__ import annotations

import pandas as pd
import pytest

from harness.report.aggregate import metric_by_stratum


def _df():
    return pd.DataFrame([
        {"model": "a", "item_id": "i1", "accuracy": 1.0},
        {"model": "a", "item_id": "i2", "accuracy": 0.0},
        {"model": "a", "item_id": "i3", "accuracy": 1.0},
        {"model": "a", "item_id": "i4", "accuracy": None},
        {"model": "b", "item_id": "i1", "accuracy": 0.0},
        {"model": "b", "item_id": "i2", "accuracy": 0.0},
        {"model": "b", "item_id": "i3", "accuracy": 1.0},
        {"model": "b", "item_id": "i9", "accuracy": 1.0},
    ])


def test_hand_computed_means_and_counts():
    out = metric_by_stratum(_df(), {"i1": "en", "i2": "en", "i3": "hi", "i4": "hi"}, name="language")
    rows = {(r.model, r.language): (r.accuracy, r.n) for r in out.itertuples()}
    assert rows[("a", "en")] == (pytest.approx(0.5), 2)
    assert rows[("a", "hi")] == (pytest.approx(1.0), 1), "the null row is not counted"
    assert rows[("b", "en")] == (pytest.approx(0.0), 2)
    assert rows[("b", "hi")] == (pytest.approx(1.0), 1)
    # i9 is not in the map: reported, not dropped.
    assert rows[("b", "unmapped")] == (pytest.approx(1.0), 1)


def test_empty_and_missing_metric_are_empty_frames():
    assert metric_by_stratum(pd.DataFrame(), {}).empty
    assert metric_by_stratum(_df(), {}, metric="faithfulness").empty
