"""
Regression tests for the CLAUDE.md §2 invariants that Phase 0 found unenforced.

Each test here exists because `docs/INVENTORY.md` recorded a specific way the
harness could report a wrong number while every other test stayed green. The
invariant each one guards is named in its docstring, because these are the tests
that must never be weakened to make a change pass (§5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.clients.base import GenResult, MissingUsageError
from harness.report.stats import (
    UnpairedItemsError,
    compare_models,
    paired_values,
)

# --------------------------------------------------------------------------- #
#  I1 — paired comparison
# --------------------------------------------------------------------------- #
#
# Phase 0 finding R-01. `paired_values` used `dropna` + `index.intersection`,
# so an item only one model produced a score for was silently discarded and
# `n_pairs` quietly shrank. That is the highest-severity class available: if a
# model errored on its hard items, the surviving set is biased toward the easy
# ones and the comparison is run on a non-random subset without anyone being
# told. CLAUDE.md §9 Phase 5 step 3 requires a raise.


def _unpaired_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "A", "item_id": "q1", "accuracy": 1.0},
        {"model": "A", "item_id": "q2", "accuracy": 0.0},
        {"model": "A", "item_id": "q3", "accuracy": 1.0},   # B never scored q3
        {"model": "B", "item_id": "q1", "accuracy": 0.0},
        {"model": "B", "item_id": "q2", "accuracy": 1.0},
    ])


def test_unpaired_items_raise_rather_than_being_dropped():
    """I1. The defect: this used to return two arrays of length 2, silently."""
    with pytest.raises(UnpairedItemsError) as exc:
        paired_values(_unpaired_frame(), "A", "B", "accuracy")
    assert "q3" in str(exc.value), "the error must name the offending item"


def test_unpaired_error_reports_which_model_is_missing_what():
    """I1. A raise the user cannot act on is barely better than a silent drop."""
    with pytest.raises(UnpairedItemsError) as exc:
        paired_values(_unpaired_frame(), "A", "B", "accuracy")
    err = exc.value
    assert err.missing_from_b == ("q3",)
    assert err.missing_from_a == ()
    assert err.n_common == 2


def test_dropping_is_possible_but_only_when_asked_for_explicitly():
    """I1. §9 Phase 5 says *silently*. An explicit, recorded opt-out is allowed."""
    a, b, dropped = paired_values(_unpaired_frame(), "A", "B", "accuracy",
                                  on_unpaired="drop", return_dropped=True)
    assert len(a) == len(b) == 2
    assert dropped == ("q3",), "the caller must be able to report what it lost"


def test_a_score_present_for_one_model_and_null_for_the_other_also_raises():
    """I1. A NaN is 'this model has no score for this item' — the same hazard.

    This is the shape the bug actually takes in a real run: the row exists
    because the item was attempted, but the metric is null because the call
    errored. `dropna` made it vanish.
    """
    df = pd.DataFrame([
        {"model": "A", "item_id": "q1", "accuracy": 1.0},
        {"model": "A", "item_id": "q2", "accuracy": 1.0},
        {"model": "B", "item_id": "q1", "accuracy": 0.0},
        {"model": "B", "item_id": "q2", "accuracy": np.nan},   # errored
    ])
    with pytest.raises(UnpairedItemsError):
        paired_values(df, "A", "B", "accuracy")


def test_fully_paired_data_is_unaffected():
    """I1. The fix must not change the answer for a well-formed run."""
    df = pd.DataFrame([
        {"model": "A", "item_id": "q1", "accuracy": 1.0},
        {"model": "A", "item_id": "q2", "accuracy": 0.0},
        {"model": "B", "item_id": "q1", "accuracy": 0.0},
        {"model": "B", "item_id": "q2", "accuracy": 1.0},
    ])
    a, b = paired_values(df, "A", "B", "accuracy")
    assert list(a) == [1.0, 0.0]
    assert list(b) == [0.0, 1.0]


def test_compare_models_propagates_the_raise():
    """I1. The decision path must not be reachable with unpaired data."""
    with pytest.raises(UnpairedItemsError):
        compare_models(_unpaired_frame(), "A", "B", metric="accuracy")


def test_no_overlap_at_all_raises_rather_than_reporting_no_difference():
    """I1. Zero overlap used to return empty arrays, which `compare_models`
    turned into a p-value of 1.0 — 'no significant difference' for two models
    that were never compared at all."""
    df = pd.DataFrame([
        {"model": "A", "item_id": "q1", "accuracy": 1.0},
        {"model": "B", "item_id": "q2", "accuracy": 0.0},
    ])
    with pytest.raises(UnpairedItemsError):
        compare_models(df, "A", "B", metric="accuracy")


# --------------------------------------------------------------------------- #
#  I3 — all spend is metered
# --------------------------------------------------------------------------- #
#
# Phase 0 finding R-02. The adapter read usage with
# `getattr(usage, "prompt_tokens", 0) or 0`, so a provider that omitted its
# usage block was billed $0.00 rather than raising. Because cost carries a
# NEGATIVE weight in the composite (I4), that model then *rose* in the ranking.
# `harness/clients/base.py` already promised the opposite in its own docstring:
# "Token counts always come from the provider's usage block."


class _FakeUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None):
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            self.completion_tokens = completion_tokens


def test_gen_result_defaults_to_measured_usage():
    """I3. `usage_estimated` must default to False so a row is honest unless
    something explicitly marked it otherwise."""
    r = GenResult(text="x", prompt_tokens=10, completion_tokens=2, latency_ms=1.0)
    assert r.usage_estimated is False


def test_extract_usage_raises_when_the_block_is_absent():
    """I3. The core of R-02: absent usage is a hard failure, not a zero."""
    from harness.clients.openai_compatible import extract_usage

    with pytest.raises(MissingUsageError) as exc:
        extract_usage(None, provider="groq", model="m", allow_estimated=False)
    assert "usage" in str(exc.value).lower()


def test_extract_usage_raises_when_the_fields_are_absent():
    """I3. A usage object present but empty is the same hazard as no object."""
    from harness.clients.openai_compatible import extract_usage

    with pytest.raises(MissingUsageError):
        extract_usage(_FakeUsage(), provider="groq", model="m",
                      allow_estimated=False)


def test_zero_tokens_reported_by_the_provider_is_not_an_error():
    """I3. A real, measured zero must survive — only *absence* raises.

    The distinction matters: `or 0` conflated them, which is how the bug hid.
    """
    from harness.clients.openai_compatible import extract_usage

    p, c, estimated = extract_usage(
        _FakeUsage(prompt_tokens=0, completion_tokens=0),
        provider="groq", model="m", allow_estimated=False)
    assert (p, c, estimated) == (0, 0, False)


def test_estimation_is_possible_but_must_flag_itself():
    """I3 / §9 Phase 2 step 3: estimation is allowed only on opt-in, and the
    row must record `usage_estimated=True` so the report can surface the rate."""
    from harness.clients.openai_compatible import extract_usage

    p, c, estimated = extract_usage(None, provider="groq", model="m",
                                    allow_estimated=True,
                                    prompt_text="a" * 40, completion_text="b" * 8)
    assert estimated is True
    assert p > 0 and c > 0


def test_measured_usage_is_never_flagged_as_estimated():
    from harness.clients.openai_compatible import extract_usage

    p, c, estimated = extract_usage(
        _FakeUsage(prompt_tokens=123, completion_tokens=45),
        provider="groq", model="m", allow_estimated=True)
    assert (p, c, estimated) == (123, 45, False)


# --------------------------------------------------------------------------- #
#  §5 — the offline guarantee
# --------------------------------------------------------------------------- #
def test_the_socket_block_is_real():
    """R-07. Before this, 'the suite is offline' was true but unenforced.

    A guarantee nothing checks is a guarantee that breaks on the first test
    someone adds while their own API key happens to be exported.
    """
    import socket

    with pytest.raises(RuntimeError, match="blocked"):
        socket.socket()


def test_create_connection_is_blocked_too():
    import socket

    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("example.com", 80))
