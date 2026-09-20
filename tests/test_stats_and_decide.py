"""
Tests for the statistical and decision layers.

These are the tests that protect against the harness being *confidently wrong*.
A metric bug produces a number someone can sanity-check; a significance bug
produces a p-value nobody can, and it will be believed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.report.aggregate import (
    elo_from_pairwise,
    judge_calibration,
    pareto_frontier,
    weighted_composite,
    win_rate_matrix,
)
from harness.report.decide import (
    Constraint,
    headroom,
    model_aggregates,
    project_cost,
    select,
)
from harness.report.gate import budget_gate, check_regression, parse_gate_spec
from harness.report.stats import (
    compare_models,
    holm_bonferroni,
    mcnemar,
    minimum_detectable_effect,
    paired_bootstrap,
    paired_values,
    power_report,
    required_n,
    significance_matrix,
)


def _frame(spec: dict[str, list[float]], pass_: str = "baseline") -> pd.DataFrame:
    """Build a trace-shaped frame from {model: [per-item scores]}."""
    rows = []
    for model, scores in spec.items():
        for i, s in enumerate(scores):
            rows.append({"model": model, "item_id": f"q{i}", "pass_": pass_,
                         "accuracy": s})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Pairing
# --------------------------------------------------------------------------- #
def test_paired_values_aligns_on_shared_items_only():
    """Pairing, when the caller has explicitly accepted the loss.

    This test previously asserted `len(a) == 2, "must drop the unpaired item"`
    against the *default* call — pinning the silent-drop behaviour that Phase 0
    recorded as finding R-01. The test was wrong, not merely outdated: CLAUDE.md
    §9 Phase 5 step 3 requires items scored by one model but not another to
    "raise, not drop silently, or pairing (I1) is broken". Dropping selects the
    comparison set by one model's failures, which biases it toward easy items.

    The behaviour it meant to check — that alignment keeps only shared items —
    is still real and still worth pinning, so it is kept here on the explicit
    `on_unpaired="drop"` path. The default now raises; see
    `tests/test_invariants.py` for that half.
    """
    df = pd.DataFrame([
        {"model": "A", "item_id": "q1", "accuracy": 1.0},
        {"model": "A", "item_id": "q2", "accuracy": 0.0},
        {"model": "A", "item_id": "q3", "accuracy": 1.0},   # B never saw q3
        {"model": "B", "item_id": "q1", "accuracy": 0.0},
        {"model": "B", "item_id": "q2", "accuracy": 1.0},
    ])
    a, b = paired_values(df, "A", "B", "accuracy", on_unpaired="drop")
    assert len(a) == len(b) == 2, "aligns on the shared items"


def test_pairing_detects_a_difference_that_marginal_intervals_would_hide():
    """The reason paired tests exist.

    A is better than B on every single item by a small, consistent margin, but
    item difficulty swamps that margin in the marginal spread. An unpaired view
    sees two heavily overlapping distributions; the paired view sees a perfectly
    consistent winner.
    """
    rng = np.random.default_rng(0)
    difficulty = rng.uniform(0, 1, 60)          # huge shared variance
    a_scores = difficulty + 0.05                # A always slightly better
    b_scores = difficulty
    df = pd.DataFrame(
        [{"model": "A", "item_id": f"q{i}", "accuracy": v}
         for i, v in enumerate(a_scores)] +
        [{"model": "B", "item_id": f"q{i}", "accuracy": v}
         for i, v in enumerate(b_scores)])

    result = compare_models(df, "A", "B", "accuracy")
    assert result.p_value < 0.05
    assert result.diff == pytest.approx(0.05, abs=1e-9)
    # And the marginal spread really is much larger than the effect, which is
    # exactly the situation where an unpaired comparison fails.
    assert np.std(a_scores) > 5 * result.diff


def test_identical_models_are_not_significant():
    df = _frame({"A": [1, 0, 1, 1, 0] * 8, "B": [1, 0, 1, 1, 0] * 8})
    assert compare_models(df, "A", "B", "accuracy").p_value >= 0.05


# --------------------------------------------------------------------------- #
#  McNemar
# --------------------------------------------------------------------------- #
def test_mcnemar_uses_only_discordant_pairs():
    """100 items, models differ on 6 - you have 6 data points, not 100."""
    a = np.array([1.0] * 50 + [0.0] * 50)
    b = np.array([1.0] * 50 + [0.0] * 50)
    b[:5] = 0.0   # a right, b wrong  (5 discordant)
    b[50] = 1.0   # b right, a wrong  (1 discordant)
    b01, b10, p = mcnemar(a, b)
    assert (b01, b10) == (1, 5)
    assert 0.0 < p <= 1.0


def test_mcnemar_all_concordant_is_p_one():
    a = np.array([1.0, 1.0, 0.0, 0.0])
    assert mcnemar(a, a.copy()) == (0, 0, 1.0)


def test_compare_models_auto_selects_mcnemar_for_binary_metrics():
    df = _frame({"A": [1, 1, 1, 1, 0, 0] * 6, "B": [0, 1, 1, 0, 0, 0] * 6})
    assert compare_models(df, "A", "B", "accuracy").test == "mcnemar_exact"


def test_compare_models_uses_bootstrap_for_continuous_metrics():
    df = _frame({"A": [0.7, 0.8, 0.65, 0.9] * 6, "B": [0.6, 0.7, 0.6, 0.8] * 6})
    assert compare_models(df, "A", "B", "accuracy").test == "paired_bootstrap"


# --------------------------------------------------------------------------- #
#  Bootstrap
# --------------------------------------------------------------------------- #
def test_paired_bootstrap_never_reports_an_impossible_p_of_zero():
    a = np.ones(50)
    b = np.zeros(50)
    _, _, _, p = paired_bootstrap(a, b, n_boot=1000, seed=0)
    assert p > 0.0, "smoothing must keep p above zero"


def test_paired_bootstrap_ci_brackets_the_observed_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(0.8, 0.1, 100)
    b = rng.normal(0.7, 0.1, 100)
    diff, lo, hi, _ = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert lo < diff < hi


def test_paired_bootstrap_handles_degenerate_input():
    assert paired_bootstrap(np.array([]), np.array([]))[3] == 1.0
    assert paired_bootstrap(np.array([1.0]), np.array([0.0]))[3] == 1.0


# --------------------------------------------------------------------------- #
#  Multiple comparisons
# --------------------------------------------------------------------------- #
def test_holm_is_monotonic_and_never_exceeds_one():
    adj, sig = holm_bonferroni([0.01, 0.02, 0.03, 0.5], alpha=0.05)
    assert all(0.0 <= p <= 1.0 for p in adj)
    assert sorted(adj) == adj, "adjusted p-values must be non-decreasing in rank"
    assert sig[-1] is False


def test_holm_is_less_conservative_than_bonferroni():
    p = [0.01, 0.02, 0.03]
    adj, _ = holm_bonferroni(p)
    bonferroni = [min(1.0, x * len(p)) for x in p]
    assert any(a < b for a, b in zip(adj, bonferroni))


def test_significance_matrix_corrects_for_the_number_of_comparisons():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(80):
        d = rng.random()
        for m, skill in (("A", 0.30), ("B", 0.15), ("C", 0.0)):
            rows.append({"model": m, "item_id": f"q{i}", "pass_": "baseline",
                         "accuracy": float(d + skill > 0.5)})
    sig = significance_matrix(pd.DataFrame(rows), metric="accuracy")
    assert len(sig) == 3                          # 3 models -> 3 pairs
    assert (sig["p_adjusted"] >= sig["p_value"]).all()


def test_significance_matrix_handles_a_single_model():
    assert significance_matrix(_frame({"A": [1, 0, 1]}), metric="accuracy").empty


# --------------------------------------------------------------------------- #
#  Power
# --------------------------------------------------------------------------- #
def test_required_n_grows_as_the_effect_shrinks():
    assert required_n(0.02, 0.3) > required_n(0.05, 0.3) > required_n(0.10, 0.3)


def test_mde_shrinks_as_the_sample_grows():
    assert minimum_detectable_effect(50, 0.3) > minimum_detectable_effect(500, 0.3)


def test_power_report_is_actionable():
    """'Add more questions' is not advice; 'you need ~340' is."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(40):
        for m, skill in (("A", 0.8), ("B", 0.75)):
            rows.append({"model": m, "item_id": f"q{i}", "pass_": "baseline",
                         "accuracy": float(rng.random() < skill)})
    pr = power_report(pd.DataFrame(rows), "accuracy")
    assert pr.n_items == 40
    assert pr.mde > 0
    assert pr.suggestions["detect_0.02"] > pr.suggestions["detect_0.1"]


# --------------------------------------------------------------------------- #
#  Pareto
# --------------------------------------------------------------------------- #
def test_pareto_does_not_reward_a_model_for_missing_data():
    """The original bug: NaN comparisons are all False, so an unmeasured model
    was never dominated and always looked optimal."""
    df = pd.DataFrame([
        {"model": "good", "pass_": "baseline", "accuracy": 0.9,
         "cost_usd": 0.001, "latency_ms": 500},
        {"model": "ghost", "pass_": "baseline", "accuracy": np.nan,
         "cost_usd": np.nan, "latency_ms": np.nan},
    ])
    pf = pareto_frontier(df)
    assert not pf[pf["model"] == "ghost"]["on_frontier"].iloc[0]


def test_pareto_keeps_genuinely_non_dominated_models():
    df = pd.DataFrame([
        {"model": "accurate", "pass_": "baseline", "accuracy": 0.95,
         "cost_usd": 0.01, "latency_ms": 3000},
        {"model": "cheap", "pass_": "baseline", "accuracy": 0.70,
         "cost_usd": 0.0001, "latency_ms": 200},
        {"model": "dominated", "pass_": "baseline", "accuracy": 0.60,
         "cost_usd": 0.02, "latency_ms": 4000},
    ])
    pf = pareto_frontier(df).set_index("model")
    assert pf.loc["accurate", "on_frontier"]
    assert pf.loc["cheap", "on_frontier"]
    assert not pf.loc["dominated", "on_frontier"]


# --------------------------------------------------------------------------- #
#  Composite
# --------------------------------------------------------------------------- #
def test_composite_respects_negative_weights_for_cost():
    df = pd.DataFrame([
        {"model": "pricey", "accuracy": 0.9, "cost_usd": 0.10},
        {"model": "thrifty", "accuracy": 0.9, "cost_usd": 0.001},
    ])
    comp = weighted_composite(df, {"accuracy": 1.0, "cost_usd": -1.0})
    assert comp.iloc[0]["model"] == "thrifty"


def test_composite_normalisation_stops_one_metric_dominating_by_scale():
    """Latency in milliseconds sits in the thousands; accuracy sits in [0,1].
    Unnormalised, a tiny latency weight silently outweighs a large accuracy one."""
    df = pd.DataFrame([
        {"model": "accurate_slow", "accuracy": 1.0, "latency_ms": 3000},
        {"model": "poor_fast", "accuracy": 0.1, "latency_ms": 100},
    ])
    weights = {"accuracy": 0.9, "latency_ms": -0.001}
    raw = weighted_composite(df, weights)
    normed = weighted_composite(df, weights, normalise=True)
    assert raw.iloc[0]["model"] == "poor_fast"        # scale wins
    assert normed.iloc[0]["model"] == "accurate_slow"  # intent wins


# --------------------------------------------------------------------------- #
#  Elo / arena
# --------------------------------------------------------------------------- #
def test_elo_orders_a_transitive_ladder():
    pairwise = [("A", "B", "A")] * 12 + [("B", "C", "A")] * 12
    table = elo_from_pairwise(pairwise)
    order = list(table["model"])
    assert order.index("A") < order.index("B") < order.index("C")


def test_elo_records_win_loss_tie_counts():
    table = elo_from_pairwise([("A", "B", "A"), ("A", "B", "tie")]).set_index("model")
    assert table.loc["A", "wins"] == 1 and table.loc["A", "ties"] == 1
    assert table.loc["A", "win_rate"] == pytest.approx(0.75)


def test_win_rate_matrix_is_complementary():
    m = win_rate_matrix([("A", "B", "A"), ("A", "B", "B")])
    assert m.loc["A", "B"] == pytest.approx(0.5)
    assert m.loc["B", "A"] == pytest.approx(0.5)


def test_elo_handles_no_comparisons():
    assert elo_from_pairwise([]).empty


# --------------------------------------------------------------------------- #
#  Decision engine
# --------------------------------------------------------------------------- #
def _decision_frame() -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows += [
            {"model": "big", "item_id": f"q{i}", "pass_": "baseline",
             "accuracy": 0.95, "faithfulness": 0.95, "cost_usd": 0.004,
             "gen_cost_usd": 0.004, "latency_ms": 2500, "error": None},
            {"model": "small", "item_id": f"q{i}", "pass_": "baseline",
             "accuracy": 0.80, "faithfulness": 0.80, "cost_usd": 0.0004,
             "gen_cost_usd": 0.0004, "latency_ms": 600, "error": None},
            {"model": "big", "item_id": f"q{i}", "pass_": "latency",
             "latency_ms": 2500, "error": None},
            {"model": "small", "item_id": f"q{i}", "pass_": "latency",
             "latency_ms": 600, "error": None},
        ]
    return pd.DataFrame(rows)


def test_constraint_parsing():
    c = Constraint.parse("faithfulness>=0.9")
    assert (c.metric, c.op, c.value) == ("faithfulness", ">=", 0.9)
    assert Constraint.parse("latency_p95_ms <= 2000").metric == "latency_p95_ms"
    with pytest.raises(ValueError):
        Constraint.parse("nonsense")


def test_select_returns_the_cheapest_model_that_clears_the_bar():
    df = _decision_frame()
    short, _ = select(df, [Constraint("accuracy", ">=", 0.75)],
                      optimise="cost_usd")
    assert short.iloc[0]["model"] == "small"


def test_select_explains_why_nothing_qualified():
    """An empty shortlist with no reason is useless; the reason is the product."""
    df = _decision_frame()
    short, candidates = select(df, [Constraint("accuracy", ">=", 0.99)])
    assert short.empty
    assert all(not c.eligible for c in candidates)
    assert any("accuracy >= 0.99" in f for c in candidates for f in c.failures)


def test_unmeasured_constraint_is_reported_as_unknown_not_as_a_failure():
    df = _decision_frame()
    _, candidates = select(df, [Constraint("injection_resisted", ">=", 0.9)])
    assert all(c.unknowns for c in candidates)
    assert all(not c.failures for c in candidates)


def test_latency_percentiles_come_from_the_latency_lane():
    agg = model_aggregates(_decision_frame()).set_index("model")
    assert "latency_p95_ms" in agg.columns
    assert agg.loc["big", "latency_p95_ms"] > agg.loc["small", "latency_p95_ms"]


def test_cost_projection_scales_to_volume():
    proj = project_cost(_decision_frame(), queries_per_day=50_000).set_index("model")
    assert proj.loc["big", "monthly_usd"] == pytest.approx(0.004 * 50_000 * 30)
    assert proj.loc["small", "monthly_usd"] < proj.loc["big", "monthly_usd"]


def test_headroom_prices_each_quality_point():
    hr = headroom(_decision_frame(), "accuracy", 50_000).set_index("model")
    # big is 15 points better and $5,400/mo more: $360 per point.
    assert hr.loc["big", "usd_per_point"] == pytest.approx(360.0, rel=1e-6)


# --------------------------------------------------------------------------- #
#  Regression gate
# --------------------------------------------------------------------------- #
def _gate_frame(scores: list[float], model: str = "M") -> pd.DataFrame:
    return pd.DataFrame([
        {"model": model, "item_id": f"q{i}", "pass_": "baseline",
         "accuracy": s, "cost_usd": 0.001}
        for i, s in enumerate(scores)])


def test_gate_passes_when_nothing_regressed():
    base = _gate_frame([1.0, 1.0, 0.0, 1.0] * 10)
    cand = _gate_frame([1.0, 1.0, 0.0, 1.0] * 10)
    result = check_regression(base, cand, ["accuracy"], tolerance=0.02)
    assert result.passed and result.exit_code() == 0


def test_gate_fails_on_a_real_regression():
    base = _gate_frame([1.0] * 40)
    cand = _gate_frame([1.0] * 20 + [0.0] * 20)
    result = check_regression(base, cand, ["accuracy"], tolerance=0.02)
    assert not result.passed and result.exit_code() == 1


def test_gate_does_not_fire_on_noise_by_default():
    """A gate that cries wolf gets ignored, and then protects nothing."""
    rng = np.random.default_rng(0)
    base = _gate_frame(list(rng.random(60) < 0.8))
    cand = _gate_frame(list(rng.random(60) < 0.79))
    result = check_regression(base, cand, ["accuracy"], tolerance=0.001,
                              require_significance=True)
    assert result.passed


def test_strict_mode_fires_on_raw_delta():
    base = _gate_frame([1.0] * 40)
    cand = _gate_frame([1.0] * 38 + [0.0, 0.0])
    strict = check_regression(base, cand, ["accuracy"], tolerance=0.001,
                              require_significance=False)
    assert not strict.passed


def test_gate_normalises_direction_for_lower_is_better_metrics():
    """A cost *increase* must read as a regression, not an improvement."""
    base = pd.DataFrame([{"model": "M", "item_id": f"q{i}", "pass_": "baseline",
                          "cost_usd": 0.001} for i in range(30)])
    cand = pd.DataFrame([{"model": "M", "item_id": f"q{i}", "pass_": "baseline",
                          "cost_usd": 0.010} for i in range(30)])
    result = check_regression(base, cand, ["cost_usd"], tolerance=0.001,
                              require_significance=False)
    assert not result.passed


def test_absolute_floor_catches_slow_erosion():
    """Delta gates permit quality sliding one tolerable step at a time."""
    base = _gate_frame([0.5] * 30)
    cand = _gate_frame([0.5] * 30)
    result = check_regression(base, cand, ["accuracy"], tolerance=0.02,
                              floors=[Constraint("accuracy", ">=", 0.8)])
    assert not result.passed


def test_budget_gate_catches_cost_regressions():
    df = _gate_frame([1.0] * 10)
    assert budget_gate(df, max_cost_per_query=0.01).passed
    assert not budget_gate(df, max_cost_per_query=0.0001).passed


def test_parse_gate_spec():
    metrics, tol = parse_gate_spec(["accuracy:0.02", "faithfulness"])
    assert metrics == ["accuracy", "faithfulness"]
    assert tol == {"accuracy": 0.02}


def test_gate_skips_rather_than_crashes_on_missing_data():
    result = check_regression(_gate_frame([1.0] * 5), _gate_frame([1.0] * 5),
                              ["nonexistent_metric"])
    assert result.passed and result.skipped


# --------------------------------------------------------------------------- #
#  Judge calibration
# --------------------------------------------------------------------------- #
def test_kappa_penalises_a_judge_that_just_follows_the_base_rate():
    """90% raw agreement, zero information. Kappa is the metric that says so."""
    df = pd.DataFrame([{"human_label": 1.0, "accuracy": 1.0} for _ in range(90)]
                      + [{"human_label": 0.0, "accuracy": 1.0} for _ in range(10)])
    cal = judge_calibration(df)
    assert cal["agreement"] == pytest.approx(0.9)
    assert cal["cohens_kappa"] == pytest.approx(0.0, abs=1e-9)
    assert "slight" in cal["interpretation"] or "not usable" in cal["interpretation"]


def test_calibration_detects_a_generous_judge():
    df = pd.DataFrame([{"human_label": 0.0, "accuracy": 1.0} for _ in range(20)]
                      + [{"human_label": 1.0, "accuracy": 1.0} for _ in range(20)])
    assert judge_calibration(df)["judge_bias"] > 0


def test_calibration_without_labels_says_so():
    assert judge_calibration(pd.DataFrame())["n"] == 0
