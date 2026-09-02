"""
Tests for the presentation layer.

Charts are usually left untested because "it's just visuals". But an Altair spec
is data, and the failure modes here are silent: a chart that renders a plausible
picture of the wrong thing is worse than one that crashes. These tests pin the
properties that would otherwise rot — that insignificant comparisons stay
neutral, that a single-series chart doesn't get a rainbow, and that every
builder survives the empty and missing-column cases a real trace store produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness.ui import DARK, LIGHT, MIDNIGHT, page_css, register_altair_theme
from harness.ui import charts as C

ALL_PALETTES = (LIGHT, DARK, MIDNIGHT)


@pytest.fixture(params=ALL_PALETTES, ids=[p.name for p in ALL_PALETTES])
def palette(request):
    register_altair_theme(request.param)
    return request.param


@pytest.fixture
def frontier() -> pd.DataFrame:
    return pd.DataFrame([
        {"model": "big", "accuracy": 0.90, "cost": 0.0040, "latency": 2400,
         "on_frontier": True},
        {"model": "small", "accuracy": 0.78, "cost": 0.0006, "latency": 900,
         "on_frontier": True},
        {"model": "dominated", "accuracy": 0.70, "cost": 0.0090, "latency": 3000,
         "on_frontier": False},
    ])


@pytest.fixture
def sig() -> pd.DataFrame:
    return pd.DataFrame([
        {"model_a": "big", "model_b": "small", "diff": 0.12,
         "significant": True, "verdict": "big is better by 0.12"},
        {"model_a": "big", "model_b": "mid", "diff": 0.03,
         "significant": False, "verdict": "no significant difference"},
    ])


def _spec(chart) -> dict:
    """Render to a Vega-Lite spec, which is also the validation step."""
    return chart.to_dict()


def _rows(spec: dict) -> list[dict]:
    """The chart's data rows.

    Altair puts them under `datasets` keyed by a content hash and leaves a
    named reference in `data`, so reading `spec["data"]["values"]` finds nothing.
    """
    if "datasets" in spec and spec["datasets"]:
        return next(iter(spec["datasets"].values()))
    return spec.get("data", {}).get("values", [])


def _colours(spec: dict) -> set[str]:
    """Hex colours the CHART chose, excluding theme defaults.

    `config` carries the registered theme's full categorical range, so walking
    it would report every series colour as "used" by every chart and make these
    assertions meaningless.
    """
    found: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "config":
                    continue
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and node.startswith("#"):
            found.add(node.lower())

    walk(spec)
    return found


# --------------------------------------------------------------------------- #
#  Palette integrity
# --------------------------------------------------------------------------- #
def test_dark_is_a_selected_palette_not_an_inversion():
    """Dark steps were chosen for the dark surface. If they were merely the
    light values reused, contrast against the dark background is unverified."""
    assert LIGHT.series_1 != DARK.series_1
    assert LIGHT.series_2 != DARK.series_2
    assert LIGHT.surface != DARK.surface
    assert DARK.is_dark and not LIGHT.is_dark


def test_status_colours_never_collide_with_series_colours_in_any_palette():
    """A status hue impersonating a series is the fastest way to make a chart
    lie about what a colour means - in every theme, not just two."""
    for p in ALL_PALETTES:
        series = {p.series_1, p.series_2, p.series_3}
        status = {p.good, p.warning, p.serious, p.critical}
        assert not (series & status), p.name


def test_status_colours_never_collide_with_series_colours():
    """A status hue impersonating a series is the fastest way to make a chart
    lie about what a colour means."""
    for p in ALL_PALETTES:
        series = {p.series_1, p.series_2, p.series_3}
        status = {p.good, p.warning, p.serious, p.critical}
        assert not (series & status), p.name


def test_diverging_pair_has_a_neutral_midpoint():
    """The midpoint must read as 'nothing'. Two cool poles or a hue in the
    middle would make 'no difference' look like a finding.

    Checked for every palette: a blue-tinted 'grey' on a navy surface reads as a
    faint win for the row model, which is precisely the false signal the neutral
    midpoint exists to avoid.
    """
    for p in ALL_PALETTES:
        lo, mid, hi = p.diverging
        assert lo == p.series_1 and hi == p.critical
        # A neutral gray has near-equal RGB channels.
        r, g, b = (int(mid[i:i + 2], 16) for i in (1, 3, 5))
        assert max(r, g, b) - min(r, g, b) <= 12, f"{mid} is not neutral"


def test_page_css_defines_every_token_it_references():
    for p in ALL_PALETTES:
        css = page_css(p)
        for token in ("--surface", "--ink", "--ink-2", "--ink-3", "--accent",
                      "--good", "--warning", "--critical", "--border"):
            assert f"{token}:" in css, f"{token} undefined for {p.name}"


# --------------------------------------------------------------------------- #
#  Chart correctness
# --------------------------------------------------------------------------- #
def test_pareto_uses_emphasis_not_a_categorical_ramp(frontier, palette):
    """Two classes only: the subject in the accent hue, context in gray.

    Scatter is an all-pairs form where more than three hues cannot clear the
    colour-vision floors, so a per-model categorical scale would be wrong here
    regardless of how good it looked.
    """
    spec = _spec(C.pareto_chart(frontier, palette))
    used = _colours(spec)
    assert palette.series_1.lower() in used
    assert palette.muted_mark.lower() in used
    assert palette.series_2.lower() not in used, "should not reach for a 2nd hue"


def test_pareto_labels_every_point_so_identity_is_never_colour_alone(frontier,
                                                                    palette):
    spec = _spec(C.pareto_chart(frontier, palette))
    assert any("text" in str(layer) for layer in spec.get("layer", []))


def test_significance_neutralises_insignificant_comparisons(sig, palette):
    """The core honesty property of this chart.

    An uncorrected heatmap of raw deltas shows a confident pattern of colour
    where the data supports none of it. Only comparisons that survived the
    correction may carry a non-zero value.
    """
    spec = _spec(C.significance_matrix(sig, palette))
    values = [row["value"] for row in _rows(spec)]
    # The significant pair contributes +0.12 and its mirror -0.12; the
    # insignificant one contributes 0.0 both ways.
    assert sorted(values) == [-0.12, 0.0, 0.0, 0.12]


def test_significance_is_symmetric(sig, palette):
    spec = _spec(C.significance_matrix(sig, palette))
    rows = _rows(spec)
    by_pair = {(r["row"], r["col"]): r["value"] for r in rows}
    for (a, b), v in by_pair.items():
        assert by_pair[(b, a)] == pytest.approx(-v)


def test_metric_bars_use_one_colour_for_every_bar(frontier, palette):
    """One series is one colour. A value-ramp across nominal categories would
    double-encode bar length as hue and fail the categorical checks by design."""
    spec = _spec(C.metric_bars(frontier, "accuracy", palette))
    used = _colours(spec)
    assert palette.series_1.lower() in used
    assert palette.series_2.lower() not in used
    assert palette.series_3.lower() not in used


def test_bars_are_capped_in_thickness(frontier, palette):
    """Thick saturated blocks read loud. The spec caps marks well under 24px."""
    spec = _spec(C.metric_bars(frontier, "accuracy", palette))
    sizes = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "bar" and "size" in node:
                sizes.append(node["size"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(spec)
    assert sizes and all(s <= 24 for s in sizes)


def test_latency_chart_shows_a_legend_for_two_series(palette):
    """Two or more series always carry a legend; identity is never colour alone."""
    agg = pd.DataFrame([
        {"model": "big", "latency_p50_ms": 2400, "latency_p95_ms": 3100},
        {"model": "small", "latency_p50_ms": 900, "latency_p95_ms": 1200},
    ])
    spec = _spec(C.latency_bars(agg, palette))
    assert spec["encoding"]["color"]["legend"] is not None


def test_no_chart_uses_a_dashed_grid(frontier, palette):
    """Dashing reads as 'threshold' or 'projection' when it is only a grid."""
    for chart in (C.pareto_chart(frontier, palette),
                  C.metric_bars(frontier, "accuracy", palette)):
        assert "gridDash" not in str(_spec(chart)),             "gridlines must be solid hairlines"


# --------------------------------------------------------------------------- #
#  Robustness — a real trace store is full of missing columns
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("builder", [
    lambda d, p: C.pareto_chart(d, p),
    lambda d, p: C.significance_matrix(d, p),
    lambda d, p: C.metric_bars(d, "accuracy", p),
    lambda d, p: C.cost_projection_bars(d, p),
    lambda d, p: C.latency_bars(d, p),
    lambda d, p: C.quality_cost_scatter(d, p),
])
def test_every_builder_survives_an_empty_frame(builder, palette):
    """A profile that never activated a metric is the normal case, not an edge
    case. A chart that raises there takes down the whole page."""
    assert _spec(builder(pd.DataFrame(), palette))


@pytest.mark.parametrize("builder", [
    lambda d, p: C.pareto_chart(d, p),
    lambda d, p: C.metric_bars(d, "accuracy", p),
    lambda d, p: C.cost_projection_bars(d, p),
    lambda d, p: C.latency_bars(d, p),
])
def test_every_builder_survives_a_missing_column(builder, palette):
    assert _spec(builder(pd.DataFrame([{"model": "a"}]), palette))


def test_metric_bars_survive_an_all_nan_column(frontier, palette):
    d = frontier.copy()
    d["faithfulness"] = np.nan
    assert _spec(C.metric_bars(d, "faithfulness", palette))


def test_long_model_names_are_shortened_for_axes(palette):
    """Provider-prefixed ids blow out the axis and squeeze the plot to a sliver.
    The tooltip keeps the full name."""
    long_name = "openrouter:meta-llama/llama-3.3-70b-instruct-turbo-preview"
    d = pd.DataFrame([{"model": long_name, "accuracy": 0.8}])
    spec = _spec(C.metric_bars(d, "accuracy", palette))
    labels = [r["label"] for r in _rows(spec)]
    assert len(labels[0]) < len(long_name)
    # The untruncated id is still present for the tooltip.
    assert _rows(spec)[0]["model"] == long_name


# =========================================================================== #
#  Capability catalogue - the Capabilities screen's data source
# =========================================================================== #
def test_catalogue_is_discovered_not_hardcoded():
    """The whole point of the Capabilities screen.

    A typed-out feature list starts lying the first time someone adds a
    provider or renames a metric, and a feature page that disagrees with the
    software is worse than none because people act on it. Every section is
    derived from the code, so these must match their sources exactly.
    """
    from harness.clients.endpoints import NON_OPENAI_PROVIDERS, PROVIDER_ENDPOINTS
    from harness.eval.scoring import DEFAULT_SCORERS
    from harness.profiles.profile import KNOWN_METRICS
    from harness.store.schema import Pass, RetrievalMode, TaskType
    from harness.ui.catalog import build_catalog

    cat = build_catalog()

    assert {p.name for p in cat.providers} == (
        set(PROVIDER_ENDPOINTS) | set(NON_OPENAI_PROVIDERS))
    assert cat.n_metrics == len(KNOWN_METRICS), "every metric must appear"
    assert [i.name for i in cat.scorers] == [s.name for s in DEFAULT_SCORERS]
    assert {i.name for i in cat.passes} == {p.value for p in Pass}
    assert {i.name for i in cat.tasks} == {t.value for t in TaskType}
    assert cat.retrieval_modes == [m.value for m in RetrievalMode]


def test_a_new_metric_appears_on_the_screen_automatically():
    """Add a metric to the harness and the page shows it, with no page edit.

    Grouping is by known name, but anything unrecognised still lands under
    'Other' rather than vanishing - a silently dropped metric is exactly the
    drift this design exists to prevent.
    """
    from harness.profiles import profile as P
    from harness.ui.catalog import build_catalog

    original = P.KNOWN_METRICS
    try:
        P.KNOWN_METRICS = set(original) | {"brand_new_metric"}
        cat = build_catalog()
        everything = {i.name for group in cat.metrics.values() for i in group}
        assert "brand_new_metric" in everything
        assert cat.n_metrics == len(original) + 1
    finally:
        P.KNOWN_METRICS = original


def test_provider_capabilities_match_the_endpoint_table():
    """The matrix must not restate capabilities by hand."""
    from harness.clients.endpoints import PROVIDER_ENDPOINTS
    from harness.ui.catalog import build_catalog

    rows = {p.name: p for p in build_catalog().providers}
    for name, spec in PROVIDER_ENDPOINTS.items():
        assert rows[name].embeddings == spec.get("supports_embeddings", True)
        assert rows[name].rerank == spec.get("supports_rerank", False)
    # The two facts most easily got wrong, asserted explicitly.
    assert not rows["anthropic"].embeddings, "Anthropic serves no embeddings"
    assert not rows["groq"].embeddings, "Groq serves no embeddings"
    assert rows["openrouter"].embeddings, "OpenRouter does serve embeddings"
    assert rows["together"].rerank, "Together is the one with a rerank endpoint"


def test_self_hosted_providers_do_not_claim_to_be_ready():
    """They carry a placeholder key the server ignores. Calling that 'ready'
    would assert a local server is up when nothing has checked."""
    from harness.ui.catalog import build_catalog

    rows = {p.name: p for p in build_catalog().providers}
    for name in ("ollama", "vllm", "lmstudio"):
        assert rows[name].local
        assert rows[name].key_state == "no key needed"
        assert not rows[name].configured


def test_a_typed_key_counts_as_configured():
    """The sidebar key never reaches the environment, so it has to be passed in."""
    from harness.ui.catalog import build_catalog

    rows = {p.name: p for p in build_catalog(extra_keys={"together": True}).providers}
    assert rows["together"].key_state == "ready"


def test_catalogue_needs_no_sdk_import():
    """The Capabilities tab renders on every rerun. Importing the openai SDK or
    qdrant_client for a table of strings added ~19s to every page load."""
    import subprocess
    import sys

    code = ("import sys; import harness.ui.catalog as c; c.build_catalog(); "
            "print('openai' in sys.modules, 'qdrant_client' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False", out.stdout


def test_every_described_name_still_exists():
    """The prose notes are keyed by discovered names. A note for a metric that
    no longer exists is dead weight; one that is missing leaves a blank line,
    which is the safe direction."""
    from harness.profiles.profile import KNOWN_METRICS
    from harness.store.schema import Pass, TaskType
    from harness.ui.catalog import METRIC_NOTES, PASS_NOTES, TASK_NOTES

    assert set(METRIC_NOTES) <= set(KNOWN_METRICS), "note for a removed metric"
    assert set(PASS_NOTES) <= {p.value for p in Pass}
    assert set(TASK_NOTES) <= {t.value for t in TaskType}


# =========================================================================== #
#  Page registry - the shell's Open/Closed seam
# =========================================================================== #
def _page_ctx(**kw):
    from pathlib import Path

    from harness.ui import MIDNIGHT
    from harness.ui.pages import PageContext

    base = dict(palette=MIDNIGHT, workspace=Path("workspace"),
                data_dir=Path("workspace/data"), store_path="workspace/traces",
                qdrant_path="workspace/qdrant", cache_dir="workspace/cache")
    base.update(kw)
    return PageContext(**base)


def test_every_shipped_screen_satisfies_the_page_protocol():
    from harness.ui.pages import Page
    from harness.ui.screens import default_registry

    for page in default_registry():
        assert isinstance(page, Page), page
        assert page.meta.key and page.meta.label


def test_a_new_screen_needs_no_shell_change():
    """The Open/Closed claim for the UI.

    A page defined here participates fully - it appears in the rail and renders
    - purely by being registered. The tabbed version required editing the shell
    for every screen, which is why the shell grew with the app.
    """
    from harness.ui.pages import PageMeta, PageRegistry

    rendered = []

    class CustomPage:
        meta = PageMeta(key="custom", label="Custom", group="Extras")

        def available(self, ctx):
            return ""

        def render(self, ctx):
            rendered.append(ctx)

    reg = PageRegistry().add(CustomPage())
    ctx = _page_ctx()
    assert [i.key for i in reg.nav_items(ctx)] == ["custom"]
    reg.get("custom").render(ctx)
    assert rendered == [ctx]


def test_registry_rejects_duplicate_keys():
    """Two pages on one key would make navigation non-deterministic."""
    from harness.ui.pages import PageMeta, PageRegistry

    class P:
        meta = PageMeta(key="dup", label="A")

        def available(self, ctx):
            return ""

        def render(self, ctx):
            pass

    reg = PageRegistry().add(P())
    with pytest.raises(ValueError, match="duplicate"):
        reg.add(P())


def test_a_blocked_page_explains_itself_rather_than_going_quiet():
    """`available()` returns a reason, not a bool. A greyed-out button that
    never says why is the failure mode this replaces."""
    from harness.ui.screens import default_registry

    reg = default_registry()
    ctx = _page_ctx()          # no store, no profile, no key
    items = {i.key: i for i in reg.nav_items(ctx)}

    assert not items["overview"].enabled
    assert items["overview"].blocked_reason, "must say why"
    assert not items["run"].enabled
    assert "dataset" in items["run"].blocked_reason.lower()
    # Screens with no preconditions stay reachable on a fresh install.
    assert items["data"].enabled
    assert items["capabilities"].enabled


def test_first_available_lands_on_a_page_that_can_actually_render():
    """A fresh install must not open on a screen that immediately says 'no data'."""
    from harness.ui.screens import default_registry

    reg = default_registry()
    key = reg.first_available(_page_ctx())
    assert not reg.get(key).available(_page_ctx())


def test_pages_depend_on_the_context_not_on_globals():
    """Dependency inversion for the UI: a page takes what it needs as an
    argument, which is what makes it renderable outside a browser."""
    import inspect

    from harness.ui.screens import default_registry

    for page in default_registry():
        params = list(inspect.signature(page.render).parameters)
        assert params == ["ctx"], f"{page.meta.key} should take only ctx"


def test_page_context_reports_readiness_without_touching_the_network():
    from harness.ui.pages import PageContext

    ctx = _page_ctx()
    assert not ctx.has_key
    assert not ctx.has_results
    assert isinstance(ctx, PageContext)
