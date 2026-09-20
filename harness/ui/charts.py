"""
Chart builders.

Each chart's *form* is picked from the data's job before any colour is chosen:

  pareto_chart: "which models are not beaten on every axis?" One class is
                        the subject, the rest are context, so this is EMPHASIS:
                        one accent hue plus gray. Not eight categorical hues,
                        scatter is an all-pairs form where more than three hues
                        cannot clear the colour-vision floors anyway.
  significance_matrix: "is A better, worse, or indistinguishable from B?" That
                        is polarity, so DIVERGING: blue <-> red with a neutral
                        gray midpoint that genuinely reads as "no difference".
  metric_bars: "compare magnitude across models." One series, so ONE
                        colour for every bar. Colouring bars darker-where-bigger
                        would double-encode length as hue and burn the only free
                        channel on information the bar already shows.
  cost_projection, magnitude again, same treatment, with the quality
                        constraint carried by a label rather than a second axis.
  latency_bars, p50 and p95 as two named series (2 slots, legend shown).

Rules held throughout: no dual axes, hairline solid grid, thin marks, a legend
whenever two or more series are present, tooltips on every mark, and text in ink
tokens rather than series colours.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from .theme import Palette

# Charts are compact by default: they sit in a dashboard column, not a report page.
_H = 260


def _empty(message: str = "No data yet.") -> alt.Chart:
    """A placeholder that keeps layout stable when a metric wasn't collected."""
    return (
        alt.Chart(pd.DataFrame({"t": [message]}))
        .mark_text(align="center", baseline="middle", fontSize=12, dy=0)
        .encode(text="t:N")
        .properties(height=120)
    )


def _short(names: pd.Series, width: int = 26) -> pd.Series:
    """Trim provider-prefixed model ids for axis labels.

    Full strings like `openrouter:meta-llama/llama-3.3-70b-instruct` blow out
    the axis and push the plot into a sliver. The tooltip carries the full name.
    """
    return names.map(
        lambda s: str(s) if len(str(s)) <= width
        else str(s)[: width - 1].rstrip("/:-") + "…")


# --------------------------------------------------------------------------- #
#  Pareto frontier: EMPHASIS
# --------------------------------------------------------------------------- #
def pareto_chart(pf: pd.DataFrame, palette: Palette,
                 height: int = 300) -> alt.LayerChart | alt.Chart:
    """Cost against quality, with the non-dominated models emphasised.

    The frontier is the decision: those are the models where buying more quality
    genuinely costs more, and everything off it is strictly beaten. So the
    frontier carries the accent hue and the rest recede to gray, identity is
    also carried by the legend and by direct labels, never by colour alone.
    """
    if pf is None or pf.empty or "accuracy" not in pf.columns:
        return _empty("Run an evaluation to see the frontier.")

    d = pf.copy()
    d["on_frontier"] = d.get("on_frontier", False).astype(bool)
    d["status"] = d["on_frontier"].map({True: "On frontier", False: "Dominated"})
    d["label"] = _short(d["model"])

    colour = alt.Color(
        "status:N",
        scale=alt.Scale(domain=["On frontier", "Dominated"],
                        range=[palette.series_1, palette.muted_mark]),
        legend=alt.Legend(title=None),
    )

    base = alt.Chart(d).encode(
        x=alt.X("cost:Q", title="cost per query (USD)",
                axis=alt.Axis(format="$.4f"),
                scale=alt.Scale(nice=True, zero=False)),
        y=alt.Y("accuracy:Q", title="accuracy",
                scale=alt.Scale(nice=True, zero=False)),
        tooltip=[
            alt.Tooltip("model:N", title="model"),
            alt.Tooltip("accuracy:Q", title="accuracy", format=".4f"),
            alt.Tooltip("cost:Q", title="cost/query", format="$.6f"),
            alt.Tooltip("latency:Q", title="latency (ms)", format=".0f"),
            alt.Tooltip("status:N", title=""),
        ],
    )

    # 2px surface ring so overlapping points stay legible; the ring is also part
    # of the hover target, which small dots badly need.
    points = base.mark_point(
        filled=True, size=170, opacity=1,
        stroke=palette.surface, strokeWidth=2,
    ).encode(color=colour)

    labels = base.mark_text(
        align="left", dx=11, dy=-1, fontSize=11, color=palette.ink_secondary,
    ).encode(text="label:N")

    return (points + labels).properties(height=height)


# --------------------------------------------------------------------------- #
#  Significance: DIVERGING
# --------------------------------------------------------------------------- #
def significance_matrix(sig: pd.DataFrame, palette: Palette,
                        height: int = 260) -> alt.LayerChart | alt.Chart:
    """Who beats whom, and by how much, with insignificant cells reading as neutral.

    The encoded value is the signed difference, but **only for comparisons that
    survived multiple-comparison correction**. Everything else is forced to zero
    so it lands on the neutral midpoint. That is the honest rendering: an
    uncorrected heatmap of raw deltas shows a confident pattern of colour where
    the data supports none of it.
    """
    if sig is None or sig.empty:
        return _empty("Need two models with overlapping items.")

    rows = []
    for _, r in sig.iterrows():
        signed = float(r["diff"]) if bool(r.get("significant")) else 0.0
        verdict = str(r.get("verdict", ""))
        rows.append({"row": r["model_a"], "col": r["model_b"],
                     "value": signed, "verdict": verdict})
        rows.append({"row": r["model_b"], "col": r["model_a"],
                     "value": -signed, "verdict": verdict})
    d = pd.DataFrame(rows)
    d["row_label"] = _short(d["row"], 22)
    d["col_label"] = _short(d["col"], 22)

    limit = max(0.01, float(d["value"].abs().max()))
    lo, mid, hi = palette.diverging

    cells = (
        alt.Chart(d)
        .mark_rect(stroke=palette.surface, strokeWidth=2)  # 2px surface gap
        .encode(
            x=alt.X("col_label:N", title=None,
                    axis=alt.Axis(labelAngle=-30, labelLimit=140)),
            y=alt.Y("row_label:N", title=None,
                    axis=alt.Axis(labelLimit=140)),
            color=alt.Color(
                "value:Q",
                title="advantage of row over column",
                scale=alt.Scale(domain=[-limit, 0, limit],
                                range=[hi, mid, lo]),
                legend=alt.Legend(format=".2f", gradientLength=140),
            ),
            tooltip=[alt.Tooltip("row:N", title="row model"),
                     alt.Tooltip("col:N", title="column model"),
                     alt.Tooltip("value:Q", title="advantage", format="+.4f"),
                     alt.Tooltip("verdict:N", title="")],
        )
        .properties(height=height)
    )
    return cells


# --------------------------------------------------------------------------- #
#  Magnitude: SINGLE HUE
# --------------------------------------------------------------------------- #
def metric_bars(df: pd.DataFrame, metric: str, palette: Palette,
                title: str = "", height: int = 240,
                fmt: str = ".3f") -> alt.LayerChart | alt.Chart:
    """One metric across models, sorted, one colour.

    Horizontal because model names are long. A value-ramp across the bars would
    encode length twice and fail the categorical checks by design, so every bar
    is slot 1.
    """
    if df is None or df.empty or metric not in df.columns:
        return _empty(f"'{metric}' was not measured in this run.")

    d = df[["model", metric]].dropna()
    if d.empty:
        return _empty(f"'{metric}' was not measured in this run.")
    d = d.sort_values(metric, ascending=False)
    d["label"] = _short(d["model"])

    base = alt.Chart(d).encode(
        y=alt.Y("label:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=190)),
        x=alt.X(f"{metric}:Q", title=metric.replace("_", " ")),
        tooltip=[alt.Tooltip("model:N", title="model"),
                 alt.Tooltip(f"{metric}:Q", title=metric, format=fmt)],
    )
    bars = base.mark_bar(color=palette.series_1, size=18,
                         cornerRadiusEnd=4)
    # Direct-label every bar only because there are few of them and the value
    # is the point; the axis alone would make this a lookup exercise.
    labels = base.mark_text(align="left", dx=5, fontSize=11,
                            color=palette.ink_secondary).encode(
        text=alt.Text(f"{metric}:Q", format=fmt))

    chart = (bars + labels).properties(height=height)
    return chart.properties(title=title) if title else chart


def cost_projection_bars(proj: pd.DataFrame, palette: Palette,
                         height: int = 240) -> alt.LayerChart | alt.Chart:
    """Projected monthly spend per model at the chosen volume."""
    if proj is None or proj.empty or "monthly_usd" not in proj.columns:
        return _empty("No cost data yet.")

    d = proj.copy().sort_values("monthly_usd")
    d["label"] = _short(d["model"])
    base = alt.Chart(d).encode(
        y=alt.Y("label:N", sort="x", title=None, axis=alt.Axis(labelLimit=190)),
        x=alt.X("monthly_usd:Q", title="projected USD / month",
                axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("model:N", title="model"),
                 alt.Tooltip("cost_per_query:Q", title="per query",
                             format="$.6f"),
                 alt.Tooltip("daily_usd:Q", title="per day", format="$,.2f"),
                 alt.Tooltip("monthly_usd:Q", title="per month", format="$,.2f"),
                 alt.Tooltip("annual_usd:Q", title="per year", format="$,.0f")],
    )
    bars = base.mark_bar(color=palette.series_1, size=18,
                         cornerRadiusEnd=4)
    labels = base.mark_text(align="left", dx=5, fontSize=11,
                            color=palette.ink_secondary).encode(
        text=alt.Text("monthly_usd:Q", format="$,.0f"))
    return (bars + labels).properties(height=height)


def latency_bars(agg: pd.DataFrame, palette: Palette,
                 height: int = 240) -> alt.LayerChart | alt.Chart:
    """p50 and p95 side by side, two named series, so a legend is present.

    Both are milliseconds on one axis. Putting latency and any other unit on a
    second y-scale would invent a relationship the data does not contain.
    """
    cols = [c for c in ("latency_p50_ms", "latency_p95_ms") if c in (agg.columns
                                                                     if agg is not None else [])]
    if agg is None or agg.empty or not cols:
        return _empty("Run the latency lane for clean p50/p95.")

    d = agg[["model", *cols]].dropna(subset=cols, how="all")
    if d.empty:
        return _empty("Run the latency lane for clean p50/p95.")

    long = d.melt(id_vars="model", value_vars=cols,
                  var_name="percentile", value_name="ms").dropna()
    long["percentile"] = long["percentile"].map(
        {"latency_p50_ms": "p50", "latency_p95_ms": "p95"})
    long["label"] = _short(long["model"])

    return (
        alt.Chart(long)
        .mark_bar(cornerRadiusEnd=4, size=9)
        .encode(
            y=alt.Y("label:N", title=None, sort="x",
                    axis=alt.Axis(labelLimit=190)),
            x=alt.X("ms:Q", title="latency (ms)"),
            yOffset=alt.YOffset("percentile:N"),
            color=alt.Color(
                "percentile:N",
                scale=alt.Scale(domain=["p50", "p95"],
                                range=[palette.series_1, palette.series_2]),
                legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("model:N", title="model"),
                     alt.Tooltip("percentile:N", title=""),
                     alt.Tooltip("ms:Q", title="ms", format=".0f")],
        )
        .properties(height=height)
    )


def quality_cost_scatter(agg: pd.DataFrame, palette: Palette,
                         quality: str = "accuracy",
                         height: int = 280) -> alt.LayerChart | alt.Chart:
    """Quality against monthly cost, sized by nothing and coloured by nothing.

    A deliberately plain scatter: two axes carry the whole story, so adding a
    size or colour channel would decorate rather than inform.
    """
    if agg is None or agg.empty or quality not in agg.columns:
        return _empty(f"'{quality}' was not measured.")
    if "cost_usd" not in agg.columns:
        return _empty("No cost data yet.")

    d = agg[["model", quality, "cost_usd"]].dropna()
    if d.empty:
        return _empty("Not enough data yet.")
    d["label"] = _short(d["model"])

    base = alt.Chart(d).encode(
        x=alt.X("cost_usd:Q", title="cost per query (USD)",
                axis=alt.Axis(format="$.4f"), scale=alt.Scale(zero=False, nice=True)),
        y=alt.Y(f"{quality}:Q", title=quality.replace("_", " "),
                scale=alt.Scale(zero=False, nice=True)),
        tooltip=[alt.Tooltip("model:N", title="model"),
                 alt.Tooltip(f"{quality}:Q", title=quality, format=".4f"),
                 alt.Tooltip("cost_usd:Q", title="cost/query", format="$.6f")],
    )
    pts = base.mark_point(filled=True, size=150, color=palette.series_1,
                          stroke=palette.surface, strokeWidth=2)
    lbl = base.mark_text(align="left", dx=10, fontSize=11,
                         color=palette.ink_secondary).encode(text="label:N")
    return (pts + lbl).properties(height=height)
