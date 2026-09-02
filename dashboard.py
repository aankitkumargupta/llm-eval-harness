"""
Results dashboard for the evaluation harness.

Read-only view over the trace store (runs/traces.parquet). It renders exactly the
aggregates the harness already computes in harness/report/aggregate.py — no new
analysis logic lives here, so the dashboard can never disagree with `main.py
report`. It just displays those numbers with filtering and charts.

Run it with:
    streamlit run dashboard.py

It does NOT make any Together calls or touch Qdrant — it only reads the Parquet
file produced by `python main.py run`. So it's safe to leave open and refresh
while runs complete.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from harness.env import load_env
from harness.profiles.profile import Profile
from harness.report.aggregate import (
    bootstrap_ci,
    pareto_frontier,
    tuning_gain,
    weighted_composite,
)
from harness.report.stats import power_report, significance_matrix
from harness.store.store import TraceStore
from harness.ui import active_palette, page_css, register_altair_theme
from harness.ui import charts as C
from harness.ui import components as U

# Consistent with the app: a .env in the project root is honoured.
load_env()

st.set_page_config(page_title="LLM Eval Harness", layout="wide")

# The dashboard and the app render the same numbers, so they render them the
# same way: one palette, one Altair theme, one set of components.
PALETTE = active_palette()
register_altair_theme(PALETTE)
st.markdown(page_css(PALETTE), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Data loading (cached so refreshes are cheap)                               #
# --------------------------------------------------------------------------- #
def _run_cfg() -> dict:
    return yaml.safe_load(Path("configs/run.yaml").read_text())


@st.cache_data(show_spinner=False)
def load_traces(store_path: str, fingerprint: str) -> pd.DataFrame:
    """Load the whole trace store.

    `fingerprint` is in the signature only so Streamlit's cache invalidates when
    the store changes on disk. It must reflect the *contents*: the store is now
    a directory of append-only Parquet parts, and a directory mtime does not
    change when an existing part is rewritten, so a plain mtime would serve
    stale results after a resumed run.
    """
    store = TraceStore(store_path)
    return store.load_all()


def _store_fingerprint(store_path: str) -> str:
    """Names, sizes and mtimes of every part - cheap, and actually sensitive
    to the writes we care about."""
    p = Path(store_path)
    parts = []
    if p.is_dir():
        for f in sorted(p.glob("part-*.parquet")):
            s = f.stat()
            parts.append(f"{f.name}:{s.st_size}:{s.st_mtime_ns}")
    elif p.exists():
        s = p.stat()
        parts.append(f"{p.name}:{s.st_size}:{s.st_mtime_ns}")
    return "|".join(parts)


def _profile_config(name: str) -> Profile | None:
    """Find a profile by name, wherever it was created.

    The CLI writes profiles to configs/profiles/; the app writes them into
    workspace/data/<name>/. Checking only the first meant every app-created run
    showed "no weights found" and lost its leaderboard.
    """
    for candidate in (Path(f"configs/profiles/{name}.yaml"),
                      Path("workspace/data") / name / "profile.yaml"):
        if candidate.exists():
            try:
                return Profile.from_yaml(str(candidate))
            except Exception:  # noqa: BLE001 - a bad profile must not blank the page
                return None
    return None


# --------------------------------------------------------------------------- #
#  Header + store status                                                       #
# --------------------------------------------------------------------------- #
st.title("Multi-Model LLM Evaluation Harness")
st.caption("Read-only dashboard over the trace store. "
           "Run `python main.py run --profile ...` to produce data.")

cfg = _run_cfg()
store_path = cfg["store_path"]

if not TraceStore(store_path).exists:
    st.info(
        f"No trace store found at `{store_path}` yet.\n\n"
        "Run the harness first:\n"
        "```\n"
        "python main.py ingest --profile configs/profiles/regulated_qa.yaml\n"
        "python main.py run    --profile configs/profiles/regulated_qa.yaml\n"
        "```\n"
        "then reload this page."
    )
    st.stop()

df = load_traces(store_path, _store_fingerprint(store_path))
if df.empty:
    st.warning("The trace store exists but is empty. Run a profile first.")
    st.stop()


# --------------------------------------------------------------------------- #
#  Sidebar filters                                                             #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Filters")
    profiles = sorted(df["profile"].dropna().unique().tolist())
    profile = st.selectbox("Profile", profiles)

    pdf_all = df[df["profile"] == profile]

    models = sorted(pdf_all["model"].dropna().unique().tolist())
    picked_models = st.multiselect("Models", models, default=models)

    st.divider()
    st.caption(f"Rows in store: {len(df):,}")
    st.caption(f"Rows for `{profile}`: {len(pdf_all):,}")

# apply model filter
pdf = pdf_all[pdf_all["model"].isin(picked_models)] if picked_models else pdf_all
profile_cfg = _profile_config(profile)

# test-split rows (baseline + adapted) are what leaderboards use
test_df = pdf[pdf["pass_"].isin(["baseline", "adapted"])]


# --------------------------------------------------------------------------- #
#  Tabs                                                                        #
# --------------------------------------------------------------------------- #
tab_board, tab_sig, tab_pareto, tab_tuning, tab_ci, tab_raw = st.tabs(
    ["Leaderboard", "Significance", "Pareto (acc/cost/latency)", "Tuning gain",
     "Confidence intervals", "Raw traces"]
)

# ---- Significance: are the leaderboard gaps real? ------------------------- #
with tab_sig:
    st.subheader("Is the difference real?")
    st.caption(
        "Paired tests on the same items, Holm-Bonferroni corrected for the "
        "number of comparisons. This is the question a leaderboard cannot "
        "answer: the per-model confidence intervals in the next tab are "
        "*marginal*, and two overlapping marginal intervals can still hide a "
        "significant difference - because both models faced identical "
        "questions, and pairing removes item difficulty from the estimate."
    )
    sig_opts = [c for c in ("accuracy", "faithfulness", "token_f1",
                            "answer_relevance", "completeness", "hit_rate_at_k",
                            "mrr", "ndcg_at_k", "context_recall",
                            "citation_supporting")
                if c in test_df.columns and test_df[c].notna().any()]
    if not sig_opts:
        st.info("No metric has data yet for this profile.")
    else:
        sig_metric = st.selectbox("Metric", sig_opts, key="dash_sig_metric")
        sig = significance_matrix(test_df, metric=sig_metric)
        if sig.empty:
            st.info("Need at least two models with overlapping items.")
        else:
            for _, r in sig.iterrows():
                line = (f"**{r['model_a']}** vs **{r['model_b']}** - "
                        f"{r['verdict']}")
                (st.success if r["significant"] else st.info)(line)
            st.dataframe(
                sig[["model_a", "model_b", "n_pairs", "mean_a", "mean_b",
                     "diff", "p_value", "p_adjusted", "significant", "test"]],
                width="stretch", hide_index=True)

        pr = power_report(test_df, sig_metric)
        st.divider()
        st.subheader("Statistical power")
        st.metric("Smallest detectable gap", f"{pr.mde:.3f}"
                  if pr.mde != float("inf") else "n/a")
        st.caption(f"With {pr.n_items} paired items, any difference smaller "
                   f"than this is indistinguishable from noise. Adding "
                   f"questions is the only fix.")
        if pr.suggestions:
            st.dataframe(
                pd.DataFrame([{"to detect a gap of": k.split("_")[1],
                               "paired items needed": v}
                              for k, v in sorted(pr.suggestions.items())]),
                width="stretch", hide_index=True)

# ---- Leaderboard: weighted composite ------------------------------------- #
with tab_board:
    st.subheader(f"Weighted composite — {profile}")
    if profile_cfg and profile_cfg.metric_weights:
        st.caption("Composite uses this profile's metric weights "
                   "(negative weights = lower-is-better, e.g. cost).")
        comp = weighted_composite(test_df, profile_cfg.metric_weights)
        st.dataframe(comp, width="stretch", hide_index=True)

        # bar chart of the headline number
        if not comp.empty:
            chart = (
                alt.Chart(comp)
                .mark_bar()
                .encode(
                    x=alt.X("composite:Q", title="weighted composite"),
                    y=alt.Y("model:N", sort="-x", title=None),
                    tooltip=list(comp.columns),
                )
                .properties(height=40 * len(comp) + 40)
            )
            st.altair_chart(chart, width="stretch")

        with st.expander("Metric weights used"):
            st.json(profile_cfg.metric_weights)
    else:
        st.warning("No profile config / weights found for this profile.")

# ---- Pareto frontier ------------------------------------------------------ #
with tab_pareto:
    st.subheader("Cost against quality")
    U.note("Models on the frontier are not beaten on every objective at once "
           "(higher accuracy, lower cost, lower latency). Everything gray is "
           "strictly worse than something on the frontier. Latency is drawn "
           "from the low-concurrency lane where available.")
    try:
        pf = pareto_frontier(pdf)
        if pf.empty or pf["accuracy"].isna().all():
            st.info("Not enough accuracy/cost/latency data yet to build a frontier.")
        else:
            # The shared builder, so this chart cannot disagree with the app's.
            # The previous inline version carried its own hardcoded green/gray
            # and encoded latency as marker SIZE, which double-encoded a value
            # the tooltip already gave and made small differences unreadable.
            st.altair_chart(C.pareto_chart(pf, PALETTE, height=420),
                            width="stretch")
            st.dataframe(pf, width="stretch", hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build Pareto view: {e}")

# ---- Tuning gain ---------------------------------------------------------- #
with tab_tuning:
    st.subheader("Tuning gain (adapted - baseline)")
    st.caption("Positive = the model improved under the equal-budget tuning pass.")
    has_both = (set(pdf["pass_"].unique()) >= {"baseline", "adapted"})
    if not has_both:
        st.info("Need both a baseline and an adapted pass in the store to show gain. "
                "Enable both passes in configs/run.yaml and re-run.")
    else:
        metrics = [m for m in (profile_cfg.active_metrics if profile_cfg else [])
                   if m in pdf.columns]
        if not metrics:
            metrics = [c for c in ["accuracy", "faithfulness", "hit_rate_at_k"]
                       if c in pdf.columns]
        gain = tuning_gain(pdf, metrics)
        st.dataframe(gain, width="stretch", hide_index=True)

        # long-form for a grouped bar chart
        gain_cols = [c for c in gain.columns if c.endswith("_gain")]
        if gain_cols:
            long = gain.melt(id_vars="model", value_vars=gain_cols,
                             var_name="metric", value_name="gain")
            long["metric"] = long["metric"].str.replace("_gain", "", regex=False)
            # Faceted rather than coloured by metric: a metric list longer
            # than the categorical cap would force hue cycling, and two cycled
            # hues are indistinguishable under colour-vision deficiency.
            # Diverging, because the sign is the point - a gain and a loss must
            # not read as the same thing at different lengths.
            lo, mid, hi = PALETTE.diverging
            limit = max(0.01, float(long["gain"].abs().max()))
            chart = (
                alt.Chart(long)
                .mark_bar(size=14, cornerRadiusEnd=4)
                .encode(
                    x=alt.X("gain:Q", title="adapted - baseline"),
                    y=alt.Y("model:N", sort="-x", title=None,
                            axis=alt.Axis(labelLimit=170)),
                    color=alt.Color("gain:Q",
                                    scale=alt.Scale(domain=[-limit, 0, limit],
                                                    range=[hi, mid, lo]),
                                    legend=None),
                    tooltip=["model", "metric", alt.Tooltip("gain:Q", format="+.4f")],
                )
                .properties(height=max(120, 34 * long["model"].nunique()))
                .facet(row=alt.Row("metric:N", title=None,
                                   header=alt.Header(labelAngle=0,
                                                     labelAnchor="start",
                                                     labelFontWeight=600)))
                .resolve_scale(x="shared")
            )
            st.altair_chart(chart, width="stretch")

# ---- Confidence intervals ------------------------------------------------- #
with tab_ci:
    st.subheader("Bootstrap 95% CIs")
    st.caption("So 'A beats B by 1 point' can be judged against its uncertainty. "
               "Computed on the baseline pass.")
    metric_choices = [c for c in
                      ["accuracy", "faithfulness", "hit_rate_at_k", "mrr",
                       "answer_relevance", "completeness", "citation_supporting"]
                      if c in pdf.columns]
    if not metric_choices:
        st.info("No metric columns available for CIs yet.")
    else:
        metric = st.selectbox("Metric", metric_choices)
        base = pdf[pdf["pass_"] == "baseline"]
        rows = []
        for model, g in base.groupby("model"):
            mean, lo, hi = bootstrap_ci(g[metric].to_numpy())
            rows.append({"model": model, "mean": mean, "lo": lo, "hi": hi})
        ci_df = pd.DataFrame(rows).dropna(subset=["mean"]).sort_values(
            "mean", ascending=False)
        if ci_df.empty:
            st.info(f"No `{metric}` data on the baseline pass.")
        else:
            base_chart = alt.Chart(ci_df)
            bars = base_chart.mark_point(size=90, filled=True).encode(
                x=alt.X("mean:Q", title=metric),
                y=alt.Y("model:N", sort="-x", title=None),
            )
            errbars = base_chart.mark_errorbar().encode(
                x="lo:Q", x2="hi:Q",
                y=alt.Y("model:N", sort="-x"),
            )
            st.altair_chart((errbars + bars).properties(height=40 * len(ci_df) + 40),
                            width="stretch")
            st.dataframe(ci_df, width="stretch", hide_index=True)

# ---- Raw traces ----------------------------------------------------------- #
with tab_raw:
    st.subheader("Raw trace rows")
    st.caption("One row per eval item. Filter by pass and inspect prompts/outputs.")
    passes = sorted(pdf["pass_"].dropna().unique().tolist())
    pick_pass = st.multiselect("Pass", passes, default=passes)
    view = pdf[pdf["pass_"].isin(pick_pass)] if pick_pass else pdf

    # keep the wide table readable: show key columns first
    front = [c for c in ["item_id", "model", "pass_", "item_type", "accuracy",
                         "faithfulness", "hit_rate_at_k", "cost_usd", "latency_ms",
                         "cache_hit", "error"] if c in view.columns]
    rest = [c for c in view.columns if c not in front]
    st.dataframe(view[front + rest], width="stretch", hide_index=True)

    # drill into a single item's prompt + output
    if not view.empty:
        with st.expander("Inspect a single item's prompt & output"):
            item_id = st.selectbox("item_id", sorted(view["item_id"].unique()))
            model_id = st.selectbox("model", sorted(view["model"].unique()))
            sel = view[(view["item_id"] == item_id) & (view["model"] == model_id)]
            if not sel.empty:
                r = sel.iloc[0]
                st.markdown("**Assembled prompt**")
                st.code(r.get("assembled_prompt", "") or "(empty)")
                st.markdown("**Model output**")
                st.code(r.get("raw_output", "") or "(empty)")
                # `or []` is wrong for a numpy array: truth-testing one raises
                # "the truth value of an empty array is ambiguous", which took
                # down the entire page rather than one field. List columns come
                # back from Parquet as arrays, so this is the normal case.
                cited = r.get("cited_ids")
                cited = list(cited) if cited is not None and len(cited) else []
                st.markdown(f"**Cited ids:** {cited or '(none)'}")
