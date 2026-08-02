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

from harness.profiles.profile import Profile
from harness.report.aggregate import (
    bootstrap_ci,
    elo_from_pairwise,
    pareto_frontier,
    tuning_gain,
    weighted_composite,
)
from harness.store.store import TraceStore

st.set_page_config(page_title="LLM Eval Harness", layout="wide")


# --------------------------------------------------------------------------- #
#  Data loading (cached so refreshes are cheap)                               #
# --------------------------------------------------------------------------- #
def _run_cfg() -> dict:
    return yaml.safe_load(Path("configs/run.yaml").read_text())


@st.cache_data(show_spinner=False)
def load_traces(store_path: str, mtime: float) -> pd.DataFrame:
    """Load the whole trace store. `mtime` is in the signature only so the cache
    invalidates automatically when the Parquet file changes on disk."""
    store = TraceStore(store_path)
    return store.load_all()


def _profile_config(name: str) -> Profile | None:
    p = Path(f"configs/profiles/{name}.yaml")
    if p.exists():
        return Profile.from_yaml(str(p))
    return None


# --------------------------------------------------------------------------- #
#  Header + store status                                                       #
# --------------------------------------------------------------------------- #
st.title("Multi-Model LLM Evaluation Harness")
st.caption("Read-only dashboard over the trace store. "
           "Run `python main.py run --profile ...` to produce data.")

cfg = _run_cfg()
store_path = cfg["store_path"]

if not Path(store_path).exists():
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

df = load_traces(store_path, Path(store_path).stat().st_mtime)
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
tab_board, tab_pareto, tab_tuning, tab_ci, tab_raw = st.tabs(
    ["Leaderboard", "Pareto (acc/cost/latency)", "Tuning gain",
     "Confidence intervals", "Raw traces"]
)

# ---- Leaderboard: weighted composite ------------------------------------- #
with tab_board:
    st.subheader(f"Weighted composite — {profile}")
    if profile_cfg and profile_cfg.metric_weights:
        st.caption("Composite uses this profile's metric weights "
                   "(negative weights = lower-is-better, e.g. cost).")
        comp = weighted_composite(test_df, profile_cfg.metric_weights)
        st.dataframe(comp, use_container_width=True, hide_index=True)

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
            st.altair_chart(chart, use_container_width=True)

        with st.expander("Metric weights used"):
            st.json(profile_cfg.metric_weights)
    else:
        st.warning("No profile config / weights found for this profile.")

# ---- Pareto frontier ------------------------------------------------------ #
with tab_pareto:
    st.subheader("Accuracy vs. cost vs. latency")
    st.caption("Points on the frontier are NOT dominated on all three objectives "
               "(higher accuracy, lower cost, lower latency). Latency is drawn "
               "from the low-concurrency lane where available.")
    try:
        pf = pareto_frontier(pdf)
        if pf.empty or pf["accuracy"].isna().all():
            st.info("Not enough accuracy/cost/latency data yet to build a frontier.")
        else:
            pf_plot = pf.copy()
            pf_plot["frontier"] = pf_plot["on_frontier"].map(
                {True: "on frontier", False: "dominated"})
            scatter = (
                alt.Chart(pf_plot)
                .mark_circle(size=200, opacity=0.85)
                .encode(
                    x=alt.X("cost:Q", title="mean cost per query (USD)"),
                    y=alt.Y("accuracy:Q", title="mean accuracy"),
                    color=alt.Color("frontier:N",
                                    scale=alt.Scale(
                                        domain=["on frontier", "dominated"],
                                        range=["#2e7d32", "#b0bec5"])),
                    size=alt.Size("latency:Q", title="latency (ms)",
                                  scale=alt.Scale(range=[80, 600])),
                    tooltip=["model", "accuracy", "cost", "latency", "frontier"],
                )
                .properties(height=460)
                .interactive()
            )
            st.altair_chart(scatter, use_container_width=True)
            st.dataframe(pf, use_container_width=True, hide_index=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not build Pareto view: {e}")

# ---- Tuning gain ---------------------------------------------------------- #
with tab_tuning:
    st.subheader("Tuning gain (adapted − baseline)")
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
        st.dataframe(gain, use_container_width=True, hide_index=True)

        # long-form for a grouped bar chart
        gain_cols = [c for c in gain.columns if c.endswith("_gain")]
        if gain_cols:
            long = gain.melt(id_vars="model", value_vars=gain_cols,
                             var_name="metric", value_name="gain")
            long["metric"] = long["metric"].str.replace("_gain", "", regex=False)
            chart = (
                alt.Chart(long)
                .mark_bar()
                .encode(
                    x=alt.X("gain:Q", title="adapted − baseline"),
                    y=alt.Y("model:N", sort="-x", title=None),
                    color=alt.Color("metric:N"),
                    tooltip=["model", "metric", "gain"],
                )
                .properties(height=max(200, 30 * len(long)))
            )
            st.altair_chart(chart, use_container_width=True)

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
                            use_container_width=True)
            st.dataframe(ci_df, use_container_width=True, hide_index=True)

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
    st.dataframe(view[front + rest], use_container_width=True, hide_index=True)

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
                st.markdown(f"**Cited ids:** {list(r.get('cited_ids') or [])}")
