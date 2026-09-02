"""
Overview: the dashboard landing screen.

Leads with the decision, because that is what the whole harness exists to
produce. A reader who has to scroll past four tables to find the conclusion has
been failed by the layout, not the data.

Order is deliberate: headline numbers, then the recommendation, then the
cost/quality frontier and the significance verdicts. Everything below that is a
supporting detail and lives on its own screen.
"""

from __future__ import annotations

import pandas as pd

from ..layout import Kpi, card, empty_state, kpi_row, page_header, pill, toolbar
from ..pages import PageContext, PageMeta


class OverviewPage:
    meta = PageMeta(
        key="overview", label="Overview", icon="◈", group="Dashboard",
        breadcrumb=("Dashboard", "Overview"),
        title="Overview",
        subtitle="Which model to use, and whether the difference is real.",
    )

    def available(self, ctx: PageContext) -> str:
        return "" if ctx.has_results else "Run an evaluation first"

    def render(self, ctx: PageContext) -> None:
        import streamlit as st

        from ...report import aggregate as A
        from ...report import stats as S
        from ...report.decide import Constraint, model_aggregates, project_cost, select
        from .. import charts as C

        page_header(list(self.meta.breadcrumb), self.meta.title,
                    self.meta.subtitle)

        df = ctx.store.load_all()
        if df.empty:
            empty_state("No results yet",
                        "Build a dataset, then run an evaluation.",
                        "Data → Run")
            return

        profiles = sorted(df["profile"].dropna().unique())
        c1, c2, c3 = st.columns([2, 1, 1])
        sel = c1.selectbox("Profile", profiles, key="ov_profile")
        pdf = df[df["profile"] == sel]
        quality = pdf[pdf["pass_"].isin(["baseline", "adapted"])]

        metric_opts = [c for c in ("accuracy", "faithfulness", "token_f1",
                                   "answer_relevance", "completeness",
                                   "hit_rate_at_k", "mrr")
                       if c in quality.columns and quality[c].notna().any()]
        metric = c2.selectbox("Quality metric", metric_opts or ["accuracy"],
                              key="ov_metric")
        qpd = c3.number_input("Queries/day", 100, 10_000_000, 10_000, step=1000,
                              key="ov_qpd")

        # ---- headline numbers ------------------------------------------- #
        err_rate = (float(pdf["error"].notna().mean())
                    if "error" in pdf.columns and len(pdf) else 0.0)
        total_cost = (float(pdf["cost_usd"].sum(skipna=True))
                      if "cost_usd" in pdf.columns else 0.0)
        best = (float(quality[metric].max()) if metric in quality.columns
                and quality[metric].notna().any() else None)

        kpi_row([
            Kpi("Models compared", f"{pdf['model'].nunique()}", icon="◇"),
            Kpi("Eval items", f"{pdf['item_id'].nunique():,}", icon="◇"),
            Kpi(f"Best {metric}", f"{best:.3f}" if best is not None else "-",
                icon="◇"),
            Kpi("Evaluation cost", f"${total_cost:,.2f}", icon="◇",
                note="what this run cost to produce"),
            Kpi("Error rate", f"{err_rate * 100:.1f}%",
                delta=err_rate, higher_is_better=False, icon="◇"),
        ])

        # ---- the decision ------------------------------------------------ #
        with card("The decision",
                  "Set your quality bar; the cheapest model clearing it wins."):
            bar = st.slider(f"Minimum acceptable {metric}", 0.0, 1.0, 0.80, 0.01,
                            key="ov_bar")
            constraints = [Constraint(metric, ">=", bar)]
            shortlist, candidates = select(pdf, constraints, optimise="cost_usd")
            proj = project_cost(pdf, qpd)

            if not shortlist.empty:
                top = shortlist.iloc[0]
                monthly = ""
                if not proj.empty:
                    row = proj[proj["model"] == top["model"]]
                    if not row.empty:
                        monthly = (f" · about ${row.iloc[0]['monthly_usd']:,.0f} "
                                   f"per month at {qpd:,} queries/day")
                st.markdown(
                    f'<div class="hx-hero"><div class="k">Recommended</div>'
                    f'<div class="v">{top["model"]}</div>'
                    f'<div class="n">Cheapest model clearing {metric} '
                    f'&ge; {bar:.2f}{monthly}</div></div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="hx-hero none"><div class="k">No model '
                    f'qualifies</div><div class="v">{metric} &ge; {bar:.2f}</div>'
                    f'<div class="n">Lower the bar, or see what each model '
                    f'missed by.</div></div>', unsafe_allow_html=True)
                for c in candidates:
                    reason = "; ".join(c.failures) or "; ".join(
                        f"{u} (not measured)" for u in c.unknowns)
                    st.markdown(f'<div class="hx-note">{c.model}: {reason}</div>',
                                unsafe_allow_html=True)

        # ---- frontier + significance ------------------------------------- #
        left, right = st.columns([3, 2])
        with left, card("Cost against quality",
                  "The frontier is where the decision lives - everything "
                  "grey is beaten on every axis at once."):
            st.altair_chart(C.pareto_chart(A.pareto_frontier(pdf),
                                           ctx.palette, height=280),
                            width="stretch")
        with right, card("Is the difference real?",
                  "Paired tests, Holm-corrected."):
            sig = S.significance_matrix(quality, metric=metric)
            if sig.empty:
                st.markdown('<div class="hx-note">Need two models with '
                            'overlapping items.</div>',
                            unsafe_allow_html=True)
            else:
                for _, r in sig.iterrows():
                    tone = "good" if r["significant"] else ""
                    word = "significant" if r["significant"] else "n.s."
                    st.markdown(
                        f'<div class="hx-verdict">{pill(word, tone)}'
                        f'<span>{r["model_a"]} vs {r["model_b"]} - '
                        f'{r["verdict"]}</span></div>',
                        unsafe_allow_html=True)
                pr = S.power_report(quality, metric)
                st.markdown(
                    f'<div class="hx-note">Smallest detectable gap '
                    f'{pr.mde:.3f} with {pr.n_items} paired items. '
                    f'Anything smaller is noise.</div>',
                    unsafe_allow_html=True)

        # ---- spend + speed ------------------------------------------------ #
        left, right = st.columns(2)
        agg = model_aggregates(pdf)
        with left, card(f"Projected spend at {qpd:,}/day",
                  "Generation only - the judge is evaluation infrastructure."):
            st.altair_chart(C.cost_projection_bars(proj, ctx.palette,
                                                   height=210),
                            width="stretch")
        with right, card("Latency",
                         "From the low-concurrency lane. p95, not mean."):
            st.altair_chart(C.latency_bars(agg, ctx.palette, height=210),
                            width="stretch")

        # ---- safety -------------------------------------------------------- #
        safety = [c for c in ("injection_resisted", "abstention_correct",
                              "pii_leaked")
                  if c in pdf.columns and pdf[c].notna().any()]
        if safety:
            with card("Robustness and security",
                      "Injection resistance below 1.0 means the model took "
                      "orders from a retrieved passage."):
                sa = (pdf.groupby("model")[safety].mean(numeric_only=True)
                      .reset_index())
                flat = [m for m in safety if sa[m].nunique(dropna=True) <= 1]
                varying = [m for m in safety if m not in flat]
                if flat:
                    kpi_row([
                        Kpi(m.replace("_", " "),
                            f"{sa[m].dropna().iloc[0]:.2f}"
                            if sa[m].notna().any() else "-",
                            note="identical across every model")
                        for m in flat])
                if varying:
                    cols = st.columns(len(varying))
                    for col, m in zip(cols, varying):
                        with col:
                            st.altair_chart(
                                C.metric_bars(sa, m, ctx.palette, height=170),
                                width="stretch")

        toolbar(left=pill(f"{len(pdf):,} rows", ""),
                right=pill(f"profile: {sel}", ""))
        _ = pd  # imported for type clarity in this module's helpers
