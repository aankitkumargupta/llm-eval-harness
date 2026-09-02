"""
Analysis screens: Reports (per-metric detail) and Capabilities.

Split out from the Overview because they answer different questions. Overview
says *which model to use*; Reports says *why*, in as much detail as you want.
Keeping them apart is what lets the Overview lead with a conclusion instead of
opening with a wall of tables.
"""

from __future__ import annotations

from ..layout import Kpi, card, empty_state, kpi_row, page_header, pill, toolbar
from ..pages import PageContext, PageMeta


class ReportsPage:
    meta = PageMeta(
        key="reports", label="Reports", icon="▦", group="Dashboard",
        breadcrumb=("Dashboard", "Reports"), title="Reports",
        subtitle="Every metric, the diagnostics, and the export.",
    )

    def available(self, ctx: PageContext) -> str:
        return "" if ctx.has_results else "Run an evaluation first"

    def render(self, ctx: PageContext) -> None:
        import pandas as pd
        import streamlit as st

        from ...report import aggregate as A
        from ...report import stats as S
        from ...report.decide import headroom, model_aggregates
        from .. import charts as C

        page_header(list(self.meta.breadcrumb), self.meta.title,
                    self.meta.subtitle)

        df = ctx.store.load_all()
        if df.empty:
            empty_state("Nothing recorded yet", "Run an evaluation first.")
            return

        profiles = sorted(df["profile"].dropna().unique())
        c1, c2 = st.columns([2, 1])
        sel = c1.selectbox("Profile", profiles, key="rep_profile")
        pdf = df[df["profile"] == sel]
        quality = pdf[pdf["pass_"].isin(["baseline", "adapted"])]

        metric_opts = [c for c in ("accuracy", "faithfulness", "token_f1",
                                   "answer_relevance", "completeness",
                                   "hit_rate_at_k", "mrr", "ndcg_at_k")
                       if c in quality.columns and quality[c].notna().any()]
        metric = c2.selectbox("Metric", metric_opts or ["accuracy"],
                              key="rep_metric")

        prof_cfg = ctx.profile if (ctx.profile and ctx.profile.name == sel) else None
        if prof_cfg is None:
            from ...profiles.profile import Profile
            for path in (ctx.data_dir / sel / "profile.yaml",
                         __import__("pathlib").Path(f"configs/profiles/{sel}.yaml")):
                if path.exists():
                    try:
                        prof_cfg = Profile.from_yaml(str(path))
                    except Exception:  # noqa: BLE001
                        prof_cfg = None
                    break

        # ---- leaderboard --------------------------------------------------- #
        with card("Leaderboard",
                  "Deterministic arithmetic over the profile's metric weights. "
                  "Lower-is-better metrics carry negative weights."):
            if prof_cfg and prof_cfg.metric_weights:
                st.dataframe(A.weighted_composite(quality,
                                                  prof_cfg.metric_weights),
                             width="stretch", hide_index=True)
            else:
                st.markdown('<div class="hx-note">No profile config found for '
                            'this run, so there are no weights to combine.</div>',
                            unsafe_allow_html=True)

        # ---- per-metric ---------------------------------------------------- #
        left, right = st.columns(2)
        with left, card(f"{metric} by model"):
            st.altair_chart(
                C.metric_bars(model_aggregates(pdf), metric, ctx.palette,
                              height=220), width="stretch")
        with right, card("Statistical power",
                  "Differences smaller than this are not measurable here."):
            pr = S.power_report(quality, metric)
            kpi_row([
                Kpi("Detectable gap",
                    f"{pr.mde:.3f}" if pr.mde != float("inf") else "-"),
                Kpi("Paired items", f"{pr.n_items:,}"),
            ])
            if pr.suggestions:
                st.dataframe(
                    pd.DataFrame([{"to detect": k.split("_")[1],
                                   "items needed": v}
                                  for k, v in sorted(pr.suggestions.items())]),
                    width="stretch", hide_index=True)

        for title, cols_, source, note in (
            ("Retrieval quality",
             ["hit_rate_at_k", "mrr", "ndcg_at_k", "context_recall",
              "context_precision", "average_precision", "rerank_mrr_delta"],
             quality, "How well retrieval found the gold passages."),
            ("Answer & citations",
             ["accuracy", "faithfulness", "answer_relevance", "completeness",
              "token_f1", "citation_valid_pointer", "citation_supporting",
              "citation_density", "abstention_correct"], quality, ""),
            ("Efficiency",
             ["latency_ms", "ttft_ms", "prompt_tokens", "completion_tokens",
              "gen_cost_usd", "judge_cost_usd", "cost_usd"], pdf,
             "Cost is split by subsystem: the judge is often the largest line."),
        ):
            present = [c for c in cols_
                       if c in source.columns and source[c].notna().any()]
            if present:
                with card(title, note):
                    st.dataframe(
                        source.groupby("model")[present]
                        .mean(numeric_only=True).reset_index(),
                        width="stretch", hide_index=True)
                    dormant = [c for c in cols_ if c not in present]
                    if dormant:
                        st.markdown('<div class="hx-note">Not exercised by this '
                                    f'dataset: {", ".join(dormant)}</div>',
                                    unsafe_allow_html=True)

        # ---- tuning gain ---------------------------------------------------- #
        if {"baseline", "adapted"} <= set(pdf["pass_"].unique()):
            with card("Tuning gain (adapted minus baseline)",
                      "Every model received an identical budget, so this is how "
                      "much each benefits from tuning - not how hard someone tried."):
                metrics = [m for m in (prof_cfg.active_metrics if prof_cfg else [])
                           if m in pdf.columns]
                st.dataframe(A.tuning_gain(pdf, metrics or ["accuracy"]),
                             width="stretch", hide_index=True)

        # ---- headroom ------------------------------------------------------- #
        hr = headroom(pdf, metric, 10_000)
        if not hr.empty and "usd_per_point" in hr.columns:
            with card("What each extra quality point costs",
                      "Relative to the cheapest model, per month at 10,000 "
                      "queries/day. A negative value marks a model that is both "
                      "worse and more expensive."):
                st.dataframe(
                    hr[[c for c in ("model", metric, "monthly_usd",
                                    "quality_delta", "usd_per_point")
                        if c in hr.columns]], width="stretch", hide_index=True)

        # ---- diagnostics ----------------------------------------------------- #
        err = A.error_attribution(pdf)
        trunc = A.truncation_report(pdf)
        cons = A.consistency_report(pdf)
        if ((not err.empty and err.get("errors", pd.Series([0])).sum())
                or (not trunc.empty and trunc["truncated"].sum())
                or not cons.empty):
            with card("Diagnostics"):
                if not err.empty and err.get("errors", pd.Series([0])).sum():
                    st.markdown("**Error attribution**")
                    st.markdown('<div class="hx-note">rate_limit means lower '
                                'your concurrency; context_length means lower '
                                'k; content_filter is a finding about the '
                                'model.</div>', unsafe_allow_html=True)
                    st.dataframe(err, width="stretch", hide_index=True)
                if not trunc.empty and trunc["truncated"].sum():
                    st.warning("Some answers were cut off by max_tokens. Their "
                               "completeness and citation scores are not valid.")
                    st.dataframe(trunc, width="stretch", hide_index=True)
                if not cons.empty:
                    st.markdown("**Paraphrase consistency**")
                    st.dataframe(cons, width="stretch", hide_index=True)

        if "human_label" in pdf.columns and pdf["human_label"].notna().any():
            with card("Judge calibration",
                      "Cohen's kappa against your human labels. Every "
                      "judge-scored number rests on this."):
                st.json(A.judge_calibration(quality))

        # ---- export ----------------------------------------------------------- #
        with card("Export"):
            e1, e2 = st.columns(2)
            with e1:
                if prof_cfg:
                    from ...report.html import build_html_report
                    st.download_button(
                        "Download HTML report",
                        build_html_report(pdf, prof_cfg, metric=metric),
                        file_name=f"{sel}_report.html", mime="text/html",
                        width="stretch")
            with e2:
                st.download_button("Download traces (CSV)", pdf.to_csv(index=False),
                                   file_name=f"{sel}_traces.csv",
                                   mime="text/csv", width="stretch")

        with st.expander("Raw traces"):
            st.dataframe(pdf, width="stretch", hide_index=True)

        toolbar(left=pill(f"{len(pdf):,} rows", ""),
                right=pill(f"profile: {sel}", ""))


# --------------------------------------------------------------------------- #
class CapabilitiesPage:
    meta = PageMeta(
        key="capabilities", label="Capabilities", icon="◈", group="System",
        breadcrumb=("Dashboard", "Capabilities"), title="Capabilities",
        subtitle="Everything this platform can do, read from the installed code.",
    )

    def available(self, ctx: PageContext) -> str:
        return ""

    def render(self, ctx: PageContext) -> None:
        import streamlit as st

        from .. import components as U
        from ..catalog import build_catalog, build_status

        page_header(list(self.meta.breadcrumb), self.meta.title,
                    self.meta.subtitle)

        # Nothing here is a hardcoded feature list: providers, metrics, scorers,
        # probes and passes are all introspected, so the page cannot drift out
        # of sync with the software.
        catalog = build_catalog(extra_keys={
            ctx.models_cfg.get("default_provider", "together"): ctx.has_key})
        status = build_status(catalog, store=ctx.store,
                              models_cfg=ctx.models_cfg, data_dir=ctx.data_dir)

        kpi_row([
            Kpi("Providers", f"{status.providers_ready}/{status.providers_total}",
                note="with a key configured"),
            Kpi("Metrics", f"{catalog.n_metrics}", note="across 5 subsystems"),
            Kpi("Probe families", f"{len(catalog.probes)}", note="adversarial"),
            Kpi("Profiles",
                f"{len(status.profiles) + len(status.app_profiles)}",
                note="workloads configured"),
            Kpi("Recorded", f"{status.rows:,}",
                note=f"rows across {status.runs} run(s)"),
        ])

        with card("Why it exists",
                  "Most internal LLM evaluations fail in one of four ways, and "
                  "none of them look like failure."):
            U.kv_rows([
                ("They measure noise",
                 "40 questions, 0.82 vs 0.78, ship it. That gap is inside the "
                 "margin of error and would flip on a re-run."),
                ("They under-count cost",
                 "Judge, embedding and rerank calls are billed but not "
                 "reported. Since cost is weighted negatively, the ranking is "
                 "wrong too - not just the dollar figure."),
                ("They skip the failure modes that matter",
                 "Accuracy says nothing about whether a model obeys an "
                 "instruction hidden in a retrieved document, or leaks data."),
                ("They answer the wrong question",
                 "A leaderboard says which model scored highest. The real "
                 "question is the cheapest model clearing your bar."),
            ])

        with card("Providers",
                  "Embeddings, reranking and judging are pinned separately "
                  "because they are the fixed apparatus - if two models saw "
                  "different retrieved passages, the run measures the embedders."):
            U.capability_matrix(catalog.providers)

        with card("What it measures",
                  f"{catalog.n_metrics} metrics. A profile activates only the "
                  f"subset relevant to its workload; anything inactive stays "
                  f"blank rather than zero."):
            for group, items in catalog.metrics.items():
                st.markdown(f"**{group}**")
                U.feature_cards(items)

        with card("Adversarial probes",
                  "Derived from your own questions, so they test your actual "
                  "system."):
            U.feature_cards(catalog.probes)
            if not status.has_probes:
                st.info("No probe items in your evalsets yet - generate them on "
                        "the Probes screen. Free and instant.")

        with card("Analysis and decisions"):
            U.kv_rows([(n, f"{d}  -  main.py {c}")
                       for n, c, d in catalog.analysis])

        left, right = st.columns(2)
        with left, card("How a run works"):
            st.markdown("**Passes**")
            U.feature_cards(catalog.passes, columns=1)
            st.markdown("**Task types**")
            U.feature_cards(catalog.tasks, columns=1)
        with right:
            with card("Scoring pipeline",
                      "Adding a metric is a new class plus a registry entry - "
                      "no existing file changes."):
                U.feature_cards(catalog.scorers, columns=1)
            with card("Retrieval"):
                st.markdown("**Modes**")
                U.chips(catalog.retrieval_modes)
                st.markdown("**Answer scorers**")
                U.chips(catalog.answer_scorers)

        with card("Reliability and spend control"):
            U.kv_rows(catalog.reliability)

        with card("Command line",
                  "Everything here is scriptable, and the gate runs in CI."):
            U.command_list([
                ("main.py validate --profile P", "Check config and routing "
                                                 "before spending"),
                ("main.py estimate --profile P", "Forecast cost, judge included"),
                ("main.py probes --profile P", "Generate adversarial items"),
                ("main.py ingest --profile P", "Embed and index the corpus"),
                ("main.py run --profile P --budget 25", "Run under a ceiling"),
                ("main.py report --profile P", "Leaderboard, significance, power"),
                ("main.py compare --profile P", "Paired significance tests"),
                ("main.py decide --profile P", "Constrained recommendation"),
                ("main.py arena --profile P", "Pairwise Elo"),
                ("main.py gate --baseline RUN", "CI gate; non-zero on regression"),
                ("main.py html --profile P", "Self-contained HTML report"),
                ("main.py runs", "List recorded runs"),
            ])

        with card("This installation"):
            a, b = st.columns(2)
            with a:
                st.markdown("**Profiles**")
                U.chips(status.profiles + status.app_profiles)
                if status.models_seen:
                    st.markdown("**Models evaluated**")
                    U.chips(status.models_seen)
            with b:
                st.markdown("**Pricing**")
                st.markdown(f'<div class="hx-note">{status.priced_models} models '
                            f'priced, as of {status.pricing_as_of}.</div>',
                            unsafe_allow_html=True)
                if status.pricing_stale:
                    st.warning(status.pricing_stale)
                if status.unpriced:
                    st.warning(f"No price for {status.unpriced}.")
                st.markdown("**Verification**")
                st.markdown(f'<div class="hx-note">{status.n_tests} test '
                            f'modules, all offline - pytest needs no key and '
                            f'costs nothing.</div>', unsafe_allow_html=True)
