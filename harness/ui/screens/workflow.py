"""
The three screens that produce data: Data, Probes, Run.

Grouped in one module because they form a single linear workflow and share the
same preconditions, you cannot probe without a dataset, or run without an
index. Each is still its own `Page`, so the registry treats them independently
and `available()` explains what is blocking.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ..layout import Kpi, card, empty_state, kpi_row, page_header, pill, toolbar
from ..pages import PageContext, PageMeta


def _job_panel(job_id, label: str) -> None:
    """Live progress for a background job.

    A fragment so only this block reruns while a job is in flight; the worker
    never touches Streamlit, it only mutates a plain Job object.
    """
    import streamlit as st

    from ...orchestration.jobs import get_job

    @st.fragment(run_every=2)
    def _render():
        job = get_job(job_id)
        if job is None:
            return
        snap = job.snapshot()
        if snap["status"] == "running":
            st.progress(snap["frac"],
                        text=f"{label}  {snap['done']:,}/{snap['total']:,}  "
                             f"({snap['frac'] * 100:.0f}%)  ·  "
                             f"{snap['elapsed']:.0f}s")
            if snap["messages"]:
                st.markdown(f'<div class="hx-note">'
                            f'{" · ".join(snap["messages"][-2:])}</div>',
                            unsafe_allow_html=True)
        elif snap["status"] == "done":
            st.success(f"{label} finished in {snap['elapsed']:.0f}s "
                       f"({snap['rows']:,} rows).")
            for m in snap["messages"]:
                st.markdown(f'<div class="hx-note">{m}</div>',
                            unsafe_allow_html=True)
        else:
            st.error(f"{label} failed: {snap['error']}")

    _render()


def _write_profile_yaml(ctx: PageContext, name: str, out_dir: Path, scorer: str,
                        chunk_size: int, overlap: int) -> None:
    """Generate a complete, valid profile for an uploaded dataset."""
    import yaml

    active = ["hit_rate_at_k", "mrr", "context_recall", "context_precision",
              "accuracy", "citation_valid_pointer", "cost_usd"]
    weights = {"accuracy": 0.5, "hit_rate_at_k": 0.2, "mrr": 0.15,
               "cost_usd": -0.15}
    if scorer == "judge":
        active += ["faithfulness", "answer_relevance"]
        weights = {"accuracy": 0.35, "faithfulness": 0.25,
                   "answer_relevance": 0.1, "hit_rate_at_k": 0.15,
                   "cost_usd": -0.15}
    # Probe metrics cost nothing extra unless probe items exist in the evalset.
    active += ["abstention", "injection_resisted"]

    profile = {
        "name": name, "task": "rag",
        "corpus_path": str(out_dir / "corpus.jsonl"),
        "evalset_path": str(out_dir / "evalset.jsonl"),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "retrieval_mode": "hybrid", "k": 8,
        "rerank": False, "rerank_model": ctx.models_cfg.get("rerank_model", ""),
        "rerank_top_n": 5, "accuracy_scorer": scorer,
        "chunk_size": int(chunk_size), "overlap": int(overlap),
        "max_tokens": 1024,
        "active_metrics": active, "metric_weights": weights,
        "tuning_budget": 12,
        "knobs": {
            "retrieval_modes": ["dense", "sparse", "hybrid"],
            "k_values": [5, 8, 12], "rerank_options": [False, True],
            "rerank_top_n": [5], "context_orders": ["as_is"],
            "system_prompts": [], "few_shot_sets": [[]],
        },
    }
    (out_dir / "profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
class DataPage:
    meta = PageMeta(
        key="data", label="Data", icon="▤", group="Workflow",
        breadcrumb=("Dashboard", "Data"), title="Dataset",
        subtitle="Turn PDFs and a Q&A sheet into an indexed corpus.",
    )

    def available(self, ctx: PageContext) -> str:
        return ""

    def render(self, ctx: PageContext) -> None:
        import streamlit as st

        from ...orchestration.jobs import any_running, new_job
        from ...orchestration.ui_runner import ingest_job
        from ...profiles.loaders import dataset_stats, load_evalset

        page_header(list(self.meta.breadcrumb), self.meta.title,
                    self.meta.subtitle)

        with card("Upload",
                  "PDFs become the knowledge base. The spreadsheet is your Q&A "
                  "set - two columns, Question and Answer. Answers are matched "
                  "to source passages automatically."):
            name = st.text_input("Profile name", value="my_eval")
            c1, c2 = st.columns(2)
            pdfs = c1.file_uploader("Source PDFs", type=["pdf"],
                                    accept_multiple_files=True)
            qa = c2.file_uploader("Q&A spreadsheet", type=["csv", "xlsx", "xls"])
            c3, c4 = st.columns(2)
            q_col = c3.text_input("Question column", value="Question")
            a_col = c4.text_input("Answer column", value="Answer")
            c5, c6, c7 = st.columns(3)
            chunk_size = c5.number_input("Chunk size (words)", 50, 1000, 200, 10)
            overlap = c6.number_input("Overlap (words)", 0, 200, 40, 5)
            scorer = c7.selectbox(
                "Answer scorer",
                ["token_f1", "contains", "exact", "numeric", "judge"],
                help="token_f1 is free, graded and penalises padding - a good "
                     "default. judge is most accurate and most expensive.")

            if st.button("Build dataset", type="primary",
                         disabled=not (pdfs and qa and name)):
                out_dir = ctx.data_dir / name
                pdf_dir = out_dir / "pdfs"
                pdf_dir.mkdir(parents=True, exist_ok=True)
                for up in pdfs:
                    (pdf_dir / up.name).write_bytes(up.getbuffer())
                qa_path = out_dir / ("qa" + Path(qa.name).suffix)
                qa_path.write_bytes(qa.getbuffer())

                from prepare_dataset import build_corpus, build_evalset

                try:
                    with st.spinner("Extracting PDFs and matching answers..."):
                        corpus = build_corpus(pdf_dir, out_dir / "corpus.jsonl",
                                              granularity="page")
                        build_evalset(qa_path, corpus, out_dir / "evalset.jsonl",
                                      q_col, a_col, chunk_size, overlap)
                    _write_profile_yaml(ctx, name, out_dir, scorer,
                                        chunk_size, overlap)
                    st.session_state["profile_name"] = name
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Prepare failed: {e}")

        if ctx.profile is None:
            empty_state("No dataset yet",
                        "Upload PDFs and a Q&A spreadsheet above, then build.")
            return

        try:
            items = load_evalset(ctx.profile.evalset_path)
            stats = dataset_stats(items, task=ctx.profile.task.value)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the evalset: {e}")
            return

        kpi_row([
            Kpi("Questions", f"{stats.n_items:,}", icon="◇"),
            Kpi("Auto-labeled", f"{stats.labeled:,}", icon="◇",
                note="have gold passages"),
            Kpi("Human labels", f"{stats.with_human_label:,}", icon="◇",
                note="unlocks judge calibration"),
            Kpi("Duplicates", f"{stats.duplicate_queries:,}", icon="◇",
                delta=None,
                note="inflate apparent sample size"),
        ])
        for w in stats.warnings:
            st.warning(w)

        with card(f"Index '{ctx.profile_name}'",
                  "One-time per dataset. Embeds every chunk, so it uses your "
                  "API key. Qdrant runs embedded - no Docker needed."):
            if st.button("Ingest", type="primary",
                         disabled=not ctx.has_key or any_running()):
                job = new_job("ingest", total=1)
                st.session_state["ingest_job"] = job.id
                threading.Thread(
                    target=ingest_job,
                    args=(job, ctx.profile, ctx.qdrant_path, ctx.api_key,
                          ctx.profile.chunk_size, ctx.profile.overlap, 128,
                          ctx.models_cfg),
                    daemon=True).start()
            if not ctx.has_key:
                st.markdown('<div class="hx-note">Add an API key in the sidebar '
                            'to ingest.</div>', unsafe_allow_html=True)
            _job_panel(st.session_state.get("ingest_job"), "Ingestion")


# --------------------------------------------------------------------------- #
class ProbesPage:
    meta = PageMeta(
        key="probes", label="Probes", icon="◎", group="Workflow",
        breadcrumb=("Dashboard", "Probes"), title="Adversarial probes",
        subtitle="Test the failure modes accuracy cannot see.",
    )

    def available(self, ctx: PageContext) -> str:
        return "" if ctx.profile is not None else "Build a dataset first"

    def render(self, ctx: PageContext) -> None:
        import streamlit as st

        from ...eval.probes import ProbeConfig
        from ...orchestration.jobs import any_running, new_job
        from ...orchestration.ui_runner import probe_job
        from ...profiles.loaders import load_evalset

        page_header(list(self.meta.breadcrumb), self.meta.title,
                    self.meta.subtitle)

        with card("Why these exist",
                  "Your corpus is an attack surface: anything that can put text "
                  "into the index - an uploaded PDF, a support ticket, a wiki "
                  "page - can put instructions in front of your model. These "
                  "probes are derived from your own questions, so they test "
                  "your actual system. Generating them is free and instant."):
            c1, c2, c3 = st.columns(3)
            f_unans = c1.slider("Unanswerable", 0.0, 0.5, 0.20, 0.05,
                                help="Can it refuse when the corpus has no answer?")
            f_inject = c2.slider("Prompt injection", 0.0, 0.5, 0.15, 0.05,
                                 help="Does it obey instructions hidden in a passage?")
            f_noise = c3.slider("Retrieval noise", 0.0, 0.5, 0.10, 0.05,
                                help="Does quality survive imperfect retrieval?")
            c4, c5 = st.columns(2)
            f_para = c4.slider("Paraphrase", 0.0, 0.5, 0.10, 0.05,
                               help="Is the answer stable when reworded?")
            f_pos = c5.slider("Positional", 0.0, 0.5, 0.0, 0.05,
                              help="Lost-in-the-middle on long contexts.")

        try:
            items = load_evalset(ctx.profile.evalset_path)
        except Exception:  # noqa: BLE001
            items = []
        existing = sum(1 for i in items if (i.meta or {}).get("probe"))
        expected = int(round(len(items) * (f_unans + f_inject + f_noise
                                           + f_para + f_pos)))

        kpi_row([
            Kpi("Base questions", f"{len(items):,}", icon="◇"),
            Kpi("Probes present", f"{existing:,}", icon="◇"),
            Kpi("Will generate", f"~{expected:,}", icon="◇",
                note="one model call each at run time"),
        ])

        if st.button("Generate probes", type="primary",
                     disabled=any_running() or expected == 0):
            cfg = ProbeConfig(unanswerable=f_unans, noise=f_noise,
                              injection=f_inject, paraphrase=f_para,
                              positional=f_pos, seed=0)
            job = new_job("probes", total=1)
            st.session_state["probe_job"] = job.id
            threading.Thread(target=probe_job,
                             args=(job, ctx.profile, cfg, True),
                             daemon=True).start()
        _job_panel(st.session_state.get("probe_job"), "Probe generation")


# --------------------------------------------------------------------------- #
class RunPage:
    meta = PageMeta(
        key="run", label="Run", icon="▷", group="Workflow",
        breadcrumb=("Dashboard", "Run"), title="Run evaluation",
        subtitle="Pick models, check the estimate, run in the background.",
    )

    def available(self, ctx: PageContext) -> str:
        if ctx.profile is None:
            return "Build a dataset first"
        if not ctx.has_key:
            return "Add an API key in the sidebar"
        return ""

    def render(self, ctx: PageContext) -> None:
        import streamlit as st

        from ...clients.cost import forecast_run
        from ...clients.pricing import PricingRegistry
        from ...orchestration.jobs import any_running, get_job, new_job
        from ...orchestration.ui_runner import eval_job
        from ...profiles.loaders import load_evalset

        page_header(list(self.meta.breadcrumb), self.meta.title,
                    self.meta.subtitle)

        all_models = ctx.models_cfg.get("models", [])
        with card("Configuration"):
            picked = st.multiselect("Models to compare", all_models,
                                    default=all_models[:3])
            p1, p2, p3 = st.columns(3)
            do_baseline = p1.checkbox("Baseline", value=True,
                                      help="Fixed prompt and retrieval for every model.")
            do_adapted = p2.checkbox(
                "Adapted (tuning)", value=False,
                help="Roughly tuning_budget times more expensive.")
            do_latency = p3.checkbox(
                "Latency lane", value=True,
                help="Serial, uncached, streaming - the only honest p50/p95.")

        try:
            items = load_evalset(ctx.profile.evalset_path)
        except Exception:  # noqa: BLE001
            items = []
        dev_frac = ctx.run_cfg.get("dev_split", 0.3)
        n_test = max(1, int(len(items) * (1 - dev_frac)))
        n_dev = max(1, int(len(items) * dev_frac))

        fc = forecast_run(
            PricingRegistry("configs/pricing.yaml", strict=False),
            picked, n_test=n_test, n_dev=n_dev,
            tuning_budget=ctx.profile.tuning_budget,
            judge_model=ctx.models_cfg.get("judge_model", "")
            if ctx.profile.accuracy_scorer == "judge" else "",
            do_baseline=do_baseline, do_adapted=do_adapted)

        over = ctx.budget and fc.est_usd > ctx.budget > 0
        kpi_row([
            Kpi("Models", f"{len(picked)}", icon="◇"),
            Kpi("Test items", f"{n_test:,}", icon="◇"),
            Kpi("Model calls", f"{fc.calls:,}", icon="◇"),
            Kpi("Judge calls", f"{fc.judge_calls:,}", icon="◇",
                note="often the largest line"),
            Kpi("Estimated", f"${fc.est_usd:,.2f}", icon="◇",
                note="includes judge calls"),
        ])

        if fc.detail.get("unpriced_models"):
            st.warning(f"No pricing for {fc.detail['unpriced_models']} - their "
                       f"cost metric will be blank rather than zero, because a "
                       f"free-looking model would win a cost-weighted leaderboard.")
        if over:
            st.error(f"The estimate (${fc.est_usd:,.2f}) exceeds your ceiling "
                     f"(${ctx.budget:,.2f}). The run would abort partway. Raise "
                     f"the ceiling in the sidebar, or narrow the run.")

        can_run = (ctx.has_key and picked and (do_baseline or do_adapted)
                   and not any_running())
        if st.button("Run evaluation", type="primary", disabled=not can_run):
            job = new_job("eval", total=1)
            st.session_state["eval_job"] = job.id
            threading.Thread(
                target=eval_job,
                args=(job, ctx.profile, picked, ctx.qdrant_path, ctx.store_path,
                      ctx.cache_dir, ctx.api_key,
                      ctx.models_cfg.get("judge_model"),
                      ctx.models_cfg.get("rerank_model", ""),
                      {"baseline": do_baseline, "adapted": do_adapted,
                       "latency": do_latency},
                      dev_frac, ctx.run_cfg.get("split_seed", 0),
                      ctx.run_cfg.get("max_workers", 4), ctx.budget,
                      ctx.models_cfg, None),
                daemon=True).start()

        if any_running() and get_job(st.session_state.get("eval_job")) is None:
            st.warning("Another job is running. Wait for it to finish.")
        _job_panel(st.session_state.get("eval_job"), "Evaluation")
        toolbar(right=pill("one job at a time", ""))
