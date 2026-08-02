"""
Multi-Model LLM Evaluation Harness — single-user app (Step 1).

An end-to-end UI: upload PDFs + a Q&A spreadsheet, prepare + ingest them, pick
models and metrics, run the evaluation in the BACKGROUND with a live progress
bar, and view the leaderboard/Pareto/tuning/CIs — all without a terminal.

Run it:
    streamlit run app.py

Design choices for a self-contained single-user app:
  * Qdrant runs EMBEDDED (a local folder), so no Docker is required.
  * The evaluation runs on a background thread; the worker only mutates a plain
    Job object (never calls Streamlit), and a fragment polls that Job every 2s.
  * Only one heavy job runs at a time (enforced), since embedded Qdrant and the
    Together bill both prefer serial runs here.

This spends your Together credits on every run — the Configure tab shows a rough
cost estimate before you start.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from harness.clients.pricing import PricingRegistry
from harness.orchestration.jobs import any_running, get_job, new_job
from harness.orchestration.ui_runner import eval_job, ingest_job
from harness.profiles.profile import Profile
from harness.report.aggregate import (
    bootstrap_ci,
    pareto_frontier,
    tuning_gain,
    weighted_composite,
)
from harness.store.store import TraceStore

st.set_page_config(page_title="LLM Eval Harness", layout="wide")

# ----- workspace: everything the app writes lives under ./workspace --------- #
WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)
DATA_DIR = WORKSPACE / "data"
QDRANT_PATH = str(WORKSPACE / "qdrant")
STORE_PATH = str(WORKSPACE / "traces.parquet")
CACHE_DIR = str(WORKSPACE / "cache")


def _models_cfg() -> dict:
    return yaml.safe_load(Path("configs/models.yaml").read_text())


def _run_cfg() -> dict:
    return yaml.safe_load(Path("configs/run.yaml").read_text())


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _write_profile_yaml(
    name: str, out_dir: Path, scorer: str, chunk_size: int, overlap: int
) -> None:
    """Generate a minimal but complete profile config for an uploaded dataset."""
    mcfg = _models_cfg()
    # sensible default metric set (retrieval metrics only kick in for labeled items)
    active = [
        "hit_rate_at_k",
        "mrr",
        "context_recall",
        "accuracy",
        "citation_valid_pointer",
        "cost_usd",
    ]
    weights = {"accuracy": 0.5, "hit_rate_at_k": 0.2, "mrr": 0.15, "cost_usd": -0.15}
    if scorer == "judge":
        active += ["faithfulness", "answer_relevance"]
        weights = {
            "accuracy": 0.35,
            "faithfulness": 0.25,
            "answer_relevance": 0.1,
            "hit_rate_at_k": 0.15,
            "cost_usd": -0.15,
        }
    profile = {
        "name": name,
        "corpus_path": str(out_dir / "corpus.jsonl"),
        "evalset_path": str(out_dir / "evalset.jsonl"),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "retrieval_mode": "hybrid",
        "k": 8,
        "rerank": False,
        "rerank_model": mcfg.get("rerank_model", ""),
        "rerank_top_n": 5,
        "accuracy_scorer": scorer,
        "active_metrics": active,
        "metric_weights": weights,
        "tuning_budget": 12,
        "knobs": {
            "retrieval_modes": ["dense", "sparse", "hybrid"],
            "k_values": [5, 8, 12],
            "rerank_options": [False, True],
            "rerank_top_n": [5],
            "system_prompts": [],
            "few_shot_sets": [[]],
        },
    }
    (out_dir / "profile.yaml").write_text(yaml.safe_dump(profile, sort_keys=False))


def _estimate_cost(models, n_items, do_baseline, do_adapted, profile, run_cfg) -> dict:
    """Very rough spend estimate so a click isn't a blind bill."""
    dev_frac = run_cfg.get("dev_split", 0.3)
    n_test = max(1, int(n_items * (1 - dev_frac)))
    n_dev = max(1, int(n_items * dev_frac))
    calls = 0
    if do_baseline:
        calls += n_test * len(models)
    if do_adapted:
        # tuning_budget candidates * dev items + winner over test, per model
        calls += (profile.tuning_budget * n_dev + n_test) * len(models)
    # assume ~1000 prompt + ~120 completion tokens/call, price ~ mid model
    try:
        pricing = PricingRegistry()
        sample = models[0]
        per_call = pricing.generation_cost(sample, 1000, 120)
    except Exception:
        per_call = 0.0015
    return {"calls": calls, "usd": calls * per_call}


@st.fragment(run_every=2)
def _render_job(job_id, label: str) -> None:
    """Live-refreshing job panel. Runs as a fragment so only THIS block reruns
    every 2s — the worker thread updates the Job, we just read its snapshot."""
    job = get_job(job_id)
    if job is None:
        return
    snap = job.snapshot()
    if snap["status"] == "running":
        st.progress(
            snap["frac"],
            text=f"{label}: {snap['done']}/{snap['total']} "
            f"({snap['frac'] * 100:.0f}%) · {snap['elapsed']:.0f}s",
        )
        if snap["messages"]:
            st.caption(" · ".join(snap["messages"][-2:]))
    elif snap["status"] == "done":
        st.success(f"{label} done in {snap['elapsed']:.0f}s ({snap['rows']} rows).")
        for m in snap["messages"]:
            st.caption(m)
    else:  # error
        st.error(f"{label} failed: {snap['error']}")


# --------------------------------------------------------------------------- #
#  Sidebar: API key + status                                                   #
# --------------------------------------------------------------------------- #
import os

with st.sidebar:
    st.header("Setup")
    default_key = os.environ.get("TOGETHER_API_KEY", "")
    api_key = st.text_input(
        "Together API key",
        value=default_key,
        type="password",
        help="Read from TOGETHER_API_KEY if set; override here if needed.",
    )
    if not api_key:
        st.warning("Enter a Together API key to ingest or run.")
    st.divider()
    st.caption("Workspace")
    st.code(str(WORKSPACE.resolve()), language=None)
    if Path(STORE_PATH).exists():
        st.caption("Trace store: present")
    else:
        st.caption("Trace store: empty")

st.title("Multi-Model LLM Evaluation Harness")

tab_upload, tab_run, tab_results = st.tabs(
    ["1 · Upload & prepare", "2 · Configure & run", "3 · Results"]
)


# =========================================================================== #
#  TAB 1 — Upload & prepare                                                    #
# =========================================================================== #
with tab_upload:
    st.subheader("Upload your data")
    st.caption(
        "PDFs become the knowledge base. The spreadsheet is your Q&A set. "
        "Answers are matched to source passages automatically."
    )

    profile_name = st.text_input(
        "Profile name", value="my_eval", help="A short id for this evaluation set."
    )
    col1, col2 = st.columns(2)
    with col1:
        pdfs = st.file_uploader("Source PDFs", type=["pdf"], accept_multiple_files=True)
    with col2:
        qa = st.file_uploader(
            "Q&A spreadsheet (CSV or XLSX)", type=["csv", "xlsx", "xls"]
        )

    colq, cola = st.columns(2)
    q_col = colq.text_input("Question column", value="Question")
    a_col = cola.text_input("Answer column", value="Answer")

    colc1, colc2, colc3 = st.columns(3)
    chunk_size = colc1.number_input("Chunk size (words)", 50, 1000, 200, 10)
    overlap = colc2.number_input("Overlap (words)", 0, 200, 40, 5)
    scorer = colc3.selectbox(
        "Answer scorer",
        ["exact", "numeric", "contains", "judge"],
        help="exact/numeric for short answers; judge for free-form.",
    )

    if st.button(
        "Build dataset", type="primary", disabled=not (pdfs and qa and profile_name)
    ):
        # save uploads to the workspace
        out_dir = DATA_DIR / profile_name
        pdf_dir = out_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        for up in pdfs:
            (pdf_dir / up.name).write_bytes(up.getbuffer())
        qa_path = out_dir / ("qa" + Path(qa.name).suffix)
        qa_path.write_bytes(qa.getbuffer())

        # reuse the prepare_dataset logic
        from prepare_dataset import build_corpus, build_evalset

        try:
            with st.spinner("Extracting PDFs and matching answers to passages..."):
                corpus = build_corpus(
                    pdf_dir, out_dir / "corpus.jsonl", granularity="page"
                )
                build_evalset(
                    qa_path,
                    corpus,
                    out_dir / "evalset.jsonl",
                    q_col,
                    a_col,
                    chunk_size,
                    overlap,
                )

            # write a profile yaml for this dataset
            _write_profile_yaml(profile_name, out_dir, scorer, chunk_size, overlap)
            st.session_state["prepared_profile"] = profile_name
            st.session_state["chunk_size"] = int(chunk_size)
            st.session_state["overlap"] = int(overlap)

            # show the labeled/unlabeled summary
            ev = [pd.read_json(str(out_dir / "evalset.jsonl"), lines=True)][0]
            labeled = ev["gold_passage_ids"].apply(lambda x: len(x) > 0).sum()
            st.success(
                f"Prepared '{profile_name}': {len(corpus)} documents, "
                f"{len(ev)} questions "
                f"({labeled} auto-labeled with source passages, "
                f"{len(ev) - labeled} without)."
            )
            with st.expander("Preview evalset"):
                st.dataframe(ev.head(20), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prepare failed: {e}")

    # ---- ingest -----------------------------------------------------------
    prepared = st.session_state.get("prepared_profile")
    if prepared:
        st.divider()
        st.subheader(f"Ingest '{prepared}' into the vector store")
        st.caption("One-time per dataset. Embeds every chunk (uses your API key).")
        if st.button("Ingest", disabled=not api_key or any_running()):
            profile = Profile.from_yaml(str(DATA_DIR / prepared / "profile.yaml"))
            job = new_job("ingest", total=1)
            st.session_state["ingest_job"] = job.id
            t = threading.Thread(
                target=ingest_job,
                args=(
                    job,
                    profile,
                    QDRANT_PATH,
                    api_key,
                    st.session_state.get("chunk_size", 200),
                    st.session_state.get("overlap", 40),
                ),
                daemon=True,
            )
            t.start()

        _render_job(st.session_state.get("ingest_job"), "Ingestion")


# =========================================================================== #
#  TAB 2 — Configure & run                                                     #
# =========================================================================== #
with tab_run:
    prepared = st.session_state.get("prepared_profile")
    if not prepared:
        st.info("Prepare a dataset in tab 1 first.")
    else:
        st.subheader(f"Run evaluation on '{prepared}'")
        mcfg = _models_cfg()
        all_models = mcfg["models"]
        picked = st.multiselect("Models to compare", all_models, default=all_models[:3])

        colp1, colp2 = st.columns(2)
        do_baseline = colp1.checkbox("Baseline pass (fixed prompt)", value=True)
        do_adapted = colp2.checkbox(
            "Adapted pass (equal-budget tuning)",
            value=False,
            help="Much more expensive: searches configs per model on a dev split.",
        )

        profile = Profile.from_yaml(str(DATA_DIR / prepared / "profile.yaml"))
        n_items = sum(1 for _ in open(profile.evalset_path))
        run_cfg = _run_cfg()

        # rough cost estimate
        est = _estimate_cost(picked, n_items, do_baseline, do_adapted, profile, run_cfg)
        st.info(
            f"Rough estimate: **~{est['calls']:,} model calls**, "
            f"**~${est['usd']:.2f}** in Together spend "
            f"(very approximate — depends on answer/context length)."
        )

        can_run = (
            api_key and picked and (do_baseline or do_adapted) and not any_running()
        )
        if st.button("Run evaluation", type="primary", disabled=not can_run):
            job = new_job("eval", total=1)
            st.session_state["eval_job"] = job.id
            t = threading.Thread(
                target=eval_job,
                args=(
                    job,
                    profile,
                    picked,
                    QDRANT_PATH,
                    STORE_PATH,
                    CACHE_DIR,
                    api_key,
                    mcfg.get("judge_model"),
                    mcfg.get("rerank_model", ""),
                    {"baseline": do_baseline, "adapted": do_adapted},
                    run_cfg.get("dev_split", 0.3),
                    run_cfg.get("split_seed", 0),
                ),
                daemon=True,
            )
            t.start()

        if any_running() and get_job(st.session_state.get("eval_job")) is None:
            st.warning("Another job is running. Wait for it to finish.")

        _render_job(st.session_state.get("eval_job"), "Evaluation")


# =========================================================================== #
#  TAB 3 — Results                                                             #
# =========================================================================== #
with tab_results:
    if not Path(STORE_PATH).exists():
        st.info("No results yet. Run an evaluation in tab 2.")
    else:
        store = TraceStore(STORE_PATH)
        df = store.load_all()
        if df.empty:
            st.info("Trace store is empty.")
        else:
            profiles = sorted(df["profile"].dropna().unique())
            sel = st.selectbox("Profile", profiles)
            pdf = df[df["profile"] == sel]
            prof_cfg_path = DATA_DIR / sel / "profile.yaml"
            prof_cfg = (
                Profile.from_yaml(str(prof_cfg_path))
                if prof_cfg_path.exists()
                else None
            )
            test_df = pdf[pdf["pass_"].isin(["baseline", "adapted"])]

            st.subheader("Leaderboard (weighted composite)")
            if prof_cfg and prof_cfg.metric_weights:
                st.dataframe(
                    weighted_composite(test_df, prof_cfg.metric_weights),
                    use_container_width=True,
                    hide_index=True,
                )

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Pareto (accuracy vs cost)")
                try:
                    pf = pareto_frontier(pdf)
                    if not pf.empty and not pf["accuracy"].isna().all():
                        st.scatter_chart(pf, x="cost", y="accuracy", color="model")
                        st.dataframe(pf, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Not enough accuracy/cost data yet.")
                except Exception as e:  # noqa: BLE001
                    st.caption(f"Pareto unavailable: {e}")
            with c2:
                st.subheader("Tuning gain")
                if {"baseline", "adapted"} <= set(pdf["pass_"].unique()):
                    metrics = [
                        m
                        for m in (prof_cfg.active_metrics if prof_cfg else [])
                        if m in pdf.columns
                    ]
                    st.dataframe(
                        tuning_gain(pdf, metrics or ["accuracy"]),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("Run both baseline and adapted to see tuning gain.")

            # ---- retrieval-quality metrics (ranking-aware) ------------------
            st.subheader("Retrieval quality")
            st.caption(
                "How well retrieval found the gold passages. These are "
                "identical across models when retrieval is shared, and "
                "differ once the adapted pass tunes retrieval per model."
            )
            retr_cols = [
                c
                for c in [
                    "hit_rate_at_k",
                    "mrr",
                    "ndcg_at_k",
                    "context_recall",
                    "rerank_hit_delta",
                ]
                if c in test_df.columns
            ]
            if retr_cols:
                retr = (
                    test_df.groupby("model")[retr_cols]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                # groupby().mean() drops all-NaN columns, so only keep survivors
                present = [
                    c for c in retr_cols if c in retr.columns and retr[c].notna().any()
                ]
                if present:
                    st.dataframe(
                        retr[["model"] + present],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption(
                        "No retrieval metrics recorded (items may lack gold passages)."
                    )
            else:
                st.caption("No retrieval metrics available for this profile.")

            # ---- answer & citation quality (per-model means) ----------------
            st.subheader("Answer & citation quality")
            qual_cols = [
                c
                for c in [
                    "accuracy",
                    "faithfulness",
                    "answer_relevance",
                    "completeness",
                    "citation_valid_pointer",
                    "citation_supporting",
                    "abstention_correct",
                ]
                if c in test_df.columns
            ]
            if qual_cols:
                qual = (
                    test_df.groupby("model")[qual_cols]
                    .mean(numeric_only=True)
                    .reset_index()
                )
                # groupby().mean() silently DROPS all-NaN columns, so only keep
                # the ones that actually survived (i.e. had data).
                present = [
                    c for c in qual_cols if c in qual.columns and qual[c].notna().any()
                ]
                dormant = [c for c in qual_cols if c not in present]
                st.dataframe(
                    qual[["model"] + present], use_container_width=True, hide_index=True
                )
                if dormant:
                    st.caption(
                        "Not exercised by this dataset (blank): " + ", ".join(dormant)
                    )

            # ---- efficiency (latency + tokens + cost) -----------------------
            st.subheader("Efficiency")
            eff_cols = [
                c
                for c in [
                    "latency_ms",
                    "ttft_ms",
                    "prompt_tokens",
                    "completion_tokens",
                    "cost_usd",
                ]
                if c in pdf.columns
            ]
            if eff_cols:
                eff = (
                    pdf.groupby("model")[eff_cols].mean(numeric_only=True).reset_index()
                )
                present = [
                    c for c in eff_cols if c in eff.columns and eff[c].notna().any()
                ]
                st.dataframe(
                    eff[["model"] + present], use_container_width=True, hide_index=True
                )

            # ---- confidence intervals for ANY metric ------------------------
            st.subheader("95% confidence intervals (baseline)")
            st.caption(
                "Wide intervals mean the sample is too small to rank "
                "confidently on this metric - add more questions."
            )
            base = pdf[pdf["pass_"] == "baseline"]
            ci_candidates = [
                c
                for c in [
                    "accuracy",
                    "faithfulness",
                    "answer_relevance",
                    "completeness",
                    "hit_rate_at_k",
                    "mrr",
                    "ndcg_at_k",
                    "context_recall",
                    "citation_supporting",
                ]
                if c in base.columns and base[c].notna().any()
            ]
            if ci_candidates:
                metric = st.selectbox("Metric", ci_candidates, key="ci_metric")
                rows = []
                for m, g in base.groupby("model"):
                    vals = g[metric].to_numpy()
                    mean, lo, hi = bootstrap_ci(vals)
                    if not (mean != mean):  # skip NaN means
                        rows.append(
                            {
                                "model": m,
                                "mean": round(mean, 4),
                                "lo": round(lo, 4),
                                "hi": round(hi, 4),
                            }
                        )
                if rows:
                    ci_df = pd.DataFrame(rows).sort_values("mean", ascending=False)
                    st.dataframe(ci_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No metrics with data on the baseline pass yet.")

            with st.expander("Raw traces (every metric, one row per item)"):
                st.dataframe(pdf, use_container_width=True, hide_index=True)
