"""
UI-facing run functions. These are what the background threads execute. They
reuse the existing engine (Ingestor, run_item, TraceStore) and translate its
progress into Job.advance() calls. They take no Streamlit dependency, so they're
importable and testable on their own.

Two entry points:
  ingest_job(...)  -> embeds + indexes a profile's corpus, ticking per batch
  eval_job(...)    -> runs baseline (and optionally adapted) over the evalset,
                      ticking per (item, model)
"""

from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient

from ..cache.cache import Cache, stable_hash
from ..clients.pricing import PricingRegistry
from ..clients.together_client import TogetherClient
from ..eval.judge import Judge
from ..profiles.loaders import load_corpus, load_evalset
from ..profiles.profile import Profile
from ..rag.ingest import Ingestor
from ..rag.prompt import PromptConfig
from ..rag.rerank import Reranker
from ..rag.retrieve import RetrievalConfig, Retriever
from ..store.schema import Pass
from ..store.store import TraceStore
from ..tuning.search import enumerate_candidates
from .jobs import Job
from .runner import RunContext, run_item


def _count_chunks(corpus_path: str, chunk_size: int, overlap: int) -> int:
    """Pre-count chunks so the ingest progress bar has a real denominator."""
    from ..rag.ingest import chunk_text
    n = 0
    for doc in load_corpus(corpus_path):
        n += len(chunk_text(doc.text, chunk_size, overlap))
    return max(1, n)


def ingest_job(job: Job, profile: Profile, qdrant_path: str,
               api_key: Optional[str], chunk_size: int, overlap: int,
               batch_size: int = 128) -> None:
    """Background ingestion. Embedded Qdrant (local folder) so no Docker needed."""
    try:
        total = _count_chunks(profile.corpus_path, chunk_size, overlap)
        job.set_total(total)
        client = TogetherClient(api_key=api_key)
        qdrant = QdrantClient(path=qdrant_path)
        ingestor = Ingestor(client, qdrant, profile.embedding_model)

        n = ingestor.ingest(
            collection=profile.collection_name(),
            documents=load_corpus(profile.corpus_path),
            chunk_size=chunk_size, overlap=overlap, batch_size=batch_size,
            progress_cb=lambda done: job.set_done_absolute(done),
        )
        try:
            qdrant.close()
        except Exception:
            pass
        job.finish(rows=n, message=f"Indexed {n} chunks into "
                                   f"'{profile.collection_name()}'.")
    except Exception as e:  # noqa: BLE001
        job.fail(f"{type(e).__name__}: {e}")


def eval_job(job: Job, profile: Profile, models: list[str],
             qdrant_path: str, store_path: str, cache_dir: str,
             api_key: Optional[str], judge_model: Optional[str],
             rerank_model: str, run_passes: dict, dev_split: float,
             split_seed: int, max_workers: int = 4) -> None:
    """
    Background evaluation. Runs baseline and (optionally) adapted over the
    profile's evalset, writing rows to the store and ticking the job per item.
    Kept sequential-per-item (not the ThreadPool path) so progress is smooth and
    latency-friendly; concurrency here is modest by design.
    """
    try:
        client = TogetherClient(api_key=api_key)
        qdrant = QdrantClient(path=qdrant_path)
        pricing = PricingRegistry()
        store = TraceStore(store_path)
        judge = Judge(client, judge_model) if judge_model else None
        retriever = Retriever(client, qdrant, profile.collection_name())
        reranker = Reranker(client, rerank_model) if rerank_model else None
        ctx = RunContext(client=client, retriever=retriever, pricing=pricing,
                         judge=judge, cache=Cache(cache_dir),
                         embedding_model=profile.embedding_model,
                         reranker=reranker)

        items = load_evalset(profile.evalset_path)
        # deterministic dev/test split
        import random
        rng = random.Random(split_seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        cut = int(len(items) * dev_split)
        dev = [items[i] for i in idx[:cut]]
        test = [items[i] for i in idx[cut:]] or items  # never empty

        do_baseline = run_passes.get("baseline", True)
        do_adapted = run_passes.get("adapted", False)

        # progress denominator: baseline is |test|*|models|; adapted adds the
        # tuning dev-evals plus another |test|*|models|.
        total = 0
        if do_baseline:
            total += len(test) * len(models)
        if do_adapted:
            cand_n = len(enumerate_candidates(profile, profile.tuning_budget))
            total += (cand_n * len(dev) + len(test)) * len(models)
        job.set_total(total)

        run_id = job.id
        rows = []

        # -- baseline pass -------------------------------------------------
        if do_baseline:
            rcfg = RetrievalConfig(mode=profile.retrieval_mode, k=profile.k,
                                   embedding_model=profile.embedding_model)
            pcfg = PromptConfig(max_context_chunks=profile.k)
            pcfg_hash = stable_hash(profile.name, "baseline")
            for it in test:
                for m in models:
                    r = run_item(it, m, profile.name, Pass.BASELINE, rcfg, pcfg,
                                 ctx, run_id, profile.rerank, profile.rerank_top_n,
                                 profile.active_metrics, profile.accuracy_scorer,
                                 pcfg_hash)
                    rows.append(r)
                    job.advance(1)
            store.write([r for r in rows])
            job.messages.append(f"Baseline: {len(test)*len(models)} rows.")

        # -- adapted pass (equal-budget tuning per model) ------------------
        if do_adapted:
            pcfg_hash = stable_hash(profile.name, "adapted")
            for m in models:
                candidates = enumerate_candidates(profile, profile.tuning_budget)
                best, best_score = None, float("-inf")
                for cand in candidates:
                    dev_rows = []
                    for it in dev:
                        r = run_item(it, m, profile.name, Pass.TUNING,
                                     cand.retrieval, cand.prompt, ctx, run_id,
                                     cand.rerank, cand.rerank_top_n,
                                     profile.active_metrics,
                                     profile.accuracy_scorer, pcfg_hash)
                        dev_rows.append(r)
                        job.advance(1)
                    s = _weighted_mean(dev_rows, profile.metric_weights)
                    store.write(dev_rows)
                    if s > best_score:
                        best, best_score = cand, s
                # run winner over test
                adapted_rows = []
                for it in test:
                    r = run_item(it, m, profile.name, Pass.ADAPTED,
                                 best.retrieval, best.prompt, ctx, run_id,
                                 best.rerank, best.rerank_top_n,
                                 profile.active_metrics, profile.accuracy_scorer,
                                 pcfg_hash)
                    adapted_rows.append(r)
                    job.advance(1)
                store.write(adapted_rows)

        try:
            qdrant.close()
        except Exception:
            pass
        # count non-error rows written this run
        written = len(store.query(
            f"SELECT * FROM traces WHERE run_id = '{run_id}'"))
        job.finish(rows=written, message="Evaluation complete.")
    except Exception as e:  # noqa: BLE001
        job.fail(f"{type(e).__name__}: {e}")


def _weighted_mean(rows, weights: dict) -> float:
    import numpy as np
    total, used = 0.0, False
    for metric, w in weights.items():
        vals = [getattr(r, metric) for r in rows
                if getattr(r, metric, None) is not None]
        if vals:
            total += w * float(np.mean(vals))
            used = True
    return total if used else float("-inf")
