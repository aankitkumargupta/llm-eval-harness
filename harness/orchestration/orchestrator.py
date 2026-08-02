"""
The matrix walk: for each (model, profile, pass) it runs every eval item and
persists the trace rows. Also drives the equal-budget tuning search that
produces the adapted pass.

Two concurrency regimes, as designed:
  * LATENCY LANE — low, fixed concurrency, models INTERLEAVED, cache BYPASSED,
    warm-up discarded. Produces clean p50/p95. Small item subset is enough.
  * THROUGHPUT LANE — high concurrency, cache ON. Produces accuracy/cost/
    retrieval/judge metrics, where queuing doesn't matter.

Keeping them separate is what stops us from benchmarking our own concurrency
instead of the model.
"""

from __future__ import annotations

import concurrent.futures as cf
import uuid
from typing import Callable, Optional

from qdrant_client import QdrantClient

from ..cache.cache import Cache, stable_hash
from ..clients.pricing import PricingRegistry
from ..clients.together_client import TogetherClient
from ..eval.judge import Judge
from ..profiles.loaders import load_evalset
from ..profiles.profile import Profile
from ..rag.prompt import PromptConfig
from ..rag.rerank import Reranker
from ..rag.retrieve import RetrievalConfig, Retriever
from ..store.schema import EvalItem, Pass, TraceRow
from ..store.store import TraceStore
from ..tuning.search import Candidate, enumerate_candidates
from .runner import RunContext, run_item


class Orchestrator:
    def __init__(self, client: TogetherClient, qdrant: QdrantClient,
                 pricing: PricingRegistry, store: TraceStore,
                 judge_model: Optional[str] = None,
                 cache_dir: str = ".cache",
                 rerank_model: str = ""):
        self.client = client
        self.qdrant = qdrant
        self.pricing = pricing
        self.store = store
        self.cache = Cache(cache_dir)
        self.judge = Judge(client, judge_model) if judge_model else None
        self.rerank_model = rerank_model

    # ------------------------------------------------------------------ #
    def _context(self, profile: Profile) -> RunContext:
        retriever = Retriever(self.client, self.qdrant, profile.collection_name())
        reranker = (Reranker(self.client, self.rerank_model)
                    if self.rerank_model else None)
        return RunContext(
            client=self.client, retriever=retriever, pricing=self.pricing,
            judge=self.judge, cache=self.cache,
            embedding_model=profile.embedding_model, reranker=reranker,
        )

    # ------------------------------------------------------------------ #
    #  Baseline pass                                                     #
    # ------------------------------------------------------------------ #
    def run_baseline(self, profile: Profile, models: list[str],
                     items: list[EvalItem], run_id: str,
                     max_workers: int = 8) -> list[TraceRow]:
        """Fixed retrieval + fixed prompt, identical for every model. Runs in the
        throughput lane (cache on, concurrent)."""
        ctx = self._context(profile)
        retrieval_cfg = RetrievalConfig(mode=profile.retrieval_mode, k=profile.k,
                                        embedding_model=profile.embedding_model)
        prompt_cfg = PromptConfig(max_context_chunks=profile.k)
        pcfg_hash = stable_hash(profile.name, profile.embedding_model,
                                profile.retrieval_mode.value, profile.k)

        rows: list[TraceRow] = []
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            # interleave (model-major inner loop) so throughput is balanced
            for item in items:
                for model in models:
                    futures.append(ex.submit(
                        run_item, item, model, profile.name, Pass.BASELINE,
                        retrieval_cfg, prompt_cfg, ctx, run_id,
                        profile.rerank, profile.rerank_top_n,
                        profile.active_metrics, profile.accuracy_scorer, pcfg_hash,
                    ))
            for f in cf.as_completed(futures):
                rows.append(f.result())
        self.store.write(rows)
        return rows

    # ------------------------------------------------------------------ #
    #  Adapted pass (with equal-budget tuning)                           #
    # ------------------------------------------------------------------ #
    def run_adapted(self, profile: Profile, models: list[str],
                    dev_items: list[EvalItem], test_items: list[EvalItem],
                    run_id: str, max_workers: int = 8) -> list[TraceRow]:
        """
        For each model: search candidates on dev (equal budget), pick the best,
        then run that winner once over test. The dev evaluations are logged as
        Pass.TUNING; the winning test run is Pass.ADAPTED.
        """
        ctx = self._context(profile)
        pcfg_hash = stable_hash(profile.name, "adapted", profile.embedding_model)
        all_rows: list[TraceRow] = []

        for model in models:
            candidates = enumerate_candidates(profile, profile.tuning_budget)

            def score_candidate(cand: Candidate) -> float:
                """Mean weighted-composite of this candidate over the dev split."""
                dev_rows = []
                for it in dev_items:
                    dev_rows.append(run_item(
                        it, model, profile.name, Pass.TUNING,
                        cand.retrieval, cand.prompt, ctx, run_id,
                        cand.rerank, cand.rerank_top_n,
                        profile.active_metrics, profile.accuracy_scorer, pcfg_hash,
                    ))
                all_rows.extend(dev_rows)
                return _weighted_mean(dev_rows, profile.metric_weights)

            best, best_score = None, float("-inf")
            for cand in candidates:
                s = score_candidate(cand)
                if s > best_score:
                    best, best_score = cand, s

            # run the winner over the test split as the adapted pass
            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(
                    run_item, it, model, profile.name, Pass.ADAPTED,
                    best.retrieval, best.prompt, ctx, run_id,
                    best.rerank, best.rerank_top_n,
                    profile.active_metrics, profile.accuracy_scorer, pcfg_hash,
                ) for it in test_items]
                for f in cf.as_completed(futures):
                    all_rows.append(f.result())

        self.store.write(all_rows)
        return all_rows

    # ------------------------------------------------------------------ #
    #  Latency lane                                                      #
    # ------------------------------------------------------------------ #
    def run_latency(self, profile: Profile, models: list[str],
                    items: list[EvalItem], run_id: str,
                    warmup: int = 1) -> list[TraceRow]:
        """
        Clean latency: low concurrency (serial), models INTERLEAVED, cache
        BYPASSED (we need real network timings), first `warmup` calls per model
        discarded. Use a small item subset — you don't need the whole evalset for
        stable p50/p95.
        """
        # temporarily disable cache by pointing at a throwaway dir
        latency_ctx = self._context(profile)
        latency_ctx.cache = Cache(".cache_latency_scratch")
        retrieval_cfg = RetrievalConfig(mode=profile.retrieval_mode, k=profile.k,
                                        embedding_model=profile.embedding_model)
        prompt_cfg = PromptConfig(max_context_chunks=profile.k)
        pcfg_hash = stable_hash(profile.name, "latency")

        warm_done = {m: 0 for m in models}
        rows: list[TraceRow] = []
        # interleaved order: item-major, model-minor, run serially
        for item in items:
            for model in models:
                r = run_item(item, model, profile.name, Pass.BASELINE,
                             retrieval_cfg, prompt_cfg, latency_ctx, run_id,
                             active_metrics=[], accuracy_scorer="none",
                             profile_cfg_hash=pcfg_hash)
                if warm_done[model] < warmup:
                    warm_done[model] += 1
                    continue  # discard warm-up timing
                rows.append(r)
        self.store.write(rows)
        return rows


def _weighted_mean(rows: list[TraceRow], weights: dict) -> float:
    """Weighted composite over a set of rows (used to score tuning candidates)."""
    import numpy as np
    total, used = 0.0, False
    for metric, w in weights.items():
        vals = [getattr(r, metric) for r in rows if getattr(r, metric, None) is not None]
        if vals:
            total += w * float(np.mean(vals))
            used = True
    return total if used else float("-inf")
