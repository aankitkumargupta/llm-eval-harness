"""
The orchestrator: wiring, not work.

This class used to *be* the matrix walk — three passes, the tuning search, the
arena, row buffering, progress reporting and resume bookkeeping, all sharing
five pieces of mutable state. That is six reasons to change one class, and the
shared state is what made it hard to reason about under concurrency.

It is now a facade. It owns exactly one responsibility: assemble the
collaborators a run needs (client, retriever, reranker, judge, cache, meter)
and hand them to whichever pass was asked for. The work lives in:

    passes.py     BaselinePass / AdaptedPass / LatencyPass
    collector.py  RowCollector — buffering, checkpointing, progress
    arena.py      ArenaService — pairwise judging over stored answers
    runner.py     run_item — one item, end to end
    scoring.py    the metric registry

The public API is unchanged (`run_baseline`, `run_adapted`, `run_latency`,
`run_arena`), so the CLI, the app and the UI runner keep working untouched. A
refactor that forces its callers to change is a rewrite wearing a disguise.

Adding a fourth kind of pass — a retrieval-free ablation, a multi-turn walk —
is now a new class in `passes.py` plus one method here, with nothing existing
modified.
"""

from __future__ import annotations

from collections.abc import Callable

from ..cache.cache import DiskCache, KeyValueCache
from ..clients.cost import CostMeter
from ..eval.judge import Judge
from ..profiles.profile import Profile
from ..rag.rerank import Reranker
from ..rag.retrieve import Retriever
from ..store.schema import EvalItem, TaskType
from .arena import ArenaService
from .collector import RowCollector
from .passes import (
    AdaptedPass,
    BaselinePass,
    LatencyPass,
    RunReport,
    weighted_mean,
)
from .runner import RunContext

# Re-exported: callers and tests import these from here, and moving a symbol is
# not a reason to break them.
__all__ = ["Orchestrator", "RunReport", "weighted_mean"]


class Orchestrator:
    def __init__(self, client, qdrant, pricing, store,
                 judge_model: str | None = None,
                 cache_dir: str = ".cache",
                 rerank_model: str = "",
                 meter: CostMeter | None = None,
                 judge_ensemble: list[str] | None = None,
                 max_workers: int = 8,
                 checkpoint_every: int = 50,
                 progress_cb: Callable[[int, int, str], None] | None = None):
        self.client = client
        self.qdrant = qdrant
        self.pricing = pricing
        self.store = store
        self.cache: KeyValueCache = DiskCache(cache_dir)
        self.meter = meter or CostMeter()
        self.rerank_model = rerank_model
        self.max_workers = max_workers
        self.judge = (
            Judge(client, judge_model, cache=self.cache, ensemble=judge_ensemble)
            if judge_model else None
        )
        self.collector = RowCollector(store, checkpoint_every, progress_cb)

    # ------------------------------------------------------------------ #
    #  Collaborator assembly — the one job this class kept
    # ------------------------------------------------------------------ #
    def _context(self, profile: Profile, cache: KeyValueCache | None = None,
                 stream: bool = False) -> RunContext:
        cache = cache if cache is not None else self.cache
        retriever = None
        if profile.task == TaskType.RAG and self.qdrant is not None:
            retriever = Retriever(self.client, self.qdrant,
                                  profile.collection_name(),
                                  cache=cache, meter=self.meter)
        reranker = (Reranker(self.client, self.rerank_model)
                    if self.rerank_model else None)
        return RunContext(
            client=self.client, retriever=retriever, pricing=self.pricing,
            judge=self.judge, cache=cache,
            embedding_model=profile.embedding_model, reranker=reranker,
            temperature=profile.temperature, max_tokens=profile.max_tokens,
            seed=profile.seed, meter=self.meter, task=profile.task,
            stream=stream, abstention_judge=profile.abstention_judge,
            label_set=list(profile.label_set),
            provider=getattr(getattr(self.client, "info", None), "name", ""),
        )

    def _pass_kwargs(self) -> dict:
        return {"collector": self.collector, "make_context": self._context,
                "meter": self.meter, "max_workers": self.max_workers}

    def _existing_keys(self, resume_from: str | None) -> set:
        """(item_id, model, pass) already recorded, for resume."""
        if not resume_from:
            return set()
        df = self.store.load_run(resume_from)
        if df.empty:
            return set()
        df = df[df["error"].isna()] if "error" in df.columns else df
        return set(zip(df["item_id"], df["model"], df["pass_"]))

    # ------------------------------------------------------------------ #
    #  Passes
    # ------------------------------------------------------------------ #
    def run_baseline(self, profile: Profile, models: list[str],
                     items: list[EvalItem], run_id: str,
                     max_workers: int | None = None,
                     resume_from: str | None = None) -> RunReport:
        return BaselinePass(**self._pass_kwargs()).run(
            profile, models, run_id, items=items, max_workers=max_workers,
            done_keys=self._existing_keys(resume_from))

    def run_adapted(self, profile: Profile, models: list[str],
                    dev_items: list[EvalItem], test_items: list[EvalItem],
                    run_id: str, max_workers: int | None = None) -> RunReport:
        return AdaptedPass(**self._pass_kwargs()).run(
            profile, models, run_id, dev_items=dev_items,
            test_items=test_items, max_workers=max_workers)

    def run_latency(self, profile: Profile, models: list[str],
                    items: list[EvalItem], run_id: str,
                    warmup: int = 1, repeats: int = 1) -> RunReport:
        return LatencyPass(**self._pass_kwargs()).run(
            profile, models, run_id, items=items, warmup=warmup,
            repeats=repeats)

    def run_arena(self, profile: Profile, models: list[str], run_id: str,
                  source_run: str | None = None,
                  max_pairs_per_item: int = 3) -> list[tuple[str, str, str]]:
        return ArenaService(self.store, self.judge).run(
            profile.name, models, source_run=source_run,
            max_pairs_per_item=max_pairs_per_item)

    # ------------------------------------------------------------------ #
    #  Back-compatible accessors
    # ------------------------------------------------------------------ #
    # The old code exposed these as attributes on the orchestrator. Tests and
    # the UI runner read them, so they stay — now delegating to the collector
    # that actually owns the state.
    @property
    def _written(self) -> int:
        return self.collector.written

    @property
    def _errors(self) -> int:
        return self.collector.errors


# Kept for backwards compatibility with any external caller of the old name.
_weighted_mean = weighted_mean
