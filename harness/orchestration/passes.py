"""
Evaluation passes, as interchangeable strategies.

A "pass" is one way of walking the model x item matrix. The harness has three,
and they differ in ways that matter:

  BaselinePass  concurrent, cached, one fixed config for every model
  AdaptedPass   searches per-model configs on a dev split under an equal budget,
                then runs the winner on test
  LatencyPass   serial, interleaved, cache genuinely bypassed, streaming on

They used to be three methods on `Orchestrator`, each reaching into the same
five pieces of shared mutable state. Adding a fourth kind of pass — a
retrieval-free ablation, a multi-turn walk — meant editing that class, which is
the Open/Closed problem in its usual form.

Each is now a class implementing `EvaluationPass`. They share the collector and
the run context; nothing else. `Orchestrator` picks one and calls `run`.

The two concurrency regimes are the reason the split is worth making explicit.
Keeping the latency lane serial, uncached and interleaved is what stops the
harness from benchmarking its own queue depth instead of the model — and that
property is much easier to see, and to protect, when it is a separate class
than when it is one branch among three in a long method.
"""

from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..cache.cache import NullCache, stable_hash
from ..clients.cost import BudgetExceeded
from ..profiles.profile import Profile
from ..rag.prompt import PromptConfig
from ..rag.retrieve import RetrievalConfig
from ..store.schema import EvalItem, Pass, TraceRow
from ..tuning.search import Candidate, enumerate_candidates
from .collector import RowCollector
from .runner import RunContext, run_item


@dataclass
class RunReport:
    """What a pass produced, beyond the rows themselves."""

    run_id: str
    rows_written: int = 0
    errors: int = 0
    aborted: bool = False
    abort_reason: str = ""
    cost: dict = field(default_factory=dict)
    winners: dict = field(default_factory=dict)   # model -> winning config
    elapsed_s: float = 0.0

    def summary(self) -> str:
        bits = [f"run {self.run_id}", f"{self.rows_written} rows",
                f"{self.errors} errors", f"{self.elapsed_s:.0f}s"]
        if self.cost:
            bits.append(f"${self.cost.get('total_usd', 0.0):.4f}")
        if self.aborted:
            bits.append(f"ABORTED: {self.abort_reason}")
        return " · ".join(bits)


class ContextFactory(Protocol):
    """Builds a `RunContext` for a profile. Supplied by the orchestrator.

    A pass needs a context but has no business knowing how retrievers,
    rerankers and judges are constructed — that is provider wiring.
    """

    def __call__(self, profile: Profile, cache=None,
                 stream: bool = False) -> RunContext: ...


@runtime_checkable
class EvaluationPass(Protocol):
    """One walk of the matrix.

    Runtime-checkable so a user-defined pass can be validated structurally,
    without being forced to inherit from anything.
    """

    name: str

    def run(self, profile: Profile, models: list[str], run_id: str,
            **kwargs) -> RunReport: ...


@dataclass
class _PassBase:
    """Shared plumbing: the collector, the context factory, the meter."""

    collector: RowCollector
    make_context: ContextFactory
    meter: object
    max_workers: int = 8

    def _finish(self, report: RunReport, started: float) -> RunReport:
        self.collector.flush()
        report.rows_written = self.collector.written
        report.errors = self.collector.errors
        report.cost = self.meter.summary() if self.meter is not None else {}
        report.elapsed_s = time.time() - started
        return report

    @staticmethod
    def _retrieval_cfg(profile: Profile) -> RetrievalConfig:
        return RetrievalConfig(
            mode=profile.retrieval_mode, k=profile.k,
            embedding_model=profile.embedding_model,
            dense_weight=profile.dense_weight,
            sparse_weight=profile.sparse_weight,
            candidate_multiplier=(3 if profile.rerank
                                  else profile.candidate_multiplier),
        )


# --------------------------------------------------------------------------- #
#  Baseline
# --------------------------------------------------------------------------- #
class BaselinePass(_PassBase):
    """Fixed retrieval + fixed prompt, identical for every model.

    Runs in the throughput lane: concurrent, cache on. Queuing doesn't matter
    here because nothing this pass reports is a timing.
    """

    name = "baseline"

    def run(self, profile: Profile, models: list[str], run_id: str,
            items: list[EvalItem] | None = None,
            max_workers: int | None = None,
            done_keys: set | None = None, **_) -> RunReport:
        started = time.time()
        items = items or []
        ctx = self.make_context(profile)
        retrieval_cfg = self._retrieval_cfg(profile)
        prompt_cfg = PromptConfig(max_context_chunks=profile.k)
        pcfg_hash = stable_hash(*profile.config_hash_parts())

        done_keys = done_keys or set()
        work = [(item, model) for item in items for model in models
                if (item.item_id, model, "baseline") not in done_keys]
        self.collector.expect(len(work))

        report = RunReport(run_id=run_id)
        with cf.ThreadPoolExecutor(max_workers=max_workers or self.max_workers) as ex:
            futures = [
                ex.submit(run_item, item, model, profile.name, Pass.BASELINE,
                          retrieval_cfg, prompt_cfg, ctx, run_id,
                          profile.rerank, profile.rerank_top_n,
                          profile.active_metrics, profile.accuracy_scorer,
                          pcfg_hash)
                for item, model in work
            ]
            tripped = self.collector.drain(futures)
        if tripped is not None:
            report.aborted, report.abort_reason = True, str(tripped)
        return self._finish(report, started)


# --------------------------------------------------------------------------- #
#  Adapted (equal-budget tuning)
# --------------------------------------------------------------------------- #
class AdaptedPass(_PassBase):
    """Search candidates on dev under an equal budget, run the winner on test.

    Dev evaluations are logged as `Pass.TUNING`; the winning test run is
    `Pass.ADAPTED`. The baseline-vs-adapted delta on test is the tuning gain.
    """

    name = "adapted"

    def run(self, profile: Profile, models: list[str], run_id: str,
            dev_items: list[EvalItem] | None = None,
            test_items: list[EvalItem] | None = None,
            max_workers: int | None = None, **_) -> RunReport:
        started = time.time()
        dev_items = dev_items or []
        test_items = test_items or []
        if not dev_items:
            raise ValueError(
                "The adapted pass needs a non-empty dev split. Raise "
                "`dev_split` in configs/run.yaml, or add more eval items.")

        ctx = self.make_context(profile)
        pcfg_hash = stable_hash(*profile.config_hash_parts(), "adapted")
        workers = max_workers or self.max_workers
        report = RunReport(run_id=run_id)

        candidates = enumerate_candidates(profile, profile.tuning_budget,
                                          seed=profile.seed)
        if not candidates:
            raise ValueError(f"Profile '{profile.name}' produced no candidates.")

        # Every model faces the identical candidate list — the equal-budget
        # guarantee, made explicit rather than assumed.
        self.collector.expect(
            (len(candidates) * len(dev_items) + len(test_items)) * len(models))

        try:
            for model in models:
                best, best_score = self._search(profile, model, candidates,
                                                dev_items, ctx, run_id,
                                                pcfg_hash, workers)
                report.winners[model] = {"config": best.describe(),
                                         "dev_score": best_score}

                with cf.ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = [
                        ex.submit(run_item, it, model, profile.name,
                                  Pass.ADAPTED, best.retrieval, best.prompt,
                                  ctx, run_id, best.rerank, best.rerank_top_n,
                                  profile.active_metrics,
                                  profile.accuracy_scorer, pcfg_hash)
                        for it in test_items
                    ]
                    tripped = self.collector.drain(futures)
                if tripped is not None:
                    raise tripped
        except BudgetExceeded as e:
            report.aborted, report.abort_reason = True, str(e)
        return self._finish(report, started)

    def _search(self, profile: Profile, model: str,
                candidates: list[Candidate], dev_items: list[EvalItem],
                ctx: RunContext, run_id: str, pcfg_hash: str,
                workers: int) -> tuple[Candidate, float]:
        """Score every candidate on dev, in parallel, and return the winner.

        Parallelising across (candidate, item) rather than candidate-by-candidate
        keeps every worker busy; nested serial loops were the single slowest
        thing in the harness and the reason this pass ships off by default.
        """
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {}
            for ci, cand in enumerate(candidates):
                for it in dev_items:
                    fut = ex.submit(
                        run_item, it, model, profile.name, Pass.TUNING,
                        cand.retrieval, cand.prompt, ctx, run_id,
                        cand.rerank, cand.rerank_top_n,
                        profile.active_metrics, profile.accuracy_scorer,
                        pcfg_hash)
                    futures[fut] = ci
            grouped, _tripped = self.collector.drain_tagged(futures)

        best_idx, best_score = 0, float("-inf")
        for ci in range(len(candidates)):
            s = weighted_mean(grouped.get(ci, []), profile.metric_weights)
            if s > best_score:
                best_idx, best_score = ci, s
        return candidates[best_idx], best_score


# --------------------------------------------------------------------------- #
#  Latency
# --------------------------------------------------------------------------- #
class LatencyPass(_PassBase):
    """Clean latency: serial, interleaved, cache truly bypassed, streaming on.

    `NullCache` rather than a scratch directory — a scratch directory warms up
    and then serves cached "network" timings on every subsequent run, which is
    worse than not measuring latency at all because it looks measured.

    Streaming is on specifically to capture time-to-first-token, the number that
    governs perceived responsiveness in an interactive UI.
    """

    name = "latency"

    def run(self, profile: Profile, models: list[str], run_id: str,
            items: list[EvalItem] | None = None, warmup: int = 1,
            repeats: int = 1, **_) -> RunReport:
        started = time.time()
        items = items or []
        ctx = self.make_context(profile, cache=NullCache(), stream=True)
        retrieval_cfg = RetrievalConfig(
            mode=profile.retrieval_mode, k=profile.k,
            embedding_model=profile.embedding_model)
        prompt_cfg = PromptConfig(max_context_chunks=profile.k)
        pcfg_hash = stable_hash(*profile.config_hash_parts(), "latency")

        warm_done = dict.fromkeys(models, 0)
        self.collector.expect(len(items) * len(models) * repeats)
        report = RunReport(run_id=run_id)

        try:
            for _ in range(repeats):
                for item in items:
                    for model in models:
                        row = run_item(item, model, profile.name, Pass.LATENCY,
                                       retrieval_cfg, prompt_cfg, ctx, run_id,
                                       active_metrics=["cost_usd"],
                                       accuracy_scorer="none",
                                       profile_cfg_hash=pcfg_hash)
                        if warm_done[model] < warmup:
                            warm_done[model] += 1
                            self.collector.skip()
                            continue  # discard warm-up timing
                        self.collector.collect(row)
        except BudgetExceeded as e:
            report.aborted, report.abort_reason = True, str(e)
        return self._finish(report, started)


# --------------------------------------------------------------------------- #
def weighted_mean(rows: list[TraceRow], weights: dict) -> float:
    """Weighted composite over a set of rows (used to score tuning candidates).

    Returns -inf when no weighted metric was measured at all, so an unscorable
    candidate can never beat a scorable one.
    """
    import numpy as np

    total, used = 0.0, False
    for metric, w in weights.items():
        vals = [getattr(r, metric) for r in rows
                if getattr(r, metric, None) is not None]
        if vals:
            total += w * float(np.mean(vals))
            used = True
    return total if used else float("-inf")
