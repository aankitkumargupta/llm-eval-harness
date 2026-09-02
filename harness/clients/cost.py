"""
Cost metering and the budget guard.

Two problems this fixes.

**Cost was under-reported.** The old runner billed generation only. But a
judge-scored profile makes *two extra model calls per item* against a large
judge model, plus a query embedding, plus an optional rerank. On a
regulated-QA-style profile the judge is routinely the majority of the bill, so
"cost per query" was reporting maybe a third of what the run actually cost — and
cost carries a negative weight in the leaderboard composite, so the ranking
itself was wrong, not just the dollar figure.

**A typo could cost real money.** `tuning_budget: 20` on a 500-item evalset with
four models is ~40,000 calls. Nothing stopped it. `BudgetGuard` gives a run a
hard ceiling and aborts cleanly when it's reached, so the blast radius of a
mis-set config is bounded by a number you chose.

Every counter here is thread-safe: the throughput lane updates it from many
worker threads at once.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised when a run would push spend past its configured ceiling.

    Deliberately not retryable: the resilience layer must let this through so
    the run stops instead of backing off and trying to overspend again.
    """

    def __init__(self, spent: float, limit: float):
        super().__init__(
            f"Budget exceeded: ${spent:.4f} spent of ${limit:.4f} limit. "
            f"Raise `budget_usd` in configs/run.yaml or narrow the run "
            f"(fewer models, smaller evalset, lower tuning_budget)."
        )
        self.spent = spent
        self.limit = limit


@dataclass
class CostBreakdown:
    """Spend split by the subsystem that caused it.

    Keeping these separate is the point: "the judge cost 3x the models under
    test" is an actionable finding, and it's invisible in a single total.
    """
    generation: float = 0.0
    judge: float = 0.0
    embedding: float = 0.0
    rerank: float = 0.0

    @property
    def total(self) -> float:
        return self.generation + self.judge + self.embedding + self.rerank

    def as_dict(self) -> dict:
        return {
            "generation_usd": self.generation,
            "judge_usd": self.judge,
            "embedding_usd": self.embedding,
            "rerank_usd": self.rerank,
            "total_usd": self.total,
        }


@dataclass
class CallCounts:
    """Call and token volumes, for throughput reporting and cost forecasting."""
    generation: int = 0
    judge: int = 0
    embedding: int = 0
    rerank: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    retries: int = 0

    def as_dict(self) -> dict:
        return dict(
            generation_calls=self.generation, judge_calls=self.judge,
            embedding_calls=self.embedding, rerank_calls=self.rerank,
            cache_hits=self.cache_hits, prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            embedding_tokens=self.embedding_tokens, retries=self.retries,
        )


class CostMeter:
    """Thread-safe running total of what a run has spent, and on what.

    One meter per run. The runner records each billed call against it, so the
    final number is measured from real usage blocks rather than estimated from
    an assumed average prompt length.
    """

    def __init__(self, budget_usd: float = 0.0):
        # budget_usd <= 0 means "no ceiling" — the default, so existing runs
        # behave exactly as before unless a limit is opted into.
        self.budget_usd = float(budget_usd)
        self.costs = CostBreakdown()
        self.counts = CallCounts()
        self._lock = threading.Lock()
        self._tripped = False

    # -- recording ------------------------------------------------------- #
    def record_generation(self, usd: float, prompt_tokens: int = 0,
                          completion_tokens: int = 0, is_judge: bool = False) -> None:
        with self._lock:
            if is_judge:
                self.costs.judge += usd
                self.counts.judge += 1
            else:
                self.costs.generation += usd
                self.counts.generation += 1
            self.counts.prompt_tokens += prompt_tokens
            self.counts.completion_tokens += completion_tokens
        self.check()

    def record_embedding(self, usd: float, tokens: int = 0) -> None:
        with self._lock:
            self.costs.embedding += usd
            self.counts.embedding += 1
            self.counts.embedding_tokens += tokens
        self.check()

    def record_rerank(self, usd: float) -> None:
        with self._lock:
            self.costs.rerank += usd
            self.counts.rerank += 1
        self.check()

    def record_cache_hit(self) -> None:
        with self._lock:
            self.counts.cache_hits += 1

    def record_retry(self, n: int = 1) -> None:
        with self._lock:
            self.counts.retries += n

    # -- reading --------------------------------------------------------- #
    @property
    def spent(self) -> float:
        with self._lock:
            return self.costs.total

    @property
    def remaining(self) -> float:
        """Headroom left, or infinity when no budget was set."""
        if self.budget_usd <= 0:
            return float("inf")
        return max(0.0, self.budget_usd - self.spent)

    def check(self) -> None:
        """Trip the guard once the ceiling is crossed.

        Checked *after* recording rather than before: we can't know a call's real
        cost until its usage block comes back, so the guard stops the run at the
        first call that crosses the line rather than pre-authorising each one.
        Overshoot is bounded by a single call.
        """
        if self.budget_usd <= 0:
            return
        with self._lock:
            if self.costs.total < self.budget_usd:
                return
            self._tripped = True
            spent = self.costs.total
        raise BudgetExceeded(spent, self.budget_usd)

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def summary(self) -> dict:
        with self._lock:
            out = {**self.costs.as_dict(), **self.counts.as_dict()}
            out["budget_usd"] = self.budget_usd
            total_calls = (self.counts.generation + self.counts.judge)
            out["cache_hit_rate"] = (
                self.counts.cache_hits / (total_calls + self.counts.cache_hits)
                if (total_calls + self.counts.cache_hits) else 0.0
            )
            return out


@dataclass
class CostForecast:
    """A projection produced *before* a run, so a click isn't a blind bill."""
    calls: int
    judge_calls: int
    embed_calls: int
    est_usd: float
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "judge_calls": self.judge_calls,
                "embed_calls": self.embed_calls, "est_usd": self.est_usd,
                **self.detail}


def forecast_run(
    pricing,
    models: list[str],
    n_test: int,
    n_dev: int = 0,
    tuning_budget: int = 0,
    *,
    judge_model: str = "",
    judge_calls_per_item: int = 2,
    do_baseline: bool = True,
    do_adapted: bool = False,
    avg_prompt_tokens: int = 1000,
    avg_completion_tokens: int = 150,
    avg_judge_prompt_tokens: int = 1400,
    avg_judge_completion_tokens: int = 120,
) -> CostForecast:
    """Estimate a run's spend, *including the judge* — which the old estimate omitted.

    Prices each model at its own rate rather than sampling the first one, since a
    matrix mixing a 20B and a 120B model has a spend profile the cheap model's
    price badly misrepresents. Still an estimate — real prompt length depends on
    your corpus — but it is now the right order of magnitude.
    """
    per_model_items = 0
    if do_baseline:
        per_model_items += n_test
    if do_adapted:
        per_model_items += tuning_budget * n_dev + n_test

    gen_calls = per_model_items * len(models)
    judge_calls = gen_calls * judge_calls_per_item if judge_model else 0

    gen_usd = 0.0
    for m in models:
        try:
            gen_usd += pricing.generation_cost(
                m, avg_prompt_tokens * per_model_items,
                avg_completion_tokens * per_model_items)
        except KeyError:
            # An unpriced model can't be forecast. Skipping it under-reports the
            # estimate, so we surface the gap in `detail` rather than hiding it.
            gen_usd += 0.0

    judge_usd = 0.0
    if judge_model and judge_calls:
        try:
            judge_usd = pricing.generation_cost(
                judge_model, avg_judge_prompt_tokens * judge_calls,
                avg_judge_completion_tokens * judge_calls)
        except KeyError:
            judge_usd = 0.0

    unpriced = [m for m in models if m not in getattr(pricing, "models", {})]
    if judge_model and judge_model not in getattr(pricing, "models", {}):
        unpriced.append(judge_model)

    return CostForecast(
        calls=gen_calls, judge_calls=judge_calls, embed_calls=per_model_items,
        est_usd=gen_usd + judge_usd,
        detail={"generation_usd": gen_usd, "judge_usd": judge_usd,
                "unpriced_models": unpriced},
    )
