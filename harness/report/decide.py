"""
The decision layer: turning a leaderboard into a recommendation.

A leaderboard answers "which model scored highest". Nobody actually gets to make
that decision. The real one is constrained:

    "cheapest model with faithfulness >= 0.90 and p95 latency <= 2000 ms,
     that we can afford at 50,000 queries a day"

The original harness produced every input to that answer — accuracy, cost,
latency, a Pareto frontier — and then stopped, leaving the user to eyeball four
tables and do the arithmetic. That gap is where benchmark results go to die: the
work was done, the decision still wasn't made.

Three things here:

  select        — constraint satisfaction over per-model aggregates, returning a
                  ranked shortlist and, crucially, *why* each rejected model was
                  rejected
  project_cost  — per-query cost extrapolated to daily/monthly spend at volume,
                  because $0.0004 per query sounds free until it's 50k/day
  headroom      — how much accuracy you buy per extra dollar, so "the expensive
                  model is 2% better" becomes "2% better for $4,200/month"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Direction = Literal["max", "min"]

# Which way is better for each metric. Getting this wrong inverts a
# recommendation, so it lives in one place rather than at each call site.
METRIC_DIRECTION: dict[str, Direction] = {
    "accuracy": "max", "faithfulness": "max", "answer_relevance": "max",
    "completeness": "max", "token_f1": "max", "hit_rate_at_k": "max",
    "mrr": "max", "ndcg_at_k": "max", "context_recall": "max",
    "context_precision": "max", "average_precision": "max",
    "citation_valid_pointer": "max", "citation_supporting": "max",
    "citation_density": "max", "citation_recall": "max",
    "abstention_correct": "max", "injection_resisted": "max",
    "cost_usd": "min", "latency_ms": "min", "ttft_ms": "min",
    "latency_p95_ms": "min", "prompt_tokens": "min", "completion_tokens": "min",
    "pii_leaked": "min", "judge_disagreement": "min", "error_rate": "min",
}


@dataclass
class Constraint:
    """One requirement a model must satisfy to be eligible."""
    metric: str
    op: Literal[">=", "<=", ">", "<", "=="]
    value: float

    def satisfied_by(self, actual: float | None) -> bool | None:
        """None when the metric wasn't measured — unknown, not failed.

        The distinction matters: silently treating an unmeasured metric as a
        failure would reject every model on a profile that didn't activate it,
        and the user would see an empty shortlist with no explanation.
        """
        if actual is None or (isinstance(actual, float) and np.isnan(actual)):
            return None
        return {
            ">=": actual >= self.value, "<=": actual <= self.value,
            ">": actual > self.value, "<": actual < self.value,
            "==": abs(actual - self.value) < 1e-9,
        }[self.op]

    def describe(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"

    @staticmethod
    def parse(text: str) -> Constraint:
        """Parse "faithfulness>=0.9" or "latency_p95_ms <= 2000"."""
        for op in (">=", "<=", "==", ">", "<"):
            if op in text:
                metric, value = text.split(op, 1)
                return Constraint(metric.strip(), op, float(value.strip()))
        raise ValueError(
            f"Cannot parse constraint '{text}'. Use e.g. 'accuracy>=0.85' "
            f"or 'latency_p95_ms<=2000'.")


@dataclass
class Candidate:
    model: str
    metrics: dict = field(default_factory=dict)
    eligible: bool = True
    failures: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"model": self.model, "eligible": self.eligible,
                "failed": "; ".join(self.failures) or "",
                "unmeasured": "; ".join(self.unknowns) or "",
                **dict(self.metrics)}


def model_aggregates(df: pd.DataFrame,
                     passes: tuple[str, ...] = ("baseline", "adapted"),
                     latency_pass: str = "latency") -> pd.DataFrame:
    """Per-model summary: quality means, cost mean, and true latency percentiles.

    Latency is taken from the latency-lane rows when they exist and only falls
    back to throughput rows otherwise. Mixing the two would report your own
    queuing delay as model latency — precisely the mistake the two-lane design
    exists to prevent, and one that is invisible in the final number.

    p95 rather than the mean, because SLAs are written on tails. A model with a
    good mean and a bad tail fails in production and looks fine in a report.
    """
    if df.empty:
        return pd.DataFrame()

    quality = df[df["pass_"].isin(passes)] if "pass_" in df.columns else df
    rows = []

    lat_source = (df[df["pass_"] == latency_pass]
                  if "pass_" in df.columns and (df["pass_"] == latency_pass).any()
                  else quality)

    for model in sorted(df["model"].dropna().unique()):
        g = quality[quality["model"] == model]
        rec: dict = {"model": model, "n_items": len(g)}

        for col in ("accuracy", "faithfulness", "answer_relevance",
                    "completeness", "token_f1", "hit_rate_at_k", "mrr",
                    "ndcg_at_k", "context_recall", "context_precision",
                    "citation_valid_pointer", "citation_supporting",
                    "citation_density", "abstention_correct",
                    "injection_resisted", "pii_leaked", "judge_disagreement",
                    "cost_usd", "prompt_tokens", "completion_tokens"):
            if col in g.columns and g[col].notna().any():
                rec[col] = float(g[col].mean(skipna=True))

        lg = lat_source[lat_source["model"] == model]
        for col, out in (("latency_ms", "latency"), ("ttft_ms", "ttft")):
            if col in lg.columns and lg[col].notna().any():
                vals = lg[col].dropna().to_numpy(float)
                rec[f"{out}_p50_ms"] = float(np.percentile(vals, 50))
                rec[f"{out}_p95_ms"] = float(np.percentile(vals, 95))

        if "error" in g.columns and len(g):
            rec["error_rate"] = float(g["error"].notna().mean())

        rows.append(rec)
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, constraints: list[Constraint],
           optimise: str = "cost_usd",
           direction: Direction | None = None) -> tuple[pd.DataFrame, list[Candidate]]:
    """Rank the models that satisfy every constraint, best on `optimise` first.

    Returns (shortlist, all_candidates). The second value is the important one
    when the shortlist comes back empty: it says exactly which constraint each
    model failed and by how much, so "no model qualifies" becomes "everything
    failed latency, relax it to 2.4s or accept the accuracy drop".
    """
    agg = model_aggregates(df) if "pass_" in df.columns else df
    if agg.empty:
        return pd.DataFrame(), []

    direction = direction or METRIC_DIRECTION.get(optimise, "max")
    candidates: list[Candidate] = []

    for _, row in agg.iterrows():
        metrics = {k: v for k, v in row.items() if k != "model"}
        cand = Candidate(model=str(row["model"]), metrics=metrics)
        for c in constraints:
            actual = metrics.get(c.metric)
            ok = c.satisfied_by(actual)
            if ok is None:
                cand.unknowns.append(c.describe())
                # An unmeasured requirement cannot be confirmed, so the model is
                # not eligible — but it's reported as "unmeasured", never as a
                # failure it didn't earn.
                cand.eligible = False
            elif not ok:
                cand.failures.append(f"{c.describe()} (actual {actual:.4g})")
                cand.eligible = False
        candidates.append(cand)

    eligible = [c for c in candidates if c.eligible and optimise in c.metrics]
    eligible.sort(key=lambda c: c.metrics[optimise],
                  reverse=(direction == "max"))
    shortlist = pd.DataFrame([c.as_dict() for c in eligible])
    return shortlist, candidates


def explain(candidates: list[Candidate], constraints: list[Constraint]) -> str:
    """Human-readable account of who qualified and why the rest didn't."""
    eligible = [c for c in candidates if c.eligible]
    lines = [f"Constraints: {', '.join(c.describe() for c in constraints) or 'none'}",
             f"{len(eligible)}/{len(candidates)} models qualify."]
    if not eligible:
        lines.append("")
        lines.append("Nothing qualifies. Closest misses:")
        for c in candidates:
            reason = "; ".join(c.failures) or "; ".join(
                f"{u} (not measured)" for u in c.unknowns)
            lines.append(f"  {c.model}: {reason}")
    for c in eligible:
        lines.append(f"  OK  {c.model}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Cost projection
# ---------------------------------------------------------------------------
@dataclass
class CostProjection:
    model: str
    cost_per_query: float
    queries_per_day: int
    daily_usd: float
    monthly_usd: float
    annual_usd: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def project_cost(df: pd.DataFrame, queries_per_day: int = 10_000,
                 include_judge: bool = False) -> pd.DataFrame:
    """Extrapolate measured per-query cost to production volume.

    `include_judge` defaults to False on purpose: the judge is *evaluation*
    infrastructure, not part of serving. Including it would inflate every
    production forecast by the cost of grading, which you will not be paying in
    production. The evaluation's own cost is reported separately.
    """
    agg = model_aggregates(df) if "pass_" in df.columns else df
    if agg.empty:
        return pd.DataFrame()

    source = df[df["pass_"].isin(("baseline", "adapted"))] if "pass_" in df.columns else df
    rows = []
    for model in agg["model"]:
        g = source[source["model"] == model]
        if include_judge or "gen_cost_usd" not in g.columns:
            col = "cost_usd"
        else:
            col = "gen_cost_usd"
        if col not in g.columns or not g[col].notna().any():
            continue
        per_query = float(g[col].mean(skipna=True))
        rows.append(CostProjection(
            model=model, cost_per_query=per_query,
            queries_per_day=queries_per_day,
            daily_usd=per_query * queries_per_day,
            monthly_usd=per_query * queries_per_day * 30,
            annual_usd=per_query * queries_per_day * 365,
        ).as_dict())
    return pd.DataFrame(rows).sort_values("monthly_usd") if rows else pd.DataFrame()


def headroom(df: pd.DataFrame, quality_metric: str = "accuracy",
             queries_per_day: int = 10_000) -> pd.DataFrame:
    """What each quality step up actually costs, relative to the cheapest model.

    Converts "the big model is 3 points better" into "3 points for $1,850 a
    month", which is the form the decision is actually made in. A negative
    `usd_per_point` marks a model that is both worse and more expensive — a
    strictly dominated option, and the fastest thing to cut from a shortlist.
    """
    proj = project_cost(df, queries_per_day)
    agg = model_aggregates(df) if "pass_" in df.columns else df
    if proj.empty or agg.empty or quality_metric not in agg.columns:
        return pd.DataFrame()

    merged = proj.merge(agg[["model", quality_metric]], on="model", how="inner")
    merged = merged.dropna(subset=[quality_metric])
    if merged.empty:
        return pd.DataFrame()

    base = merged.loc[merged["monthly_usd"].idxmin()]
    merged["quality_delta"] = merged[quality_metric] - base[quality_metric]
    merged["monthly_delta_usd"] = merged["monthly_usd"] - base["monthly_usd"]
    merged["usd_per_point"] = np.where(
        merged["quality_delta"].abs() > 1e-9,
        merged["monthly_delta_usd"] / (merged["quality_delta"] * 100),
        np.nan,
    )
    merged["baseline_model"] = base["model"]
    return merged.sort_values(quality_metric, ascending=False)
