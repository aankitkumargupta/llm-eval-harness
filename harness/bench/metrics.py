"""
Benchmark metrics (§10.7).

Every function here is pure and every one has a reason to exist beyond
completeness:

**`chance_adjusted`**, 25% on 4-way multiple choice is not "25% good", it is
zero information. Reporting raw accuracy next to a chance level of 0.25 invites
exactly that misreading, and the adjustment is one line, so there is no excuse
for omitting it. Negative values are kept rather than clamped: a model *below*
chance is a real and interesting finding (usually an extraction bug or an
inverted label), and clamping to zero hides it.

**`pass_at_k`**, the unbiased estimator from Chen et al. (2021), not
`any(correct)`. Sampling n completions and reporting "did any of k pass" is
biased upward whenever n > k, and pass@1 from a single sample is a coin flip
with a standard error nobody quotes.

**The rate family**: `extraction_failure_rate`, `refusal_rate`,
`over_refusal_rate`, `format_violation_rate`, `truncation_rate`,
`sandbox_violation_rate`. These exist so that I7 is arithmetic rather than
good intentions: each counts items that are *excluded* from the accuracy
denominator, so a harness cannot quietly convert "could not parse" into
"got it wrong".

**`cost_per_correct_answer`**, §10.7 calls this "the number that actually
drives procurement", and it is the one figure that changes a decision: a model
twice as accurate at five times the price loses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


def chance_adjusted(accuracy: float, chance_level: float) -> float:
    """`(acc − chance) / (1 − chance)`.

    0.0 means "no better than guessing", 1.0 means perfect. Returns raw
    accuracy when `chance_level` is 0, so non-MC families can call it freely.
    """
    if chance_level <= 0.0:
        return accuracy
    if chance_level >= 1.0:
        raise ValueError("chance_level must be < 1.0")
    return (accuracy - chance_level) / (1.0 - chance_level)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: `1 − C(n−c, k) / C(n, k)`.

    `n` samples drawn, `c` of them correct, estimating the chance that at least
    one of `k` draws passes. Chen et al. (2021), "Evaluating Large Language
    Models Trained on Code", §2.1.

    The naive alternative, draw n, report whether any of the first k passed,
    is biased upward and noisier at exactly the sample sizes people can afford.
    """
    if n < 0 or c < 0 or k < 1:
        raise ValueError(f"invalid pass@k inputs: n={n}, c={c}, k={k}")
    if c > n:
        raise ValueError(f"more correct ({c}) than sampled ({n})")
    if n - c < k:
        return 1.0
    if k > n:
        raise ValueError(f"k ({k}) exceeds samples drawn ({n})")
    # Product form rather than factorials: C(n, k) overflows float for the
    # sample counts a real code benchmark uses.
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def _rate(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 0.0
    vals = df[col].dropna()
    return float(vals.mean()) if len(vals) else 0.0


@dataclass
class BenchSummary:
    """One model on one benchmark. Rates sit *beside* the score, never inside."""
    model: str
    benchmark: str
    n_items: int
    n_scored: int                    # items that produced a usable answer
    accuracy: float | None
    accuracy_chance_adjusted: float | None
    chance_level: float
    extraction_failure_rate: float
    refusal_rate: float
    truncation_rate: float
    format_violation_rate: float
    cost_usd: float
    cost_per_correct_answer: float | None
    latency_p50_ms: float | None
    usage_estimated_rate: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def headline(self) -> str:
        """One line that cannot be quoted misleadingly.

        The excluded-item count is part of the sentence, not a footnote: a
        score computed over 62 of 100 items means something different from the
        same score over 100, and whoever reads it needs both numbers together.
        """
        if self.accuracy is None:
            return (f"{self.model} on {self.benchmark}: no scoreable items "
                    f"({self.n_items} attempted, all excluded)")
        adj = ""
        if self.chance_level > 0 and self.accuracy_chance_adjusted is not None:
            adj = f" (chance-adjusted {self.accuracy_chance_adjusted:.3f})"
        excluded = self.n_items - self.n_scored
        tail = f", {excluded} excluded" if excluded else ""
        return (f"{self.model} on {self.benchmark}: {self.accuracy:.3f}{adj} "
                f"over {self.n_scored}/{self.n_items} items{tail}")


def summarise(df: pd.DataFrame, benchmark: str, chance_level: float = 0.0
              ) -> pd.DataFrame:
    """Per-model summary for one benchmark.

    `accuracy` is the mean over items that produced a usable answer only. Items
    whose answer could not be extracted carry `accuracy = NaN` and are absent
    from both numerator and denominator, that is the whole of I7, expressed as
    a `dropna`.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    for model, g in df.groupby("model"):
        scored = g["accuracy"].dropna() if "accuracy" in g.columns else pd.Series(dtype=float)
        acc = float(scored.mean()) if len(scored) else None
        cost = float(g["cost_usd"].fillna(0).sum()) if "cost_usd" in g.columns else 0.0
        n_correct = float(scored.sum()) if len(scored) else 0.0

        rows.append(BenchSummary(
            model=str(model),
            benchmark=benchmark,
            n_items=int(len(g)),
            n_scored=int(len(scored)),
            accuracy=acc,
            accuracy_chance_adjusted=(chance_adjusted(acc, chance_level)
                                      if acc is not None else None),
            chance_level=chance_level,
            extraction_failure_rate=_rate(g, "extraction_failed"),
            refusal_rate=_rate(g, "refused"),
            truncation_rate=_rate(g, "truncated"),
            format_violation_rate=_rate(g, "format_violation"),
            cost_usd=cost,
            cost_per_correct_answer=(cost / n_correct) if n_correct else None,
            latency_p50_ms=(float(g["latency_ms"].dropna().median())
                            if "latency_ms" in g.columns
                            and len(g["latency_ms"].dropna()) else None),
            usage_estimated_rate=_rate(g, "usage_estimated"),
        ).as_dict())

    out = pd.DataFrame(rows)
    return out.sort_values("accuracy", ascending=False, na_position="last")


def suite_composite(per_benchmark: pd.DataFrame, weights: dict[str, float]
                    ) -> pd.DataFrame:
    """Cross-benchmark composite, only ever with declared weights (§10.7.2).

    Raises on an unweighted benchmark rather than assuming equal weighting.
    "Averaging benchmarks hides the weighting" (§10.1), and a default that
    looks neutral is the most effective way to hide it: nobody questions an
    arithmetic mean, and nobody can reconstruct what it assumed.
    """
    if per_benchmark.empty:
        return pd.DataFrame()
    missing = sorted(set(per_benchmark["benchmark"]) - set(weights))
    if missing:
        raise ValueError(
            f"No declared weight for {missing}. A cross-benchmark composite "
            f"requires weights in config/suites/<name>.yaml, printed beside "
            f"the number every time, there is no implicit equal weighting.")

    rows = []
    for model, g in per_benchmark.groupby("model"):
        total_w = sum(weights[b] for b in g["benchmark"])
        if total_w == 0:
            continue
        score = sum(weights[b] * (v if pd.notna(v) else 0.0)
                    for b, v in zip(g["benchmark"], g["accuracy"]))
        rows.append({"model": model, "composite": score / total_w,
                     "benchmarks": len(g),
                     "weighting": ", ".join(f"{b}={weights[b]:g}"
                                            for b in sorted(g["benchmark"]))})
    return (pd.DataFrame(rows).sort_values("composite", ascending=False)
            if rows else pd.DataFrame())
