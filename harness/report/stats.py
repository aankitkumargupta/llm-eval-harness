"""
Statistical significance for model comparisons.

The problem this solves is the one that makes most internal LLM benchmarks
untrustworthy: someone runs 40 questions, sees 0.82 against 0.78, and ships the
"better" model. On 40 items that gap is comfortably inside the noise, and half
the time the ranking would flip on a re-run.

The original harness reported a bootstrap CI **per model**, which is the wrong
interval for the decision being made. Two overlapping per-model CIs do *not*
imply the difference is insignificant — the models were evaluated on the **same
items**, so the comparison is paired, and pairing removes item difficulty (the
dominant variance component) from the estimate. A paired test routinely finds a
real difference where two marginal intervals overlap heavily.

What's here:

  paired_bootstrap    — CI and p-value on the per-item *difference*
  mcnemar             — exact test for paired binary outcomes (correct/incorrect)
  holm_bonferroni     — multiple-comparison correction across a model matrix
  required_n          — how many questions you'd need to detect a given gap
  significance_matrix — every pair, corrected, with a plain-language verdict

All of it is pure numpy/pandas over trace rows: no network, free to re-run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Paired comparison
# ---------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    model_a: str
    model_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    diff: float                    # mean_a - mean_b
    ci_low: float
    ci_high: float
    p_value: float
    test: str = "paired_bootstrap"
    p_adjusted: float | None = None
    significant: bool | None = None

    def verdict(self, alpha: float = 0.05) -> str:
        """A sentence a non-statistician can act on."""
        p = self.p_adjusted if self.p_adjusted is not None else self.p_value
        if self.n_pairs < 2:
            return "not enough paired items to compare"
        if p >= alpha:
            return (f"no significant difference (p={p:.3f}, "
                    f"{self.n_pairs} paired items)")
        better = self.model_a if self.diff > 0 else self.model_b
        return (f"{better} is better by {abs(self.diff):.3f} "
                f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] (p={p:.3f})")

    def as_dict(self) -> dict:
        return {
            "model_a": self.model_a, "model_b": self.model_b,
            "metric": self.metric, "n_pairs": self.n_pairs,
            "mean_a": self.mean_a, "mean_b": self.mean_b, "diff": self.diff,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_value": self.p_value, "p_adjusted": self.p_adjusted,
            "significant": self.significant, "test": self.test,
            "verdict": self.verdict(),
        }


def paired_values(df: pd.DataFrame, model_a: str, model_b: str,
                  metric: str,
                  item_col: str = "item_id") -> tuple[np.ndarray, np.ndarray]:
    """Line up two models' scores on the SAME items.

    Pairing is the whole point. Comparing marginal means throws away the fact
    that both models faced identical questions, and item difficulty is usually a
    much bigger source of variance than the model difference you're trying to
    detect.
    """
    if metric not in df.columns:
        return np.array([]), np.array([])
    a = (df[df["model"] == model_a].dropna(subset=[metric])
         .groupby(item_col)[metric].mean())
    b = (df[df["model"] == model_b].dropna(subset=[metric])
         .groupby(item_col)[metric].mean())
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.array([]), np.array([])
    return a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                     ci: float = 0.95, seed: int = 0) -> tuple[float, float, float, float]:
    """Bootstrap the per-item difference. Returns (diff, lo, hi, p_value).

    Resamples *item indices*, keeping each pair together, so the distribution
    being bootstrapped is the one that matters: the mean difference.

    The p-value is two-sided and computed as the fraction of bootstrap means on
    the wrong side of zero, doubled. It uses `+1` smoothing so that a resample
    that never crosses zero reports `p ~ 1/n_boot` rather than an impossible
    p = 0.
    """
    n = len(a)
    if n < 2:
        return (float(np.mean(a - b)) if n else 0.0, np.nan, np.nan, 1.0)

    diffs = a - b
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diffs[idx].mean(axis=1)

    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))

    n_wrong_side = (np.sum(boot <= 0) if observed > 0 else np.sum(boot >= 0))
    p = min(1.0, 2.0 * (n_wrong_side + 1) / (n_boot + 1))
    return observed, lo, hi, float(p)


def mcnemar(a: np.ndarray, b: np.ndarray,
            threshold: float = 0.5) -> tuple[int, int, float]:
    """Exact McNemar test for paired binary outcomes. Returns (b01, b10, p).

    The right test when the metric is pass/fail (accuracy with an exact or
    judge-binary scorer). It looks only at *discordant* pairs — items where one
    model was right and the other wrong — because items both got right and items
    both got wrong carry no information about which is better. On a 100-item set
    where the models differ on 6 questions, this correctly reports that you have
    6 data points, not 100.

    Uses the exact binomial test rather than the chi-square approximation, which
    is unreliable at exactly the small discordant counts that are typical here.
    """
    a_bin = a >= threshold
    b_bin = b >= threshold
    b01 = int(np.sum(~a_bin & b_bin))   # b right, a wrong
    b10 = int(np.sum(a_bin & ~b_bin))   # a right, b wrong
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0

    # Two-sided exact binomial against p=0.5.
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return b01, b10, float(min(1.0, 2.0 * tail))


def compare_models(df: pd.DataFrame, model_a: str, model_b: str,
                   metric: str = "accuracy", n_boot: int = 5000,
                   seed: int = 0, binary_threshold: float | None = None
                   ) -> ComparisonResult:
    """Full paired comparison of two models on one metric.

    Automatically switches to McNemar when the metric is actually binary — using
    a bootstrap on 0/1 data works but wastes information the exact test uses.
    """
    a, b = paired_values(df, model_a, model_b, metric)
    if len(a) == 0:
        return ComparisonResult(model_a, model_b, metric, 0, np.nan, np.nan,
                                np.nan, np.nan, np.nan, 1.0)

    diff, lo, hi, p = paired_bootstrap(a, b, n_boot=n_boot, seed=seed)
    test = "paired_bootstrap"

    values = np.concatenate([a, b])
    looks_binary = np.all(np.isin(values, (0.0, 1.0)))
    if looks_binary or binary_threshold is not None:
        _, _, p_mc = mcnemar(a, b, binary_threshold or 0.5)
        p = p_mc
        test = "mcnemar_exact"

    return ComparisonResult(
        model_a=model_a, model_b=model_b, metric=metric, n_pairs=len(a),
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
        diff=diff, ci_low=lo, ci_high=hi, p_value=p, test=test,
    )


# ---------------------------------------------------------------------------
#  Multiple comparisons
# ---------------------------------------------------------------------------
def holm_bonferroni(p_values: list[float], alpha: float = 0.05
                    ) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down correction. Returns (adjusted_p, significant).

    Comparing 5 models pairwise is 10 tests; at alpha=0.05 you expect one false
    "winner" by chance alone roughly 40% of the time. Correcting is not optional
    once you are ranking a matrix rather than testing one hypothesis.

    Holm rather than plain Bonferroni: uniformly more powerful, same guarantee
    on the family-wise error rate.
    """
    m = len(p_values)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p_values[i])
        running = max(running, val)  # enforce monotonicity down the sequence
        adjusted[i] = running
    return adjusted, [p <= alpha for p in adjusted]


def significance_matrix(df: pd.DataFrame, models: list[str] | None = None,
                        metric: str = "accuracy", alpha: float = 0.05,
                        n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Every pairwise comparison, corrected for multiplicity, with verdicts.

    This is the table that answers "which of these differences are real?" — the
    question a leaderboard alone cannot answer.
    """
    models = models or sorted(df["model"].dropna().unique())
    results: list[ComparisonResult] = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            results.append(compare_models(df, a, b, metric, n_boot, seed))

    usable = [r for r in results if r.n_pairs >= 2]
    if usable:
        adjusted, significant = holm_bonferroni([r.p_value for r in usable], alpha)
        for r, p_adj, sig in zip(usable, adjusted, significant):
            r.p_adjusted, r.significant = p_adj, sig
    for r in results:
        if r.significant is None:
            r.significant = False

    return pd.DataFrame([r.as_dict() for r in results])


# ---------------------------------------------------------------------------
#  Power / sample size
# ---------------------------------------------------------------------------
def required_n(effect: float, std: float, alpha: float = 0.05,
               power: float = 0.80) -> int:
    """Paired items needed to detect a difference of `effect` with `power`.

    Answers the question every wide confidence interval raises and the harness
    previously left hanging: *how many more questions do I need?* "Add more
    items" is not actionable; "you need about 340" is.

    Normal-approximation formula, adequate at the sample sizes where this
    question comes up.
    """
    if effect <= 0 or std <= 0:
        return 0
    z_alpha = 1.959963985  # two-sided 0.05
    z_beta = {0.80: 0.8416212336, 0.90: 1.2815515655,
              0.95: 1.6448536270}.get(round(power, 2), 0.8416212336)
    if abs(alpha - 0.05) > 1e-9:
        z_alpha = {0.01: 2.5758293035, 0.10: 1.6448536270}.get(round(alpha, 2),
                                                               1.959963985)
    n = ((z_alpha + z_beta) * std / effect) ** 2
    return int(math.ceil(n))


def minimum_detectable_effect(n: int, std: float, alpha: float = 0.05,
                              power: float = 0.80) -> float:
    """The smallest true difference this sample size can reliably detect.

    The honest headline for a small evalset: "with 40 items you cannot detect a
    gap smaller than 0.13, so every ranking inside that band is noise."
    """
    if n <= 0 or std <= 0:
        return float("inf")
    z_alpha = 1.959963985
    z_beta = {0.80: 0.8416212336, 0.90: 1.2815515655,
              0.95: 1.6448536270}.get(round(power, 2), 0.8416212336)
    return (z_alpha + z_beta) * std / math.sqrt(n)


@dataclass
class PowerReport:
    metric: str
    n_items: int
    observed_std: float
    mde: float
    suggestions: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.metric}: {self.n_items} paired items, "
                f"smallest reliably detectable gap ~{self.mde:.3f}")


def power_report(df: pd.DataFrame, metric: str = "accuracy",
                 targets: tuple[float, ...] = (0.10, 0.05, 0.02)) -> PowerReport:
    """How much resolution the current evalset actually has, and what it'd cost.

    `observed_std` is the standard deviation of the per-item *paired difference*
    where two or more models exist — the quantity the test actually operates on.
    Using the raw score spread instead would overstate the noise and demand a far
    larger evalset than necessary.
    """
    models = sorted(df["model"].dropna().unique())
    diffs: list[np.ndarray] = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            va, vb = paired_values(df, a, b, metric)
            if len(va) >= 2:
                diffs.append(va - vb)

    if diffs:
        pooled = np.concatenate(diffs)
        std = float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0
        n = int(np.median([len(d) for d in diffs]))
    elif metric in df.columns:
        vals = df[metric].dropna().to_numpy(float)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        n = len(vals)
    else:
        std, n = 0.0, 0

    return PowerReport(
        metric=metric, n_items=n, observed_std=std,
        mde=minimum_detectable_effect(n, std) if n and std else float("inf"),
        suggestions={f"detect_{t:g}": required_n(t, std) for t in targets if std},
    )
