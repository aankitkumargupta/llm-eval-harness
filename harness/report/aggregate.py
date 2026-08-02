"""
Reporting layer. Reads TraceRows (via the store) and computes the cross-model
aggregates that DON'T live on individual rows:

  weighted_composite  — per-profile leaderboard number from metric weights
  tuning_gain         — adapted-pass minus baseline-pass, per metric
  pareto_frontier     — non-dominated models on (accuracy, cost, latency)
  elo_from_pairwise   — Bradley-Terry / Elo ranking from pairwise judge results
  bootstrap_ci        — confidence intervals so "A beats B" isn't just noise

All functions operate on pandas DataFrames of trace rows, so they run entirely
offline against the cached store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Weighted composite (the single leaderboard number, per profile)
# ---------------------------------------------------------------------------
def weighted_composite(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """
    Mean each weighted metric per model, then combine with the profile's weights.
    Cost and latency are 'lower is better', so weights for them should be NEGATIVE
    in the profile config (the caller decides sign). Metrics absent from a row are
    ignored via NaN-aware means.
    """
    grouped = df.groupby("model")
    out = []
    for model, g in grouped:
        score = 0.0
        detail = {}
        for metric, w in weights.items():
            if metric in g.columns:
                val = g[metric].mean(skipna=True)
                if not np.isnan(val):
                    score += w * val
                    detail[metric] = val
        out.append({"model": model, "composite": score, **detail})
    return pd.DataFrame(out).sort_values("composite", ascending=False)


# ---------------------------------------------------------------------------
#  Tuning gain (adapted - baseline)
# ---------------------------------------------------------------------------
def tuning_gain(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """
    For each model, mean(metric | adapted) - mean(metric | baseline). Positive
    means the model benefited from tuning. This is the headline 'how much does a
    model gain from tuning' result.
    """
    rows = []
    for model, g in df.groupby("model"):
        base = g[g["pass_"] == "baseline"]
        adap = g[g["pass_"] == "adapted"]
        rec = {"model": model}
        for m in metrics:
            if m in g.columns:
                b = base[m].mean(skipna=True)
                a = adap[m].mean(skipna=True)
                rec[f"{m}_gain"] = (a - b) if not (np.isnan(a) or np.isnan(b)) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Pareto frontier on (accuracy up, cost down, latency down)
# ---------------------------------------------------------------------------
def pareto_frontier(df: pd.DataFrame,
                    acc_col: str = "accuracy",
                    cost_col: str = "cost_usd",
                    lat_col: str = "latency_ms") -> pd.DataFrame:
    """
    Aggregate to per-model means, then mark models NOT dominated on all three
    objectives (higher accuracy, lower cost, lower latency). A model is dominated
    if another is at least as good on every objective and strictly better on one.
    """
    agg = df.groupby("model").agg(
        accuracy=(acc_col, "mean"),
        cost=(cost_col, "mean"),
        latency=(lat_col, "mean"),
    ).reset_index()

    def dominated(row) -> bool:
        for _, other in agg.iterrows():
            if other["model"] == row["model"]:
                continue
            at_least = (other["accuracy"] >= row["accuracy"] and
                        other["cost"] <= row["cost"] and
                        other["latency"] <= row["latency"])
            strictly = (other["accuracy"] > row["accuracy"] or
                        other["cost"] < row["cost"] or
                        other["latency"] < row["latency"])
            if at_least and strictly:
                return True
        return False

    agg["on_frontier"] = ~agg.apply(dominated, axis=1)
    return agg.sort_values("accuracy", ascending=False)


# ---------------------------------------------------------------------------
#  Elo / Bradley-Terry from pairwise comparisons
# ---------------------------------------------------------------------------
def elo_from_pairwise(pairwise: list[tuple[str, str, str]],
                      k: float = 32.0, base: float = 1500.0,
                      iterations: int = 20, seed: int = 0) -> pd.DataFrame:
    """
    pairwise: list of (model_a, model_b, winner) where winner in {"A","B","tie"}.
    Runs repeated Elo updates over shuffled comparisons to reduce order
    sensitivity. Returns a rating table, highest first.
    """
    import random
    models = sorted({m for a, b, _ in pairwise for m in (a, b)})
    ratings = {m: base for m in models}
    rng = random.Random(seed)
    games = list(pairwise)
    for _ in range(iterations):
        rng.shuffle(games)
        for a, b, w in games:
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            eb = 1.0 - ea
            if w == "A":
                sa, sb = 1.0, 0.0
            elif w == "B":
                sa, sb = 0.0, 1.0
            else:
                sa, sb = 0.5, 0.5
            ratings[a] += k * (sa - ea)
            ratings[b] += k * (sb - eb)
    return (pd.DataFrame({"model": list(ratings), "elo": list(ratings.values())})
            .sort_values("elo", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
#  Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """
    Returns (mean, lo, hi) for a metric via percentile bootstrap. This is what
    turns 'Model A scores 0.71' into 'A: 0.71 [0.68, 0.74]', so a 1-point gap can
    be judged against its uncertainty.
    """
    values = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boots, (1 - ci) / 2 * 100)
    hi = np.percentile(boots, (1 + ci) / 2 * 100)
    return (float(values.mean()), float(lo), float(hi))
