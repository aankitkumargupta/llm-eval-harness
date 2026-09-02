"""
Reporting layer. Reads TraceRows and computes the cross-model aggregates that
DON'T live on individual rows:

  weighted_composite  — per-profile leaderboard number from metric weights
  tuning_gain         — adapted minus baseline, per metric
  pareto_frontier     — non-dominated models on (accuracy, cost, latency)
  elo_from_pairwise   — Bradley-Terry / Elo ranking from pairwise judge results
  bootstrap_ci        — confidence intervals, so "A beats B" isn't just noise
  error_attribution   — *why* items failed, not just how many
  judge_calibration   — how far the LLM judge is from human labels

All functions operate on pandas DataFrames of trace rows, so they run entirely
offline against the cached store.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
#  Weighted composite (the single leaderboard number, per profile)
# ---------------------------------------------------------------------------
def weighted_composite(df: pd.DataFrame, weights: dict[str, float],
                       normalise: bool = False) -> pd.DataFrame:
    """Mean each weighted metric per model, then combine with the profile's weights.

    Cost and latency are lower-is-better, so their weights must be negative
    (the profile validator enforces this).

    `normalise=True` min-max scales each metric across models before weighting.
    Without it the composite is dominated by whichever metric has the largest
    raw range — latency in milliseconds sits in the thousands while accuracy
    sits in [0,1], so a 0.05 latency weight can silently outweigh a 0.5 accuracy
    weight by four orders of magnitude. Off by default to preserve the original
    behaviour and keep scores comparable across runs.
    """
    if df.empty or "model" not in df.columns:
        return pd.DataFrame()

    present = [m for m in weights if m in df.columns]
    ranges: dict[str, tuple[float, float]] = {}
    if normalise:
        for m in present:
            per_model = df.groupby("model")[m].mean(numeric_only=True)
            lo, hi = float(per_model.min()), float(per_model.max())
            ranges[m] = (lo, hi)

    out = []
    for model, g in df.groupby("model"):
        score = 0.0
        detail: dict = {}
        contributed = 0
        for metric, w in weights.items():
            if metric not in g.columns:
                continue
            val = g[metric].mean(skipna=True)
            if pd.isna(val):
                continue
            detail[metric] = float(val)
            scaled = float(val)
            if normalise and metric in ranges:
                lo, hi = ranges[metric]
                scaled = (scaled - lo) / (hi - lo) if hi > lo else 0.5
            score += w * scaled
            contributed += 1
        out.append({"model": model, "composite": score,
                    "metrics_used": contributed, **detail})

    result = pd.DataFrame(out)
    return result.sort_values("composite", ascending=False) if not result.empty \
        else result


# ---------------------------------------------------------------------------
#  Tuning gain (adapted - baseline)
# ---------------------------------------------------------------------------
def tuning_gain(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Per model: mean(metric | adapted) - mean(metric | baseline).

    Positive means the model benefited from tuning. This is the headline "how
    much does a model gain from tuning" result.
    """
    if df.empty or "pass_" not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, g in df.groupby("model"):
        base = g[g["pass_"] == "baseline"]
        adap = g[g["pass_"] == "adapted"]
        rec: dict = {"model": model, "n_baseline": len(base), "n_adapted": len(adap)}
        for m in metrics:
            if m not in g.columns:
                continue
            b = base[m].mean(skipna=True) if len(base) else np.nan
            a = adap[m].mean(skipna=True) if len(adap) else np.nan
            rec[f"{m}_gain"] = (a - b) if not (pd.isna(a) or pd.isna(b)) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Pareto frontier
# ---------------------------------------------------------------------------
def pareto_frontier(df: pd.DataFrame,
                    acc_col: str = "accuracy",
                    cost_col: str = "cost_usd",
                    lat_col: str = "latency_ms") -> pd.DataFrame:
    """Models not dominated on (accuracy up, cost down, latency down).

    Two fixes over the original:

    **NaN handling.** Every comparison against NaN is False, so a model missing
    one objective was never dominated and always landed on the frontier — the
    models with the *least* data looked the most attractive. Missing objectives
    are now imputed to the worst observed value, so a model can't win by not
    being measured.

    **O(n log n) instead of O(n²).** The original ran a full DataFrame scan
    inside `.apply`, re-iterating every row for every row.
    """
    needed = [c for c in (acc_col, cost_col, lat_col) if c in df.columns]
    if df.empty or not needed:
        return pd.DataFrame()

    agg = df.groupby("model").agg(
        accuracy=(acc_col, "mean") if acc_col in df.columns else ("model", "size"),
        cost=(cost_col, "mean") if cost_col in df.columns else ("model", "size"),
        latency=(lat_col, "mean") if lat_col in df.columns else ("model", "size"),
    ).reset_index()

    # A model with NO measured objective cannot be placed on the frontier at
    # all. Imputing it to the worst observed value isn't enough: when it is the
    # only other model, "worst observed" equals "best observed" and the ghost
    # ties its way onto the frontier.
    measurable = agg[["accuracy", "cost", "latency"]].notna().any(axis=1)

    # Partially-measured models ARE comparable, so impute their gaps to the
    # worst observed value — being unmeasured must never be an advantage.
    if agg["accuracy"].notna().any():
        agg["accuracy"] = agg["accuracy"].fillna(agg["accuracy"].min())
    if agg["cost"].notna().any():
        agg["cost"] = agg["cost"].fillna(agg["cost"].max())
    if agg["latency"].notna().any():
        agg["latency"] = agg["latency"].fillna(agg["latency"].max())
    agg = agg.fillna({"accuracy": 0.0, "cost": 0.0, "latency": 0.0})
    agg["measurable"] = measurable.values

    # Sort by cost then latency ascending, accuracy descending. Sweeping in that
    # order means a model is dominated exactly when some earlier model already
    # had accuracy >= its own.
    order = agg.sort_values(["cost", "latency", "accuracy"],
                            ascending=[True, True, False]).reset_index(drop=True)
    on_frontier = []
    best_acc = -np.inf
    best_lat = np.inf
    for _, row in order.iterrows():
        if not row["measurable"]:
            on_frontier.append(False)
            continue
        dominated = (row["accuracy"] <= best_acc) and (row["latency"] >= best_lat)
        on_frontier.append(not dominated)
        if not dominated:
            best_acc = max(best_acc, row["accuracy"])
            best_lat = min(best_lat, row["latency"])
    order["on_frontier"] = on_frontier
    return (order.drop(columns=["measurable"])
            .sort_values("accuracy", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
#  Elo / Bradley-Terry from pairwise comparisons
# ---------------------------------------------------------------------------
def elo_from_pairwise(pairwise: list[tuple[str, str, str]],
                      k: float = 32.0, base: float = 1500.0,
                      iterations: int = 20, seed: int = 0) -> pd.DataFrame:
    """Elo ratings from [(model_a, model_b, winner)] with winner in {A, B, tie}.

    Repeated shuffled passes reduce order sensitivity — a single pass over a
    fixed ordering lets whoever played first accumulate an artefact.
    """
    if not pairwise:
        return pd.DataFrame(columns=["model", "elo", "games", "wins", "losses",
                                     "ties", "win_rate"])

    models = sorted({m for a, b, _ in pairwise for m in (a, b)})
    ratings = dict.fromkeys(models, base)
    record = {m: {"games": 0, "wins": 0, "losses": 0, "ties": 0} for m in models}

    for a, b, w in pairwise:
        record[a]["games"] += 1
        record[b]["games"] += 1
        if w == "A":
            record[a]["wins"] += 1
            record[b]["losses"] += 1
        elif w == "B":
            record[b]["wins"] += 1
            record[a]["losses"] += 1
        else:
            record[a]["ties"] += 1
            record[b]["ties"] += 1

    rng = random.Random(seed)
    games = list(pairwise)
    for _ in range(iterations):
        rng.shuffle(games)
        for a, b, w in games:
            ea = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            eb = 1.0 - ea
            sa, sb = {"A": (1.0, 0.0), "B": (0.0, 1.0)}.get(w, (0.5, 0.5))
            ratings[a] += k * (sa - ea)
            ratings[b] += k * (sb - eb)

    rows = []
    for m in models:
        r = record[m]
        rows.append({
            "model": m, "elo": ratings[m], "games": r["games"],
            "wins": r["wins"], "losses": r["losses"], "ties": r["ties"],
            "win_rate": (r["wins"] + 0.5 * r["ties"]) / r["games"]
                        if r["games"] else 0.0,
        })
    return (pd.DataFrame(rows).sort_values("elo", ascending=False)
            .reset_index(drop=True))


def win_rate_matrix(pairwise: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Head-to-head win rates: cell [a][b] is a's win rate against b.

    More legible than a single Elo number when you want to know whether the
    top model is actually better than the runner-up or merely beat up on a
    weak third model.
    """
    models = sorted({m for a, b, _ in pairwise for m in (a, b)})
    wins = {a: dict.fromkeys(models, 0.0) for a in models}
    played = {a: dict.fromkeys(models, 0) for a in models}
    for a, b, w in pairwise:
        played[a][b] += 1
        played[b][a] += 1
        if w == "A":
            wins[a][b] += 1
        elif w == "B":
            wins[b][a] += 1
        else:
            wins[a][b] += 0.5
            wins[b][a] += 0.5
    data = {a: {b: (wins[a][b] / played[a][b] if played[a][b] else np.nan)
                for b in models} for a in models}
    return pd.DataFrame(data).T


# ---------------------------------------------------------------------------
#  Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(values, n_boot: int = 2000, ci: float = 0.95,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap. Returns (mean, lo, hi).

    Turns "Model A scores 0.71" into "A: 0.71 [0.68, 0.74]". For *comparing two
    models*, use `stats.compare_models` instead — these marginal intervals
    ignore that both models saw the same items, and overlapping intervals do
    not mean the difference is insignificant.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan, np.nan)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))

    rng = np.random.default_rng(seed)
    # Vectorised: the original built each resample in a Python loop, which is
    # ~50x slower and gets called once per model per metric in the UI.
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = arr[idx].mean(axis=1)
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return (float(arr.mean()), lo, hi)


def ci_table(df: pd.DataFrame, metric: str, n_boot: int = 2000,
             seed: int = 0) -> pd.DataFrame:
    """Per-model mean with a 95% CI for one metric."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, g in df.groupby("model"):
        vals = g[metric].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        if not np.isnan(mean):
            rows.append({"model": model, "mean": mean, "ci_low": lo,
                         "ci_high": hi, "n": int(np.sum(~np.isnan(vals)))})
    return (pd.DataFrame(rows).sort_values("mean", ascending=False)
            if rows else pd.DataFrame())


# ---------------------------------------------------------------------------
#  Error attribution
# ---------------------------------------------------------------------------
def error_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: failure rate broken down by cause.

    "8% of items failed" is not actionable. "8% failed, all context_length"
    means lower your `k`; "all rate_limit" means lower your concurrency; "all
    content_filter" is a finding about the model. Same number, three different
    responses — which is why the taxonomy exists.
    """
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, g in df.groupby("model"):
        failed = g[g["error"].notna()]
        rec = {"model": model, "n": len(g), "errors": len(failed),
               "error_rate": len(failed) / len(g) if len(g) else 0.0}
        if "error_kind" in g.columns and len(failed):
            for kind, count in failed["error_kind"].value_counts().items():
                rec[f"kind_{kind}"] = int(count)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False)


def truncation_report(df: pd.DataFrame) -> pd.DataFrame:
    """How often each model's answer was cut off by `max_tokens`.

    A high rate invalidates completeness and citation scores for that model —
    it was measured mid-sentence. This is a config problem masquerading as a
    quality finding, and without this column it looks exactly like one.
    """
    if df.empty or "truncated" not in df.columns:
        return pd.DataFrame()
    rows = []
    for model, g in df.groupby("model"):
        vals = g["truncated"].dropna()
        if len(vals):
            rows.append({"model": model, "n": len(vals),
                         "truncated": int(vals.sum()),
                         "truncation_rate": float(vals.mean())})
    return (pd.DataFrame(rows).sort_values("truncation_rate", ascending=False)
            if rows else pd.DataFrame())


# ---------------------------------------------------------------------------
#  Judge calibration
# ---------------------------------------------------------------------------
def judge_calibration(df: pd.DataFrame, human_col: str = "human_label",
                      judge_col: str = "accuracy",
                      threshold: float = 0.5) -> dict:
    """How well the LLM judge agrees with human labels.

    Every judge-scored number in this harness rests on an unexamined assumption:
    that the judge agrees with a person. This measures it on whatever subset of
    items carries a human label.

    Cohen's kappa rather than raw agreement, because raw agreement is inflated
    by the base rate — a judge that marks everything correct scores 90%
    agreement on a set that's 90% correct while carrying no information at all.
    Kappa corrects for chance and would report ~0 there.
    """
    if df.empty or human_col not in df.columns or judge_col not in df.columns:
        return {"n": 0, "note": "no human labels available"}

    pairs = df[[human_col, judge_col]].dropna()
    if pairs.empty:
        return {"n": 0, "note": "no items carry both a human and a judge score"}

    human = pairs[human_col].to_numpy(float)
    judge = pairs[judge_col].to_numpy(float)
    n = len(pairs)

    h_bin = human >= threshold
    j_bin = judge >= threshold
    agreement = float(np.mean(h_bin == j_bin))

    p_h, p_j = float(np.mean(h_bin)), float(np.mean(j_bin))
    p_chance = p_h * p_j + (1 - p_h) * (1 - p_j)
    kappa = ((agreement - p_chance) / (1 - p_chance)
             if abs(1 - p_chance) > 1e-12 else 0.0)

    corr = float(np.corrcoef(human, judge)[0, 1]) if n > 1 and \
        np.std(human) > 0 and np.std(judge) > 0 else np.nan

    return {
        "n": n,
        "agreement": agreement,
        "cohens_kappa": float(kappa),
        "correlation": corr,
        # Positive bias = the judge is more generous than the humans, which is
        # the usual direction and the reason judge scores read optimistically.
        "judge_bias": float(np.mean(judge - human)),
        "mean_absolute_error": float(np.mean(np.abs(judge - human))),
        "interpretation": _kappa_label(kappa),
    }


def _kappa_label(kappa: float) -> str:
    """Landis & Koch bands, so the number comes with a verdict."""
    if kappa < 0.0:
        return "worse than chance - the judge is not usable as configured"
    if kappa < 0.20:
        return "slight agreement - do not trust judge-scored metrics"
    if kappa < 0.40:
        return "fair agreement - judge scores are weak evidence"
    if kappa < 0.60:
        return "moderate agreement - usable with caution"
    if kappa < 0.80:
        return "substantial agreement - judge scores are reliable"
    return "almost perfect agreement"


# ---------------------------------------------------------------------------
#  Consistency (paraphrase probes)
# ---------------------------------------------------------------------------
def consistency_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: how stable answers are across paraphrases of the same question.

    Reads the `consistency_group` column that paraphrase probes populate. A
    model scoring well on accuracy but poorly here answers correctly only when
    asked in exactly the right words — which is not a system you can ship to
    users who phrase things however they like.
    """
    from ..eval.metrics import token_f1

    if df.empty or "consistency_group" not in df.columns:
        return pd.DataFrame()
    grouped = df.dropna(subset=["consistency_group"])
    if grouped.empty:
        return pd.DataFrame()

    rows = []
    for model, g in grouped.groupby("model"):
        scores = []
        for _, family in g.groupby("consistency_group"):
            answers = [str(a) for a in family["raw_output"].dropna() if str(a).strip()]
            if len(answers) >= 2:
                pairs = [token_f1(answers[i], answers[j])
                         for i in range(len(answers))
                         for j in range(i + 1, len(answers))]
                if pairs:
                    scores.append(sum(pairs) / len(pairs))
        if scores:
            rows.append({"model": model, "n_groups": len(scores),
                         "consistency": float(np.mean(scores))})
    return (pd.DataFrame(rows).sort_values("consistency", ascending=False)
            if rows else pd.DataFrame())
