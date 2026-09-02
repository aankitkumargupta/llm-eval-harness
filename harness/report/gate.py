"""
Regression gating: turning an evaluation into a CI check.

The harness could tell you which model is best *today*. It could not tell you
whether the prompt change you just made broke anything — and that is the
question a team asks far more often than "which model should we buy".

A gate compares a candidate run against a stored baseline and exits non-zero
when a guarded metric has regressed beyond tolerance. Drop it in a pipeline and
prompt changes, retrieval changes, and model upgrades all stop being acts of
faith.

Two design decisions worth stating:

**Significance, not just delta.** A naive gate fires on any drop, so on a small
evalset it fires constantly on noise, everyone learns to ignore it, and it
stops protecting anything. This gate requires a regression to be both larger
than the tolerance *and* statistically significant under a paired test before
it fails the build. `--strict` disables that and fails on raw delta, for teams
that would rather over-alert.

**Absolute floors as well as deltas.** "No worse than last time" lets quality
erode one tolerable step at a time. A floor (`accuracy>=0.8`) catches the slow
slide that a pure delta gate never notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .decide import METRIC_DIRECTION, Constraint
from .stats import compare_models, paired_values


@dataclass
class GateCheck:
    metric: str
    model: str
    kind: str                      # "delta" | "floor"
    baseline: float | None
    candidate: float | None
    delta: float | None
    tolerance: float
    passed: bool
    significant: bool | None = None
    p_value: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        if self.kind == "floor":
            return (f"[{status}] {self.model} {self.metric} "
                    f"{self.candidate:.4f} (floor {self.tolerance:.4f})")
        sig = ""
        if self.p_value is not None:
            sig = f", p={self.p_value:.3f}" + ("" if self.significant else ", n.s.")
        base = "n/a" if self.baseline is None else f"{self.baseline:.4f}"
        cand = "n/a" if self.candidate is None else f"{self.candidate:.4f}"
        delta = "n/a" if self.delta is None else f"{self.delta:+.4f}"
        return (f"[{status}] {self.model} {self.metric} "
                f"{base} -> {cand} ({delta}{sig})")


@dataclass
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def report(self) -> str:
        lines = [c.line() for c in self.checks]
        for s in self.skipped:
            lines.append(f"[SKIP] {s}")
        lines.append("")
        lines.append("GATE PASSED" if self.passed
                     else f"GATE FAILED ({len(self.failures)} regression(s))")
        return "\n".join(lines)

    def exit_code(self) -> int:
        return 0 if self.passed else 1


def check_regression(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    metrics: list[str],
    tolerance: float = 0.02,
    per_metric_tolerance: dict[str, float] | None = None,
    floors: list[Constraint] | None = None,
    require_significance: bool = True,
    alpha: float = 0.05,
    models: list[str] | None = None,
) -> GateResult:
    """Compare a candidate run against a baseline run, metric by metric, model by model.

    Both frames must come from the same profile and evalset for the paired test
    to be meaningful; comparing across evalsets measures the questions, not the
    change.
    """
    per_metric_tolerance = per_metric_tolerance or {}
    result = GateResult(passed=True)

    if baseline_df.empty:
        result.skipped.append("baseline run is empty - nothing to compare")
        return result
    if candidate_df.empty:
        result.passed = False
        result.skipped.append("candidate run is empty - treating as failure")
        return result

    shared_models = models or sorted(
        set(baseline_df["model"].dropna()) & set(candidate_df["model"].dropna()))
    if not shared_models:
        result.skipped.append(
            "no models in common between the two runs - nothing to compare")
        return result

    for metric in metrics:
        tol = per_metric_tolerance.get(metric, tolerance)
        lower_better = METRIC_DIRECTION.get(metric, "max") == "min"

        for model in shared_models:
            b = baseline_df[baseline_df["model"] == model]
            c = candidate_df[candidate_df["model"] == model]
            if metric not in b.columns or metric not in c.columns:
                result.skipped.append(f"{model}/{metric}: not present in both runs")
                continue
            if not (b[metric].notna().any() and c[metric].notna().any()):
                result.skipped.append(f"{model}/{metric}: no data in both runs")
                continue

            b_mean = float(b[metric].mean(skipna=True))
            c_mean = float(c[metric].mean(skipna=True))
            # Normalise so a positive delta always means "better", whichever
            # direction the metric runs. Without this, a cost *increase* reads
            # as an improvement.
            delta = (b_mean - c_mean) if lower_better else (c_mean - b_mean)

            significant: bool | None = None
            p_value: float | None = None
            if require_significance:
                merged = pd.concat([
                    b.assign(model="__baseline__"),
                    c.assign(model="__candidate__"),
                ])
                va, vb = paired_values(merged, "__candidate__", "__baseline__",
                                       metric)
                if len(va) >= 2:
                    cmp_ = compare_models(merged, "__candidate__",
                                          "__baseline__", metric)
                    p_value = cmp_.p_value
                    significant = p_value < alpha

            regressed = delta < -tol
            if regressed and require_significance and significant is False:
                passed, note = True, "within noise (not significant)"
            else:
                passed, note = (not regressed), ("" if not regressed
                                                 else "regression beyond tolerance")

            check = GateCheck(
                metric=metric, model=model, kind="delta", baseline=b_mean,
                candidate=c_mean, delta=delta, tolerance=tol, passed=passed,
                significant=significant, p_value=p_value, note=note)
            result.checks.append(check)
            if not passed:
                result.passed = False

    # -- absolute floors --------------------------------------------------- #
    for floor in (floors or []):
        for model in shared_models:
            c = candidate_df[candidate_df["model"] == model]
            if floor.metric not in c.columns or not c[floor.metric].notna().any():
                result.skipped.append(
                    f"{model}/{floor.metric}: floor not checkable (no data)")
                continue
            value = float(c[floor.metric].mean(skipna=True))
            ok = bool(floor.satisfied_by(value))
            result.checks.append(GateCheck(
                metric=floor.metric, model=model, kind="floor", baseline=None,
                candidate=value, delta=None, tolerance=floor.value, passed=ok,
                note="" if ok else f"below floor {floor.describe()}"))
            if not ok:
                result.passed = False

    return result


def budget_gate(candidate_df: pd.DataFrame, max_cost_per_query: float,
                models: list[str] | None = None) -> GateResult:
    """Fail when measured cost per query exceeds a ceiling.

    Quality regressions get caught by review; cost regressions do not, because
    nothing about a prompt that grew by 400 tokens looks wrong in a diff. This
    is the check that notices.
    """
    result = GateResult(passed=True)
    if candidate_df.empty or "cost_usd" not in candidate_df.columns:
        result.skipped.append("no cost data in candidate run")
        return result

    for model in (models or sorted(candidate_df["model"].dropna().unique())):
        g = candidate_df[candidate_df["model"] == model]
        if not g["cost_usd"].notna().any():
            continue
        value = float(g["cost_usd"].mean(skipna=True))
        ok = value <= max_cost_per_query
        result.checks.append(GateCheck(
            metric="cost_usd", model=model, kind="floor", baseline=None,
            candidate=value, delta=None, tolerance=max_cost_per_query,
            passed=ok, note="" if ok else "cost per query above ceiling"))
        if not ok:
            result.passed = False
    return result


def parse_gate_spec(specs: list[str]) -> tuple[list[str], dict[str, float]]:
    """Parse `--gate accuracy:0.02 --gate faithfulness:0.01` into metrics + tolerances.

    A bare metric name uses the global tolerance.
    """
    metrics: list[str] = []
    tolerances: dict[str, float] = {}
    for spec in specs:
        if ":" in spec:
            metric, tol = spec.split(":", 1)
            metric = metric.strip()
            tolerances[metric] = float(tol)
        else:
            metric = spec.strip()
        metrics.append(metric)
    return metrics, tolerances
