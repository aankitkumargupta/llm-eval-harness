"""
The profile half of the dashboard.

`api.py` covers the §10 benchmark subsystem. This module covers the original
apparatus, profiles, the weighted composite, the Pareto frontier, the decision
layer, the CI gate, so that one dashboard presents both.

**They belong in one UI, not two.** A benchmark run and a profile run differ
only in where the items came from: they share the trace store, the cost meter,
the scorer registry and the significance implementation, which is exactly what
§10 insisted on ("no second trace format, no second cost meter, no second
significance implementation"). Shipping two separate dashboards would quietly
assert the opposite, that these are different systems producing numbers of
different kinds, and that would be the most misleading thing in the product.

Same rule as `api.py`: presentation holds no business logic. Every table here
comes from `harness.report.*`. This module turns DataFrames into JSON and
nothing else. A metric computed in this file would be a second definition of
something the library already owns (§14.3).
"""

from __future__ import annotations

from pathlib import Path

from .api import ApiError, _records


def list_profiles(search_dir: str = "configs/profiles") -> dict:
    """Profiles on disk, including the ones that would fail to load.

    An invalid profile is listed with its error rather than filtered out: a
    profile that silently vanishes from the picker is harder to debug than one
    that says what is wrong with it.
    """
    from ..profiles.profile import Profile

    out = []
    for path in sorted(Path(search_dir).glob("*.yaml")):
        try:
            p = Profile.from_yaml(str(path))
        except Exception as e:      # noqa: BLE001 - listed, not raised
            # Deliberately broad. `ProfileError` alone is not enough: a bad
            # `task:` value surfaces as a bare `ValueError` from the enum, and
            # a truncated file as a YAML error. Letting any of those escape
            # would make one malformed profile 500 the endpoint and blank the
            # whole dashboard, for a screen whose entire job is to say which
            # profiles are broken.
            out.append({"name": path.stem, "invalid": f"{type(e).__name__}: {e}"})
            continue
        # Whether the data is actually on disk. Four of the six shipped
        # profiles are configured examples whose datasets are not in the repo,
        # and a picker that offers them without saying so opens every screen
        # on "file not found", which reads as a broken app rather than an
        # unconfigured profile.
        evalset_ok = bool(p.evalset_path) and Path(p.evalset_path).exists()
        corpus_ok = (not p.corpus_path) or Path(p.corpus_path).exists()
        out.append({
            "name": p.name,
            "task": p.task.value,
            "evalset_exists": evalset_ok,
            "corpus_exists": corpus_ok,
            "runnable": evalset_ok and corpus_ok,
            "description": " ".join((p.description or "").split()),
            "evalset_path": p.evalset_path,
            "corpus_path": p.corpus_path,
            "embedding_model": p.embedding_model,
            "accuracy_scorer": p.accuracy_scorer,
            "active_metrics": list(p.active_metrics),
            "metric_weights": dict(p.metric_weights),
            "max_tokens": p.max_tokens,
            "tuning_budget": p.tuning_budget,
        })
    return {"profiles": out}


def _aborted(run_dir: Path) -> bool:
    import json

    f = run_dir / "manifest.json"
    if not f.exists():
        return False
    try:
        return bool(json.loads(f.read_text(encoding="utf-8")).get("aborted"))
    except (ValueError, OSError):
        return False


def all_runs(store_path: str) -> dict:
    """Every run in the store, benchmark and profile alike, newest first.

    One list on purpose. "Which runs do I have?" is a single question, and
    answering it in two places is how a user comes to believe the two halves
    are separate systems.
    """
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        return {"runs": []}
    df = store.load_all()
    if df.empty:
        return {"runs": []}

    has_bench = "benchmark" in df.columns
    out = []
    for run_id, g in df.groupby("run_id"):
        bench = (str(g["benchmark"].dropna().iloc[0])
                 if has_bench and g["benchmark"].notna().any() else "")
        subject = bench or (str(g["profile"].dropna().iloc[0])
                            if g["profile"].notna().any() else "?")
        out.append({
            "run_id": str(run_id),
            "kind": "benchmark" if bench else "profile",
            "subject": subject,
            "models": sorted({str(m) for m in g["model"].dropna()}),
            "rows": int(len(g)),
            "errors": int(g["error"].notna().sum()) if "error" in g.columns else 0,
            "passes": (sorted({str(p) for p in g["pass_"].dropna()})
                       if "pass_" in g.columns else []),
            "cost_usd": (float(g["cost_usd"].fillna(0).sum())
                         if "cost_usd" in g.columns else 0.0),
            "ts": float(g["ts"].max()) if "ts" in g.columns else 0.0,
            "has_manifest": (Path(store_path).parent / str(run_id)
                             / "manifest.json").exists(),
            # A run stopped by the operator or the budget must not look like
            # one that finished. Read from the manifest, which is where the
            # runner records it; absent a manifest the answer is unknown,
            # reported as False rather than guessed.
            "aborted": _aborted(Path(store_path).parent / str(run_id)),
        })
    out.sort(key=lambda r: r["ts"], reverse=True)
    return {"runs": out}


def profile_results(store_path: str, *, profile: str = "", run_id: str = "",
                    metric: str = "accuracy",
                    allow_unpaired: bool = False) -> dict:
    """The profile report, as the CLI prints it.

    Leaderboard, Pareto frontier, bootstrap CIs, paired significance, power,
    tuning gain, error attribution, truncation and the estimated-usage rate.
    """
    from ..profiles.profile import Profile, ProfileError
    from ..report import aggregate as A
    from ..report import stats as S
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet. Run a profile or a benchmark first.",
                       404)

    df = store.load_run(run_id) if run_id else store.load_profile(profile)
    if df.empty:
        raise ApiError(
            f"No rows for {('run ' + run_id) if run_id else repr(profile)}.", 404)

    # A profile is not a run. Pooling every run of a profile averaged a
    # retrieval-corrupted attempt with a clean one and reported 0.69 where the
    # clean run said 1.00, and it duplicates item ids across runs, which the
    # paired test joins on. Runs differ in apparatus and dataset hashes (I2,
    # I9) and are not poolable. So a profile resolves to its NEWEST run, and
    # the others are named so the reader can pick one deliberately.
    other_runs: list[str] = []
    if not run_id and "run_id" in df.columns and df["run_id"].nunique() > 1:
        latest = (df.groupby("run_id")["ts"].max().idxmax()
                  if "ts" in df.columns else sorted(df["run_id"].unique())[-1])
        other_runs = sorted(str(r) for r in df["run_id"].unique() if r != latest)
        df = df[df["run_id"] == latest]
        run_id = str(latest)

    name = profile or (str(df["profile"].dropna().iloc[0])
                       if df["profile"].notna().any() else "")

    weights: dict = {}
    active: list[str] = []
    try:
        p = Profile.from_yaml(f"configs/profiles/{name}.yaml")
        weights, active = dict(p.metric_weights), list(p.active_metrics)
    except (ProfileError, FileNotFoundError, OSError):
        pass

    quality = (df[df["pass_"].isin(["baseline", "adapted"])]
               if "pass_" in df.columns else df)
    unpaired = "drop" if allow_unpaired else "raise"

    def safe(fn, *a, **kw):
        """Run one report section, reporting its failure rather than losing it.

        A single section raising must not blank the page. An
        `UnpairedItemsError` from the significance table is a *finding* about
        the run, and the leaderboard beside it is still worth reading, so the
        error is shown in place of that table, not instead of the report.
        """
        try:
            return fn(*a, **kw), ""
        except Exception as e:                      # noqa: BLE001 - surfaced
            return None, f"{type(e).__name__}: {e}"

    composite, composite_err = (safe(A.weighted_composite, quality, weights)
                                if weights else (None, ""))
    frontier, _ = safe(A.pareto_frontier, df)
    ci, _ = safe(A.ci_table, quality, metric)
    sig, sig_err = safe(S.significance_matrix, quality, metric=metric,
                        on_unpaired=unpaired)
    power_obj, _ = safe(S.power_report, quality, metric, on_unpaired=unpaired)
    errors, _ = safe(A.error_attribution, df)
    trunc, _ = safe(A.truncation_report, df)
    estimated, _ = safe(A.estimated_usage_report, df)
    cpc, _ = safe(A.cost_per_correct, quality, metric)

    gain = None
    if "pass_" in df.columns and {"baseline", "adapted"} <= set(df["pass_"].unique()):
        gain, _ = safe(A.tuning_gain, df,
                       [m for m in active if m in df.columns] or [metric])

    return {
        "profile": name,
        "run_id": run_id,
        "other_runs": other_runs,
        "metric": metric,
        "n_rows": int(len(df)),
        "models": sorted({str(m) for m in df["model"].dropna()}),
        "weights": weights,
        "active_metrics": active,
        "composite": _maybe(composite),
        "composite_error": composite_err,
        "pareto": _maybe(frontier),
        "ci": _maybe(ci),
        "significance": _maybe(sig),
        "significance_error": sig_err,
        "power": power_obj.summary() if power_obj is not None else "",
        "power_suggestions": (dict(power_obj.suggestions)
                              if power_obj is not None else {}),
        "tuning_gain": _maybe(gain),
        "error_attribution": _maybe(errors),
        "truncation": _maybe(trunc),
        "estimated_usage": _maybe(estimated),
        "cost_per_correct": _maybe(cpc),
    }


def decide(store_path: str, *, profile: str = "", run_id: str = "",
           require: list[str] | None = None, optimise: str = "cost_usd",
           quality_metric: str = "accuracy", qpd: int = 10_000) -> dict:
    """Which model to actually buy, at what volume.

    §1's fourth silent failure is answering the wrong question: "highest
    score" instead of "cheapest model clearing our bar". Nobody gets to make
    the first decision, so the second one belongs on the dashboard rather than
    only in the CLI.
    """
    from ..report.decide import explain, headroom, project_cost, select
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet.", 404)
    df = store.load_run(run_id) if run_id else store.load_profile(profile)
    if df.empty:
        raise ApiError("No rows for that selection.", 404)

    constraints = []
    for spec in (require or []):
        try:
            constraints.append(parse_constraint(spec))
        except ValueError as e:
            raise ApiError(str(e), 400) from e

    shortlist, candidates = select(df, constraints, optimise=optimise)
    proj = project_cost(df, queries_per_day=qpd)
    head = headroom(df, quality_metric=quality_metric, queries_per_day=qpd)

    return {
        "profile": profile,
        "run_id": run_id,
        "constraints": [f"{c.metric} {c.op} {c.value:g}" for c in constraints],
        "optimise": optimise,
        "qpd": qpd,
        "qualified": 0 if shortlist is None or shortlist.empty else int(len(shortlist)),
        "considered": len(candidates),
        "shortlist": _maybe(shortlist) or [],
        "recommended": ("" if shortlist is None or shortlist.empty
                        else str(shortlist.iloc[0]["model"])),
        # The important half when the shortlist is empty: which constraint each
        # model failed, and by how much. "No model qualifies" is not actionable;
        # "everything failed latency, relax it to 2.4s" is.
        "explanation": explain(candidates, constraints),
        "projection": _maybe(proj) or [],
        "headroom": _maybe(head) or [],
    }


def gate(store_path: str, *, baseline: str, candidate: str,
         metrics: list[str] | None = None, tolerance: float = 0.02,
         strict: bool = False, allow_unpaired: bool = False) -> dict:
    """The CI regression gate, run from the browser.

    Worth surfacing because the gate's *reason* is the interesting part:
    "within noise (not significant)" is what stops a team from tuning the
    tolerance until the build goes green.
    """
    from ..report.gate import check_regression
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet.", 404)

    b, c = store.load_run(baseline), store.load_run(candidate)
    if b.empty:
        raise ApiError(f"No rows for baseline run {baseline!r}.", 404)
    if c.empty:
        raise ApiError(f"No rows for candidate run {candidate!r}.", 404)

    try:
        result = check_regression(
            b, c, metrics=list(metrics or ["accuracy"]), tolerance=tolerance,
            require_significance=not strict,
            on_unpaired="drop" if allow_unpaired else "raise")
    except Exception as e:                          # noqa: BLE001 - a finding
        # An UnpairedItemsError here means the two runs did not score the same
        # items, which is a real reason not to gate on them.
        raise ApiError(f"{type(e).__name__}: {e}", 400) from e

    return {
        "baseline": baseline,
        "candidate": candidate,
        "passed": bool(result.passed),
        "exit_code": int(result.exit_code()),
        "report": result.report(),
        "checks": [
            {"model": getattr(ch, "model", ""),
             "metric": getattr(ch, "metric", ""),
             "baseline": _num(getattr(ch, "baseline", None)),
             "candidate": _num(getattr(ch, "candidate", None)),
             "delta": _num(getattr(ch, "delta", None)),
             "passed": bool(getattr(ch, "passed", False)),
             "note": getattr(ch, "note", ""),
             "significant": getattr(ch, "significant", None),
             "p_value": _num(getattr(ch, "p_value", None))}
            for ch in getattr(result, "checks", [])
        ],
    }


def parse_constraint(spec: str):
    """`"faithfulness>=0.90"` -> Constraint. Same shape as the CLI's flag."""
    from ..report.decide import Constraint

    for op in (">=", "<=", "==", ">", "<"):
        if op in spec:
            metric, _, value = spec.partition(op)
            try:
                return Constraint(metric.strip(), op, float(value))
            except ValueError as e:
                raise ValueError(
                    f"Bad constraint {spec!r}: {value!r} is not a number.") from e
    raise ValueError(
        f"Bad constraint {spec!r}. Expected something like 'accuracy>=0.9'.")


def _maybe(df):
    """Records, or None for an absent or empty table."""
    return None if df is None or getattr(df, "empty", True) else _records(df)


def _num(v):
    import pandas as pd

    if v is None:
        return None
    try:
        return None if pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None
