"""
The JSON API, as plain functions.

Separated from the HTTP plumbing in `server.py` so the API is testable without
a socket, which matters here more than usual, because §5 blocks sockets for
the whole test suite. Every function below takes plain data and returns plain
data, so the tests exercise the same code a browser reaches.

§3's layering rule applies: this is presentation, so it holds **no business
logic**. Every number comes from `harness.bench.*`, `harness.report.stats` or
the trace store. Where you see arithmetic here it is formatting, turning a
DataFrame into JSON, never computing a metric. A metric computed in this file
would be a second definition of something, which is §14.3's failure mode.

One deliberate asymmetry: reads are cheap and synchronous, runs are not. A run
returns a job id immediately and the browser polls, because a benchmark over a
real provider takes minutes and an HTTP request that blocks for minutes is a
request that times out halfway through something you are paying for.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

from ..bench import registry as bench_registry
from ..bench.extract import EXTRACTORS, run_chain
from ..bench.spec import BenchmarkSpec, SpecError, available_specs, load_spec

#: Local models, always offered. The point of a deterministic offline provider
#: is that the run path can be tried before any key exists; requiring config to
#: see it would defeat that.
FAKE_MODELS = ["fake:alpha", "fake:beta", "fake:gamma"]


class ApiError(Exception):
    """A client error, rendered as a JSON body with an HTTP status."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------- #
#  Reads
# --------------------------------------------------------------------------- #
def _spec_json(spec: BenchmarkSpec) -> dict:
    """A spec as the UI needs it.

    Everything in the `hashed` block feeds `spec_hash`. The UI groups them
    under that heading deliberately: a reader should be able to see, without
    reading the YAML, exactly which knobs make two runs incomparable.
    """
    fs = spec.prompt.few_shot
    return {
        "id": spec.id,
        "version": spec.version,
        "family": spec.family,
        "task": spec.task,
        "spec_hash": spec.spec_hash(),
        "licence": spec.source.licence,
        "commercial_use": spec.source.commercial_use,
        "needs_licence_ack": spec.requires_licence_ack(),
        "citation": spec.source.citation,
        "source_kind": spec.source.kind,
        "source_ref": spec.source.ref,
        "hashed": {
            "mode": spec.scoring.mode,
            "chance_level": spec.scoring.chance_level,
            "few_shot_n": fs.n,
            "few_shot_pool": fs.pool_split,
            "few_shot_selection": fs.selection,
            "max_tokens": spec.prompt.decoding.get("max_tokens"),
            "temperature": spec.prompt.decoding.get("temperature"),
            "samples_per_item": spec.sampling.samples_per_item,
            "seed": spec.sampling.seed,
            "stratify_by": spec.sampling.stratify_by,
        },
        "chain": [dict(link or {}) for link in spec.scoring.chain],
        "primary_metric": spec.primary_metric,
        "also_report": list(spec.also_report),
        "problems": spec.validate(),
    }


def list_benchmarks() -> dict:
    """Every spec on disk, with whether an adapter exists for it.

    A spec without an adapter is a half-built benchmark that fails only when
    someone selects it, so it is reported rather than filtered out.
    """
    out = []
    for path in available_specs():
        try:
            spec = BenchmarkSpec.from_yaml(path)
        except SpecError as e:
            out.append({"id": path.stem, "invalid": str(e), "has_adapter": False})
            continue
        row = _spec_json(spec)
        row["has_adapter"] = bench_registry.has_adapter(spec)
        out.append(row)
    return {"benchmarks": out}


def list_models(models_cfg: dict) -> dict:
    """Local models first, then whatever models.yaml configures."""
    configured = [m for m in (models_cfg.get("models") or []) if m]
    return {
        "local": FAKE_MODELS,
        "configured": configured,
        "note": ("fake: models run locally with no key and no network. They "
                 "exercise the whole run path; their scores are not results."),
    }


def extract_preview(benchmark: str, raw: str) -> dict:
    """Run a spec's extraction chain against arbitrary text.

    Pure and instant, no provider, no store. It exists because extraction is
    quietly the whole benchmark: the gap between a reported 0.61 and a reported
    0.78 is usually not the model, it is whether the harness could find the
    answer in the prose. Being able to see that before a run is the cheapest
    bug-catch in the system.
    """
    spec = _require_spec(benchmark)
    got = run_chain(raw, spec.scoring.chain)

    # What every extractor would return independently, so a reader can see
    # which link to add and in what order.
    all_rows = []
    in_chain = {(link or {}).get("kind") for link in spec.scoring.chain}
    for name, fn in EXTRACTORS.items():
        link: dict[str, Any] = {"kind": name}
        if name == "label_set":
            link["labels"] = list(spec.label_set) or ["yes", "no"]
        if name == "regex":
            # Borrow the spec's own first regex, or the link is meaningless.
            first = next((dict(c) for c in spec.scoring.chain
                          if (c or {}).get("kind") == "regex"), None)
            if first is None:
                all_rows.append({"extractor": name, "result": None,
                                 "in_chain": name in in_chain,
                                 "note": "needs a pattern"})
                continue
            link = first
        try:
            value = fn(raw, link)
        except Exception as e:                      # noqa: BLE001 - shown, not hidden
            value, note = None, f"{type(e).__name__}: {e}"
        else:
            note = ""
        all_rows.append({"extractor": name, "result": value,
                         "in_chain": name in in_chain, "note": note})

    return {
        "benchmark": spec.id,
        "chain": [dict(link or {}) for link in spec.scoring.chain],
        "ok": got.ok,
        "value": got.value,
        "via": got.via,
        "reason": got.reason,
        "consequence": (
            "Scores 1.0 or 0.0 against the gold answer."
            if got.ok else
            "Scores accuracy=None and counts toward extraction_failure_rate. "
            "It is NOT counted as a wrong answer (I7)."),
        "extractors": all_rows,
    }


def list_runs(store_path: str) -> dict:
    """Benchmark runs in the trace store, newest first."""
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        return {"runs": []}
    df = store.load_all()
    if "benchmark" not in df.columns:
        return {"runs": []}
    df = df[df["benchmark"].notna()]
    if df.empty:
        return {"runs": []}

    runs = []
    for run_id, g in df.groupby("run_id"):
        runs.append({
            "run_id": str(run_id),
            "benchmark": str(g["benchmark"].iloc[0]),
            "models": sorted({str(m) for m in g["model"].dropna()}),
            "rows": int(len(g)),
            "ts": float(g["ts"].max()) if "ts" in g.columns else 0.0,
            "spec_hashes": sorted({str(h) for h in g["spec_hash"].dropna()}),
        })
    runs.sort(key=lambda r: r["ts"], reverse=True)
    return {"runs": runs}


def results(store_path: str, run_id: str) -> dict:
    """Leaderboard, failure rates and the paired significance test.

    All three come from the existing library, §10 forbids a second trace
    format, a second cost meter and a second significance implementation, and
    this is where that rule would be easiest to break.
    """
    from ..bench.metrics import summarise
    from ..report import stats as S
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet. Run a benchmark first.", 404)

    g = store.load_run(run_id)
    if g.empty:
        raise ApiError(f"No rows for run {run_id!r}.", 404)
    if "benchmark" not in g.columns or g["benchmark"].isna().all():
        raise ApiError(f"Run {run_id!r} is a profile run, not a benchmark run.", 400)

    bid = str(g["benchmark"].dropna().iloc[0])

    # §10.1: never mix formats in one table. Two runs of the "same" benchmark
    # under different prompt formats are not the same benchmark, and averaging
    # them produces a number that describes neither.
    hashes = sorted({str(h) for h in g["spec_hash"].dropna()})
    if len(hashes) > 1:
        return {
            "run_id": run_id, "benchmark": bid, "refused": True,
            "spec_hashes": hashes,
            "reason": (f"These rows span {len(hashes)} different spec_hash "
                       f"values. Prompt format, few-shot count or extraction "
                       f"differs, so the numbers are not comparable and will "
                       f"not be averaged."),
        }

    try:
        chance = load_spec(bid).scoring.chance_level
    except SpecError:
        chance = 0.0

    table = summarise(g, benchmark=bid, chance_level=chance)
    rows = [] if table.empty else _records(table)

    # Bootstrap CIs, so the chart can draw an interval. A bar chart of accuracy
    # with no interval is precisely the artefact this harness exists to argue
    # against, and the data for one is already in the rows.
    from ..report import aggregate as A
    try:
        ci = A.ci_table(g, "accuracy")
        ci_rows = [] if ci.empty else _records(ci)
    except Exception:                               # noqa: BLE001 - optional
        ci_rows = []

    significance, power, sig_error = [], "", ""
    if len(rows) >= 2:
        try:
            sig = S.significance_matrix(g, metric="accuracy")
            significance = [] if sig.empty else _records(
                sig[["model_a", "model_b", "n_pairs", "diff", "p_value",
                     "p_adjusted", "significant", "test", "verdict"]])
            power = S.power_report(g, "accuracy").summary()
        except Exception as e:                      # noqa: BLE001 - surfaced
            # Most likely UnpairedItemsError, which is a finding, not a crash:
            # the models were not scored on the same items and the harness
            # refuses to paper over it.
            sig_error = f"{type(e).__name__}: {e}"

    return {
        "run_id": run_id,
        "benchmark": bid,
        "chance_level": chance,
        "spec_hash": hashes[0] if hashes else "",
        "summary": rows,
        "ci": ci_rows,
        "significance": significance,
        "significance_error": sig_error,
        "power": power,
        "excluded_total": int(sum(r["n_items"] - r["n_scored"] for r in rows)),
    }


def rows(store_path: str, run_id: str, limit: int = 100) -> dict:
    """Raw trace rows, for reading what the model actually said."""
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet.", 404)
    g = store.load_run(run_id)
    if g.empty:
        raise ApiError(f"No rows for run {run_id!r}.", 404)

    cols = [c for c in ("model", "item_id", "accuracy", "extraction_failed",
                        "extracted_via", "finish_reason", "truncated",
                        "prompt_tokens", "completion_tokens", "latency_ms",
                        "raw_output", "error") if c in g.columns]
    return {"run_id": run_id, "rows": _records(g[cols].head(limit))}


def manifest(store_path: str, run_id: str) -> dict:
    """The run manifest, what makes a run a result rather than a number (I9)."""
    from ..store.manifest import RunManifest

    d = Path(store_path).parent / run_id
    if not (d / "manifest.json").exists():
        raise ApiError(f"No manifest for run {run_id!r}.", 404)
    m = RunManifest.read(d)
    import json
    return json.loads(m.to_json())


# --------------------------------------------------------------------------- #
#  Runs
# --------------------------------------------------------------------------- #
def start_run(*, benchmark: str, models: list[str], limit: int, seed: int,
              store_path: str, models_cfg: dict, budget: float = 0.0,
              acknowledge_licence: bool = False) -> dict:
    """Kick off a benchmark in a background thread. Returns a job id.

    Validation happens *here*, synchronously, so a bad request fails with a
    message the browser can show rather than dying silently in a thread.
    """
    from ..orchestration.jobs import any_running, new_job

    spec = _require_spec(benchmark)
    if not bench_registry.has_adapter(spec):
        raise ApiError(f"No adapter registered for {spec.id!r}.", 400)
    if spec.requires_licence_ack() and not acknowledge_licence:
        raise ApiError(
            f"{spec.id} is licensed {spec.source.licence!r} (non-commercial) "
            f"and needs explicit acknowledgement.", 400)
    if not models:
        raise ApiError("Pick at least one model.", 400)
    if limit < 1:
        raise ApiError("limit must be >= 1.", 400)
    if any_running():
        raise ApiError("A run is already in flight. One at a time.", 409)

    job = new_job("bench", total=max(1, len(models) * limit))
    run_id = f"bench_{uuid.uuid4().hex[:8]}"
    threading.Thread(
        target=_run_worker,
        args=(job, run_id, spec.id, list(models), int(limit), int(seed),
              store_path, dict(models_cfg or {}), float(budget)),
        daemon=True).start()
    return {"job_id": job.id, "run_id": run_id, "total": job.total}


def job_status(job_id: str) -> dict:
    from ..orchestration.jobs import get_job

    job = get_job(job_id)
    if job is None:
        raise ApiError(f"No job {job_id!r}.", 404)
    return job.snapshot()


def _run_worker(job, run_id: str, benchmark_id: str, models: list[str],
                limit: int, seed: int, store_path: str, models_cfg: dict,
                budget: float) -> None:
    """Worker thread. Mutates `job` and nothing else (see `jobs.py`)."""
    try:
        from ..bench.runner import run_benchmark
        from ..clients.cost import CostMeter
        from ..clients.pricing import PricingRegistry
        from ..clients.registry import build_from_config
        from ..store.manifest import RunManifest, apparatus_hash, dataset_hash
        from ..store.store import TraceStore

        spec = load_spec(benchmark_id)
        adapter = bench_registry.build(spec)
        pricing = PricingRegistry("configs/pricing.yaml", strict=False)
        meter = CostMeter(budget_usd=budget or 0.0)

        cfg = dict(models_cfg)
        cfg["models"] = models
        client = build_from_config(cfg, meter=meter, pricing=pricing)
        store = TraceStore(store_path)

        items = adapter.load(seed=seed, limit=limit)
        job.set_total(max(1, len(models) * len(items)))

        # I9 applies to a run started from a browser exactly as from the CLI.
        mf = RunManifest.capture(
            run_id, spec.id, models=tuple(models),
            dataset_hash=dataset_hash(items),
            apparatus_hash=apparatus_hash(
                judge_model=cfg.get("judge_model", "") or "",
                judge_provider=cfg.get("judge_provider", "") or ""),
            spec_hash=spec.spec_hash(),
            judge_model=cfg.get("judge_model", "") or "",
            seeds={"sampling_seed": seed},
            pricing_as_of=getattr(pricing, "as_of", "unknown"),
            budget_usd=float(budget or 0.0))
        mdir = Path(store_path).parent / run_id
        mf.write(mdir)

        rep = run_benchmark(
            adapter=adapter, spec=spec, models=models, client=client,
            run_id=run_id, store=store, meter=meter, pricing=pricing,
            limit=limit, seed=seed,
            progress_cb=lambda done, total, model: job.set_done_absolute(done))

        mf.n_rows = len(rep.rows)
        mf.n_errors = rep.errors
        mf.aborted = rep.aborted
        mf.total_cost_usd = float(meter.summary().get("total_usd", 0.0))
        mf.write(mdir)

        msg = (f"{run_id} · {len(rep.rows)} rows · {rep.errors} errors · "
               f"${meter.spent:.4f}")
        if rep.aborted:
            job.fail(f"aborted at the budget ceiling, {msg}")
        else:
            job.finish(rows=len(rep.rows), message=msg)
    except Exception as e:                          # noqa: BLE001 - reported
        job.fail(f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
def _require_spec(benchmark: str) -> BenchmarkSpec:
    try:
        return load_spec(benchmark)
    except SpecError as e:
        raise ApiError(str(e), 404) from e


def _records(df) -> list[dict]:
    """DataFrame to JSON-safe records.

    NaN becomes `null`, not `0`. That distinction is the whole of I7 at this
    layer: an accuracy of `null` means *not applicable*, the item could not be
    scored, and rendering it as zero would be the exact fold-in the harness
    refuses everywhere else.
    """
    import numpy as np
    import pandas as pd

    out = []
    for rec in df.to_dict("records"):
        clean = {}
        for k, v in rec.items():
            # bool before int: numpy and Python both make bool a subclass of
            # int, so the int branch would swallow it and render True as 1.
            if isinstance(v, (bool, np.bool_)):
                clean[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                clean[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                clean[k] = None if pd.isna(v) else float(v)
            elif v is None:
                clean[k] = None
            elif isinstance(v, (str, list, dict)):
                clean[k] = v
            else:
                clean[k] = None if pd.isna(v) else str(v)
        out.append(clean)
    return out
