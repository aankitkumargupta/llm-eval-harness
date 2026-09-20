"""
`bench` subcommands (§10.8).

Deliberately thin. Every one of these delegates to machinery that already
exists, the trace store, the cost meter, the significance layer, the gate,
because §10 forbids a parallel universe of benchmark-only code. If a command
here starts computing something itself, that is the signal that a seam is in
the wrong place.

The list/validate/estimate trio exists for the same reason `validate` and
`estimate` exist on the profile side: the expensive mistakes in benchmarking
are made before the first paid call, and they are all catchable by reading
config.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd

from ..bench import registry
from ..bench.metrics import summarise
from ..bench.runner import run_benchmark
from ..bench.spec import BenchmarkSpec, SpecError, available_specs, load_spec


def _specs(ids: list[str] | None = None) -> list[BenchmarkSpec]:
    if ids:
        return [load_spec(i) for i in ids]
    out = []
    for p in available_specs():
        try:
            out.append(BenchmarkSpec.from_yaml(p))
        except SpecError as e:
            print(f"[skip] {p}: {e}")
    return out


# --------------------------------------------------------------------------- #
def cmd_bench_list(args) -> int:
    rows = []
    for spec in _specs():
        rows.append({
            "id": spec.id,
            "family": spec.family,
            "version": spec.version,
            "licence": spec.source.licence,
            "commercial": "yes" if spec.source.commercial_use else "ACK REQUIRED",
            "chance": spec.scoring.chance_level,
            "mode": spec.scoring.mode,
            "adapter": "yes" if spec.id in registry.known() else "MISSING",
            "spec_hash": spec.spec_hash()[:12],
        })
    if args.family:
        rows = [r for r in rows if r["family"] == args.family]
    if not rows:
        print("No benchmark specs in configs/benchmarks/.")
        return 1
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2))
    else:
        print(pd.DataFrame(rows).to_string(index=False))
    return 0


def cmd_bench_validate(args) -> int:
    """Spec + licence + adapter preflight, before anything is spent."""
    ok = True
    for spec in _specs(args.benchmark):
        problems = spec.validate()
        if problems:
            ok = False
            print(f"[fail] {spec.id}")
            for p in problems:
                print(f"  - {p}")
            continue

        notes = []
        if spec.id not in registry.known():
            ok = False
            notes.append("no adapter registered")
        if spec.requires_licence_ack() and not args.acknowledge_licence:
            ok = False
            notes.append(f"licence {spec.source.licence!r} is non-commercial; "
                         f"re-run with --acknowledge-licence")
        data = Path(spec.source.ref)
        if spec.source.kind == "local" and not data.exists():
            ok = False
            notes.append(f"data file not found: {data}")

        if notes:
            print(f"[fail] {spec.id}")
            for n in notes:
                print(f"  - {n}")
        else:
            print(f"[ok]   {spec.id}  spec_hash={spec.spec_hash()[:12]}  "
                  f"chance={spec.scoring.chance_level}  "
                  f"mode={spec.scoring.mode}  licence={spec.source.licence}")
    return 0 if ok else 1


def cmd_bench_fetch(args) -> int:
    """Acquire a benchmark's dataset (§10.8).

    Separate from `run` on purpose: fetching reaches the network and writes to
    a shared cache, and it is the one step that can silently change what every
    later run measures. It should be a deliberate act, not a side effect of
    the first person who happened to run the benchmark.
    """
    from ..bench.fetch import FetchError, LicenceError, fetch_for_spec

    ok = True
    for spec in _specs(args.benchmark):
        try:
            res = fetch_for_spec(spec, acknowledge_licence=args.acknowledge_licence,
                                 limit=args.limit)
        except LicenceError as e:
            ok = False
            print(f"[licence] {spec.id}: {e}")
            continue
        except FetchError as e:
            ok = False
            print(f"[fail] {spec.id}: {e}")
            continue

        state = "cached" if res.cached else f"fetched in {res.elapsed_s:.1f}s"
        print(f"[ok]   {spec.id}: {res.n_rows} rows, {state}")
        print(f"       {res.path}")
        print(f"       {res.checksum}")
        if spec.source.checksum and spec.source.checksum != res.checksum:
            ok = False
            print("       ** checksum differs from the spec, see the message above **")
        elif not spec.source.checksum and spec.source.kind != "local":
            print(f"       spec records no checksum. Paste the line above into "
                  f"configs/benchmarks/{spec.id}.yaml so a future change is caught.")
    return 0 if ok else 1


def cmd_bench_run(args) -> int:
    from ..clients.cost import CostMeter
    from ..clients.pricing import PricingRegistry
    from ..clients.registry import build_from_config
    from ..store.manifest import RunManifest, apparatus_hash, dataset_hash
    from ..store.store import TraceStore

    run_cfg = _load_yaml(args.run_config)
    models_cfg = _load_yaml(args.models_config)

    spec = load_spec(args.benchmark)
    if spec.requires_licence_ack() and not args.acknowledge_licence:
        print(f"[fail] {spec.id} is licensed {spec.source.licence!r} "
              f"(non-commercial). Re-run with --acknowledge-licence.")
        return 1

    adapter = registry.build(spec)
    models = args.models or models_cfg.get("models", [])
    if not models:
        print("No models. Pass --models or set them in configs/models.yaml.")
        return 1

    pricing = PricingRegistry(run_cfg.get("pricing_path", "configs/pricing.yaml"),
                              strict=False)
    meter = CostMeter(budget_usd=args.budget or 0.0)
    client = build_from_config(models_cfg, meter=meter, pricing=pricing)
    store = TraceStore(run_cfg.get("store_path", "runs/traces"))
    run_id = args.run_id or uuid.uuid4().hex[:12]

    items_preview = adapter.load(seed=args.seed, limit=args.limit)

    # Same manifest contract as a profile run (I9): a bench run without one is
    # no more a result than a profile run without one.
    manifest = RunManifest.capture(
        run_id, spec.id,
        models=tuple(models),
        dataset_hash=dataset_hash(items_preview),
        apparatus_hash=apparatus_hash(
            judge_model=models_cfg.get("judge_model", "") or "",
            judge_provider=models_cfg.get("judge_provider", "") or ""),
        spec_hash=spec.spec_hash(),
        judge_model=models_cfg.get("judge_model", "") or "",
        seeds={"sampling_seed": args.seed or spec.sampling.seed},
        pricing_as_of=getattr(pricing, "as_of", "unknown"),
        budget_usd=float(args.budget or 0.0),
    )
    manifest_dir = Path(run_cfg.get("store_path", "runs/traces")).parent / run_id
    manifest.write(manifest_dir)

    print(f"[bench] {spec.id} v{spec.version} run_id={run_id} "
          f"models={len(models)} items={len(items_preview)} "
          f"spec_hash={spec.spec_hash()[:12]}")

    def progress(done, total, model):
        if total and done % 20 == 0:
            print(f"  {done}/{total} ({100 * done / total:.0f}%)  "
                  f"${meter.spent:.4f}", flush=True)

    rep = run_benchmark(
        adapter=adapter, spec=spec, models=models, client=client,
        run_id=run_id, store=store, meter=meter, pricing=pricing,
        limit=args.limit, seed=args.seed, progress_cb=progress,
        checkpoint_every=int(run_cfg.get("checkpoint_every", 50)),
    )
    print(f"[bench] {rep.summary()}")

    costs = meter.summary()
    manifest.finished_ts = manifest.started_ts
    manifest.n_rows = len(rep.rows)
    manifest.n_errors = rep.errors
    manifest.aborted = rep.aborted
    manifest.total_cost_usd = float(costs.get("total_usd", 0.0))
    manifest.write(manifest_dir)

    print(f"[cost] {costs}")
    print(f"[manifest] {manifest_dir / 'manifest.json'}")
    _print_summary(pd.DataFrame([r.to_dict() for r in rep.rows]), spec)
    return 2 if rep.aborted else 0


def cmd_bench_report(args) -> int:
    from ..store.store import TraceStore

    run_cfg = _load_yaml(args.run_config)
    store = TraceStore(run_cfg.get("store_path", "runs/traces"))
    df = store.load_run(args.run_id) if args.run_id else store.load_all()
    if df.empty:
        print("No traces found.")
        return 1
    if "benchmark" in df.columns:
        df = df[df["benchmark"].notna()]
    if args.benchmark:
        df = df[df["benchmark"] == args.benchmark]
    if df.empty:
        print("No benchmark rows for that selection.")
        return 1

    for bid, g in df.groupby("benchmark"):
        try:
            spec = load_spec(str(bid))
            chance = spec.scoring.chance_level
        except SpecError:
            chance = 0.0
        _print_summary(g, None, benchmark=str(bid), chance=chance,
                       by=args.by, as_json=getattr(args, "json", False))
    return 0


# --------------------------------------------------------------------------- #
def _print_summary(df: pd.DataFrame, spec, benchmark: str | None = None,
                   chance: float | None = None, by: str = "",
                   as_json: bool = False) -> None:
    if df.empty:
        return
    bid = benchmark or (spec.id if spec else "?")
    ch = chance if chance is not None else (spec.scoring.chance_level if spec else 0.0)

    # Refuse to mix formats in one table (§10.1): two runs of the same
    # benchmark under different prompt formats are not the same benchmark.
    if "spec_hash" in df.columns:
        hashes = sorted(set(df["spec_hash"].dropna()))
        if len(hashes) > 1:
            print(f"\n[refused] {bid}: rows span {len(hashes)} different "
                  f"spec_hash values {[h[:8] for h in hashes]}. Prompt format, "
                  f"few-shot count or extraction differs, so these numbers are "
                  f"not comparable. Report them one spec_hash at a time.")
            return

    out = summarise(df, benchmark=bid, chance_level=ch)
    if out.empty:
        return

    if as_json:
        print(out.to_json(orient="records", indent=2))
        return

    print(f"\n=== {bid} ===")
    cols = ["model", "n_items", "n_scored", "accuracy",
            "accuracy_chance_adjusted", "extraction_failure_rate",
            "truncation_rate", "cost_usd", "cost_per_correct_answer"]
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))

    if ch > 0:
        print(f"  chance level {ch:g}, chance-adjusted is the honest column")
    for _, r in out.iterrows():
        excluded = int(r["n_items"] - r["n_scored"])
        if excluded:
            print(f"  note: {r['model']} had {excluded} item(s) excluded from "
                  f"accuracy (extraction failure / error), not counted wrong")

    if by and by in df.columns:
        print(f"\n--- by {by} (n shown per stratum) ---")
        strat = (df.groupby([by, "model"])["accuracy"]
                 .agg(["count", "mean"]).reset_index())
        print(strat.to_string(index=False))


def _load_yaml(path: str) -> dict:
    import yaml
    p = Path(path)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


# --------------------------------------------------------------------------- #
def add_bench_parser(sub) -> None:
    """Attach `bench` and its subcommands to the main CLI."""
    pb = sub.add_parser("bench", help="Run public or private benchmarks (§10)")
    bsub = pb.add_subparsers(dest="bench_cmd", required=True)

    pl = bsub.add_parser("list", help="Benchmarks this harness can run")
    pl.add_argument("--family", default="")
    pl.add_argument("--json", action="store_true")
    pl.set_defaults(func=cmd_bench_list)

    pv = bsub.add_parser("validate", help="Spec + licence + adapter preflight")
    pv.add_argument("--benchmark", nargs="*", default=None)
    pv.add_argument("--acknowledge-licence", action="store_true")
    pv.set_defaults(func=cmd_bench_validate)

    pf = bsub.add_parser("fetch", help="Download a benchmark's dataset")
    pf.add_argument("--benchmark", nargs="*", default=None)
    pf.add_argument("--limit", type=int, default=None,
                    help="Rows to download. Omit for the whole split.")
    pf.add_argument("--acknowledge-licence", action="store_true",
                    help="Required for datasets not marked for commercial use.")
    pf.set_defaults(func=cmd_bench_fetch)

    pr = bsub.add_parser("run", help="Run a benchmark across models")
    pr.add_argument("--benchmark", required=True)
    pr.add_argument("--models", nargs="*", default=None)
    pr.add_argument("--limit", type=int, default=None)
    pr.add_argument("--seed", type=int, default=None)
    pr.add_argument("--budget", type=float, default=0.0)
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--acknowledge-licence", action="store_true")
    pr.add_argument("--run-config", default="configs/run.yaml")
    pr.add_argument("--models-config", default="configs/models.yaml")
    pr.set_defaults(func=cmd_bench_run)

    prep = bsub.add_parser("report", help="Summarise a benchmark run")
    prep.add_argument("--run-id", default=None)
    prep.add_argument("--benchmark", default="")
    prep.add_argument("--by", default="", help="stratify by a column, e.g. subject")
    prep.add_argument("--json", action="store_true")
    prep.add_argument("--run-config", default="configs/run.yaml")
    prep.set_defaults(func=cmd_bench_report)
