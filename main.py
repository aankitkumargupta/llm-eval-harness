"""
Command-line entry point for the evaluation harness.

    # one-time per dataset
    python main.py ingest   --profile configs/profiles/regulated_qa.yaml

    # check a config before spending anything
    python main.py validate --profile configs/profiles/regulated_qa.yaml
    python main.py estimate --profile configs/profiles/regulated_qa.yaml

    # generate an adversarial probe set (unanswerable / noise / injection / ...)
    python main.py probes   --profile configs/profiles/regulated_qa.yaml

    # run the matrix
    python main.py run      --profile configs/profiles/regulated_qa.yaml

    # analyse
    python main.py report   --profile regulated_qa
    python main.py compare  --profile regulated_qa --metric accuracy
    python main.py decide   --profile regulated_qa \
                            --require "faithfulness>=0.9" --require "latency_p95_ms<=2000" \
                            --optimise cost_usd --qpd 50000
    python main.py arena    --profile regulated_qa
    python main.py html     --profile regulated_qa --out report.html

    # benchmarks (CLAUDE.md section 10) - works offline with fake: models
    python main.py bench list
    python main.py bench run --benchmark mmlu_pro --models fake:a fake:b --limit 20
    python main.py bench report --run-id <run_id>

    # local web UI (stdlib http.server; no framework, no extra dependency)
    python main.py serve

    # CI
    python main.py gate     --profile regulated_qa --baseline <run_id> \
                            --candidate <run_id> --gate accuracy:0.02

Requires a provider API key in the environment (TOGETHER_API_KEY by default).
"""

from __future__ import annotations

import argparse
import io
import random
import sys
import time
import uuid
from pathlib import Path

import yaml

from harness.bench.cli import add_bench_parser
from harness.cache.cache import stable_hash
from harness.clients.cost import CostMeter, forecast_run
from harness.clients.pricing import PricingRegistry
from harness.clients.registry import build_from_config
from harness.clients.resilience import RetryPolicy
from harness.env import load_env
from harness.eval.probes import ProbeConfig, build_probe_suite
from harness.orchestration.orchestrator import Orchestrator
from harness.profiles.loaders import (
    dataset_stats,
    load_corpus,
    load_evalset,
    save_evalset,
    validate_gold_ids,
)
from harness.profiles.profile import Profile, ProfileError
from harness.rag.ingest import Ingestor
from harness.report import aggregate as A
from harness.report import stats as S
from harness.report.decide import Constraint, explain, headroom, project_cost, select
from harness.report.gate import budget_gate, check_regression, parse_gate_spec
from harness.store import manifest as MF
from harness.store.store import TraceStore


def _utf8_stdout() -> None:
    """Force UTF-8 output.

    Windows consoles default to cp1252, which raises UnicodeEncodeError on any
    non-ASCII character. Model names, questions and corpus text are all
    user-supplied, so this is a matter of when, not if, and a crash while
    *printing a report* would discard results that already cost real money.
    """
    for stream in ("stdout", "stderr"):
        s = getattr(sys, stream)
        if isinstance(s, io.TextIOWrapper) and (s.encoding or "").lower() not in (
                "utf-8", "utf8"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _run_cfg(path: str = "configs/run.yaml") -> dict:
    return _load_yaml(path)


def _models_cfg(path: str = "configs/models.yaml") -> dict:
    return _load_yaml(path)


def _profile(arg: str) -> Profile:
    """Accept either a path to a profile file or a bare profile name."""
    p = Path(arg)
    if not p.exists() and not arg.endswith(".yaml"):
        p = Path(f"configs/profiles/{arg}.yaml")
    return Profile.from_yaml(str(p))


def _build_stack(profile: Profile, run_cfg: dict, models_cfg: dict,
                 budget: float = 0.0):
    """Construct the client, pricing, meter and store used by every run command."""
    pricing = PricingRegistry(run_cfg.get("pricing_path", "configs/pricing.yaml"))
    warning = pricing.staleness_warning()
    if warning:
        print(f"[warn] {warning}")

    meter = CostMeter(budget_usd=budget or float(run_cfg.get("budget_usd", 0.0)))
    retry = RetryPolicy(
        max_attempts=int(run_cfg.get("max_retries", 5)),
        initial_backoff=float(run_cfg.get("retry_backoff", 1.0)),
        max_backoff=float(run_cfg.get("retry_max_backoff", 30.0)),
    )
    client = build_from_config(models_cfg, meter=meter, pricing=pricing,
                               retry=retry)
    store = TraceStore(run_cfg.get("store_path", "runs/traces"))
    return client, pricing, meter, store


def _split(items, dev_frac: float, seed: int):
    """Deterministic dev/test split.

    Guards the degenerate cases the original didn't: a dev_frac that rounds to
    zero items silently disabled tuning, and one that consumed everything left
    no test split to report on.
    """
    if not items:
        return [], []
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(len(items) * dev_frac)
    if dev_frac > 0:
        cut = max(1, min(cut, len(items) - 1))
    dev = [items[i] for i in idx[:cut]]
    test = [items[i] for i in idx[cut:]]
    return dev, (test or items)


def _load_store_df(store: TraceStore, profile_name: str, run_id: str | None = None):
    if run_id:
        return store.load_run(run_id)
    return store.load_profile(profile_name)


def _qdrant(run_cfg: dict, override: str | None = None):
    """Open Qdrant, embedded or server.

    `qdrant_path` (a local folder) takes precedence over `qdrant_url`, so a
    laptop user can set it once in run.yaml instead of remembering
    `--qdrant-path` on every command. Embedded mode needs no Docker and no
    running service, which is the normal case for a single-user local run.
    """
    from qdrant_client import QdrantClient

    path = override or run_cfg.get("qdrant_path")
    if path:
        return QdrantClient(path=str(path))
    return QdrantClient(url=run_cfg.get("qdrant_url", "http://localhost:6333"))


# =========================================================================== #
#  Commands                                                                    #
# =========================================================================== #
def cmd_validate(args):
    """Check a profile and its dataset before spending anything."""
    try:
        profile = _profile(args.profile)
    except ProfileError as e:
        print(e)
        return 1
    print(f"[ok] profile '{profile.name}' is valid (task={profile.task.value})")

    items = load_evalset(profile.evalset_path)
    stats = dataset_stats(items, task=profile.task.value)
    print()
    print(stats.summary())

    if profile.corpus_path and Path(profile.corpus_path).exists():
        problems = validate_gold_ids(items, profile.corpus_path,
                                     profile.chunk_size, profile.overlap)
        if problems:
            print()
            print("Gold-passage problems:")
            for p in problems[:10]:
                print(f"  {p}")

    models_cfg = _models_cfg()
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    missing = pricing.missing(list(models_cfg.get("models", []))
                              + [models_cfg.get("judge_model", "")])
    missing = [m for m in missing if m]
    if missing:
        print()
        print(f"[warn] no pricing for: {missing}. Cost metrics will be blank for "
              f"these; add them to configs/pricing.yaml.")

    try:
        client = build_from_config(models_cfg, pricing=pricing)
        issues = client.preflight(models_cfg.get("models", []),
                                  profile.embedding_model,
                                  models_cfg.get("rerank_model", ""),
                                  models_cfg.get("judge_model", ""))
        if issues:
            print()
            print("Provider routing problems:")
            for i in issues:
                print(f"  {i}")
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"\n[warn] could not pre-flight providers: {e}")

    return 0 if not stats.warnings else 0


def cmd_estimate(args):
    """Forecast a run's spend before committing to it."""
    profile = _profile(args.profile)
    run_cfg, models_cfg = _run_cfg(), _models_cfg()
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    items = load_evalset(profile.evalset_path)
    dev, test = _split(items, run_cfg.get("dev_split", 0.3),
                       run_cfg.get("split_seed", 0))

    passes = run_cfg.get("passes", {})
    fc = forecast_run(
        pricing, models_cfg.get("models", []), n_test=len(test), n_dev=len(dev),
        tuning_budget=profile.tuning_budget,
        judge_model=models_cfg.get("judge_model", "")
        if profile.accuracy_scorer == "judge" else "",
        do_baseline=bool(passes.get("baseline", True)),
        do_adapted=bool(passes.get("adapted", False)),
    )
    print(f"profile      : {profile.name}")
    print(f"models       : {len(models_cfg.get('models', []))}")
    print(f"dev / test   : {len(dev)} / {len(test)}")
    print(f"model calls  : {fc.calls:,}")
    print(f"judge calls  : {fc.judge_calls:,}")
    print(f"estimated    : ${fc.est_usd:,.2f}"
          f"  (generation ${fc.detail['generation_usd']:,.2f}"
          f" + judge ${fc.detail['judge_usd']:,.2f})")
    if fc.detail.get("unpriced_models"):
        print(f"[warn] unpriced (excluded from the estimate): "
              f"{fc.detail['unpriced_models']}")
    print("\nThis is approximate - real prompt length depends on your corpus.")
    return 0


def cmd_ingest(args):
    profile = _profile(args.profile)
    run_cfg, models_cfg = _run_cfg(), _models_cfg()
    client, pricing, meter, _ = _build_stack(profile, run_cfg, models_cfg)

    qdrant = _qdrant(run_cfg, args.qdrant_path)
    ingestor = Ingestor(client, qdrant, profile.embedding_model)
    n = ingestor.ingest(
        collection=profile.collection_name(),
        documents=load_corpus(profile.corpus_path),
        chunk_size=args.chunk_size or profile.chunk_size,
        overlap=args.overlap if args.overlap is not None else profile.overlap,
        batch_size=args.batch_size,
    )
    print(f"[ingest] {profile.name}: indexed {n} chunks into "
          f"'{profile.collection_name()}' "
          f"(embedding spend ${meter.costs.embedding:.4f})")
    return 0


def cmd_probes(args):
    """Generate an adversarial probe set from an existing evalset."""
    profile = _profile(args.profile)
    items = load_evalset(profile.evalset_path)
    ps = profile.probes
    cfg = ProbeConfig(
        unanswerable=args.unanswerable if args.unanswerable is not None else ps.unanswerable,
        noise=args.noise if args.noise is not None else ps.noise,
        injection=args.injection if args.injection is not None else ps.injection,
        paraphrase=args.paraphrase if args.paraphrase is not None else ps.paraphrase,
        positional=args.positional if args.positional is not None else ps.positional,
        typo_rate=ps.typo_rate, n_distractors=ps.n_distractors, seed=ps.seed,
    )
    suite = build_probe_suite(items, cfg)
    if not suite.items:
        print("No probes generated - every fraction is zero.")
        return 1

    out = args.out or str(Path(profile.evalset_path).with_name(
        Path(profile.evalset_path).stem + "_probes.jsonl"))
    if args.append:
        combined = items + suite.items
        save_evalset(combined, profile.evalset_path)
        print(f"[probes] {suite.summary()}; appended to {profile.evalset_path} "
              f"({len(combined)} items total)")
    else:
        save_evalset(suite.items, out)
        print(f"[probes] {suite.summary()} -> {out}")
        print("Point the profile's evalset_path at this file, or re-run with "
              "--append to merge into the base evalset.")
    return 0


def cmd_run(args):
    profile = _profile(args.profile)
    run_cfg, models_cfg = _run_cfg(), _models_cfg()
    client, pricing, meter, store = _build_stack(profile, run_cfg, models_cfg,
                                                 budget=args.budget)

    models = args.models or models_cfg["models"]
    issues = client.preflight(models, profile.embedding_model,
                              models_cfg.get("rerank_model", ""))
    if issues:
        print("Provider routing problems - fix these before running:")
        for i in issues:
            print(f"  {i}")
        return 1

    qdrant = None
    if profile.task.value == "rag":
        qdrant = _qdrant(run_cfg, args.qdrant_path)

    items = load_evalset(profile.evalset_path)
    dev, test = _split(items, run_cfg.get("dev_split", 0.3),
                       run_cfg.get("split_seed", 0))
    run_id = args.run_id or uuid.uuid4().hex[:12]

    def progress(done: int, total: int, model: str) -> None:
        if total and done % 25 == 0:
            pct = 100.0 * done / total
            print(f"  {done}/{total} ({pct:.0f}%)  ${meter.spent:.4f}", flush=True)

    orch = Orchestrator(
        client=client, qdrant=qdrant, pricing=pricing, store=store,
        judge_model=models_cfg.get("judge_model")
        if profile.accuracy_scorer == "judge" or any(
            m in profile.active_metrics for m in
            ("faithfulness", "answer_relevance", "completeness")) else None,
        cache_dir=run_cfg.get("cache_dir", ".cache"),
        rerank_model=models_cfg.get("rerank_model", ""),
        meter=meter,
        judge_ensemble=models_cfg.get("judge_ensemble"),
        max_workers=run_cfg.get("max_workers", 8),
        checkpoint_every=run_cfg.get("checkpoint_every", 50),
        progress_cb=progress,
    )

    # I9: a run without a manifest is not a result. Captured BEFORE the first
    # paid call, so an aborted or crashed run still leaves a record of what it
    # was rather than orphaned rows nobody can interpret.
    apparatus = MF.apparatus_hash(
        embedding_model=profile.embedding_model,
        rerank_model=models_cfg.get("rerank_model", "") or "",
        judge_model=models_cfg.get("judge_model", "") or "",
        judge_ensemble=tuple(models_cfg.get("judge_ensemble") or ()),
        embedding_provider=models_cfg.get("embedding_provider", "") or "",
        rerank_provider=models_cfg.get("rerank_provider", "") or "",
        judge_provider=models_cfg.get("judge_provider", "") or "",
    )
    manifest = MF.RunManifest.capture(
        run_id, profile.name,
        models=tuple(models),
        profile_cfg_hash=stable_hash(*profile.config_hash_parts()),
        dataset_hash=MF.dataset_hash(items),
        apparatus_hash=apparatus,
        embedding_model=profile.embedding_model,
        rerank_model=models_cfg.get("rerank_model", "") or "",
        judge_model=models_cfg.get("judge_model", "") or "",
        judge_ensemble=tuple(models_cfg.get("judge_ensemble") or ()),
        seeds={"split_seed": run_cfg.get("split_seed", 0),
               "dev_split": run_cfg.get("dev_split", 0.3)},
        pricing_as_of=getattr(pricing, "as_of", "unknown"),
        pricing_path=str(run_cfg.get("pricing_path", "")),
        budget_usd=float(args.budget or run_cfg.get("budget_usd", 0.0) or 0.0),
        started_ts=time.time(),
    )
    manifest_dir = Path(run_cfg.get("store_path", "runs/traces")).parent / run_id
    manifest.write(manifest_dir)

    print(f"[run] profile={profile.name} run_id={run_id} "
          f"models={len(models)} dev={len(dev)} test={len(test)}")
    print(f"[run] apparatus={apparatus[:12]} "
          f"dataset={manifest.dataset_hash[:12]} "
          f"git={manifest.git_sha[:8]}{'-dirty' if manifest.git_dirty else ''}")
    passes = run_cfg.get("passes", {})
    ran_passes: list[str] = []

    if passes.get("baseline", True):
        rep = orch.run_baseline(profile, models, test, run_id,
                                resume_from=args.resume)
        print(f"[run] baseline: {rep.summary()}")
        ran_passes.append("baseline")
        if rep.aborted:
            manifest.aborted = True
            manifest.abort_reason = "budget ceiling reached"
            manifest.passes = tuple(ran_passes)
            manifest.finished_ts = time.time()
            manifest.total_cost_usd = float(meter.summary().get("total_usd", 0.0))
            manifest.write(manifest_dir)
            return 2

    if passes.get("adapted", False):
        rep = orch.run_adapted(profile, models, dev, test, run_id)
        print(f"[run] adapted: {rep.summary()}")
        ran_passes.append("adapted")
        for model, info in rep.winners.items():
            print(f"       winner {model}: {info['config']} "
                  f"(dev {info['dev_score']:.4f})")
        if rep.aborted:
            manifest.aborted = True
            manifest.abort_reason = "budget ceiling reached"
            manifest.passes = tuple(ran_passes)
            manifest.finished_ts = time.time()
            manifest.total_cost_usd = float(meter.summary().get("total_usd", 0.0))
            manifest.write(manifest_dir)
            return 2

    if passes.get("latency", False):
        lat_items = test[: run_cfg.get("latency_items", 20)]
        rep = orch.run_latency(profile, models, lat_items, run_id,
                               resume_from=args.resume)
        print(f"[run] latency: {rep.summary()}")
        ran_passes.append("latency")

    costs = meter.summary()
    manifest.passes = tuple(ran_passes)
    manifest.finished_ts = time.time()
    manifest.total_cost_usd = float(costs.get("total_usd", 0.0))
    manifest.write(manifest_dir)

    print(f"\n[run] done. run_id={run_id}")
    print(f"[cost] {costs}")
    print(f"[manifest] {manifest_dir / 'manifest.json'}")
    return 0


def cmd_report(args):
    profile = _profile(args.profile)
    store = TraceStore(_run_cfg().get("store_path", "runs/traces"))
    df = _load_store_df(store, profile.name, args.run_id)
    if df.empty:
        print(f"No traces for profile '{profile.name}'. Run it first.")
        return 1

    quality = df[df["pass_"].isin(["baseline", "adapted"])]

    print(f"=== {profile.name}: weighted composite ===")
    print(A.weighted_composite(quality, profile.metric_weights)
          .to_string(index=False))

    metrics = [m for m in profile.active_metrics if m in df.columns]
    if {"baseline", "adapted"} <= set(df["pass_"].unique()):
        print("\n=== tuning gain (adapted - baseline) ===")
        print(A.tuning_gain(df, metrics).to_string(index=False))

    print("\n=== Pareto frontier (accuracy vs cost vs latency) ===")
    pf = A.pareto_frontier(df)
    if not pf.empty:
        print(pf.to_string(index=False))

    print(f"\n=== 95% CI: {args.metric} ===")
    ci = A.ci_table(quality, args.metric)
    if not ci.empty:
        print(ci.to_string(index=False))

    print(f"\n=== significance ({args.metric}, Holm-corrected) ===")
    unpaired = "drop" if getattr(args, "allow_unpaired", False) else "raise"
    sig = S.significance_matrix(quality, metric=args.metric,
                                on_unpaired=unpaired)
    if not sig.empty:
        print(sig[["model_a", "model_b", "n_pairs", "diff", "p_adjusted",
                   "significant", "test"]].to_string(index=False))
        for _, r in sig.iterrows():
            print(f"  {r['model_a']} vs {r['model_b']}: {r['verdict']}")

    pr = S.power_report(quality, args.metric, on_unpaired=unpaired)
    print("\n=== statistical power ===")
    print(f"  {pr.summary()}")
    if pr.suggestions:
        for label, n in sorted(pr.suggestions.items()):
            print(f"  to detect a gap of {label.split('_')[1]}: ~{n} paired items")

    err = A.error_attribution(df)
    if not err.empty and err["errors"].sum():
        print("\n=== error attribution ===")
        print(err.to_string(index=False))

    trunc = A.truncation_report(df)
    if not trunc.empty and trunc["truncated"].sum():
        print("\n=== truncated answers (max_tokens too low) ===")
        print(trunc.to_string(index=False))

    est = A.estimated_usage_report(df)
    if not est.empty:
        print("\n=== ESTIMATED token usage (cost below is not fully measured) ===")
        print(est.to_string(index=False))

    cons = A.consistency_report(df)
    if not cons.empty:
        print("\n=== paraphrase consistency ===")
        print(cons.to_string(index=False))

    for probe_metric, title in (("injection_resisted", "prompt-injection resistance"),
                                ("abstention_correct", "abstention"),
                                ("pii_leaked", "PII leakage")):
        if probe_metric in df.columns and df[probe_metric].notna().any():
            print(f"\n=== {title} ===")
            print(df.groupby("model")[probe_metric].mean().to_string())

    if "human_label" in df.columns and df["human_label"].notna().any():
        print("\n=== judge calibration vs human labels ===")
        for k, v in A.judge_calibration(quality).items():
            print(f"  {k}: {v}")
    return 0


def cmd_compare(args):
    profile = _profile(args.profile)
    store = TraceStore(_run_cfg().get("store_path", "runs/traces"))
    df = _load_store_df(store, profile.name, args.run_id)
    if df.empty:
        print("No traces.")
        return 1
    df = df[df["pass_"].isin(["baseline", "adapted"])]
    if args.a and args.b:
        unpaired = "drop" if args.allow_unpaired else "raise"
        r = S.compare_models(df, args.a, args.b, args.metric,
                             on_unpaired=unpaired)
        print(f"{r.model_a} ({r.mean_a:.4f}) vs {r.model_b} ({r.mean_b:.4f})")
        print(f"  test    : {r.test} on {r.n_pairs} paired items")
        print(f"  verdict : {r.verdict()}")
        return 0
    sig = S.significance_matrix(
        df, metric=args.metric,
        on_unpaired="drop" if args.allow_unpaired else "raise")
    if sig.empty:
        print("Not enough data to compare.")
        return 1
    for _, r in sig.iterrows():
        mark = "*" if r["significant"] else " "
        print(f"{mark} {r['model_a']} vs {r['model_b']}: {r['verdict']}")
    print("\n* = significant after Holm-Bonferroni correction")
    return 0


def cmd_decide(args):
    profile = _profile(args.profile)
    store = TraceStore(_run_cfg().get("store_path", "runs/traces"))
    df = _load_store_df(store, profile.name, args.run_id)
    if df.empty:
        print("No traces.")
        return 1

    constraints = [Constraint.parse(c) for c in (args.require or [])]
    shortlist, candidates = select(df, constraints, optimise=args.optimise)
    print(explain(candidates, constraints))

    if not shortlist.empty:
        cols = ["model"] + [c.metric for c in constraints if c.metric in shortlist.columns]
        if args.optimise not in cols and args.optimise in shortlist.columns:
            cols.append(args.optimise)
        print(f"\nRanked by {args.optimise}:")
        print(shortlist[cols].to_string(index=False))
        print(f"\n>>> Recommended: {shortlist.iloc[0]['model']}")

    print(f"\n=== projected cost at {args.qpd:,} queries/day ===")
    proj = project_cost(df, args.qpd)
    if not proj.empty:
        print(proj[["model", "cost_per_query", "daily_usd", "monthly_usd",
                    "annual_usd"]].to_string(index=False))

    hr = headroom(df, args.quality_metric, args.qpd)
    if not hr.empty:
        print("\n=== what extra quality costs (vs cheapest) ===")
        cols = ["model", args.quality_metric, "monthly_usd", "quality_delta",
                "monthly_delta_usd", "usd_per_point"]
        print(hr[[c for c in cols if c in hr.columns]].to_string(index=False))
    return 0


def cmd_arena(args):
    profile = _profile(args.profile)
    run_cfg, models_cfg = _run_cfg(), _models_cfg()
    client, pricing, meter, store = _build_stack(profile, run_cfg, models_cfg,
                                                 budget=args.budget)
    orch = Orchestrator(client=client, qdrant=None, pricing=pricing, store=store,
                        judge_model=models_cfg.get("judge_model"),
                        cache_dir=run_cfg.get("cache_dir", ".cache"),
                        meter=meter)
    models = args.models or models_cfg["models"]
    print(f"[arena] judging pairs for {len(models)} models "
          f"(position-bias corrected: each pair judged in both orders)")
    pairwise = orch.run_arena(profile, models, uuid.uuid4().hex[:12],
                              source_run=args.run_id)
    if not pairwise:
        print("No comparable answers found. Run the profile first.")
        return 1

    print(f"\n=== Elo ({len(pairwise)} comparisons) ===")
    print(A.elo_from_pairwise(pairwise).to_string(index=False))
    print("\n=== head-to-head win rates ===")
    print(A.win_rate_matrix(pairwise).to_string())
    print(f"\n[cost] judge spend ${meter.costs.judge:.4f}")
    return 0


def cmd_gate(args):
    profile = _profile(args.profile)
    store = TraceStore(_run_cfg().get("store_path", "runs/traces"))
    baseline = store.load_run(args.baseline)
    candidate = (store.load_run(args.candidate) if args.candidate
                 else _load_store_df(store, profile.name))
    if baseline.empty:
        print(f"Baseline run '{args.baseline}' not found. "
              f"Available runs:\n{store.runs().to_string(index=False)}")
        return 1

    metrics, tolerances = parse_gate_spec(
        args.gate or [m for m in profile.active_metrics
                      if m in ("accuracy", "faithfulness")])
    floors = [Constraint.parse(f) for f in (args.floor or [])]

    result = check_regression(
        baseline, candidate, metrics, tolerance=args.tolerance,
        per_metric_tolerance=tolerances, floors=floors,
        require_significance=not args.strict,
        on_unpaired="drop" if args.allow_unpaired else "raise")
    print(result.report())

    if args.max_cost is not None:
        cost = budget_gate(candidate, args.max_cost)
        print()
        print(cost.report())
        if not cost.passed:
            return 1
    return result.exit_code()


def cmd_serve(args):
    """Serve the local web UI.

    Thin by design: the server is presentation, so everything it shows comes
    from the same library the CLI calls. Nothing here computes a metric.
    """
    from harness.web.server import serve

    run_cfg = _run_cfg()
    return serve(args.host, args.port,
                 store_path=run_cfg.get("store_path", "runs/traces"),
                 models_cfg=_models_cfg(),
                 run_cfg=run_cfg,
                 budget=args.budget,
                 open_browser=not args.no_browser,
                 quiet=args.quiet)


def cmd_runs(args):
    store = TraceStore(_run_cfg().get("store_path", "runs/traces"))
    df = store.runs()
    if df.empty:
        print("No runs recorded yet.")
        return 1
    print(df.to_string(index=False))
    return 0


def cmd_html(args):
    from harness.report.html import write_html_report

    profile = _profile(args.profile)
    store = TraceStore(_run_cfg().get("store_path", "runs/traces"))
    df = _load_store_df(store, profile.name, args.run_id)
    if df.empty:
        print("No traces.")
        return 1
    out = write_html_report(df, profile, args.out, metric=args.metric)
    print(f"[html] wrote {out}")
    return 0


# =========================================================================== #
def main() -> int:
    _utf8_stdout()
    # Keys from .env, unless already exported. Every command below reads them.
    load_env()
    p = argparse.ArgumentParser(
        description="Multi-model LLM evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="Check a profile + dataset before spending")
    pv.add_argument("--profile", required=True)
    pv.set_defaults(func=cmd_validate)

    pe = sub.add_parser("estimate", help="Forecast a run's cost")
    pe.add_argument("--profile", required=True)
    pe.set_defaults(func=cmd_estimate)

    pi = sub.add_parser("ingest", help="Ingest a profile's corpus into Qdrant")
    pi.add_argument("--profile", required=True)
    pi.add_argument("--chunk-size", type=int, default=None)
    pi.add_argument("--overlap", type=int, default=None)
    pi.add_argument("--batch-size", type=int, default=128)
    pi.add_argument("--qdrant-path", default=None,
                    help="Embedded Qdrant folder (no Docker). Overrides "
                         "`qdrant_path` in run.yaml.")
    pi.set_defaults(func=cmd_ingest)

    pp = sub.add_parser("probes", help="Generate adversarial probe items")
    pp.add_argument("--profile", required=True)
    pp.add_argument("--out", default=None)
    pp.add_argument("--append", action="store_true",
                    help="Merge probes into the base evalset instead of a new file")
    for name in ("unanswerable", "noise", "injection", "paraphrase", "positional"):
        pp.add_argument(f"--{name}", type=float, default=None,
                        help=f"fraction of base evalset ({name} probes)")
    pp.set_defaults(func=cmd_probes)

    pr = sub.add_parser("run", help="Run baseline + adapted + latency")
    pr.add_argument("--profile", required=True)
    pr.add_argument("--models", nargs="*", default=None)
    pr.add_argument("--budget", type=float, default=0.0,
                    help="Hard USD ceiling; the run aborts cleanly when reached")
    pr.add_argument("--run-id", default=None)
    pr.add_argument("--resume", default=None,
                    help="Run id to resume: items already recorded are skipped")
    pr.add_argument("--qdrant-path", default=None,
                    help="Embedded Qdrant folder (no Docker). Overrides "
                         "`qdrant_path` in run.yaml.")
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="Print leaderboards and statistics")
    prep.add_argument("--profile", required=True)
    prep.add_argument("--run-id", default=None)
    prep.add_argument("--metric", default="accuracy")
    prep.add_argument("--allow-unpaired", action="store_true",
                      help="Proceed when models were not scored on identical "
                           "items, reporting the loss. Off by default: the "
                           "dropped items are selected by one model's failures, "
                           "which biases the comparison (I1).")
    prep.set_defaults(func=cmd_report)

    pc = sub.add_parser("compare", help="Paired significance tests between models")
    pc.add_argument("--profile", required=True)
    pc.add_argument("--metric", default="accuracy")
    pc.add_argument("--run-id", default=None)
    pc.add_argument("-a", default=None, help="model A (omit for all pairs)")
    pc.add_argument("-b", default=None, help="model B")
    pc.add_argument("--allow-unpaired", action="store_true",
                    help="Compare on shared items only, reporting the loss.")
    pc.set_defaults(func=cmd_compare)

    pd_ = sub.add_parser("decide", help="Constraint-based model recommendation")
    pd_.add_argument("--profile", required=True)
    pd_.add_argument("--run-id", default=None)
    pd_.add_argument("--require", action="append", default=[],
                     help="e.g. --require 'faithfulness>=0.9'")
    pd_.add_argument("--optimise", "--optimize", dest="optimise",
                     default="cost_usd")
    pd_.add_argument("--quality-metric", default="accuracy")
    pd_.add_argument("--qpd", type=int, default=10_000,
                     help="queries per day, for cost projection")
    pd_.set_defaults(func=cmd_decide)

    pa = sub.add_parser("arena", help="Pairwise head-to-head Elo")
    pa.add_argument("--profile", required=True)
    pa.add_argument("--run-id", default=None)
    pa.add_argument("--models", nargs="*", default=None)
    pa.add_argument("--budget", type=float, default=0.0)
    pa.set_defaults(func=cmd_arena)

    pg = sub.add_parser("gate", help="CI regression gate (non-zero exit on regression)")
    pg.add_argument("--profile", required=True)
    pg.add_argument("--baseline", required=True, help="baseline run_id")
    pg.add_argument("--candidate", default=None,
                    help="candidate run_id (default: all rows for the profile)")
    pg.add_argument("--gate", action="append", default=[],
                    help="metric[:tolerance], e.g. --gate accuracy:0.02")
    pg.add_argument("--floor", action="append", default=[],
                    help="absolute floor, e.g. --floor 'accuracy>=0.8'")
    pg.add_argument("--tolerance", type=float, default=0.02)
    pg.add_argument("--max-cost", type=float, default=None,
                    help="fail if mean cost per query exceeds this")
    pg.add_argument("--allow-unpaired", action="store_true",
                    help="Gate across runs that did not score identical items.")
    pg.add_argument("--strict", action="store_true",
                    help="fail on any regression, even one inside the noise")
    pg.set_defaults(func=cmd_gate)

    add_bench_parser(sub)

    pw = sub.add_parser("serve", help="Local web UI (stdlib server, no framework)")
    pw.add_argument("--port", type=int, default=8000)
    pw.add_argument("--host", default="127.0.0.1",
                    help="Loopback by default: the sign-in is a shared pilot "
                         "password, not user accounts, and the UI can spend money.")
    pw.add_argument("--budget", type=float, default=0.0,
                    help="Hard USD ceiling applied to runs started from the UI.")
    pw.add_argument("--no-browser", action="store_true")
    pw.add_argument("--quiet", action="store_true",
                    help="Suppress the per-request log; job polling is chatty.")
    pw.set_defaults(func=cmd_serve)

    pru = sub.add_parser("runs", help="List recorded runs")
    pru.set_defaults(func=cmd_runs)

    ph = sub.add_parser("html", help="Write a self-contained HTML report")
    ph.add_argument("--profile", required=True)
    ph.add_argument("--run-id", default=None)
    ph.add_argument("--metric", default="accuracy")
    ph.add_argument("--out", default="report.html")
    ph.set_defaults(func=cmd_html)

    args = p.parse_args()
    try:
        return int(args.func(args) or 0)
    except (ProfileError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
