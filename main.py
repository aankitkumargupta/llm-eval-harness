"""
Command-line entry point for the evaluation harness.

Usage:
  # 1) One-time: ingest each profile's corpus into Qdrant
  python main.py ingest --profile configs/profiles/regulated_qa.yaml

  # 2) Run the full matrix (baseline + adapted + latency) for one profile
  python main.py run --profile configs/profiles/regulated_qa.yaml

  # 3) Produce the leaderboards/reports from the trace store
  python main.py report --profile regulated_qa

Requires:
  * TOGETHER_API_KEY in the environment
  * A running Qdrant (docker run -p 6333:6333 qdrant/qdrant)
"""

from __future__ import annotations

import argparse
import random
import uuid

import yaml
from qdrant_client import QdrantClient

from harness.clients.pricing import PricingRegistry
from harness.clients.together_client import TogetherClient
from harness.orchestration.orchestrator import Orchestrator
from harness.profiles.loaders import load_corpus, load_evalset
from harness.profiles.profile import Profile
from harness.rag.ingest import Ingestor
from harness.report.aggregate import (
    bootstrap_ci, pareto_frontier, tuning_gain, weighted_composite,
)
from harness.store.store import TraceStore


def _load_run_cfg(path: str = "configs/run.yaml") -> dict:
    return yaml.safe_load(open(path).read())


def _load_models_cfg(path: str = "configs/models.yaml") -> dict:
    return yaml.safe_load(open(path).read())


def cmd_ingest(args):
    run_cfg = _load_run_cfg()
    profile = Profile.from_yaml(args.profile)
    client = TogetherClient()
    qdrant = QdrantClient(url=run_cfg["qdrant_url"])
    ingestor = Ingestor(client, qdrant, profile.embedding_model)

    n = ingestor.ingest(
        collection=profile.collection_name(),
        documents=load_corpus(profile.corpus_path),
        chunk_size=args.chunk_size, overlap=args.overlap,
        batch_size=args.batch_size,
    )
    print(f"[ingest] {profile.name}: indexed {n} chunks into "
          f"'{profile.collection_name()}'")


def _split(items, dev_frac: float, seed: int):
    """Deterministic dev/test split."""
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    cut = int(len(items) * dev_frac)
    dev = [items[i] for i in idx[:cut]]
    test = [items[i] for i in idx[cut:]]
    return dev, test


def cmd_run(args):
    run_cfg = _load_run_cfg()
    models_cfg = _load_models_cfg()
    profile = Profile.from_yaml(args.profile)

    client = TogetherClient()
    qdrant = QdrantClient(url=run_cfg["qdrant_url"])
    pricing = PricingRegistry()
    store = TraceStore(run_cfg["store_path"])
    orch = Orchestrator(
        client=client, qdrant=qdrant, pricing=pricing, store=store,
        judge_model=models_cfg.get("judge_model"),
        cache_dir=run_cfg["cache_dir"],
        rerank_model=models_cfg.get("rerank_model", ""),
    )

    models = models_cfg["models"]
    items = load_evalset(profile.evalset_path)
    dev, test = _split(items, run_cfg["dev_split"], run_cfg["split_seed"])
    run_id = uuid.uuid4().hex[:12]
    print(f"[run] profile={profile.name} run_id={run_id} "
          f"models={len(models)} dev={len(dev)} test={len(test)}")

    if run_cfg["passes"].get("baseline"):
        rows = orch.run_baseline(profile, models, test, run_id,
                                 max_workers=run_cfg["max_workers"])
        print(f"[run] baseline: {len(rows)} rows")

    if run_cfg["passes"].get("adapted"):
        rows = orch.run_adapted(profile, models, dev, test, run_id,
                                max_workers=run_cfg["max_workers"])
        print(f"[run] adapted (incl. tuning): {len(rows)} rows")

    if run_cfg["passes"].get("latency"):
        lat_items = test[: run_cfg["latency_items"]]
        rows = orch.run_latency(profile, models, lat_items, run_id)
        print(f"[run] latency: {len(rows)} clean-timing rows")

    print(f"[run] done. traces at {run_cfg['store_path']}")


def cmd_report(args):
    run_cfg = _load_run_cfg()
    profile = Profile.from_yaml(f"configs/profiles/{args.profile}.yaml")
    store = TraceStore(run_cfg["store_path"])
    df = store.query(
        f"SELECT * FROM traces WHERE profile = '{args.profile}'"
    )
    if df.empty:
        print(f"No traces for profile '{args.profile}'. Run it first.")
        return

    print(f"\n=== {args.profile}: weighted composite (test split) ===")
    test_df = df[df["pass_"].isin(["baseline", "adapted"])]
    print(weighted_composite(test_df, profile.metric_weights).to_string(index=False))

    print(f"\n=== tuning gain (adapted - baseline) ===")
    metrics = [m for m in profile.active_metrics if m in df.columns]
    print(tuning_gain(df, metrics).to_string(index=False))

    print(f"\n=== Pareto frontier (accuracy vs cost vs latency) ===")
    # merge latency rows in for the latency dimension
    print(pareto_frontier(df).to_string(index=False))

    print(f"\n=== accuracy 95% CI per model (baseline) ===")
    base = df[df["pass_"] == "baseline"]
    for model, g in base.groupby("model"):
        if "accuracy" in g:
            mean, lo, hi = bootstrap_ci(g["accuracy"].to_numpy())
            print(f"  {model:<48} {mean:.3f}  [{lo:.3f}, {hi:.3f}]")


def main():
    p = argparse.ArgumentParser(description="Multi-model LLM evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="Ingest a profile's corpus into Qdrant")
    pi.add_argument("--profile", required=True)
    pi.add_argument("--chunk-size", type=int, default=200)
    pi.add_argument("--overlap", type=int, default=40)
    pi.add_argument("--batch-size", type=int, default=128)
    pi.set_defaults(func=cmd_ingest)

    pr = sub.add_parser("run", help="Run baseline + adapted + latency for a profile")
    pr.add_argument("--profile", required=True)
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="Print leaderboards from the trace store")
    prep.add_argument("--profile", required=True, help="profile NAME, e.g. regulated_qa")
    prep.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
