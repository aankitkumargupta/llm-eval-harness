"""
UI-facing run functions — what the app's background threads execute.

These reuse the same engine as the CLI (Ingestor, Orchestrator, TraceStore) and
translate its progress into `Job` updates. They take no Streamlit dependency, so
they stay importable and testable on their own.

Rewritten to go through `Orchestrator` rather than reimplementing the matrix
walk. The old version was a second, subtly different copy of the same logic:
it wrote the whole store on every tuning candidate (O(n^2)), had no budget
ceiling, no resume, no cost metering, and mutated `Job.messages` without the
lock the rest of the class is careful to hold. Two implementations of "run the
matrix" also meant a bug fixed in one stayed alive in the other.

Entry points:
  ingest_job(...)  -> embeds + indexes a profile's corpus, ticking per batch
  eval_job(...)    -> baseline (+ optional adapted/latency), ticking per item
  probe_job(...)   -> generates an adversarial probe set (no network, instant)
"""

from __future__ import annotations

import random

from ..clients.cost import CostMeter
from ..clients.pricing import PricingRegistry
from ..clients.registry import build_from_config
from ..clients.resilience import RetryPolicy
from ..eval.probes import ProbeConfig, build_probe_suite
from ..profiles.loaders import load_corpus, load_evalset, save_evalset
from ..profiles.profile import Profile
from ..store.schema import TaskType
from ..store.store import TraceStore
from .jobs import Job
from .orchestrator import Orchestrator


def _qdrant(path: str):
    """Open the embedded vector store.

    Imported here rather than at module scope: `qdrant_client` costs ~25s to
    import on a cold install, and `app.py` imports this module at startup. Paying
    that only when a job actually opens the store keeps the app's first paint
    fast, and hides the cost inside an operation that takes minutes anyway.
    """
    from qdrant_client import QdrantClient

    return QdrantClient(path=path)


def _count_chunks(corpus_path: str, chunk_size: int, overlap: int) -> int:
    """Pre-count chunks so the ingest progress bar has a real denominator."""
    from ..rag.ingest import chunk_text

    n = 0
    for doc in load_corpus(corpus_path):
        n += len(chunk_text(doc.text, chunk_size, overlap))
    return max(1, n)


def _split(items, dev_frac: float, seed: int):
    """Deterministic dev/test split that can never produce an empty side."""
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


def ingest_job(job: Job, profile: Profile, qdrant_path: str,
               api_key: str | None, chunk_size: int, overlap: int,
               batch_size: int = 128, models_cfg: dict | None = None) -> None:
    """Background ingestion. Embedded Qdrant (a local folder), so no Docker."""
    qdrant = None
    try:
        total = _count_chunks(profile.corpus_path, chunk_size, overlap)
        job.set_total(total)

        pricing = PricingRegistry("configs/pricing.yaml", strict=False)
        meter = CostMeter()
        client = build_from_config(models_cfg or {}, meter=meter,
                                   pricing=pricing, api_key=api_key)
        from ..rag.ingest import Ingestor

        qdrant = _qdrant(qdrant_path)
        ingestor = Ingestor(client, qdrant, profile.embedding_model)

        n = ingestor.ingest(
            collection=profile.collection_name(),
            documents=load_corpus(profile.corpus_path),
            chunk_size=chunk_size, overlap=overlap, batch_size=batch_size,
            progress_cb=job.set_done_absolute,
        )
        job.finish(rows=n, message=f"Indexed {n} chunks into "
                                   f"'{profile.collection_name()}' "
                                   f"(${meter.costs.embedding:.4f} in embeddings).")
    except Exception as e:  # noqa: BLE001
        job.fail(f"{type(e).__name__}: {e}")
    finally:
        if qdrant is not None:
            try:
                qdrant.close()
            except Exception:
                pass


def eval_job(job: Job, profile: Profile, models: list[str],
             qdrant_path: str, store_path: str, cache_dir: str,
             api_key: str | None, judge_model: str | None,
             rerank_model: str, run_passes: dict, dev_split: float,
             split_seed: int, max_workers: int = 4,
             budget_usd: float = 0.0,
             models_cfg: dict | None = None,
             resume_from: str | None = None) -> None:
    """Background evaluation, driven by the same Orchestrator the CLI uses."""
    qdrant = None
    try:
        models_cfg = dict(models_cfg or {})
        models_cfg.setdefault("judge_model", judge_model)
        models_cfg.setdefault("rerank_model", rerank_model)

        pricing = PricingRegistry("configs/pricing.yaml", strict=False)
        meter = CostMeter(budget_usd=budget_usd)
        client = build_from_config(models_cfg, meter=meter, pricing=pricing,
                                   api_key=api_key, retry=RetryPolicy())

        problems = client.preflight(models, profile.embedding_model, rerank_model)
        if problems:
            job.fail("Provider routing problems: " + "; ".join(problems))
            return

        if profile.task == TaskType.RAG:
            qdrant = _qdrant(qdrant_path)

        store = TraceStore(store_path)
        items = load_evalset(profile.evalset_path)
        dev, test = _split(items, dev_split, split_seed)

        do_baseline = run_passes.get("baseline", True)
        do_adapted = run_passes.get("adapted", False)
        do_latency = run_passes.get("latency", False)

        # Progress denominator, so the bar is honest about how long this takes:
        # the adapted pass is roughly tuning_budget x bigger than the baseline,
        # which is exactly why it is off by default.
        total = 0
        if do_baseline:
            total += len(test) * len(models)
        if do_adapted:
            total += (profile.tuning_budget * len(dev) + len(test)) * len(models)
        if do_latency:
            total += min(20, len(test)) * len(models)
        job.set_total(max(1, total))

        completed = {"n": 0}

        def progress(done: int, sub_total: int, model: str) -> None:
            # Each pass reports progress within itself, so accumulate across
            # passes rather than letting the bar reset at every pass boundary.
            job.set_done_absolute(completed["n"] + done)

        orch = Orchestrator(
            client=client, qdrant=qdrant, pricing=pricing, store=store,
            judge_model=judge_model, cache_dir=cache_dir,
            rerank_model=rerank_model, meter=meter,
            judge_ensemble=models_cfg.get("judge_ensemble"),
            max_workers=max_workers, checkpoint_every=25,
            progress_cb=progress,
        )

        run_id = job.id
        if do_baseline:
            rep = orch.run_baseline(profile, models, test, run_id,
                                    resume_from=resume_from)
            completed["n"] += len(test) * len(models)
            job.advance(0, f"Baseline: {rep.rows_written} rows, "
                           f"{rep.errors} errors, ${meter.spent:.4f}.")
            if rep.aborted:
                job.fail(rep.abort_reason)
                return

        if do_adapted:
            rep = orch.run_adapted(profile, models, dev, test, run_id)
            completed["n"] += (profile.tuning_budget * len(dev) + len(test)) * len(models)
            for model, info in rep.winners.items():
                job.advance(0, f"{model} best config: {info['config']}")
            if rep.aborted:
                job.fail(rep.abort_reason)
                return

        if do_latency:
            rep = orch.run_latency(profile, models, test[:20], run_id)
            job.advance(0, f"Latency: {rep.rows_written} clean-timing rows.")

        cost = meter.summary()
        job.finish(
            rows=orch._written,
            message=(f"Done. ${cost['total_usd']:.4f} "
                     f"(generation ${cost['generation_usd']:.4f}, "
                     f"judge ${cost['judge_usd']:.4f}, "
                     f"embeddings ${cost['embedding_usd']:.4f}); "
                     f"cache hit rate {cost['cache_hit_rate']:.0%}."))
    except Exception as e:  # noqa: BLE001
        job.fail(f"{type(e).__name__}: {e}")
    finally:
        if qdrant is not None:
            try:
                qdrant.close()
            except Exception:
                pass


def probe_job(job: Job, profile: Profile, cfg: ProbeConfig,
              append: bool = True) -> None:
    """Generate adversarial probe items. Pure CPU: no network, no spend."""
    try:
        items = load_evalset(profile.evalset_path)
        job.set_total(max(1, len(items)))
        suite = build_probe_suite(items, cfg)
        if not suite.items:
            job.finish(rows=0, message="No probes generated (all fractions zero).")
            return
        if append:
            combined = items + suite.items
            save_evalset(combined, profile.evalset_path)
            msg = (f"{suite.summary()}; evalset is now {len(combined)} items. "
                   f"Re-ingest is NOT needed - probes reuse the same corpus.")
        else:
            out = str(profile.evalset_path).replace(".jsonl", "_probes.jsonl")
            save_evalset(suite.items, out)
            msg = f"{suite.summary()} written to {out}."
        job.set_done_absolute(len(items))
        job.finish(rows=len(suite.items), message=msg)
    except Exception as e:  # noqa: BLE001
        job.fail(f"{type(e).__name__}: {e}")
