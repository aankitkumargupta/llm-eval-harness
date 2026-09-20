"""
The rest of the platform: preflight, probes, arena, providers, profile runs.

`api.py` covers benchmarks and `profile_api.py` covers the profile report and
the decision layer. What was still only reachable from the CLI is here, because
a dashboard that shows results but cannot answer *"is this config sound and
what will it cost"* leaves the expensive half of the workflow in a terminal.

The ordering is not decorative. The two endpoints that matter most.
`validate` and `estimate`, come first because they are the ones you run
*before* spending money, and §1's whole argument is that the costly mistakes in
evaluation are made before the first API call, not after it.

Same rule as its siblings: presentation holds no business logic. Every figure
comes from `harness.profiles`, `harness.clients.cost`, `harness.eval.probes`
or `harness.report.aggregate`.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from .api import ApiError, _records


# --------------------------------------------------------------------------- #
#  Preflight, the cheap half of the workflow
# --------------------------------------------------------------------------- #
def validate(profile: str, models_cfg: dict | None = None) -> dict:
    """Check a profile, its dataset and its provider routing before spending.

    Returns problems rather than raising on them: the point of this screen is
    to list everything wrong at once, and an exception would surface only the
    first one.
    """
    from ..clients.pricing import PricingRegistry
    from ..profiles.loaders import dataset_stats, load_evalset, validate_gold_ids
    from ..profiles.profile import Profile

    out: dict = {"profile": profile, "ok": False, "problems": [],
                 "warnings": [], "dataset": None, "gold_problems": []}
    try:
        p = Profile.from_yaml(f"configs/profiles/{profile}.yaml")
    except Exception as e:                          # noqa: BLE001 - reported
        out["problems"].append(f"{type(e).__name__}: {e}")
        return out

    out["task"] = p.task.value
    out["accuracy_scorer"] = p.accuracy_scorer
    out["max_tokens"] = p.max_tokens

    try:
        items = load_evalset(p.evalset_path)
    except Exception as e:                          # noqa: BLE001 - reported
        out["problems"].append(f"evalset: {type(e).__name__}: {e}")
        return out

    stats = dataset_stats(items, task=p.task.value)
    out["dataset"] = {
        "n_items": len(items),
        "summary": stats.summary(),
        "evalset_path": p.evalset_path,
        "corpus_path": p.corpus_path,
    }

    if p.corpus_path and Path(p.corpus_path).exists():
        try:
            out["gold_problems"] = list(
                validate_gold_ids(items, p.corpus_path, p.chunk_size,
                                  p.overlap))[:20]
        except Exception as e:                      # noqa: BLE001 - reported
            out["warnings"].append(f"gold check: {type(e).__name__}: {e}")

    cfg = models_cfg or {}
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    missing = [m for m in pricing.missing(
        list(cfg.get("models") or []) + [cfg.get("judge_model", "")]) if m]
    if missing:
        # A warning, not a failure: an unpriced model raises at run time by
        # design (I3), so this is the early notice, not the enforcement.
        out["warnings"].append(
            f"No pricing for {missing}. Their cost metric would be blank "
            f"rather than zero, a free-looking model would win a "
            f"cost-weighted leaderboard.")

    out["ok"] = not out["problems"]
    return out


def estimate(profile: str, models: list[str], models_cfg: dict | None = None,
             run_cfg: dict | None = None, *, do_adapted: bool = False) -> dict:
    """Forecast a run's spend, including the judge.

    The judge is included because on a judge-scored profile it is routinely the
    largest line, and an estimate that omits it is the one people quote before
    being surprised by the bill.
    """
    from ..clients.cost import forecast_run
    from ..clients.pricing import PricingRegistry
    from ..profiles.loaders import load_evalset
    from ..profiles.profile import Profile

    try:
        p = Profile.from_yaml(f"configs/profiles/{profile}.yaml")
        items = load_evalset(p.evalset_path)
    except Exception as e:                          # noqa: BLE001 - reported
        raise ApiError(f"{type(e).__name__}: {e}", 400) from e

    if not models:
        raise ApiError("Pick at least one model.", 400)

    rcfg = run_cfg or {}
    dev_frac = float(rcfg.get("dev_split", 0.3))
    n_dev = max(1, int(len(items) * dev_frac))
    n_test = max(1, len(items) - n_dev)

    cfg = models_cfg or {}
    judge = (cfg.get("judge_model", "")
             if p.accuracy_scorer == "judge" or any(
                 m in p.active_metrics for m in
                 ("faithfulness", "answer_relevance", "completeness")) else "")

    fc = forecast_run(
        PricingRegistry("configs/pricing.yaml", strict=False),
        models, n_test=n_test, n_dev=n_dev,
        tuning_budget=p.tuning_budget, judge_model=judge or "",
        do_baseline=True, do_adapted=do_adapted)

    return {
        "profile": profile,
        "models": list(models),
        "n_test": n_test,
        "n_dev": n_dev,
        "calls": int(fc.calls),
        "judge_calls": int(fc.judge_calls),
        "est_usd": float(fc.est_usd),
        "detail": dict(fc.detail),
        "judge_model": judge,
        "note": ("Approximate: real prompt length depends on your corpus. "
                 "Judge calls are included because on a judge-scored profile "
                 "they are routinely the largest line."),
    }


# --------------------------------------------------------------------------- #
#  Adversarial probes
# --------------------------------------------------------------------------- #
PROBE_FAMILIES = {
    "injection": "A hostile instruction planted in a retrieved passage, "
                 "carrying a canary. If the canary comes back, the model took "
                 "orders from its data.",
    "unanswerable": "A question the corpus provably cannot answer, phrased so "
                    "retrieval still returns confident-looking passages. Does "
                    "the model refuse, or invent a statute?",
    "noise": "Distractor passages mixed into context. Does quality survive "
             "imperfect retrieval, the only kind there is in production?",
    "paraphrase": "The same question reworded. An answer that changes when a "
                  "user rephrases is unreliable even when each one looks fine.",
    "positional": "The gold passage forced to the middle of a long context. "
                  "Separates models that read their whole context from models "
                  "that skim the ends.",
}


def probe_preview(profile: str, *, seed: int = 0, **fractions) -> dict:
    """Generate a probe suite and return what it would add.

    Free and deterministic: probe generation is pure, touches no network and no
    vector store, so this is a preview in the real sense, the same items a run
    would use, not a mock-up of them.
    """
    from ..eval.probes import ProbeConfig, build_probe_suite
    from ..profiles.loaders import load_evalset
    from ..profiles.profile import Profile

    try:
        p = Profile.from_yaml(f"configs/profiles/{profile}.yaml")
        items = load_evalset(p.evalset_path)
    except Exception as e:                          # noqa: BLE001 - reported
        raise ApiError(f"{type(e).__name__}: {e}", 400) from e

    cfg = ProbeConfig(seed=seed, **{k: float(v) for k, v in fractions.items()
                                    if k in ProbeConfig.__dataclass_fields__
                                    and k != "seed"})
    if not cfg.any_enabled():
        raise ApiError("Every probe family is set to zero.", 400)

    suite = build_probe_suite(items, cfg)
    probes = list(getattr(suite, "items", []) or [])

    by_family: dict[str, int] = {}
    samples = []
    for it in probes:
        fam = str(it.meta.get("probe", it.meta.get("family", "?")))
        by_family[fam] = by_family.get(fam, 0) + 1
        if len(samples) < 12:
            samples.append({
                "item_id": it.item_id,
                "family": fam,
                "item_type": getattr(it.item_type, "value", str(it.item_type)),
                "query": it.query[:220],
                # Canaries are NOT echoed here. They are generated as plaintext
                # and live in the item's meta, but printing one into a browser
                # (and a browser history, and a screenshot) is exactly the leak
                # I12 is about.
                "has_canary": bool(it.meta.get("canary")),
            })

    return {
        "profile": profile,
        "base_items": len(items),
        "generated": len(probes),
        "by_family": by_family,
        "families": PROBE_FAMILIES,
        "samples": samples,
        "seed": seed,
        "note": ("Deterministic given the seed, costs nothing, and needs no "
                 "re-ingest, probes reuse the existing corpus."),
    }


# --------------------------------------------------------------------------- #
#  Arena
# --------------------------------------------------------------------------- #
def arena(store_path: str, run_id: str = "", profile: str = "") -> dict:
    """Elo and head-to-head win rates over answers already in the store.

    Read-only over stored answers: no new generation, so no new spend. The
    judging itself would cost money, which is why this endpoint reports what a
    previous `arena` run recorded rather than starting one, a dashboard button
    that quietly bills per click is a bad button.
    """
    from ..report import aggregate as A
    from ..store.store import TraceStore

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet.", 404)
    df = store.load_run(run_id) if run_id else store.load_profile(profile)
    if df.empty:
        raise ApiError("No rows for that selection.", 404)

    if "arena_winner" not in df.columns or df["arena_winner"].isna().all():
        return {
            "run_id": run_id, "profile": profile, "available": False,
            "reason": ("No pairwise judgements stored for this run. Arena "
                       "judging costs a judge call per pair, so it runs from "
                       "the CLI (`python main.py arena`) rather than from a "
                       "button that would bill per click."),
        }

    pairwise = [(str(a), str(b), str(w)) for a, b, w in zip(
        df["arena_a"], df["arena_b"], df["arena_winner"]) if w]
    elo = A.elo_from_pairwise(pairwise)
    wins = A.win_rate_matrix(pairwise)
    return {
        "run_id": run_id, "profile": profile, "available": True,
        "comparisons": len(pairwise),
        "elo": _records(elo) if not elo.empty else [],
        "win_rates": _records(wins.reset_index()) if not wins.empty else [],
    }


# --------------------------------------------------------------------------- #
#  Providers
# --------------------------------------------------------------------------- #
def providers() -> dict:
    """Every provider the harness knows, and what it can actually do.

    Read from `endpoints.py`, which is deliberately SDK-free, importing an
    adapter just to ask "can this vendor embed?" used to cost ~19s on first
    page load. Capability comes from the table and the adapter agreeing, so
    the two cannot drift (see `clients/base.py`).
    """
    import os

    from ..clients.endpoints import NON_OPENAI_PROVIDERS, all_providers

    rows = []
    for name, spec in sorted(all_providers().items()):
        env = spec.get("api_key_env", "")
        rows.append({
            "provider": name,
            "base_url": spec.get("base_url", "") or ("(SDK default)"
                                                     if name in NON_OPENAI_PROVIDERS
                                                     else ""),
            "api_key_env": env,
            "key_present": bool(env and os.environ.get(env)) or bool(
                spec.get("default_api_key")),
            "embeddings": bool(spec.get("supports_embeddings", True)),
            "rerank": bool(spec.get("supports_rerank", False)),
            "seed": bool(spec.get("supports_seed", True)),
            "local": bool(spec.get("local", False)),
        })
    return {"providers": rows}


# --------------------------------------------------------------------------- #
#  Profile runs
# --------------------------------------------------------------------------- #
def start_profile_run(*, profile: str, models: list[str], store_path: str,
                      models_cfg: dict, run_cfg: dict, budget: float = 0.0,
                      passes: dict | None = None) -> dict:
    """Run a profile from the browser. Same path the CLI takes."""
    from ..orchestration.jobs import any_running, new_job
    from ..profiles.profile import Profile

    try:
        Profile.from_yaml(f"configs/profiles/{profile}.yaml")
    except Exception as e:                          # noqa: BLE001 - reported
        raise ApiError(f"{type(e).__name__}: {e}", 400) from e
    if not models:
        raise ApiError("Pick at least one model.", 400)
    if any_running():
        raise ApiError("A run is already in flight. One at a time.", 409)

    job = new_job("profile", total=1)
    run_id = f"prof_{uuid.uuid4().hex[:8]}"
    threading.Thread(
        target=_profile_worker,
        args=(job, run_id, profile, list(models), store_path,
              dict(models_cfg or {}), dict(run_cfg or {}), float(budget),
              dict(passes or {"baseline": True, "latency": True})),
        daemon=True).start()
    return {"job_id": job.id, "run_id": run_id, "total": job.total}


def _profile_worker(job, run_id, profile_name, models, store_path, models_cfg,
                    run_cfg, budget, passes):
    """Worker thread. Mutates `job` and nothing else (see `jobs.py`)."""
    import random

    try:
        from ..cache.cache import stable_hash
        from ..clients.cost import CostMeter
        from ..clients.pricing import PricingRegistry
        from ..clients.registry import build_from_config
        from ..orchestration.orchestrator import Orchestrator
        from ..profiles.loaders import load_evalset
        from ..profiles.profile import Profile
        from ..store import manifest as MF
        from ..store.store import TraceStore

        p = Profile.from_yaml(f"configs/profiles/{profile_name}.yaml")
        items = load_evalset(p.evalset_path)

        rng = random.Random(int(run_cfg.get("split_seed", 0)))
        idx = list(range(len(items)))
        rng.shuffle(idx)
        cut = max(1, min(int(len(items) * float(run_cfg.get("dev_split", 0.3))),
                         len(items) - 1)) if items else 0
        # The dev split is held out, not scored: the adapted pass is a CLI
        # concern and this endpoint runs baseline + latency only.
        test = [items[i] for i in idx[cut:]] or items

        pricing = PricingRegistry("configs/pricing.yaml", strict=False)
        meter = CostMeter(budget_usd=budget or 0.0)
        cfg = dict(models_cfg)
        cfg["models"] = models
        client = build_from_config(cfg, meter=meter, pricing=pricing)
        store = TraceStore(store_path)

        needs_judge = (p.accuracy_scorer == "judge" or any(
            m in p.active_metrics for m in
            ("faithfulness", "answer_relevance", "completeness")))
        judge_model = cfg.get("judge_model") if needs_judge else None

        apparatus = MF.apparatus_hash(
            embedding_model=p.embedding_model,
            rerank_model=cfg.get("rerank_model", "") or "",
            judge_model=judge_model or "",
            embedding_provider=cfg.get("embedding_provider", "") or "",
            judge_provider=cfg.get("judge_provider", "") or "")
        mf = MF.RunManifest.capture(
            run_id, p.name, models=tuple(models),
            profile_cfg_hash=stable_hash(*p.config_hash_parts()),
            dataset_hash=MF.dataset_hash(items), apparatus_hash=apparatus,
            embedding_model=p.embedding_model, judge_model=judge_model or "",
            seeds={"split_seed": run_cfg.get("split_seed", 0)},
            pricing_as_of=getattr(pricing, "as_of", "unknown"),
            budget_usd=float(budget or 0.0))
        mdir = Path(store_path).parent / run_id
        mf.write(mdir)

        # A RAG profile needs a vector store. Passing qdrant=None for one would
        # run it with no retrieval at all and still write rows, a silently
        # context-free run that looks like a legitimately bad model. Refuse
        # instead, and say what to do.
        qdrant = None
        if p.task.value == "rag":
            qdrant = _open_qdrant(run_cfg)
            if qdrant is None:
                raise RuntimeError(
                    f"Profile '{p.name}' is a RAG profile, so it needs a vector "
                    f"store. Set `qdrant_path` in configs/run.yaml for embedded "
                    f"mode, or point `qdrant_url` at a running server, then "
                    f"ingest the corpus with "
                    f"`python main.py ingest --profile {p.name}`.")
            if not _collection_ready(qdrant, p):
                raise RuntimeError(
                    f"No ingested corpus for '{p.name}'. Run "
                    f"`python main.py ingest --profile {p.name}` first, a RAG "
                    f"run against an empty index retrieves nothing and scores "
                    f"every item as a model failure.")

        job.set_total(max(1, len(models) * len(test)))
        orch = Orchestrator(
            client=client, qdrant=qdrant, pricing=pricing, store=store,
            judge_model=judge_model,
            cache_dir=run_cfg.get("cache_dir", ".cache"),
            rerank_model=cfg.get("rerank_model", ""), meter=meter,
            max_workers=int(run_cfg.get("max_workers", 4)),
            progress_cb=lambda done, total, model: job.set_done_absolute(done))

        rows = 0
        ran = []
        if passes.get("baseline", True):
            rep = orch.run_baseline(p, models, test, run_id)
            rows += len(getattr(rep, "rows", []) or [])
            ran.append("baseline")
            if getattr(rep, "aborted", False):
                raise RuntimeError("aborted at the budget ceiling")
        if passes.get("latency", False):
            lat = test[: int(run_cfg.get("latency_items", 20))]
            orch.run_latency(p, models, lat, run_id)
            ran.append("latency")

        mf.passes = tuple(ran)
        mf.n_rows = rows
        mf.total_cost_usd = float(meter.summary().get("total_usd", 0.0))
        mf.write(mdir)
        job.finish(rows=rows,
                   message=f"{run_id} · {', '.join(ran)} · ${meter.spent:.4f}")
    except Exception as e:                          # noqa: BLE001 - reported
        job.fail(f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
#  Vector store
# --------------------------------------------------------------------------- #
def _open_qdrant(run_cfg: dict):
    """Open Qdrant the same way the CLI does: embedded folder, or a server.

    Returns None when neither is configured, so the caller can refuse a RAG
    run with a message instead of running it without retrieval.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return None

    path = run_cfg.get("qdrant_path")
    if path:
        return QdrantClient(path=str(path))
    url = run_cfg.get("qdrant_url")
    if not url:
        return None
    try:
        client = QdrantClient(url=url)
        client.get_collections()        # fail fast rather than mid-run
    except Exception:                   # noqa: BLE001 - reported by the caller
        return None
    return client


def _collection_ready(qdrant, profile) -> bool:
    """Whether this profile's corpus has actually been ingested.

    An empty index is the worst failure shape available: retrieval returns
    nothing, every answer is unsupported, and the report reads as a uniformly
    terrible set of models rather than a missing ingest.
    """
    name = getattr(profile, "collection_name", None)
    name = name() if callable(name) else name
    if not name:
        return False
    try:
        return qdrant.collection_exists(name) and qdrant.count(name).count > 0
    except Exception:                   # noqa: BLE001 - treated as not ready
        return False


def rag_status(run_cfg: dict, search_dir: str = "configs/profiles") -> dict:
    """Which RAG profiles are ingested and ready to run.

    The dashboard needs this because "RAG profile with an empty index" and
    "RAG profile ready" look identical until a run produces uniformly awful
    numbers.
    """
    from ..profiles.profile import Profile

    qdrant = _open_qdrant(run_cfg)
    mode = ("embedded" if run_cfg.get("qdrant_path")
            else ("server" if run_cfg.get("qdrant_url") else "none"))
    rows = []
    for path in sorted(Path(search_dir).glob("*.yaml")):
        try:
            p = Profile.from_yaml(str(path))
        except Exception:               # noqa: BLE001 - listed elsewhere
            continue
        if p.task.value != "rag":
            continue
        name = getattr(p, "collection_name", None)
        name = name() if callable(name) else name
        count = 0
        if qdrant is not None and name:
            try:
                if qdrant.collection_exists(name):
                    count = int(qdrant.count(name).count)
            except Exception:           # noqa: BLE001
                count = 0
        rows.append({
            "profile": p.name,
            "collection": name or "",
            "embedding_model": p.embedding_model,
            "corpus_path": p.corpus_path,
            "corpus_present": bool(p.corpus_path) and Path(p.corpus_path).exists(),
            "chunks_indexed": count,
            "ready": count > 0,
        })
    return {
        "vector_store": mode,
        "reachable": qdrant is not None,
        "profiles": rows,
        "note": ("Changing the embedder changes the collection name, so a "
                 "re-ingest is required, vectors from two different embedders "
                 "are not comparable."),
    }


# --------------------------------------------------------------------------- #
#  Provider connections
# --------------------------------------------------------------------------- #
#
# The network calls themselves live in `harness/clients/discovery.py`, because
# §3 says the provider layer is the only code permitted to touch the network.
# These two functions are the thin presentation wrapper over it.


def connections(include_models: bool = False) -> dict:
    """Which providers are reachable right now, from the environment."""
    from ..clients.discovery import LISTABLE, check_all

    rows = [r.as_dict() for r in check_all(include_models=include_models)]
    return {
        "connections": rows,
        "listable": list(LISTABLE),
        "note": ("A key being *set* and a key *working* are different facts. "
                 "These rows come from a real /models request, so 'ok' means "
                 "the provider answered, not that an env var exists."),
    }


def connection_check(provider: str, api_key: str | None = None) -> dict:
    """Check one provider and list its catalogue.

    `api_key`, when given, is used for this single request and then dropped. It
    is never written to a config, a cache key, the trace store or a log: §6
    keeps secrets in the environment, and a diagnostic endpoint is the classic
    place that rule gets quietly broken. The response carries only a masked
    form of whatever key was used.
    """
    from ..clients.discovery import check_provider

    res = check_provider(provider, api_key=api_key, include_models=True)
    out = res.as_dict()
    if not res.ok:
        # A failed check is the normal case on this screen, not an exception.
        out["hint"] = _connection_hint(provider, res)
    return out


def _connection_hint(provider: str, res) -> str:
    from ..clients.endpoints import PROVIDER_ENDPOINTS

    env = PROVIDER_ENDPOINTS.get(provider, {}).get("api_key_env", "")
    if res.key_source == "none":
        return (f"Set {env} in your shell, or put `{env}=...` in a .env file "
                f"next to main.py, the harness loads it, and a real "
                f"environment variable always wins over the file.")
    if res.status == 401:
        return (f"The key in {env} was rejected. Keys are per-account and per "
                f"-project; check you copied the right one.")
    return "See the error above."


def suggested_models_yaml(provider: str, model_ids: list[str]) -> dict:
    """The models.yaml fragment for a chosen set of models.

    Offered as text to copy rather than written to disk. §7 lists provider
    routing and model choice among the things that change *results*, so the
    harness proposes and a human commits.
    """
    if not model_ids:
        raise ApiError("Pick at least one model.", 400)
    lines = ["models:"] + [f'  - "{provider}:{m}"' for m in model_ids]
    return {
        "provider": provider,
        "yaml": "\n".join(lines),
        "note": ("Paste into configs/models.yaml. Add a pricing entry for each "
                 "model in configs/pricing.yaml as well, an unpriced model is "
                 "a hard validation failure by design, because a free-looking "
                 "model would win a cost-weighted leaderboard."),
    }
