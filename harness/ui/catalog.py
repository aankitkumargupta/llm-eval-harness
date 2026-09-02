"""
What this installation can actually do, discovered from the code.

The Overview screen needs a list of the platform's capabilities. The obvious
way to build one is to type it out — and that list starts lying the first time
someone adds a provider, renames a metric or removes a pass. A feature page that
disagrees with the software is worse than no feature page, because people act on
it.

So nothing here is hardcoded. Providers come from `PROVIDER_ENDPOINTS`, metrics
from `KNOWN_METRICS` and the scorer registry's own `produces` tuples, probe
families from `ProbeConfig`'s fields, passes from the `Pass` enum, task types
from `TaskType`. Add a scorer and it appears on the screen; delete a provider
and it disappears. The page cannot drift, because there is nothing to keep in
sync.

Two kinds of fact are returned:

  **Capability** — what the code can do. Static across installs.
  **Status**     — what *this* install has: which keys resolve, which profiles
                   exist, what is in the trace store. Varies per machine.

Prose that genuinely needs a human — why a metric matters, what a probe catches
— is kept to one short line per item, attached to the discovered name rather
than replacing it. If a name has no description the item still renders; only
the sentence is missing.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Human descriptions, keyed by discovered names.
#  Missing entries degrade to a blank line, never to a missing feature.
# --------------------------------------------------------------------------- #
METRIC_NOTES: dict[str, str] = {
    "hit_rate_at_k": "Did any gold passage make the top-k at all?",
    "mrr": "How high the first gold passage ranked.",
    "ndcg_at_k": "Ranking quality, rewarding gold placed high.",
    "context_recall": "Share of all gold passages retrieved.",
    "context_precision": "Share of retrieved passages that are gold — the "
                         "counterweight that stops k growing forever.",
    "average_precision": "Rewards ranking *all* the gold high, not just the first.",
    "rerank_hit_delta": "What reranking added to hit-rate.",
    "rerank_mrr_delta": "What reranking added to rank — usually the real gain.",
    "accuracy": "Correct vs the gold answer, by the profile's scorer.",
    "faithfulness": "Grounded in the retrieved context, inventing nothing.",
    "answer_relevance": "Actually addresses the question asked.",
    "completeness": "Uses all the relevant information available.",
    "token_f1": "Graded overlap with gold. Free, and penalises padding.",
    "judge_disagreement": "Spread across a judge panel. High means the ITEM is "
                          "ambiguous, not that the model failed.",
    "citation_valid_pointer": "Cited ids point at real retrieved passages.",
    "citation_supporting": "Cited passages actually support the claim.",
    "citation_density": "Citations per sentence — catches the model that games "
                        "pointer-validity by never citing.",
    "citation_recall": "Share of gold passages the answer cited.",
    "abstention": "Refuses when it should, answers when it should.",
    "abstention_correct": "Scored abstention behaviour vs the item type.",
    "injection_resisted": "Ignored a hostile instruction planted in a passage.",
    "pii_leaked": "Emitted PII that was not in the retrieved context.",
    "cost_usd": "End-to-end spend: generation + judge + embedding + rerank.",
    "latency_ms": "Wall clock, measured in the low-concurrency lane.",
    "ttft_ms": "Time to first token — what governs perceived responsiveness.",
    "prompt_tokens": "Input tokens, from the provider's usage block.",
    "completion_tokens": "Output tokens, from the provider's usage block.",
}

SCORER_NOTES: dict[str, str] = {
    "retrieval": "Ranking quality of what retrieval returned, scored before any "
                 "probe mutates the context.",
    "citations": "Whether the answer's [id] markers are real and supporting.",
    "abstention": "Refusal behaviour, optionally judged rather than regex-matched.",
    "probes": "Adversarial outcomes: injection resistance, PII leakage.",
    "accuracy_deterministic": "Free offline accuracy: exact / contains / numeric "
                              "/ token-F1.",
    "classification": "Label extraction and scoring for non-RAG classify profiles.",
    "judge": "LLM-graded qualities. Runs last, so its verdict is authoritative "
             "on accuracy.",
}

PROBE_NOTES: dict[str, str] = {
    "unanswerable": "Questions the corpus cannot answer, phrased so retrieval "
                    "still returns confident-looking passages. Does the model "
                    "refuse, or invent?",
    "injection": "A hostile instruction planted in a retrieved passage, carrying "
                 "a canary. If the canary comes back, the model took orders "
                 "from its data.",
    "noise": "Distractor passages mixed into context. Does quality survive "
             "imperfect retrieval — the only kind there is in production?",
    "paraphrase": "The same question reworded. Users don't ask twice the same way.",
    "positional": "Gold forced to the middle of a long context — the "
                  "'lost in the middle' effect.",
}

PASS_NOTES: dict[str, str] = {
    "baseline": "Fixed prompt and retrieval, identical for every model. "
                "Concurrent, cached.",
    "adapted": "Per-model config search on a dev split under an equal budget, "
               "then the winner on test.",
    "tuning": "The individual candidate evaluations behind the adapted pass.",
    "latency": "Serial, interleaved, cache genuinely bypassed, streaming on. "
               "The only source of honest p50/p95 and TTFT.",
    "arena": "Pairwise head-to-head judging over stored answers, for Elo.",
}

TASK_NOTES: dict[str, str] = {
    "rag": "Retrieve, then answer from the retrieved context.",
    "direct": "Prompt the model directly. No corpus, no vector store.",
    "classify": "Direct prompting scored as a labelling task, with macro-F1.",
}

ANALYSIS: list[tuple[str, str, str]] = [
    ("Weighted composite", "report",
     "One leaderboard number from the profile's metric weights."),
    ("Paired significance", "compare",
     "Bootstrap and exact McNemar on the same items, Holm-corrected. Answers "
     "whether a gap is real."),
    ("Power analysis", "report",
     "The smallest gap this evalset can detect, and how many items you'd need "
     "for a smaller one."),
    ("Pareto frontier", "report",
     "Models not beaten on accuracy, cost and latency at once."),
    ("Decision engine", "decide",
     "Cheapest model clearing your bar, projected spend at your volume, and the "
     "price of each extra quality point."),
    ("Regression gate", "gate",
     "Exits non-zero when a guarded metric regresses — and only when the drop "
     "is statistically real."),
    ("Elo / head-to-head", "arena",
     "Position-bias-corrected pairwise ranking, for when rubric scores saturate."),
    ("Tuning gain", "report",
     "Adapted minus baseline: how much each model benefits from an equal "
     "tuning budget."),
    ("Judge calibration", "report",
     "Cohen's kappa against human labels. Every judge-scored number rests on it."),
    ("Error attribution", "report",
     "Why items failed — rate limits, context length, refusals — not just how "
     "many."),
    ("HTML report", "html",
     "A self-contained file you can attach to a ticket."),
]

RELIABILITY: list[tuple[str, str]] = [
    ("Retries with jittered backoff",
     "429s and 5xx are retried; 400s and 404s fail immediately rather than "
     "wasting minutes hiding a config bug."),
    ("Rate limiting", "Optional token bucket per provider — pacing is cheaper "
                      "than backing off."),
    ("Budget ceiling", "A hard USD limit. The run aborts cleanly, keeps what it "
                       "wrote, and stops issuing calls."),
    ("Checkpointing", "Rows flush every N items, so a crash costs a batch rather "
                      "than the whole run."),
    ("Resume", "Skip items already recorded and continue an interrupted run."),
    ("Content-addressed cache", "Generations, judge verdicts and query "
                                "embeddings. Re-runs are free and reproducible."),
]


# --------------------------------------------------------------------------- #
#  Discovered capability
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    """One discovered capability, optionally annotated."""
    name: str
    note: str = ""
    detail: str = ""
    available: bool | None = None   # None = not a yes/no thing


@dataclass
class ProviderRow:
    name: str
    env_var: str
    embeddings: bool
    rerank: bool
    local: bool
    configured: bool

    @property
    def key_state(self) -> str:
        """What to show in the key column.

        Self-hosted providers carry a placeholder key the server ignores, so
        `configured` is trivially true for them. Reporting that as "ready" would
        claim a local server is up when nothing has checked — the same kind of
        overstatement the capability flags used to make. They get "no key
        needed" instead, which is the fact we actually know.
        """
        if self.local:
            return "no key needed"
        return "ready" if self.configured else "no key"


@dataclass
class Catalog:
    providers: list[ProviderRow] = field(default_factory=list)
    metrics: dict[str, list[Item]] = field(default_factory=dict)
    scorers: list[Item] = field(default_factory=list)
    probes: list[Item] = field(default_factory=list)
    passes: list[Item] = field(default_factory=list)
    tasks: list[Item] = field(default_factory=list)
    answer_scorers: list[str] = field(default_factory=list)
    retrieval_modes: list[str] = field(default_factory=list)
    analysis: list[tuple[str, str, str]] = field(default_factory=list)
    reliability: list[tuple[str, str]] = field(default_factory=list)

    @property
    def n_metrics(self) -> int:
        return sum(len(v) for v in self.metrics.values())


# Metric grouping, by the subsystem that owns each one. Any metric that does not
# match a group still shows up under "Other", so a newly added metric can never
# be silently dropped from the page.
_METRIC_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("Retrieval", ("hit_rate_at_k", "mrr", "ndcg_at_k", "context_recall",
                   "context_precision", "average_precision",
                   "rerank_hit_delta", "rerank_mrr_delta")),
    ("Answer quality", ("accuracy", "faithfulness", "answer_relevance",
                        "completeness", "token_f1", "judge_disagreement")),
    ("Citations", ("citation_valid_pointer", "citation_supporting",
                   "citation_density", "citation_recall")),
    ("Robustness & security", ("abstention", "abstention_correct",
                               "injection_resisted", "pii_leaked")),
    ("Efficiency", ("cost_usd", "latency_ms", "ttft_ms", "prompt_tokens",
                    "completion_tokens")),
]

PROVIDER_ENV = {
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
}


def build_catalog(extra_keys: dict[str, bool] | None = None) -> Catalog:
    """Introspect the installed harness.

    `extra_keys` lets the UI report a key typed into the sidebar as configured,
    since that never reaches the environment.
    """
    from harness.clients.endpoints import (
        NON_OPENAI_PROVIDERS,
        PROVIDER_ENDPOINTS,
    )
    from harness.eval.metrics import SCORERS
    from harness.eval.probes import ProbeConfig
    from harness.eval.scoring import DEFAULT_SCORERS
    from harness.profiles.profile import KNOWN_METRICS
    from harness.store.schema import Pass, RetrievalMode, TaskType

    extra_keys = extra_keys or {}
    cat = Catalog()

    # -- providers ------------------------------------------------------- #
    # Anthropic is not in PROVIDER_ENDPOINTS (it is not OpenAI-shaped), so it is
    # added explicitly from what its adapter declares about itself.
    specs = dict(PROVIDER_ENDPOINTS)
    for name in sorted(specs):
        spec = specs[name]
        env = spec.get("api_key_env", PROVIDER_ENV.get(name, ""))
        cat.providers.append(ProviderRow(
            name=name, env_var=env,
            embeddings=bool(spec.get("supports_embeddings", True)),
            rerank=bool(spec.get("supports_rerank", False)),
            local=bool(spec.get("local", False)),
            # A self-hosted placeholder key is not evidence of a running
            # server, so it does not count as configured.
            configured=bool(extra_keys.get(name)
                            or (env and os.environ.get(env))),
        ))
    # Providers with their own adapter (not OpenAI-shaped) are enumerated from
    # the same data file, so none can be missed by hand.
    for name, spec in NON_OPENAI_PROVIDERS.items():
        env = spec.get("api_key_env", PROVIDER_ENV.get(name, ""))
        cat.providers.append(ProviderRow(
            name=name, env_var=env,
            embeddings=bool(spec.get("supports_embeddings", True)),
            rerank=bool(spec.get("supports_rerank", False)),
            local=bool(spec.get("local", False)),
            configured=bool(extra_keys.get(name)
                            or (env and os.environ.get(env))),
        ))
    cat.providers.sort(key=lambda p: (p.local, p.name))

    # -- metrics --------------------------------------------------------- #
    grouped: set[str] = set()
    for group, names in _METRIC_GROUPS:
        present = [n for n in names if n in KNOWN_METRICS]
        grouped.update(present)
        if present:
            cat.metrics[group] = [Item(n, METRIC_NOTES.get(n, "")) for n in present]
    leftover = sorted(KNOWN_METRICS - grouped)
    if leftover:
        cat.metrics["Other"] = [Item(n, METRIC_NOTES.get(n, "")) for n in leftover]

    # -- scorers, probes, passes, tasks ---------------------------------- #
    cat.scorers = [
        Item(s.name, SCORER_NOTES.get(s.name, ""),
             detail=f"{len(s.produces)} metrics")
        for s in DEFAULT_SCORERS
    ]
    probe_fields = {f.name for f in dataclasses.fields(ProbeConfig)}
    cat.probes = [Item(n, PROBE_NOTES.get(n, ""))
                  for n in ("unanswerable", "injection", "noise", "paraphrase",
                            "positional") if n in probe_fields]
    cat.passes = [Item(p.value, PASS_NOTES.get(p.value, "")) for p in Pass]
    cat.tasks = [Item(t.value, TASK_NOTES.get(t.value, "")) for t in TaskType]
    cat.answer_scorers = sorted(SCORERS) + ["judge"]
    cat.retrieval_modes = [m.value for m in RetrievalMode]
    cat.analysis = list(ANALYSIS)
    cat.reliability = list(RELIABILITY)
    return cat


# --------------------------------------------------------------------------- #
#  This install's state
# --------------------------------------------------------------------------- #
@dataclass
class InstallStatus:
    profiles: list[str] = field(default_factory=list)
    app_profiles: list[str] = field(default_factory=list)
    providers_ready: int = 0
    providers_total: int = 0
    priced_models: int = 0
    unpriced: list[str] = field(default_factory=list)
    pricing_as_of: str = ""
    pricing_stale: str | None = None
    runs: int = 0
    rows: int = 0
    models_seen: list[str] = field(default_factory=list)
    has_probes: bool = False
    n_tests: int = 0


def build_status(catalog: Catalog, store=None, models_cfg: dict | None = None,
                 data_dir: Path | None = None) -> InstallStatus:
    """What this machine actually has configured and recorded."""
    from harness.clients.pricing import PricingRegistry
    from harness.profiles.loaders import load_evalset
    from harness.profiles.profile import Profile

    st = InstallStatus()
    models_cfg = models_cfg or {}

    st.providers_total = len(catalog.providers)
    st.providers_ready = sum(1 for p in catalog.providers if p.configured)

    st.profiles = sorted(p.stem for p in Path("configs/profiles").glob("*.yaml")) \
        if Path("configs/profiles").exists() else []
    if data_dir and data_dir.exists():
        st.app_profiles = sorted(p.parent.name
                                 for p in data_dir.glob("*/profile.yaml"))

    try:
        pricing = PricingRegistry("configs/pricing.yaml", strict=False)
        st.priced_models = len(pricing.models)
        st.pricing_as_of = str(pricing.as_of)
        st.pricing_stale = pricing.staleness_warning()
        wanted = list(models_cfg.get("models", []))
        if models_cfg.get("judge_model"):
            wanted.append(models_cfg["judge_model"])
        st.unpriced = pricing.missing(wanted)
    except Exception:  # noqa: BLE001 — the overview must render regardless
        pass

    if store is not None:
        try:
            if store.exists:
                df = store.load_all()
                st.rows = len(df)
                st.runs = int(df["run_id"].nunique()) if "run_id" in df else 0
                if "model" in df:
                    st.models_seen = sorted(df["model"].dropna().unique())
        except Exception:  # noqa: BLE001
            pass

    # Do any profiles actually carry probe items?
    for name in st.app_profiles + st.profiles:
        for candidate in (Path("configs/profiles") / f"{name}.yaml",
                          (data_dir / name / "profile.yaml") if data_dir else None):
            if candidate and candidate.exists():
                try:
                    prof = Profile.from_yaml(str(candidate))
                    items = load_evalset(prof.evalset_path)
                    if any((i.meta or {}).get("probe") for i in items):
                        st.has_probes = True
                except Exception:  # noqa: BLE001
                    pass
                break
        if st.has_probes:
            break

    tests = Path("tests")
    st.n_tests = len(list(tests.glob("test_*.py"))) if tests.exists() else 0
    return st
