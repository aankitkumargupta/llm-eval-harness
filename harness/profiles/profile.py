"""
Profile = the fixed apparatus, declared as config.

A profile bundles everything held constant while the model under test varies:
the corpus + evalset, the fixed embedding model, the retrieval config, the
tuning knobs and their ranges, the metric weights that build this profile's
leaderboard number, and which metrics are active at all.

Adding a profile is a config edit, never a code change. That contract is now
enforced rather than assumed: `from_yaml` validates, reports *every* problem at
once, and refuses to construct a profile whose weights reference metrics it
never collects, a mistake that silently produced a leaderboard where a
carefully weighted metric contributed nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from ..store.schema import RetrievalMode, TaskType

# Metrics that are lower-is-better. Their weights must be negative in a
# composite, and a positive weight on one is almost always a typo that inverts
# the leaderboard, a model gets rewarded for being expensive and slow.
LOWER_IS_BETTER = {"cost_usd", "latency_ms", "ttft_ms", "prompt_tokens",
                   "completion_tokens", "pii_leaked", "judge_disagreement"}

# Metric names a profile may activate. Validated so a typo in active_metrics
# fails at load instead of quietly never being measured.
KNOWN_METRICS = {
    "hit_rate_at_k", "mrr", "ndcg_at_k", "context_recall", "context_precision",
    "average_precision", "rerank_hit_delta", "rerank_mrr_delta",
    "accuracy", "faithfulness", "answer_relevance", "completeness", "token_f1",
    "judge_disagreement",
    "citation_valid_pointer", "citation_supporting", "citation_density",
    "citation_recall",
    "abstention", "abstention_correct", "injection_resisted", "pii_leaked",
    "native_script_ratio",
    "cost_usd", "latency_ms", "ttft_ms", "prompt_tokens", "completion_tokens",
}

VALID_SCORERS = {"exact", "contains", "numeric", "token_f1", "judge", "none"}


class ProfileError(ValueError):
    """A profile config is invalid. Lists every problem, not just the first."""


@dataclass
class TuningKnobs:
    """What the ADAPTED pass may change, and over what ranges.

    The search draws candidates from exactly these ranges and nothing else,
    which is what keeps "equal budget" meaningful rather than aspirational.
    """
    retrieval_modes: list[str] = field(default_factory=lambda: ["dense"])
    k_values: list[int] = field(default_factory=lambda: [10])
    rerank_options: list[bool] = field(default_factory=lambda: [False])
    rerank_top_n: list[int] = field(default_factory=lambda: [5])
    system_prompts: list[str] = field(default_factory=list)
    few_shot_sets: list[list[dict]] = field(default_factory=list)
    # Context ordering is a real knob with a real effect on long-context models,
    # and it was unreachable from any profile until now.
    context_orders: list[str] = field(default_factory=lambda: ["as_is"])


@dataclass
class ProbeSettings:
    """Adversarial probe generation, as a fraction of the base evalset."""
    enabled: bool = False
    unanswerable: float = 0.15
    noise: float = 0.15
    injection: float = 0.15
    paraphrase: float = 0.15
    positional: float = 0.0
    typo_rate: float = 0.0
    n_distractors: int = 3
    seed: int = 0


@dataclass
class Profile:
    name: str
    corpus_path: str = ""      # optional for non-RAG tasks
    evalset_path: str = ""
    embedding_model: str = ""

    task: TaskType = TaskType.RAG

    # baseline (fixed) retrieval config
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    k: int = 10
    rerank: bool = False
    rerank_model: str = ""
    rerank_top_n: int = 5
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    candidate_multiplier: int = 1

    accuracy_scorer: str = "judge"
    active_metrics: list[str] = field(default_factory=list)
    metric_weights: dict = field(default_factory=dict)
    tuning_budget: int = 20
    knobs: TuningKnobs = field(default_factory=TuningKnobs)
    probes: ProbeSettings = field(default_factory=ProbeSettings)

    # generation params, per profile rather than hardcoded in the runner: a
    # long-document profile needs a far bigger answer budget than a chat one,
    # and one global default silently truncates the former.
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0

    abstention_judge: bool = False   # use the judge for abstention, not regex
    label_set: list[str] = field(default_factory=list)  # CLASSIFY profiles
    # Script the answer is expected in, for translation and native-language
    # tasks ("devanagari"; "" means not scored). Drives native_script_ratio.
    target_script: str = ""
    # The BASELINE pass's system prompt. Empty means the task default
    # (RAG / direct / classify). Set it when the task needs an instruction the
    # default does not carry ("translate this", "draft an application"): the
    # tuning knobs' system_prompts are candidates for the ADAPTED pass only
    # and never reach the baseline, so without this field a direct-task
    # profile could not say what the task was.
    system_prompt: str = ""
    chunk_size: int = 200
    overlap: int = 40
    description: str = ""

    # ------------------------------------------------------------------ #
    @staticmethod
    def from_yaml(path: str) -> Profile:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return Profile.from_dict(data, source=str(path))

    @staticmethod
    def from_dict(data: dict, source: str = "<dict>") -> Profile:
        data = dict(data)
        knobs = TuningKnobs(**(data.pop("knobs", None) or {}))
        probes = ProbeSettings(**(data.pop("probes", None) or {}))

        data["retrieval_mode"] = RetrievalMode(data.get("retrieval_mode", "dense"))
        data["task"] = TaskType(data.get("task", "rag"))

        # Drop unknown keys with a clear message instead of a bare TypeError
        # from the dataclass constructor, which names the key but not the file.
        valid = {f.name for f in fields(Profile)}
        unknown = sorted(set(data) - valid)
        for key in unknown:
            data.pop(key)

        profile = Profile(knobs=knobs, probes=probes, **data)
        problems = profile.validate()
        if unknown:
            problems.append(f"Unknown key(s) ignored: {', '.join(unknown)}")
        fatal = [p for p in problems if not p.startswith("Unknown key")]
        if fatal:
            raise ProfileError(
                f"Invalid profile '{source}':\n  - " + "\n  - ".join(fatal))
        return profile

    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Return every problem with this profile, so one load fixes them all."""
        problems: list[str] = []

        if not self.name:
            problems.append("`name` is required.")
        if self.accuracy_scorer not in VALID_SCORERS:
            problems.append(
                f"accuracy_scorer '{self.accuracy_scorer}' is not one of "
                f"{sorted(VALID_SCORERS)}.")

        bad_metrics = [m for m in self.active_metrics if m not in KNOWN_METRICS]
        if bad_metrics:
            problems.append(
                f"Unknown active_metrics: {bad_metrics}. Valid names: "
                f"{sorted(KNOWN_METRICS)}")

        # A weight on a metric the profile never collects contributes nothing
        # to the composite, the leaderboard silently ignores it, and the run
        # looks like it measured something it didn't.
        collected = set(self.active_metrics) | {"abstention_correct"}
        if "abstention" in self.active_metrics:
            collected.add("abstention_correct")
        orphan = [m for m in self.metric_weights
                  if m not in collected and m in KNOWN_METRICS]
        if orphan:
            problems.append(
                f"metric_weights reference metrics not in active_metrics: "
                f"{orphan}. They would contribute nothing to the composite: "
                f"add them to active_metrics or drop the weights.")

        unknown_w = [m for m in self.metric_weights if m not in KNOWN_METRICS]
        if unknown_w:
            problems.append(f"Unknown metric_weights keys: {unknown_w}")

        wrong_sign = [m for m, w in self.metric_weights.items()
                      if m in LOWER_IS_BETTER and w > 0]
        if wrong_sign:
            problems.append(
                f"{wrong_sign} are lower-is-better and must have NEGATIVE "
                f"weights; a positive weight rewards models for being worse.")

        if self.task == TaskType.RAG:
            if not self.embedding_model:
                problems.append("RAG profiles need an `embedding_model`.")
            if not self.corpus_path:
                problems.append("RAG profiles need a `corpus_path`.")
        if self.target_script:
            from ..eval.languages import resolve_script
            try:
                resolve_script(self.target_script)
            except ValueError as e:
                problems.append(f"target_script: {e}")
        if self.task == TaskType.CLASSIFY and not self.label_set:
            problems.append("CLASSIFY profiles need a non-empty `label_set`.")
        if not self.evalset_path:
            problems.append("`evalset_path` is required.")

        if self.k <= 0:
            problems.append("`k` must be >= 1.")
        if self.tuning_budget < 0:
            problems.append("`tuning_budget` must be >= 0.")
        if self.max_tokens <= 0:
            problems.append("`max_tokens` must be >= 1.")

        bad_orders = [o for o in self.knobs.context_orders
                      if o not in ("as_is", "reverse", "middle_gold", "shuffle")]
        if bad_orders:
            problems.append(f"Unknown context_orders: {bad_orders}")

        return problems

    # ------------------------------------------------------------------ #
    def collection_name(self) -> str:
        """Qdrant collection namespaced by profile + embedder.

        Switching embedder forces a distinct index, because vectors from two
        different embedders are not comparable and silently mixing them would
        produce retrieval that looks like it works.
        """
        safe_emb = self.embedding_model.replace("/", "_").replace(":", "_")
        return f"{self.name}__{safe_emb}"

    def config_hash_parts(self) -> tuple:
        """The fields that define this profile's identity for provenance hashing."""
        return (self.name, self.task.value, self.embedding_model,
                self.retrieval_mode.value, self.k, self.rerank,
                self.rerank_top_n, self.accuracy_scorer,
                tuple(sorted(self.active_metrics)),
                tuple(sorted(self.metric_weights.items())),
                self.temperature, self.max_tokens, self.seed,
                # Only when set, so profiles without one keep their hash and
                # their earlier runs stay comparable.
                *((self.system_prompt,) if self.system_prompt else ()))

    def to_dict(self) -> dict:
        from dataclasses import asdict
        d = asdict(self)
        d["retrieval_mode"] = self.retrieval_mode.value
        d["task"] = self.task.value
        return d
