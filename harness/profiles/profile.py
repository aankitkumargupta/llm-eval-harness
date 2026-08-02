"""
Profile = the fixed apparatus, declared as config.

A profile bundles everything that is held constant while the model-under-test
varies: the corpus + evalset, the fixed embedding model, the retrieval config,
the tuning knobs (and their allowed ranges), the metric weights used to build
this profile's leaderboard number, and which metrics are even active.

Your five profiles (regulated_qa, tech_support, long_doc, tabular, high_vol_chat)
are five YAML files that deserialise into this dataclass. Adding a profile is a
config edit, never a code change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..store.schema import RetrievalMode


@dataclass
class TuningKnobs:
    """What the ADAPTED pass is allowed to change, and over what ranges.
    The tuning search draws candidates from exactly these ranges — nothing else
    is touched, which is what keeps 'equal budget' meaningful."""
    retrieval_modes: list[str] = field(default_factory=lambda: ["dense"])
    k_values: list[int] = field(default_factory=lambda: [10])
    rerank_options: list[bool] = field(default_factory=lambda: [False])
    rerank_top_n: list[int] = field(default_factory=lambda: [5])
    system_prompts: list[str] = field(default_factory=list)   # candidate prompts
    few_shot_sets: list[list[dict]] = field(default_factory=list)


@dataclass
class Profile:
    name: str
    corpus_path: str          # documents to ingest
    evalset_path: str         # eval items (query + gold)
    embedding_model: str      # FIXED across all models under test

    # baseline (fixed) retrieval config
    retrieval_mode: RetrievalMode = RetrievalMode.DENSE
    k: int = 10
    rerank: bool = False
    rerank_model: str = ""
    rerank_top_n: int = 5

    accuracy_scorer: str = "judge"   # exact | contains | numeric | judge
    active_metrics: list[str] = field(default_factory=list)
    metric_weights: dict = field(default_factory=dict)  # -> weighted composite
    tuning_budget: int = 20          # N candidate configs, identical for every model
    knobs: TuningKnobs = field(default_factory=TuningKnobs)

    @staticmethod
    def from_yaml(path: str) -> "Profile":
        data = yaml.safe_load(Path(path).read_text())
        knobs = TuningKnobs(**data.pop("knobs", {}))
        data["retrieval_mode"] = RetrievalMode(data.get("retrieval_mode", "dense"))
        return Profile(knobs=knobs, **data)

    def collection_name(self) -> str:
        """Qdrant collection is namespaced by profile + embedder, so switching
        embedder forces a distinct index (they're not comparable)."""
        safe_emb = self.embedding_model.replace("/", "_")
        return f"{self.name}__{safe_emb}"
