"""
Core data contracts for the evaluation harness.

Two objects matter most:

  EvalItem  — the unit of iteration: one question + its ground truth + a type tag.
  TraceRow  — the flat, one-row-per-eval-item record produced by a single run.

Design rules:
  * TraceRow holds ONLY per-item measurements. Cross-model aggregates
    (tuning gain, Pareto, Elo, bootstrap CIs, significance) are computed in
    `report` FROM these rows and never stored here.
  * Every metric field is Optional. A profile activates only a subset; inactive
    metrics stay None rather than 0.0, so "not measured" stays distinguishable
    from "measured as zero". This is load-bearing — a metric defaulting to 0.0
    would drag every mean built on it toward zero and silently penalise models
    on dimensions nobody evaluated.
  * `item_type` distinguishes normal items from behavioural probes, which are
    structurally different eval items rather than different metric columns.
  * Config hashes (not full configs) are stored, so every row traces back to an
    exact, reproducible experiment definition.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum


class ItemType(str, Enum):
    """What kind of eval item this is — governs which metrics are meaningful."""
    ANSWERABLE = "answerable"          # gold answer + gold passages exist in corpus
    UNANSWERABLE = "unanswerable"      # abstention probe: model should refuse
    NOISE_INJECTED = "noise_injected"  # robustness: distractor chunks in context
    INJECTION = "injection"            # security: hostile instruction in a passage


class Pass(str, Enum):
    """Which experimental pass produced this row."""
    BASELINE = "baseline"  # fixed prompt + fixed retrieval, identical per model
    ADAPTED = "adapted"    # winner of the equal-budget tuning search
    TUNING = "tuning"      # a candidate evaluated during the search (dev split)
    LATENCY = "latency"    # clean-timing row from the low-concurrency lane
    ARENA = "arena"        # pairwise head-to-head comparison


class RetrievalMode(str, Enum):
    DENSE = "dense"      # cosine/dot over embeddings
    SPARSE = "sparse"    # BM25 lexical
    HYBRID = "hybrid"    # dense + sparse fused via RRF


class TaskType(str, Enum):
    """What kind of workload this profile evaluates.

    RAG was the only shape the harness originally supported, but most LLM
    evaluation isn't retrieval-augmented. DIRECT and CLASSIFY skip retrieval
    entirely, which turns the same apparatus — matrix walk, cost metering,
    caching, significance testing, regression gating — into a general-purpose
    prompt evaluator instead of a RAG-only one.
    """
    RAG = "rag"              # retrieve -> prompt -> generate -> score
    DIRECT = "direct"        # prompt -> generate -> score (no retrieval)
    CLASSIFY = "classify"    # direct, scored as a labelling task


@dataclass
class EvalItem:
    """One question and its ground truth. The unit the orchestrator iterates over."""
    item_id: str
    query: str
    item_type: ItemType = ItemType.ANSWERABLE

    # Ground truth. Which fields are populated depends on item_type:
    #   ANSWERABLE   -> gold_answer + gold_passage_ids
    #   UNANSWERABLE -> neither (the correct behaviour is to abstain)
    gold_answer: str | None = None
    gold_passage_ids: list[str] = field(default_factory=list)

    # For multi-turn items: prior turns as (role, content) pairs.
    history: list[tuple[str, str]] = field(default_factory=list)

    # Free-form per-item metadata. Probe items carry their attack parameters
    # here (canary, distractor seed, forced gold position) so the runner can
    # mutate context without the probe generator touching the network.
    meta: dict = field(default_factory=dict)

    # Optional human label, for judge calibration. When present, the report can
    # measure how well the LLM judge agrees with a person — the only way to know
    # whether the judge's numbers mean anything.
    human_label: float | None = None

    @property
    def is_answerable(self) -> bool:
        return self.item_type != ItemType.UNANSWERABLE


@dataclass
class RetrievedChunk:
    """One chunk returned by retrieval, with everything needed to score it."""
    chunk_id: str
    doc_id: str
    text: str
    score: float                      # similarity or fused score, mode-dependent
    source_uri: str | None = None  # for citation resolution
    rank: int = 0                     # 0-indexed position in the returned list


@dataclass
class TraceRow:
    """
    One row per (model, profile, pass, eval_item). The boundary between execution
    (which writes it) and analysis (which reads it). Flat by design so it
    serialises cleanly to Parquet and queries cleanly in DuckDB.
    """

    # ---- identity / provenance -------------------------------------------
    run_id: str                 # groups all rows from one invocation
    item_id: str
    model: str                  # the variable under test
    profile: str
    pass_: Pass
    item_type: ItemType
    ts: float = field(default_factory=time.time)

    # Config hashes — the reproducibility anchor. Full configs live in the run
    # manifest; a row stores only hashes, pointing at an exact setup.
    profile_cfg_hash: str = ""
    retrieval_cfg_hash: str = ""
    prompt_cfg_hash: str = ""
    gen_params_hash: str = ""
    provider: str = ""          # which vendor served this row

    # ---- what actually happened ------------------------------------------
    retrieval_mode: RetrievalMode | None = None
    retrieved_ids: list[str] = field(default_factory=list)   # ordered
    reranked_ids: list[str] = field(default_factory=list)    # ordered, post-rerank
    assembled_prompt: str = ""        # the exact prompt sent
    raw_output: str = ""              # the model's answer, verbatim
    cited_ids: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    truncated: bool | None = None  # ran out of max_tokens mid-answer

    # ---- retrieval-quality metrics ---------------------------------------
    hit_rate_at_k: float | None = None
    mrr: float | None = None
    ndcg_at_k: float | None = None
    context_recall: float | None = None
    context_precision: float | None = None
    average_precision: float | None = None
    rerank_hit_delta: float | None = None
    rerank_mrr_delta: float | None = None

    # ---- answer-quality metrics ------------------------------------------
    accuracy: float | None = None
    faithfulness: float | None = None
    answer_relevance: float | None = None
    completeness: float | None = None
    token_f1: float | None = None
    # Spread across an ensemble of judges. High values flag an ambiguous ITEM
    # rather than a bad model — a dataset finding, not a model finding.
    judge_disagreement: float | None = None

    # ---- citation-quality metrics ----------------------------------------
    citation_valid_pointer: float | None = None
    citation_supporting: float | None = None
    citation_density: float | None = None
    citation_recall: float | None = None

    # ---- behavioural / robustness / security -----------------------------
    abstained: bool | None = None
    abstention_correct: float | None = None
    injection_resisted: float | None = None   # ignored a hostile passage
    pii_leaked: float | None = None           # emitted PII absent from context
    consistency_group: str | None = None      # links paraphrases of one question

    # ---- classification (non-RAG profiles) -------------------------------
    predicted_label: str | None = None
    gold_label: str | None = None

    # Carried over from the EvalItem so judge calibration can join a human
    # verdict against the judge's on the same row. Without it the calibration
    # report has nothing to compare and silently reports "no human labels".
    human_label: float | None = None

    # ---- efficiency metrics ----------------------------------------------
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    # Cost is split by subsystem: on a judge-scored profile the judge is
    # routinely the largest line, and a single total hides that entirely.
    cost_usd: float | None = None          # total for this item
    gen_cost_usd: float | None = None
    judge_cost_usd: float | None = None
    embed_cost_usd: float | None = None
    rerank_cost_usd: float | None = None

    # ---- bookkeeping ------------------------------------------------------
    cache_hit: bool = False
    error: str | None = None
    error_kind: str | None = None   # coarse taxonomy, for error attribution
    attempts: int = 1                  # how many tries this item needed

    def to_dict(self) -> dict:
        d = asdict(self)
        # Enums -> their string values, for clean Parquet columns.
        for key, enum_cls in (("pass_", Pass), ("item_type", ItemType),
                              ("retrieval_mode", RetrievalMode)):
            val = getattr(self, key)
            d[key] = val.value if isinstance(val, enum_cls) else val
        return d


# Coarse error taxonomy. Error *attribution* is the point: "12% of this model's
# items failed" is only actionable once you know whether that was rate limiting
# (your concurrency), context overflow (your k), or refusals (the model).
ERROR_KINDS = (
    "rate_limit", "timeout", "context_length", "auth", "not_found",
    "content_filter", "parse", "capability", "budget", "other",
)


def classify_error(exc: BaseException) -> str:
    """Map an exception onto `ERROR_KINDS`."""
    name = type(exc).__name__.lower()
    text = f"{name} {exc}".lower()
    status = getattr(exc, "status_code", None)

    if "budget" in name:
        return "budget"
    if "capability" in name:
        return "capability"
    if status == 429 or "rate limit" in text or "too many requests" in text:
        return "rate_limit"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if ("context length" in text or "maximum context" in text
            or "too long" in text or "context_length" in text):
        return "context_length"
    if status in (401, 403) or "unauthor" in text or "api key" in text:
        return "auth"
    if status == 404 or "not found" in text:
        return "not_found"
    if "content filter" in text or "content_policy" in text:
        return "content_filter"
    if "json" in name or "parse" in name or "decode" in name:
        return "parse"
    return "other"
