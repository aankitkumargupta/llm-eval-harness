"""
Core data contracts for the evaluation harness.

Two objects matter most here:

  EvalItem  — the unit of iteration: one question + its ground truth + a type tag.
  TraceRow  — the flat, one-row-per-eval-item record produced by a single run.

Design rules (decided during architecture):
  * TraceRow holds ONLY per-item measurements. Cross-model aggregates
    (tuning-gain, Pareto, Elo, bootstrap CIs) are computed in `report`
    FROM these rows and never stored here.
  * Every metric field is Optional. A given profile activates only a
    subset of metrics; inactive ones stay None rather than 0.0, so we
    can tell "not measured" apart from "measured as zero".
  * `item_type` distinguishes normal items from the behavioural probes
    (unanswerable / noise-injected / injection) which are structurally
    different eval items, not different metric columns.
  * Config hashes (not full configs) are stored, so every row traces back
    to an exact, reproducible experiment definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time


class ItemType(str, Enum):
    """What kind of eval item this is — governs which metrics are meaningful."""
    ANSWERABLE = "answerable"          # normal: gold answer + gold passages exist in corpus
    UNANSWERABLE = "unanswerable"      # abstention probe: gold NOT in corpus, model should refuse
    NOISE_INJECTED = "noise_injected"  # robustness: irrelevant chunks added to context
    INJECTION = "injection"            # security: malicious instruction hidden in a retrieved doc


class Pass(str, Enum):
    """Which experimental pass produced this row."""
    BASELINE = "baseline"  # fixed prompt + fixed retrieval, identical for every model
    ADAPTED = "adapted"    # winner of the equal-budget tuning search
    TUNING = "tuning"      # a candidate evaluated during the tuning search (on the dev split)


class RetrievalMode(str, Enum):
    DENSE = "dense"      # cosine/dot over embeddings
    SPARSE = "sparse"    # BM25 lexical
    HYBRID = "hybrid"    # dense + sparse fused via RRF


@dataclass
class EvalItem:
    """One question and its ground truth. The unit the orchestrator iterates over."""
    item_id: str
    query: str
    item_type: ItemType = ItemType.ANSWERABLE

    # Ground truth. Which fields are populated depends on item_type:
    #   ANSWERABLE   -> gold_answer + gold_passage_ids
    #   UNANSWERABLE -> neither (the "right" behaviour is to abstain)
    gold_answer: Optional[str] = None
    gold_passage_ids: list[str] = field(default_factory=list)

    # For multi-turn (high-vol chat) items: prior turns as (role, content) pairs.
    history: list[tuple[str, str]] = field(default_factory=list)

    # Free-form per-item metadata (e.g. injected-passage id for the injection probe,
    # or the intended gold-passage position for the lost-in-the-middle probe).
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """One chunk returned by retrieval, with everything needed to score it."""
    chunk_id: str
    doc_id: str
    text: str
    score: float                      # similarity or fused score, mode-dependent
    source_uri: Optional[str] = None  # for citation resolution
    rank: int = 0                     # 0-indexed position in the returned list


@dataclass
class TraceRow:
    """
    One row per (model, profile, pass, eval_item). The boundary between
    execution (which writes it) and analysis (which reads it). Flat by design
    so it serialises cleanly to Parquet and queries cleanly in DuckDB.
    """

    # ---- identity / provenance -------------------------------------------
    run_id: str                 # groups all rows from one invocation of the harness
    item_id: str
    model: str                  # Together model string, the variable under test
    profile: str
    pass_: Pass
    item_type: ItemType
    ts: float = field(default_factory=time.time)

    # config hashes — the reproducibility anchor. Full configs live in the run
    # manifest; here we store only their hashes so a row points to an exact setup.
    profile_cfg_hash: str = ""
    retrieval_cfg_hash: str = ""
    prompt_cfg_hash: str = ""
    gen_params_hash: str = ""

    # ---- what actually happened ------------------------------------------
    retrieval_mode: Optional[RetrievalMode] = None
    retrieved_ids: list[str] = field(default_factory=list)   # ordered, for ranking metrics
    reranked_ids: list[str] = field(default_factory=list)    # ordered, post-rerank (if used)
    assembled_prompt: str = ""        # the exact prompt sent (reproducibility + token audit)
    raw_output: str = ""              # the model's answer, verbatim (cached by hash)
    cited_ids: list[str] = field(default_factory=list)       # ids the model claimed to cite

    # ---- retrieval-quality metrics (steps 3-4; free from ids vs gold) ----
    hit_rate_at_k: Optional[float] = None      # binary presence: any gold in top-k
    mrr: Optional[float] = None                # 1/rank of first gold passage
    ndcg_at_k: Optional[float] = None          # graded ranking quality
    context_recall: Optional[float] = None     # fraction of all gold passages retrieved
    rerank_hit_delta: Optional[float] = None   # hit-rate after rerank minus before

    # ---- answer-quality metrics (step 8) ---------------------------------
    accuracy: Optional[float] = None           # vs gold; scorer varies by profile
    faithfulness: Optional[float] = None       # grounded in context, no fabrication (judge)
    answer_relevance: Optional[float] = None   # addresses the question (judge)
    completeness: Optional[float] = None       # used all relevant info (judge)

    # ---- citation-quality metrics (step 7) -------------------------------
    citation_valid_pointer: Optional[float] = None    # cited ids are real retrieved passages
    citation_supporting: Optional[float] = None       # cited passage actually supports the claim

    # ---- behavioural / robustness (probe items only) ---------------------
    abstained: Optional[bool] = None           # did the model refuse? (scored vs item_type)
    abstention_correct: Optional[float] = None # refused-when-should / answered-when-should
    injection_resisted: Optional[float] = None # ignored hidden malicious instruction

    # ---- efficiency metrics (step 6) -------------------------------------
    prompt_tokens: Optional[int] = None        # from the usage block (authoritative)
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None         # total; measured in the low-concurrency lane
    ttft_ms: Optional[float] = None            # time-to-first-token, if streaming
    cost_usd: Optional[float] = None           # tokens x dated price (embed+rerank+gen)

    # ---- bookkeeping ------------------------------------------------------
    cache_hit: bool = False                    # was the generation served from cache?
    error: Optional[str] = None                # populated if the run failed for this item

    def to_dict(self) -> dict:
        d = asdict(self)
        # enums -> their string values for clean Parquet columns
        d["pass_"] = self.pass_.value if isinstance(self.pass_, Pass) else self.pass_
        d["item_type"] = self.item_type.value if isinstance(self.item_type, ItemType) else self.item_type
        d["retrieval_mode"] = self.retrieval_mode.value if isinstance(self.retrieval_mode, RetrievalMode) else self.retrieval_mode
        return d
