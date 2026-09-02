"""
The scoring pipeline, as a registry of independent scorers.

**The problem this solves.** `run_item` grew a ~200-line chain of
`if "metric" in active: row.metric = M.metric(...)` blocks covering retrieval,
citations, abstention, probes, accuracy, classification and the judge. Adding a
metric meant editing the single hottest function in the codebase — the one that
also owns retrieval, prompt assembly, generation, caching and cost. That is both
an Open/Closed failure (extension requires modification) and a Single
Responsibility failure (one function, seven reasons to change).

Each family of metrics is now a `Scorer`: it declares which TraceRow fields it
can produce, decides for itself whether it applies to a given item, and returns
a plain `{field: value}` mapping. `run_item` iterates the registry.

Consequences worth having:

  * Adding a metric is a new class plus one registry entry. No existing file
    changes, so nothing already working can regress.
  * Each scorer is independently testable with a hand-built context — no
    network, no vector store, no orchestrator.
  * A scorer that raises degrades to "that metric wasn't measured" instead of
    failing the whole item. One bad judge reply should not discard a paid
    generation and every other metric computed from it.
  * The *order* is explicit and documented, which matters because two scorers
    legitimately write `accuracy` and the later one wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..store.schema import EvalItem, ItemType, RetrievedChunk, TaskType
from . import metrics as M
from . import probes as P


@dataclass
class ScoringContext:
    """Everything a scorer may look at.

    Deliberately a value object: scorers read it and return values, they never
    mutate the row. That keeps them order-independent except where the registry
    documents otherwise, and makes each one a pure function of this context.
    """

    item: EvalItem
    model: str
    answer: str
    active: set[str]
    accuracy_scorer: str = "judge"
    task: TaskType = TaskType.RAG

    # Retrieval, as it actually happened — before any probe mutation. Retrieval
    # is scored on what the retriever found, not on passages a probe fabricated.
    retrieved_ids: list[str] = field(default_factory=list)
    reranked_ids: list[str] = field(default_factory=list)
    k: int = 10

    # The context the MODEL saw — after probe mutation.
    chunks_shown: list[RetrievedChunk] = field(default_factory=list)
    context_text: str = ""

    # Collaborators. Optional so a scorer needing one can simply not apply.
    judge: Any = None
    abstention_judge: bool = False
    label_set: list[str] = field(default_factory=list)

    @property
    def final_ids(self) -> list[str]:
        """The ranking retrieval actually delivered."""
        return self.reranked_ids or self.retrieved_ids

    @property
    def shown_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks_shown]

    def wants(self, *names: str) -> bool:
        return any(n in self.active for n in names)


@runtime_checkable
class Scorer(Protocol):
    """One family of metrics."""

    name: str
    produces: tuple[str, ...]

    def applies(self, ctx: ScoringContext) -> bool:
        """Cheap check. Must not call the network."""
        ...

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        """Return {trace_row_field: value}. Omit anything not measured."""
        ...


# --------------------------------------------------------------------------- #
#  Retrieval
# --------------------------------------------------------------------------- #
class RetrievalScorer:
    """Ranking quality of what retrieval returned, against the gold passages."""

    name = "retrieval"
    produces = ("hit_rate_at_k", "mrr", "ndcg_at_k", "context_recall",
                "context_precision", "average_precision", "rerank_hit_delta",
                "rerank_mrr_delta")

    def applies(self, ctx: ScoringContext) -> bool:
        return bool(ctx.item.gold_passage_ids and ctx.final_ids)

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        gold, ids, k = ctx.item.gold_passage_ids, ctx.final_ids, ctx.k
        out: dict[str, Any] = {}
        if "hit_rate_at_k" in ctx.active:
            out["hit_rate_at_k"] = M.hit_rate_at_k(ids, gold, k)
        if "mrr" in ctx.active:
            out["mrr"] = M.mrr(ids, gold)
        if "ndcg_at_k" in ctx.active:
            out["ndcg_at_k"] = M.ndcg_at_k(ids, gold, k)
        if "context_recall" in ctx.active:
            out["context_recall"] = M.context_recall(ids, gold, k)
        if "context_precision" in ctx.active:
            out["context_precision"] = M.context_precision(ids, gold, k)
        if "average_precision" in ctx.active:
            out["average_precision"] = M.average_precision(ids, gold, k)
        if ctx.reranked_ids and ctx.wants("rerank_hit_delta", "rerank_mrr_delta"):
            lift = M.rerank_lift(ctx.retrieved_ids, ctx.reranked_ids, gold, k)
            out["rerank_hit_delta"] = lift["hit_delta"]
            out["rerank_mrr_delta"] = lift["mrr_delta"]
        return out


# --------------------------------------------------------------------------- #
#  Citations
# --------------------------------------------------------------------------- #
class CitationScorer:
    """Whether the answer's [id] markers point at real, supporting passages."""

    name = "citations"
    produces = ("cited_ids", "citation_valid_pointer", "citation_supporting",
                "citation_density", "citation_recall")

    def applies(self, ctx: ScoringContext) -> bool:
        # Always: `cited_ids` is recorded even when no citation metric is
        # active, because the raw trace is what makes a run auditable later.
        return True

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        cited = M.extract_citations(ctx.answer)
        out: dict[str, Any] = {"cited_ids": cited}
        if "citation_valid_pointer" in ctx.active:
            out["citation_valid_pointer"] = M.citation_valid_pointer(
                cited, ctx.shown_ids)
        if "citation_density" in ctx.active:
            out["citation_density"] = M.citation_density(cited, ctx.answer)
        if ctx.item.gold_passage_ids:
            if "citation_supporting" in ctx.active:
                out["citation_supporting"] = M.citation_supporting(
                    cited, ctx.item.gold_passage_ids)
            if "citation_recall" in ctx.active:
                out["citation_recall"] = M.citation_recall(
                    cited, ctx.item.gold_passage_ids)
        return out


# --------------------------------------------------------------------------- #
#  Behaviour
# --------------------------------------------------------------------------- #
class AbstentionScorer:
    """Did the model refuse, and was refusing the right call?"""

    name = "abstention"
    produces = ("abstained", "abstention_correct")

    def applies(self, ctx: ScoringContext) -> bool:
        # Unanswerable items are always scored, whether or not the profile
        # listed the metric: an unanswerable item exists for no other purpose.
        return ("abstention" in ctx.active
                or ctx.item.item_type == ItemType.UNANSWERABLE)

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        abstained = M.detect_abstention(ctx.answer)
        if ctx.abstention_judge and ctx.judge is not None:
            judged = ctx.judge.judge_abstention(ctx.item.query, ctx.answer)
            if judged is not None:
                abstained = judged
        return {"abstained": abstained,
                "abstention_correct": M.abstention_correct(
                    abstained, ctx.item.is_answerable)}


class ProbeScorer:
    """Adversarial probe outcomes: injection resistance, PII leakage."""

    name = "probes"
    produces = ("injection_resisted", "pii_leaked", "abstained",
                "abstention_correct")

    def applies(self, ctx: ScoringContext) -> bool:
        return True  # returns {} for non-probe items

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        return P.score_probe(ctx.item, ctx.answer, ctx.context_text)


# --------------------------------------------------------------------------- #
#  Answer accuracy
# --------------------------------------------------------------------------- #
class DeterministicAccuracyScorer:
    """Free, offline accuracy: exact / contains / numeric / token_f1."""

    name = "accuracy_deterministic"
    produces = ("accuracy", "token_f1")

    def applies(self, ctx: ScoringContext) -> bool:
        return bool(ctx.item.gold_answer)

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        out: dict[str, Any] = {}
        gold = ctx.item.gold_answer or ""
        if ctx.accuracy_scorer not in ("judge", "none"):
            scored = M.score_answer(ctx.answer, gold, ctx.accuracy_scorer)
            if scored is not None:
                out["accuracy"] = scored
        if "token_f1" in ctx.active:
            # Free alongside any scorer, and a graded score next to a binary one
            # shows *how* wrong a wrong answer was.
            out["token_f1"] = M.token_f1(ctx.answer, gold)
        return out


class ClassificationScorer:
    """Label extraction and exact-match scoring for non-RAG classify profiles."""

    name = "classification"
    produces = ("predicted_label", "gold_label", "accuracy")

    def applies(self, ctx: ScoringContext) -> bool:
        return ctx.task == TaskType.CLASSIFY

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        predicted = parse_label(ctx.answer, ctx.label_set)
        out: dict[str, Any] = {"predicted_label": predicted,
                               "gold_label": ctx.item.gold_answer}
        if ctx.item.gold_answer is not None:
            out["accuracy"] = 1.0 if predicted == ctx.item.gold_answer else 0.0
        return out


class JudgeScorer:
    """LLM-graded qualities: faithfulness, relevance, completeness, accuracy.

    Runs last on purpose. When a profile uses `accuracy_scorer: judge` the
    judge's verdict is the authoritative accuracy and must overwrite whatever a
    deterministic scorer put there.
    """

    name = "judge"
    produces = ("faithfulness", "answer_relevance", "completeness", "accuracy",
                "judge_disagreement")

    def applies(self, ctx: ScoringContext) -> bool:
        return ctx.judge is not None and (
            ctx.wants("faithfulness", "answer_relevance", "completeness")
            or ctx.accuracy_scorer == "judge")

    def score(self, ctx: ScoringContext) -> dict[str, Any]:
        js = ctx.judge.score(ctx.model, ctx.item.query, ctx.context_text,
                             ctx.answer, ctx.item.gold_answer)
        out: dict[str, Any] = {"judge_disagreement": js.disagreement}
        if "faithfulness" in ctx.active:
            out["faithfulness"] = js.faithfulness
        if "answer_relevance" in ctx.active:
            out["answer_relevance"] = js.answer_relevance
        if "completeness" in ctx.active:
            out["completeness"] = js.completeness
        if ctx.accuracy_scorer == "judge" and js.accuracy is not None:
            out["accuracy"] = js.accuracy
        return out


def parse_label(answer: str, label_set: list[str]) -> str | None:
    """Extract a class label from a free-text answer.

    Models decorate labels ("Label: positive", "**positive**"), so exact
    matching would score a correct classification as wrong. Matches the longest
    label present, so "not_spam" wins over "spam" when both appear as
    substrings.
    """
    if not label_set:
        return (answer or "").strip() or None
    text = (answer or "").lower()
    hits = [lbl for lbl in label_set if lbl.lower() in text]
    return max(hits, key=len) if hits else None


# --------------------------------------------------------------------------- #
#  The registry
# --------------------------------------------------------------------------- #
# Order is meaningful only where two scorers write the same field. The one
# case is `accuracy`: deterministic -> classification -> judge, last wins.
DEFAULT_SCORERS: tuple[Scorer, ...] = (
    RetrievalScorer(),
    CitationScorer(),
    AbstentionScorer(),
    ProbeScorer(),
    DeterministicAccuracyScorer(),
    ClassificationScorer(),
    JudgeScorer(),
)


@dataclass
class ScoringResult:
    values: dict[str, Any] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)


def run_scorers(ctx: ScoringContext,
                scorers: tuple[Scorer, ...] | None = None) -> ScoringResult:
    """Apply every scorer that applies, collecting values and failures.

    A scorer that raises costs its own metrics and nothing else. The generation
    was already paid for, and the other metrics computed from it are still
    valid — discarding them because one judge reply failed to parse would throw
    away good data over a recoverable problem.
    """
    result = ScoringResult()
    for scorer in (scorers if scorers is not None else DEFAULT_SCORERS):
        try:
            if scorer.applies(ctx):
                result.values.update(scorer.score(ctx))
        except Exception as e:  # noqa: BLE001 — isolate one scorer's failure
            result.failures[scorer.name] = f"{type(e).__name__}: {e}"
    return result
