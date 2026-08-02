"""
Pure metric functions. Everything here takes plain data and returns a number —
no network, no state — so it's fully unit-testable and re-runnable against cached
outputs for free.

Grouped by subsystem:
  retrieval  — hit_rate_at_k, mrr, ndcg_at_k, context_recall
  citation   — citation_valid_pointer, citation_supporting
  answer     — accuracy (exact / numeric), abstention scoring
The judge-based metrics (faithfulness, relevance, completeness) live in judge.py
because they need a model call.
"""

from __future__ import annotations

import math
import re


# ---------------------------------------------------------------------------
#  Retrieval-quality metrics (computed from ordered ids vs. gold ids)
# ---------------------------------------------------------------------------
def hit_rate_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Binary: did ANY gold passage appear in the top-k retrieved?"""
    if not gold_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    return 1.0 if topk & set(gold_ids) else 0.0


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """Reciprocal rank of the FIRST gold passage (0 if none retrieved)."""
    gold = set(gold_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """
    Normalised Discounted Cumulative Gain with binary relevance (gold=1 else 0).
    Rewards putting gold passages high AND retrieving more of them.
    """
    gold = set(gold_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        if rid in gold:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-based; +2 gives log2(rank+1)
    # ideal DCG: all gold passages ranked first (capped at k)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def context_recall(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """Of all gold passages, what fraction appear in the top-k?"""
    if not gold_ids:
        return 0.0
    topk = set(retrieved_ids[:k])
    return len(topk & set(gold_ids)) / len(set(gold_ids))


# ---------------------------------------------------------------------------
#  Citation-quality metrics
# ---------------------------------------------------------------------------
CITATION_RE = re.compile(r"\[([^\[\]]+?)\]")  # matches [doc1#3] style markers


def extract_citations(answer: str) -> list[str]:
    """Pull [id] markers out of the model's answer text."""
    return [m.strip() for m in CITATION_RE.findall(answer)]


def citation_valid_pointer(cited_ids: list[str], retrieved_ids: list[str]) -> float:
    """
    Fraction of cited ids that actually point at a retrieved passage (i.e. are
    not hallucinated). 1.0 if the model cited nothing (vacuously valid) — callers
    that want to require citations should check citation COUNT separately.
    """
    if not cited_ids:
        return 1.0
    retrieved = set(retrieved_ids)
    valid = sum(1 for c in cited_ids if c in retrieved)
    return valid / len(cited_ids)


def citation_supporting(cited_ids: list[str], gold_ids: list[str]) -> float:
    """
    Fraction of cited ids that are gold (actually support the answer). This is a
    proxy that uses gold-passage overlap; the stronger version routes each cited
    passage through the judge. Kept separate from pointer-validity on purpose.
    """
    if not cited_ids:
        return 0.0
    gold = set(gold_ids)
    return sum(1 for c in cited_ids if c in gold) / len(cited_ids)


# ---------------------------------------------------------------------------
#  Answer-accuracy scorers (profile chooses which one)
# ---------------------------------------------------------------------------
def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def exact_match(answer: str, gold: str) -> float:
    """Strict normalised string equality. Good for short factoid answers."""
    return 1.0 if _normalise(answer) == _normalise(gold) else 0.0


def contains_match(answer: str, gold: str) -> float:
    """Gold string appears somewhere in the answer. Lenient factoid check."""
    return 1.0 if _normalise(gold) in _normalise(answer) else 0.0


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def numeric_match(answer: str, gold: str, tol: float = 1e-6) -> float:
    """
    Exact numeric match for the TABULAR profile, where "1,000" vs "10,000" must
    NOT count as similar. Extracts the first number from each and compares within
    a tiny tolerance. Returns 0 if either has no number.
    """
    def first_num(s: str):
        m = _NUM_RE.search(s.replace(",", ""))
        return float(m.group()) if m else None
    a, g = first_num(answer), first_num(gold)
    if a is None or g is None:
        return 0.0
    return 1.0 if abs(a - g) <= tol * max(1.0, abs(g)) else 0.0


# ---------------------------------------------------------------------------
#  Abstention (behavioural probe scoring)
# ---------------------------------------------------------------------------
_REFUSAL_MARKERS = [
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "not contained in the context", "no information", "unable to answer",
    "not enough information", "insufficient information", "not provided",
]


def detect_abstention(answer: str) -> bool:
    """Heuristic: did the model refuse / say it doesn't know?"""
    a = _normalise(answer)
    return any(m in a for m in _REFUSAL_MARKERS)


def abstention_correct(abstained: bool, is_answerable: bool) -> float:
    """
    On an UNANSWERABLE item the correct behaviour is to abstain (1.0 if it did).
    On an ANSWERABLE item the correct behaviour is to answer (1.0 if it did NOT
    abstain). This turns abstention into a single comparable score.
    """
    if is_answerable:
        return 0.0 if abstained else 1.0
    return 1.0 if abstained else 0.0
