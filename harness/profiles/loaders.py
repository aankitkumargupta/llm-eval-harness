"""
Loaders that turn on-disk corpus/evalset files into typed objects.

Formats (both JSONL, one record per line):

  corpus.jsonl  — {"doc_id": "...", "text": "...", "source_uri": "..."}
  evalset.jsonl — {"item_id": "...", "query": "...", "item_type": "answerable",
                   "gold_answer": "...", "gold_passage_ids": ["docA#3"],
                   "history": [["user","..."],["assistant","..."]],
                   "human_label": 1.0, "meta": {}}

`item_type` defaults to "answerable"; probe items set a different type and omit
gold fields.

Beyond parsing, this module now *validates*. A malformed evalset used to fail
thousands of items into a paid run with a KeyError naming no line number. It now
fails at load with the line number and the problem, and `dataset_stats` reports
the quality issues that silently wreck a benchmark: duplicate questions,
unlabeled items, gold answers absent from the corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..rag.documents import Document
from ..store.schema import EvalItem, ItemType


class DatasetError(ValueError):
    """A corpus or evalset file is malformed, with the offending line named."""


def load_corpus(path: str) -> Iterator[Document]:
    """Streamed, so a large corpus never sits fully in memory during ingest."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Corpus file not found: {path}")
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON - {e}") from e
            if "doc_id" not in r or "text" not in r:
                raise DatasetError(
                    f"{path}:{lineno}: corpus records need 'doc_id' and 'text'; "
                    f"got keys {sorted(r)}")
            yield Document(doc_id=str(r["doc_id"]), text=r["text"],
                           source_uri=r.get("source_uri", ""))


def load_evalset(path: str, strict: bool = True) -> list[EvalItem]:
    """Load eval items, validating as we go.

    `strict=False` skips malformed lines instead of raising — useful when
    salvaging a hand-edited file, but it silently shrinks your evalset, so the
    default is to fail loudly.
    """
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Evalset file not found: {path}")

    items: list[EvalItem] = []
    seen_ids: set[str] = set()

    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    raise DatasetError(f"{path}:{lineno}: invalid JSON - {e}") from e
                continue

            item_id = str(r.get("item_id") or f"item_{lineno}")
            if "query" not in r:
                if strict:
                    raise DatasetError(
                        f"{path}:{lineno}: eval items need a 'query' field; "
                        f"got keys {sorted(r)}")
                continue
            if item_id in seen_ids:
                # Duplicate ids break the paired significance tests, which join
                # two models' rows on item_id — a duplicate silently pairs the
                # wrong answers together.
                if strict:
                    raise DatasetError(
                        f"{path}:{lineno}: duplicate item_id '{item_id}'. Item "
                        f"ids must be unique; paired statistics join on them.")
                item_id = f"{item_id}__dup{lineno}"
            seen_ids.add(item_id)

            raw_type = r.get("item_type", "answerable")
            try:
                item_type = ItemType(raw_type)
            except ValueError as e:
                if strict:
                    raise DatasetError(
                        f"{path}:{lineno}: unknown item_type '{raw_type}'. "
                        f"Valid: {[t.value for t in ItemType]}") from e
                item_type = ItemType.ANSWERABLE

            items.append(EvalItem(
                item_id=item_id,
                query=str(r["query"]),
                item_type=item_type,
                gold_answer=r.get("gold_answer"),
                gold_passage_ids=list(r.get("gold_passage_ids", []) or []),
                history=[tuple(h) for h in (r.get("history") or [])],
                meta=r.get("meta") or {},
                human_label=r.get("human_label"),
            ))

    if not items:
        raise DatasetError(f"{path}: no eval items found.")
    return items


def save_evalset(items: list[EvalItem], path: str) -> int:
    """Write eval items back out as JSONL. Used to persist generated probe sets."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "item_id": it.item_id, "query": it.query,
                "item_type": it.item_type.value,
                "gold_answer": it.gold_answer,
                "gold_passage_ids": it.gold_passage_ids,
            }
            if it.history:
                rec["history"] = [list(h) for h in it.history]
            if it.meta:
                rec["meta"] = it.meta
            if it.human_label is not None:
                rec["human_label"] = it.human_label
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(items)


# ---------------------------------------------------------------------------
#  Dataset quality
# ---------------------------------------------------------------------------
@dataclass
class DatasetStats:
    n_items: int = 0
    by_type: dict = field(default_factory=dict)
    labeled: int = 0            # items with gold_passage_ids
    with_gold_answer: int = 0
    with_human_label: int = 0
    duplicate_queries: int = 0
    mean_query_len: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{self.n_items} items ({', '.join(f'{k}={v}' for k, v in sorted(self.by_type.items()))})",
            f"{self.labeled} with gold passages, {self.with_gold_answer} with gold answers",
        ]
        if self.with_human_label:
            lines.append(f"{self.with_human_label} carry a human label "
                         f"(judge calibration available)")
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        return "\n".join(lines)


def dataset_stats(items: list[EvalItem], min_items: int = 30,
                  task: str = "rag") -> DatasetStats:
    """Assess an evalset before spending money on it.

    The warnings here are the ones that make a benchmark meaningless in ways the
    final numbers never reveal:

      * too few items to separate models (wide CIs, unstable rankings)
      * no gold passages, so every retrieval metric will be blank
      * duplicate questions, which inflate apparent sample size while adding no
        information — 200 items of which 80 are dupes is a 120-item benchmark
        reporting 200-item confidence
      * no unanswerable items, so abstention is never tested and a model that
        fabricates confidently scores identically to one that refuses honestly

    `task` gates the RAG-only warnings. Telling a classification profile that it
    has no gold passages is noise, and a validator that cries wolf is one people
    stop reading — which costs you the warnings that do matter.
    """
    is_rag = task == "rag"
    stats = DatasetStats(n_items=len(items))
    stats.by_type = dict(Counter(i.item_type.value for i in items))
    stats.labeled = sum(1 for i in items if i.gold_passage_ids)
    stats.with_gold_answer = sum(1 for i in items if i.gold_answer)
    stats.with_human_label = sum(1 for i in items if i.human_label is not None)

    queries = [i.query.strip().lower() for i in items]
    counts = Counter(queries)
    stats.duplicate_queries = sum(c - 1 for c in counts.values() if c > 1)
    stats.mean_query_len = (sum(len(q.split()) for q in queries) / len(queries)
                            if queries else 0.0)

    if stats.n_items < min_items:
        stats.warnings.append(
            f"only {stats.n_items} items - too few to rank models confidently. "
            f"Expect wide confidence intervals; {min_items}+ is a practical floor.")
    if is_rag:
        if stats.labeled == 0:
            stats.warnings.append(
                "no items have gold_passage_ids - every retrieval metric "
                "(hit-rate, MRR, NDCG, recall) will be blank.")
        elif stats.labeled < stats.n_items * 0.5:
            stats.warnings.append(
                f"only {stats.labeled}/{stats.n_items} items have gold passages; "
                f"retrieval metrics rest on that subset only.")
        if ItemType.UNANSWERABLE.value not in stats.by_type:
            stats.warnings.append(
                "no unanswerable items - abstention is untested, so a model "
                "that fabricates scores the same as one that honestly refuses. "
                "Run `python main.py probes --profile <p>` to generate them.")
    if stats.duplicate_queries:
        stats.warnings.append(
            f"{stats.duplicate_queries} duplicate question(s) - they inflate "
            f"apparent sample size without adding information.")
    if stats.with_gold_answer == 0:
        stats.warnings.append(
            "no gold answers - accuracy cannot be scored.")
    return stats


def corpus_chunk_ids(corpus_path: str, chunk_size: int,
                     overlap: int) -> set[str]:
    """Every chunk id ingestion will create, for validating gold_passage_ids.

    Gold ids that don't match real chunks score as permanent retrieval misses,
    which reads as "retrieval is broken" when the real problem is that the
    labeling used a different chunk size than the ingest.
    """
    from ..rag.ingest import chunk_text

    ids: set[str] = set()
    for doc in load_corpus(corpus_path):
        for i, _ in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            ids.add(f"{doc.doc_id}#{i}")
    return ids


def validate_gold_ids(items: list[EvalItem], corpus_path: str,
                      chunk_size: int, overlap: int) -> list[str]:
    """Report gold_passage_ids that don't correspond to any real chunk.

    This is the single most common silent failure in a hand-built RAG evalset:
    labels generated at one chunk size, index built at another. Retrieval looks
    catastrophically broken and the model gets blamed.
    """
    valid = corpus_chunk_ids(corpus_path, chunk_size, overlap)
    problems: list[str] = []
    for it in items:
        missing = [g for g in it.gold_passage_ids if g not in valid]
        if missing:
            problems.append(
                f"{it.item_id}: gold passage id(s) not in corpus: {missing[:3]}"
                + (" ..." if len(missing) > 3 else ""))
    if problems:
        problems.append(
            f"({len(problems)} item(s) affected. Gold ids must use the SAME "
            f"chunk_size/overlap as ingestion: chunk_size={chunk_size}, "
            f"overlap={overlap}.)")
    return problems
