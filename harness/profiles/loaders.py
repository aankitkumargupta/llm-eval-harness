"""
Loaders that turn on-disk corpus/evalset files into typed objects.

Formats (both JSONL, one record per line):

  corpus.jsonl  — {"doc_id": "...", "text": "...", "source_uri": "..."}
  evalset.jsonl — {"item_id": "...", "query": "...", "item_type": "answerable",
                   "gold_answer": "...", "gold_passage_ids": ["docA#3"],
                   "history": [["user","..."],["assistant","..."]], "meta": {}}

item_type defaults to "answerable". Unanswerable/probe items simply set a
different item_type and omit gold fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ..rag.ingest import Document
from ..store.schema import EvalItem, ItemType


def load_corpus(path: str) -> Iterator[Document]:
    """Streamed, so a large corpus never sits fully in memory during ingest."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield Document(
                doc_id=r["doc_id"], text=r["text"], source_uri=r.get("source_uri", "")
            )


def load_evalset(path: str) -> list[EvalItem]:
    items: list[EvalItem] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items.append(
                EvalItem(
                    item_id=r["item_id"],
                    query=r["query"],
                    item_type=ItemType(r.get("item_type", "answerable")),
                    gold_answer=r.get("gold_answer"),
                    gold_passage_ids=r.get("gold_passage_ids", []),
                    history=[tuple(h) for h in r.get("history", [])],
                    meta=r.get("meta", {}),
                )
            )
    return items
