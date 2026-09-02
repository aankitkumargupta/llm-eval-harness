"""
Optional reranking (pipeline step 4).

Takes the retriever's candidates and reorders them with a cross-encoder, keeping
the top-n. Reranking often doesn't change which gold passages are PRESENT
(hit-rate) but does improve their RANK (MRR) — the harness logs both so that
distinction is visible rather than averaged away.

Two additions:

  * **A capability failure is not an item failure.** If the configured provider
    has no rerank endpoint, every item would raise and be recorded as a model
    error — thousands of rows blaming the model for a config mistake. The
    reranker now degrades to identity once, warns, and lets the run produce
    valid (unreranked) results.
  * **Over-fetch awareness.** A reranker can only reorder what it was handed.
    Reranking a list of exactly k does nothing but reorder the same k; the gain
    comes from fetching a wider pool and letting the cross-encoder promote from
    deeper down. `RetrievalConfig.candidate_multiplier` does the fetching; this
    is where the note lives so the two stay connected.
"""

from __future__ import annotations

import threading

from ..clients.base import CapabilityError
from ..store.schema import RetrievedChunk


class Reranker:
    def __init__(self, client, model: str, fail_soft: bool = True):
        self.client = client
        self.model = model
        self.fail_soft = fail_soft
        self._disabled = False
        self._lock = threading.Lock()
        self.warnings: list[str] = []

    @property
    def disabled(self) -> bool:
        with self._lock:
            return self._disabled

    def _disable(self, reason: str) -> None:
        with self._lock:
            if not self._disabled:
                self._disabled = True
                self.warnings.append(reason)

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        if self.disabled:
            return chunks[:top_n]

        docs = [c.text for c in chunks]
        try:
            order = self.client.rerank(self.model, query, docs, top_n=top_n)
        except CapabilityError as e:
            if not self.fail_soft:
                raise
            self._disable(f"Reranking disabled for this run: {e}")
            return chunks[:top_n]

        out = []
        for new_rank, (orig_idx, score) in enumerate(order):
            # Defensive: a provider returning an out-of-range index would
            # otherwise IndexError and be recorded as a per-item model failure.
            if not (0 <= orig_idx < len(chunks)):
                continue
            ch = chunks[orig_idx]
            ch.score = score
            ch.rank = new_rank
            out.append(ch)
        return out or chunks[:top_n]
