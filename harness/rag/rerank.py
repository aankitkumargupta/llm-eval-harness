"""
Optional reranking (pipeline step 4).

Takes the retriever's candidates and reorders them with a cross-encoder
(Together's rerank endpoint), keeping the top-n. Reranking often doesn't change
which gold passages are PRESENT (hit-rate) but improves their RANK (MRR) — the
harness logs both so that distinction shows up.
"""

from __future__ import annotations

from ..clients.together_client import TogetherClient
from ..store.schema import RetrievedChunk


class Reranker:
    def __init__(self, client: TogetherClient, model: str):
        self.client = client
        self.model = model

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        docs = [c.text for c in chunks]
        order = self.client.rerank(self.model, query, docs, top_n=top_n)
        out = []
        for new_rank, (orig_idx, score) in enumerate(order):
            ch = chunks[orig_idx]
            ch.score = score
            ch.rank = new_rank
            out.append(ch)
        return out
