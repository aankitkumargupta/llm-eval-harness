"""
Retrieval (pipeline step 3), the three modes plus RRF fusion.

  dense  — cosine/dot over embeddings (semantic; catches paraphrase)
  sparse — BM25 lexical (exact tokens: part numbers, statutes, error codes)
  hybrid — run both, fuse ranks with Reciprocal Rank Fusion (RRF)

RRF is score-agnostic: it merges two ranked lists by rank POSITION, so we never
have to reconcile a cosine score against a BM25 score. That's why it's the
default fusion here.

The `reciprocal_rank_fusion` function is deliberately pure (no Qdrant) so it can
be unit-tested directly — it's the one piece of retrieval logic worth testing in
isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient, models

from ..clients.together_client import TogetherClient
from ..store.schema import RetrievedChunk, RetrievalMode


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """
    Fuse several ranked ID lists into one. For each list, an item at 0-based
    position `rank` contributes 1 / (k + rank + 1) to its RRF score. Items are
    then sorted by summed score. k=60 is the standard constant from the original
    RRF paper; it damps the influence of very high ranks.

    Returns [(id, fused_score), ...] best-first.
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item_id in enumerate(lst):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class RetrievalConfig:
    mode: RetrievalMode
    k: int = 10                 # candidates to retrieve
    rrf_k: int = 60             # RRF damping constant
    embedding_model: str = ""   # must match the corpus's embedder


class Retriever:
    def __init__(self, client: TogetherClient, qdrant: QdrantClient,
                 collection: str):
        self.client = client
        self.qdrant = qdrant
        self.collection = collection

    def _points_to_chunks(self, points) -> list[RetrievedChunk]:
        chunks = []
        for rank, p in enumerate(points):
            payload = p.payload or {}
            chunks.append(RetrievedChunk(
                chunk_id=payload.get("chunk_id", str(p.id)),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                score=float(p.score) if p.score is not None else 0.0,
                source_uri=payload.get("source_uri"),
                rank=rank,
            ))
        return chunks

    def _dense(self, query: str, k: int) -> list[RetrievedChunk]:
        vec = self.client.embed(self._embed_model, [query]).vectors[0]
        res = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec, using="dense", limit=k, with_payload=True,
        ).points
        return self._points_to_chunks(res)

    def _sparse(self, query: str, k: int) -> list[RetrievedChunk]:
        res = self.qdrant.query_points(
            collection_name=self.collection,
            query=models.Document(text=query, model="Qdrant/bm25"),
            using="bm25", limit=k, with_payload=True,
        ).points
        return self._points_to_chunks(res)

    def retrieve(self, query: str, cfg: RetrievalConfig) -> list[RetrievedChunk]:
        """Dispatch on mode. Returns candidates ordered best-first."""
        self._embed_model = cfg.embedding_model  # used by _dense

        if cfg.mode == RetrievalMode.DENSE:
            return self._dense(query, cfg.k)

        if cfg.mode == RetrievalMode.SPARSE:
            return self._sparse(query, cfg.k)

        # HYBRID: retrieve both, fuse with RRF, then map fused ids back to chunks.
        dense = self._dense(query, cfg.k)
        sparse = self._sparse(query, cfg.k)
        by_id = {c.chunk_id: c for c in dense}
        for c in sparse:
            by_id.setdefault(c.chunk_id, c)
        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in dense], [c.chunk_id for c in sparse]],
            k=cfg.rrf_k,
        )
        out = []
        for new_rank, (cid, fscore) in enumerate(fused[:cfg.k]):
            ch = by_id[cid]
            ch.score = fscore
            ch.rank = new_rank
            out.append(ch)
        return out
