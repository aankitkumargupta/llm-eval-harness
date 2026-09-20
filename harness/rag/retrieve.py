"""
Retrieval (pipeline step 3): three modes plus RRF fusion.

  dense, cosine/dot over embeddings (semantic; catches paraphrase)
  sparse: BM25 lexical (exact tokens: part numbers, statutes, error codes)
  hybrid, run both, fuse ranks with Reciprocal Rank Fusion

RRF is score-agnostic: it merges ranked lists by *position*, so we never have to
reconcile a cosine score against a BM25 score. That's why it's the default fusion.

Two fixes over the original:

**A data race.** `retrieve()` used to stash the embedding model on `self`
(`self._embed_model = cfg.embedding_model`) and read it back inside `_dense()`.
One `Retriever` is shared across every worker thread in the throughput lane, so
two threads with different configs, exactly what the tuning search produces,
could interleave between the write and the read and embed a query with the wrong
model. The corrupted retrieval would then be scored as a real result. It's now a
parameter, so the config can't leak across threads.

**Redundant embedding spend.** The same query is embedded once per model, per
tuning candidate, per pass, the identical vector bought a hundred times over on
a 4-model, 20-candidate run. Query embeddings are now cached by (model, text).

**A second data race, in the dependency.** Embedded Qdrant (`QdrantClient(path=...)`)
is not designed for concurrent use, its own documentation says so, and the
harness runs eight workers against it. Lazily built state (the BM25 model on
the first `Document` query, per-collection search structures) is mutated
without locks, so threads that arrive together fail with "dictionary changed
size during iteration" or, once the model name cannot be resolved, fall into
the dense path and raise "Dense vector bm25 is not found in the collection".
Worse than the nine errored rows in the first live five-model run: after a
lost race the process's sparse inference kept returning *wrong* lists without
raising, so every item's hybrid retrieval was silently corrupted and the run
scored at chance. Reproduced offline in tests/test_retrieval_race.py, with a
correctness assertion, not just an exception count.

A guard around only the first sparse inference was tried and was not enough,
the dense path trips the same class of race. So every call into an EMBEDDED
client is serialised under one process-wide lock. A query is microseconds;
the LLM call it precedes is seconds; the lock is invisible. A server-mode
client (HTTP) is thread-safe and is not locked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..store.schema import RetrievalMode, RetrievedChunk

if TYPE_CHECKING:  # annotation only - see the note in _sparse
    from qdrant_client import QdrantClient


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked ID lists into one.

    An item at 0-based position `rank` in a list contributes
    `weight / (k + rank + 1)`. Items are then sorted by summed score. k=60 is the
    constant from the original RRF paper; it damps the influence of very high
    ranks so one list can't dominate on its top hit alone.

    `weights` lets a profile bias the fusion, a corpus of statutes and part
    numbers wants lexical weighted above semantic, and a paraphrase-heavy
    support corpus wants the reverse. Defaults to equal weighting.
    """
    scores: dict[str, float] = {}
    for i, lst in enumerate(ranked_lists):
        w = weights[i] if weights and i < len(weights) else 1.0
        for rank, item_id in enumerate(lst):
            scores[item_id] = scores.get(item_id, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


@dataclass
class RetrievalConfig:
    mode: RetrievalMode
    k: int = 10                 # candidates to retrieve
    rrf_k: int = 60             # RRF damping constant
    embedding_model: str = ""   # must match the corpus's embedder
    dense_weight: float = 1.0   # hybrid fusion bias
    sparse_weight: float = 1.0
    # Over-fetch before reranking. A reranker can only reorder what retrieval
    # handed it, so asking for exactly k leaves it nothing to promote; fetching
    # a wider pool is where rerank gain actually comes from.
    candidate_multiplier: int = 1


#: Serialises every call into an embedded Qdrant (see the module docstring).
#: Process-wide rather than per-Retriever because the unsafe state lives in
#: the qdrant_client package and the collection files, not in any instance.
_LOCAL_QDRANT_LOCK = threading.Lock()


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _is_embedded(qdrant) -> bool:
    """True for `QdrantClient(path=...)` / `(":memory:")`, False for a server."""
    inner = getattr(qdrant, "_client", None)
    return type(inner).__name__ == "QdrantLocal"


class Retriever:
    def __init__(self, client, qdrant: QdrantClient, collection: str,
                 cache=None, meter=None):
        self.client = client
        self.qdrant = qdrant
        self.collection = collection
        self.cache = cache
        self.meter = meter
        # Decided once: the lock is taken on every query for an embedded
        # client and never for a server, and that must not depend on which
        # thread asks.
        self._qlock = _LOCAL_QDRANT_LOCK if _is_embedded(qdrant) else _NoLock()

    # ------------------------------------------------------------------ #
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

    def embed_query(self, text: str, embedding_model: str) -> list[float]:
        """Embed one query, served from cache when we've seen it before.

        The evalset is fixed while models and tuning candidates vary, so nearly
        every query embedding after the first pass is a repeat.
        """
        if self.cache is not None:
            key = self.cache.key_for_embedding(embedding_model, text)
            hit = self.cache.get(key)
            if hit is not None:
                if self.meter is not None:
                    self.meter.record_cache_hit()
                return hit["vector"]

        vec = self.client.embed(embedding_model, [text]).vectors[0]
        if self.cache is not None:
            self.cache.set(key, {"vector": vec})
        return vec

    def _dense(self, query: str, k: int, embedding_model: str) -> list[RetrievedChunk]:
        # embedding_model is a parameter, not instance state, see module docstring.
        vec = self.embed_query(query, embedding_model)
        with self._qlock:
            res = self.qdrant.query_points(
                collection_name=self.collection,
                query=vec, using="dense", limit=k, with_payload=True,
            ).points
        return self._points_to_chunks(res)

    def _sparse(self, query: str, k: int) -> list[RetrievedChunk]:
    # Imported inside the method: `qdrant_client` costs ~8s to import,
    # and the app imports this module at startup while only *using* it
    # during an ingest or a retrieval.
        from qdrant_client import models

        with self._qlock:
            res = self.qdrant.query_points(
                collection_name=self.collection,
                query=models.Document(text=query, model="Qdrant/bm25"),
                using="bm25", limit=k, with_payload=True,
            ).points
        return self._points_to_chunks(res)

    def retrieve(self, query: str, cfg: RetrievalConfig) -> list[RetrievedChunk]:
        """Dispatch on mode. Returns candidates ordered best-first."""
        fetch_k = max(cfg.k, cfg.k * max(1, cfg.candidate_multiplier))

        if cfg.mode == RetrievalMode.DENSE:
            return self._dense(query, fetch_k, cfg.embedding_model)
        if cfg.mode == RetrievalMode.SPARSE:
            return self._sparse(query, fetch_k)

        # HYBRID: retrieve both, fuse with RRF, map fused ids back to chunks.
        dense = self._dense(query, fetch_k, cfg.embedding_model)
        sparse = self._sparse(query, fetch_k)
        by_id = {c.chunk_id: c for c in dense}
        for c in sparse:
            by_id.setdefault(c.chunk_id, c)
        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in dense], [c.chunk_id for c in sparse]],
            k=cfg.rrf_k, weights=[cfg.dense_weight, cfg.sparse_weight],
        )
        out = []
        for new_rank, (cid, fscore) in enumerate(fused[:fetch_k]):
            ch = by_id[cid]
            ch.score = fscore
            ch.rank = new_rank
            out.append(ch)
        return out
