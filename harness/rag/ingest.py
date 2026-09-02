"""
Offline corpus ingestion (pipeline step 0).

Runs ONCE per profile, outside the timed/billed evaluation path. Chunks each
document, embeds the chunks (dense) AND builds BM25 statistics (sparse), then
upserts everything into Qdrant with citation metadata (doc_id, source_uri).

Memory discipline: ingestion is the one memory-spiky moment on an 8 GB machine,
so we stream in batches and never hold the whole corpus in RAM at once.

Qdrant stores BOTH a dense vector and a sparse (BM25) vector per point, which is
what lets retrieve.py serve dense / sparse / hybrid from one index without a
separate lexical engine.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

from .documents import Document

if TYPE_CHECKING:  # annotations only - importing these SDKs costs ~20s
    from qdrant_client import QdrantClient

    from ..clients.base import Embedder

__all__ = ["Document", "Ingestor", "chunk_text"]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Simple word-window chunker. chunk_size and overlap are in words. Deterministic
    so the same corpus + config always yields identical chunks (reproducibility).
    Swap in a sentence/markdown-aware splitter here if a profile needs it.
    """
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        return [text]
    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


class Ingestor:
    def __init__(self, client: Embedder, qdrant: QdrantClient,
                 embedding_model: str):
        self.client = client
        self.qdrant = qdrant
        self.embedding_model = embedding_model

    def _ensure_collection(self, collection: str, dim: int) -> None:
        """Create the collection with a dense vector + a BM25 sparse vector.
        Idempotent: recreates cleanly so re-ingesting a profile is safe."""
    # Imported inside the method: `qdrant_client` costs ~8s to import,
    # and the app imports this module at startup while only *using* it
    # during an ingest or a retrieval.
        from qdrant_client import models

        if self.qdrant.collection_exists(collection):
            self.qdrant.delete_collection(collection)
        self.qdrant.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # IDF modifier makes this behave like BM25 rather than raw counts.
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    def ingest(self, collection: str, documents: Iterator[Document],
               chunk_size: int, overlap: int, batch_size: int = 128,
               progress_cb=None) -> int:
        """
        Chunk -> embed -> upsert, in batches. Returns the number of chunks indexed.

        We build (chunk_id, doc_id, text, source_uri) tuples, embed the batch's
        texts in one API call, attach a BM25 sparse vector, and upsert. Memory
        stays flat because we never accumulate more than `batch_size` chunks.

        progress_cb, if given, is called as progress_cb(chunks_done) after each
        batch — used by the UI to drive a progress bar. It must not raise.
        """
        # We need the embedding dimension before creating the collection, so we
        # embed the very first batch, learn the dim, then create + upsert.
        buffer: list[tuple[str, str, str, str]] = []  # (chunk_id, doc_id, text, uri)
        total = 0
        created = False

        from qdrant_client import models

        def flush(buf):
            nonlocal created, total
            if not buf:
                return
            texts = [t for (_, _, t, _) in buf]
            emb = self.client.embed(self.embedding_model, texts)
            if not created:
                self._ensure_collection(collection, dim=len(emb.vectors[0]))
                created = True
            points = []
            for (cid, did, txt, uri), vec in zip(buf, emb.vectors):
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": vec,
                        # Qdrant computes BM25 term stats server-side from the
                        # text via the IDF modifier; we pass the document text
                        # through a sparse encoder. Here we hand Qdrant the raw
                        # text using its built-in BM25 document representation.
                        "bm25": models.Document(text=txt, model="Qdrant/bm25"),
                    },
                    payload={"chunk_id": cid, "doc_id": did,
                             "text": txt, "source_uri": uri},
                ))
            self.qdrant.upsert(collection_name=collection, points=points)
            total += len(points)
            if progress_cb is not None:
                try:
                    progress_cb(total)
                except Exception:
                    pass  # progress reporting must never break ingestion

        for doc in documents:
            chunks = chunk_text(doc.text, chunk_size, overlap)
            for i, ch in enumerate(chunks):
                chunk_id = f"{doc.doc_id}#{i}"
                buffer.append((chunk_id, doc.doc_id, ch, doc.source_uri))
                if len(buffer) >= batch_size:
                    flush(buffer)
                    buffer = []
        flush(buffer)  # trailing partial batch
        return total
