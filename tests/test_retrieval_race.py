"""
Regression: embedded Qdrant under concurrent retrieval.

Found in the first live five-model RAG run. Nine items across four models
errored with "dictionary changed size during iteration" or "Dense vector bm25
is not found in the collection" — two strings, one cause: `QdrantClient(path=…)`
is not designed for concurrent use, its lazily built state (the BM25 model on
the first `Document` query, per-collection search structures) is mutated
without locks, and the harness runs eight workers. The rows were recorded as
errors (I7 — not scored as wrong), but the run was silently unpaired (I1).

Worse, and only visible by re-querying afterwards: for the rest of that
process every item's sparse list was *wrong without raising*, so hybrid
retrieval — sparse-weighted 1.5 to dense 1.0 — ranked junk above the gold
passage and the whole run scored at chance. That silent variant could not be
reproduced offline (here a lost race stays loud), so it is not asserted; the
same serialisation removes the race that causes both.

Two tests, because the honest fixture for each is different:

  * cold + concurrent: no `Document` is used on the main thread, so the
    first BM25 inference happens inside the workers. Asserts liveness. The
    negative control (guard bypassed) fails this ~30 times in 128 calls.
  * warm + concurrent: the collection is ingested through `Document`, so
    sparse retrieval has a real signal, and sixteen threads must each get
    their own section back first. Asserts correctness.

Both offline: embedded Qdrant and the cached BM25 model need no network, and
the suite-wide socket block would fail them if either tried.
"""

from __future__ import annotations

import collections
import os
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

N_DOCS = 30
N_THREADS = 16
PER_THREAD = 8


def _bm25_cached() -> bool:
    root = os.environ.get("FASTEMBED_CACHE_PATH") or str(
        Path(os.environ.get("TEMP", "/tmp")) / "fastembed_cache")
    return Path(root).is_dir() and any("bm25" in p.name.lower()
                                       for p in Path(root).iterdir())


def _collection(tmp_path, fake_client, *, warm: bool):
    qdrant_client = pytest.importorskip("qdrant_client")
    if not _bm25_cached():
        pytest.skip("Qdrant/bm25 weights not in the FastEmbed cache")
    from qdrant_client import models

    # Sized from the fake client's own vectors, never a literal: an earlier
    # version hard-coded 16 and the fake embeds 8.
    dim = len(fake_client.embed("fake-embed", ["probe"]).vectors[0])
    q = qdrant_client.QdrantClient(path=str(tmp_path / "q"))
    q.create_collection(
        "c",
        vectors_config={"dense": models.VectorParams(
            size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF)})

    def sparse(i: int, text: str):
        if warm:
            # Real BM25 vectors — and the model load that comes with them.
            return models.Document(text=text, model="Qdrant/bm25")
        # Placeholder vectors: no inference on this thread, so the first
        # model load happens wherever the first QUERY happens.
        return models.SparseVector(indices=[i, i + 1], values=[1.0, 0.5])

    q.upsert("c", points=[
        models.PointStruct(
            id=i,
            vector={"dense": [((i * 7 + d) % 13) / 13 for d in range(dim)],
                    "bm25": sparse(i, f"Section {i} sets a fine of {i * 100} dollars")},
            payload={"chunk_id": f"d{i}#0", "doc_id": f"d{i}",
                     "text": f"Section {i} sets a fine of {i * 100} dollars"})
        for i in range(N_DOCS)])
    return q


def _hammer(retriever, cfg, query_for):
    """Sixteen threads, eight retrievals each. Returns (errors, results)."""
    errors: collections.Counter = collections.Counter()
    results: list[tuple[int, list[str]]] = []
    lock = threading.Lock()

    def worker(n: int) -> None:
        for j in range(PER_THREAD):
            try:
                out = retriever.retrieve(query_for(n, j), cfg)
                with lock:
                    results.append((n, [c.chunk_id for c in out]))
            except Exception as e:                       # noqa: BLE001
                with lock:
                    errors[f"{type(e).__name__}: {str(e)[:60]}"] += 1

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors, results


def test_cold_concurrent_first_use_never_errors(tmp_path, fake_client):
    from harness.rag import retrieve as R
    from harness.store.schema import RetrievalMode

    q = _collection(tmp_path, fake_client, warm=False)
    try:
        r = R.Retriever(fake_client, q, "c")
        cfg = R.RetrievalConfig(mode=RetrievalMode.HYBRID, k=5,
                                embedding_model="fake-embed")
        errors, results = _hammer(r, cfg, lambda n, j: f"section {n} fine {j}")
    finally:
        q.close()

    assert not errors, f"concurrent first-use of embedded Qdrant failed: {dict(errors)}"
    assert len(results) == N_THREADS * PER_THREAD
    assert all(ids for _, ids in results), "every hybrid retrieval must return chunks"


def test_warm_concurrent_sparse_retrieval_is_correct(tmp_path, fake_client):
    """Each thread asks for its own section by number. 'section' is in every
    document (IDF ~ 0); the number is in exactly one; so the top sparse hit
    is unambiguous. Sparse mode only: the fake dense embedder is hash noise
    and would make a hybrid top-1 meaningless."""
    from harness.rag import retrieve as R
    from harness.store.schema import RetrievalMode

    q = _collection(tmp_path, fake_client, warm=True)
    try:
        r = R.Retriever(fake_client, q, "c")
        cfg = R.RetrievalConfig(mode=RetrievalMode.SPARSE, k=3,
                                embedding_model="fake-embed")
        errors, results = _hammer(r, cfg, lambda n, j: f"section {n}")
    finally:
        q.close()

    assert not errors, dict(errors)
    wrong = [(n, ids[:3]) for n, ids in results if not ids or ids[0] != f"d{n}#0"]
    assert not wrong, f"sparse retrieval returned wrong lists under concurrency: {wrong[:5]}"


def test_an_embedded_client_is_serialised_and_a_server_client_is_not(fake_client):
    """The lock exists for embedded Qdrant's sake. A server client is
    thread-safe, and locking it would throttle the one deployment shape that
    scales — so the decision is made from the client's type, once."""
    from harness.rag import retrieve as R

    class _Local:
        pass

    _Local.__name__ = "QdrantLocal"

    class _Embedded:
        _client = _Local()

    class _Server:
        _client = object()

    assert R._is_embedded(_Embedded())
    assert not R._is_embedded(_Server())
    assert R.Retriever(fake_client, _Embedded(), "c")._qlock is R._LOCAL_QDRANT_LOCK
    assert isinstance(R.Retriever(fake_client, _Server(), "c")._qlock, R._NoLock)
