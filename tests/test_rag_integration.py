"""
The RAG path, end to end, offline.

This is the integration test §5 asks for ("full fake-provider matrix run, real
Parquet") applied to the half of the harness that had never been executed by
anything. Phase 0 recorded `harness/rag/ingest.py` at **0% coverage** and the
`rag` package at 38.5%, and every session since has been unable to improve it:
the RAG path needs an embedding provider, and no reachable key served one.

The local `fake` provider changes that. It declares `supports_embeddings` and
returns deterministic 16-dimensional vectors from a sha256, so the whole
pipeline — ingest, chunk, embed, upsert, dense/sparse/hybrid retrieve, rerank,
prompt assembly, generation, scoring, retrieval metrics, trace write — runs
with no key and no network against an embedded Qdrant in `tmp_path`.

**What these tests do and do not establish.** They exercise the *plumbing*.
They do not establish retrieval *quality*, and no number produced here should
ever be reported as a result: hash vectors carry no semantics, so a hit-rate
from this test measures the harness's wiring, not its retrieval. That
distinction is the same one the fake provider's own docstring makes, and it is
worth being blunt about — a green RAG integration test is easy to mistake for
evidence that retrieval works.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.slow]

EMBED_MODEL = "fake-embed"


@pytest.fixture
def corpus():
    from harness.rag.documents import Document

    return [
        Document(doc_id="reg-1", source_uri="reg-1.txt", text=(
            "Section 12. A registered institution shall retain transaction "
            "records for seven years from the date of the transaction. "
            "Records may be held electronically provided they remain legible "
            "and retrievable throughout the retention period.")),
        Document(doc_id="reg-2", source_uri="reg-2.txt", text=(
            "Section 19. An institution must report a suspicious transaction "
            "to the supervisory authority within three business days of "
            "forming the suspicion. Late reporting attracts a penalty.")),
        Document(doc_id="reg-3", source_uri="reg-3.txt", text=(
            "Section 31. Customer identification must be completed before an "
            "account is opened. Identification documents must be verified "
            "against an independent source.")),
    ]


@pytest.fixture
def qdrant(tmp_path):
    """Embedded Qdrant in a temp folder: no Docker, no server, no port."""
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    yield client
    client.close()


@pytest.fixture
def fake():
    from harness.clients.fake_client import FakeClient

    return FakeClient(accuracy=1.0)


# --------------------------------------------------------------------------- #
#  Ingest
# --------------------------------------------------------------------------- #
def test_ingest_writes_chunks_that_can_be_counted(qdrant, fake, corpus):
    """`rag/ingest.py` was at 0% coverage — nothing in the repo had ever run it."""
    from harness.rag.ingest import Ingestor

    n = Ingestor(fake, qdrant, EMBED_MODEL).ingest(
        "t_reg", iter(corpus), chunk_size=200, overlap=40)
    assert n > 0
    assert qdrant.count("t_reg").count == n


def test_ingest_is_NOT_idempotent_in_embedded_mode(qdrant, fake, corpus):
    """Pins a BUG, not a desired behaviour. See docs/DEBT.md R-18.

    `Ingestor._ensure_collection` documents itself as "Idempotent: recreates
    cleanly so re-ingesting a profile is safe", and it does call
    `delete_collection` before `create_collection`. In **embedded** Qdrant —
    the mode `configs/run.yaml` calls "the normal choice for a single-user
    local run", and the only mode the Streamlit app uses — that delete does not
    purge the on-disk records. Verified directly: after an explicit
    `delete_collection`, `collection_exists` returns False and the next ingest
    still leaves double the points.

    The consequence is quiet and serious. Re-ingesting the same corpus — after
    fixing a typo, after an interrupted ingest, after changing `chunk_size` —
    doubles every chunk. Duplicates then compete for the top-k slots, so
    hit-rate@k, MRR, NDCG and context precision all move, and nothing in the
    report says why.

    This test asserts the CURRENT behaviour so the finding cannot regress
    unnoticed (§7: write a test describing current behaviour before changing
    it). When R-18 is fixed this test must fail, and it should then be
    rewritten to assert idempotence rather than deleted.
    """
    from harness.rag.ingest import Ingestor

    ing = Ingestor(fake, qdrant, EMBED_MODEL)
    first = ing.ingest("t_idem", iter(corpus), chunk_size=50, overlap=0)
    second = ing.ingest("t_idem", iter(corpus), chunk_size=50, overlap=0)

    assert first == second, "each ingest reports the same chunk count"
    assert qdrant.count("t_idem").count == first * 2, (
        "BUG (R-18): the second ingest appended instead of replacing. If this "
        "assertion now fails, embedded-mode delete finally purges and the test "
        "should be rewritten to assert idempotence.")


def test_chunking_respects_the_configured_size(qdrant, fake, corpus):
    """A smaller chunk size must produce more chunks, or the knob is inert and
    every tuning result over `chunk_size` is noise.

    `chunk_size` is in WORDS, not characters — `chunk_text`'s docstring says so
    and it is worth pinning, because the profiles express it as a bare number
    (`chunk_size: 220`) that reads naturally as either.
    """
    from harness.rag.ingest import Ingestor

    big = Ingestor(fake, qdrant, EMBED_MODEL).ingest(
        "t_big", iter(corpus), chunk_size=1000, overlap=0)
    small = Ingestor(fake, qdrant, EMBED_MODEL).ingest(
        "t_small", iter(corpus), chunk_size=8, overlap=0)
    assert small > big, "chunk_size is in words; 8 words must split these docs"


def test_chunk_size_is_measured_in_words():
    from harness.rag.ingest import chunk_text

    text = " ".join(f"w{i}" for i in range(25))
    assert len(chunk_text(text, chunk_size=10, overlap=0)) == 3
    assert len(chunk_text(text, chunk_size=25, overlap=0)) == 1


def test_overlap_repeats_words_between_adjacent_chunks():
    """Overlap exists so an answer spanning a boundary is not cut in half."""
    from harness.rag.ingest import chunk_text

    text = " ".join(f"w{i}" for i in range(20))
    chunks = chunk_text(text, chunk_size=10, overlap=4)
    assert len(chunks) > 2
    tail = chunks[0].split()[-4:]
    assert chunks[1].split()[:4] == tail


# --------------------------------------------------------------------------- #
#  Retrieve
# --------------------------------------------------------------------------- #
@pytest.fixture
def indexed(qdrant, fake, corpus):
    from harness.rag.ingest import Ingestor

    Ingestor(fake, qdrant, EMBED_MODEL).ingest(
        "t_ret", iter(corpus), chunk_size=200, overlap=40)
    return "t_ret"


def _retriever(fake, qdrant, indexed):
    from harness.rag.retrieve import Retriever

    return Retriever(fake, qdrant, indexed)


def _cfg(mode, k=3):
    """The embedder is part of the CONFIG, not the retriever: two runs with
    different embedders are not comparable, so it travels with the query."""
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import RetrievalMode

    return RetrievalConfig(mode=RetrievalMode(mode), k=k,
                           embedding_model=EMBED_MODEL)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_retrieval_mode_returns_chunks(fake, qdrant, indexed, mode):
    """All three modes must work against one index. The profile's `knobs`
    offer all three to the tuning search, so a mode that silently returns
    nothing would look like a legitimately bad configuration."""
    r = _retriever(fake, qdrant, indexed)
    chunks = r.retrieve("How long must records be retained?", _cfg(mode))
    assert chunks, f"{mode} returned nothing"
    assert all(c.chunk_id and c.doc_id and c.text for c in chunks)
    assert [c.rank for c in chunks] == list(range(len(chunks)))


def test_sparse_retrieval_finds_the_literal_match(fake, qdrant, indexed):
    """BM25 is the half of hybrid that does carry meaning here: the fake
    embedder's vectors are hashes, but the sparse index is real. A query using
    the corpus's own words must surface the document containing them."""
    r = _retriever(fake, qdrant, indexed)
    chunks = r.retrieve("suspicious transaction supervisory authority",
                        _cfg("sparse"))
    assert chunks[0].doc_id == "reg-2", (
        "BM25 must rank the document containing the query's own words first")


def test_k_is_respected(fake, qdrant, indexed):
    r = _retriever(fake, qdrant, indexed)
    for k in (1, 2, 3):
        assert len(r.retrieve("retention period", _cfg("hybrid", k))) <= k


def test_retrieval_is_deterministic(fake, qdrant, indexed):
    """Two models in one run must see the identical retrieved context, or the
    comparison measures the retriever rather than the models (I1, I2)."""
    r = _retriever(fake, qdrant, indexed)
    cfg = _cfg("hybrid")
    a = [c.chunk_id for c in r.retrieve("identification documents", cfg)]
    b = [c.chunk_id for c in r.retrieve("identification documents", cfg)]
    assert a == b


# --------------------------------------------------------------------------- #
#  Prompt assembly
# --------------------------------------------------------------------------- #
def test_context_ordering_changes_the_prompt(fake, qdrant, indexed):
    """`context_order` is a tuning knob. If reordering produced an identical
    prompt the knob would be inert, and any measured 'gain' from it noise."""
    from harness.rag.prompt import PromptConfig, assemble_messages

    r = _retriever(fake, qdrant, indexed)
    chunks = r.retrieve("retention period", _cfg("hybrid"))
    assert len(chunks) >= 2

    normal = assemble_messages("How long?", chunks, PromptConfig(context_order="as_is"))
    reverse = assemble_messages("How long?", chunks, PromptConfig(context_order="reverse"))
    assert str(normal) != str(reverse)


def test_the_prompt_carries_the_retrieved_text(fake, qdrant, indexed):
    from harness.rag.prompt import PromptConfig, assemble_messages

    r = _retriever(fake, qdrant, indexed)
    chunks = r.retrieve("suspicious transaction", _cfg("sparse", 2))
    blob = str(assemble_messages("When must it be reported?", chunks,
                                 PromptConfig()))
    assert any(c.text[:40] in blob for c in chunks)


# --------------------------------------------------------------------------- #
#  Retrieval metrics
# --------------------------------------------------------------------------- #
def test_retrieval_metrics_reward_a_hit_and_punish_a_miss():
    """These are the numbers a RAG evaluation exists to produce, and until now
    nothing had checked they move in the right direction."""
    from harness.eval.metrics import hit_rate_at_k, mrr, ndcg_at_k

    gold = ["reg-2"]
    hit_first = ["reg-2", "reg-1", "reg-3"]
    hit_third = ["reg-1", "reg-3", "reg-2"]
    miss = ["reg-1", "reg-3"]

    assert hit_rate_at_k(hit_first, gold, 3) == 1.0
    assert hit_rate_at_k(miss, gold, 3) == 0.0
    # Rank matters: the same hit further down is worth less.
    assert mrr(hit_first, gold) > mrr(hit_third, gold) > 0.0
    assert mrr(miss, gold) == 0.0
    assert ndcg_at_k(hit_first, gold, 3) > ndcg_at_k(hit_third, gold, 3)
