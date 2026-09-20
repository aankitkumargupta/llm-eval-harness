"""
in_statute_qa: the profile validates as a hybrid RAG profile over the
multilingual embedder, every passage is one chunk so gold passage ids are
exact, every gold id exists, the summaries are labelled as summaries, the
builder is idempotent (probes included), and the probe counts match the
profile so nobody appends them twice. Offline throughout.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from harness.profiles.loaders import load_corpus, load_evalset
from harness.profiles.profile import Profile
from harness.rag.ingest import chunk_text
from harness.store.schema import ItemType

PROFILE = "configs/profiles/in_statute_qa.yaml"
DATA = Path("data/in_statute_qa")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def test_the_profile_validates_and_pins_the_apparatus():
    p = Profile.from_yaml(PROFILE)
    assert p.task.value == "rag"
    assert p.validate() == [], p.validate()
    assert "multilingual" in p.embedding_model
    assert p.retrieval_mode.value == "hybrid"
    assert p.sparse_weight > p.dense_weight, "section numbers are lexical tokens"
    assert p.abstention_judge is True
    assert p.metric_weights["cost_usd"] < 0
    assert "citation_supporting" in p.active_metrics


def test_every_passage_is_one_chunk_labelled_as_a_summary_and_gold_ids_exist():
    p = Profile.from_yaml(PROFILE)
    docs = list(load_corpus(p.corpus_path))
    assert len(docs) >= 40
    chunk_ids = set()
    for d in docs:
        chunks = chunk_text(d.text, p.chunk_size, p.overlap)
        assert len(chunks) == 1, f"{d.doc_id}: {len(chunks)} chunks"
        assert "summary" in d.text.lower() or "guidance" in d.text.lower(), \
            f"{d.doc_id}: must say it is a summary, not the statute"
        chunk_ids.add(f"{d.doc_id}#0")
    items = load_evalset(p.evalset_path)
    for it in items:
        for gid in it.gold_passage_ids:
            assert gid in chunk_ids, f"{it.item_id}: gold {gid} not in corpus"


def test_the_language_mix_and_unanswerables_are_as_documented():
    base = [json.loads(line) for line in
            (DATA / "evalset_base.jsonl").read_text(encoding="utf-8").splitlines()]
    langs = {}
    for r in base:
        langs[r["meta"]["language"]] = langs.get(r["meta"]["language"], 0) + 1
    assert langs["en"] >= 25 and langs["hi"] >= 12 and langs["hinglish"] >= 10, langs
    for r in base:
        if r["meta"]["language"] == "hi":
            assert DEVANAGARI.search(r["query"]), r["item_id"]
            if r["gold_answer"]:
                assert DEVANAGARI.search(r["gold_answer"]), r["item_id"]
    unans = [r for r in base if r["item_type"] == "unanswerable"]
    assert len(unans) == 6
    assert all(r["gold_answer"] is None and r["gold_passage_ids"] == [] for r in unans)


def test_probe_counts_match_the_profile_so_nobody_appends_twice():
    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)
    base = [it for it in items if "probe" not in it.meta]
    answerable = [it for it in base if it.item_type == ItemType.ANSWERABLE]
    probes = [it for it in items if "probe" in it.meta]
    counts = {}
    for it in probes:
        counts[it.meta["probe"]] = counts.get(it.meta["probe"], 0) + 1
    n = len(answerable)
    for kind, frac in (("unanswerable", p.probes.unanswerable), ("injection", p.probes.injection),
                       ("noise", p.probes.noise), ("paraphrase", p.probes.paraphrase)):
        assert abs(counts.get(kind, 0) - round(n * frac)) <= 1, (kind, counts, n)
    for it in probes:
        if it.meta["probe"] == "injection":
            canary = it.meta["canary"]
            assert canary and canary not in it.query
            assert all(canary not in (b.gold_answer or "") for b in base)


def test_the_builder_is_idempotent():
    before = {f: (DATA / f).read_text(encoding="utf-8")
              for f in ("corpus.jsonl", "evalset_base.jsonl", "evalset.jsonl")}
    import runpy
    runpy.run_path(str(DATA / "build_dataset.py"), run_name="__main__")
    after = {f: (DATA / f).read_text(encoding="utf-8") for f in before}
    assert after == before
