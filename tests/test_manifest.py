"""
I9 (reproducibility) and I2 (pinned apparatus).

Phase 0 finding R-03: `harness/store/schema.py` documented a run manifest that
was never built, so every TraceRow carried config hashes pointing at nothing and
no run could be described, let alone reproduced. R-04: no `apparatus_hash`
existed, so a run whose embedder or judge had changed was silently comparable
with one where it had not.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.store.manifest import (
    RunManifest,
    apparatus_hash,
    dataset_hash,
    git_provenance,
    package_versions,
)


@dataclass
class _Item:
    item_id: str
    query: str = "q"
    gold_answer: str = "a"
    item_type: str = "answerable"


# --------------------------------------------------------------------------- #
#  I2 — apparatus hash
# --------------------------------------------------------------------------- #
def test_apparatus_hash_is_stable_for_the_same_apparatus():
    a = apparatus_hash(embedding_model="e", judge_model="j", rerank_model="r")
    b = apparatus_hash(embedding_model="e", judge_model="j", rerank_model="r")
    assert a == b


def test_changing_the_judge_changes_the_apparatus_hash():
    """I2. The judge is apparatus. Swapping it invalidates cross-run comparison
    even when the profile, dataset and models under test are identical."""
    a = apparatus_hash(embedding_model="e", judge_model="j1")
    b = apparatus_hash(embedding_model="e", judge_model="j2")
    assert a != b


def test_changing_the_embedder_changes_the_apparatus_hash():
    """I2. A different embedder means the models saw different retrieved
    passages — the run then measures the embedders, not the models."""
    assert apparatus_hash(embedding_model="e1") != apparatus_hash(embedding_model="e2")


def test_judge_ensemble_order_does_not_change_the_apparatus():
    """A reordered config is the same apparatus; it must not read as a change,
    or every config tidy-up would invalidate history for no reason."""
    a = apparatus_hash(judge_ensemble=("a", "b"))
    b = apparatus_hash(judge_ensemble=("b", "a"))
    assert a == b


# --------------------------------------------------------------------------- #
#  Dataset hash
# --------------------------------------------------------------------------- #
def test_dataset_hash_detects_an_edit_that_preserves_item_count():
    """The quietest way two runs stop being comparable: an evalset edited in
    place. Length is unchanged, so a count-based check would miss it."""
    before = [_Item("q1", gold_answer="paris"), _Item("q2")]
    after = [_Item("q1", gold_answer="PARIS-changed"), _Item("q2")]
    assert dataset_hash(before) != dataset_hash(after)


def test_dataset_hash_is_order_independent():
    """A shuffled evalset is the same evalset; the split seed decides order."""
    a = [_Item("q1"), _Item("q2")]
    assert dataset_hash(a) == dataset_hash(list(reversed(a)))


# --------------------------------------------------------------------------- #
#  I9 — the manifest itself
# --------------------------------------------------------------------------- #
def test_manifest_round_trips(tmp_path):
    m = RunManifest(run_id="r1", profile="p", models=("a", "b"),
                    apparatus_hash="h", seeds={"split_seed": 0})
    m.write(tmp_path)
    back = RunManifest.read(tmp_path)
    assert back.run_id == "r1"
    assert back.models == ("a", "b")
    assert back.apparatus_hash == "h"
    assert back.seeds == {"split_seed": 0}


def test_manifest_reader_tolerates_unknown_future_fields(tmp_path):
    """Manifests evolve additively; an older reader must not choke on a newer
    run's extra keys, or upgrading the harness orphans your history."""
    import json
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "r1", "profile": "p", "a_field_from_2027": 1}),
        encoding="utf-8")
    assert RunManifest.read(tmp_path).run_id == "r1"


def test_capture_records_provenance_even_when_it_cannot_be_determined(tmp_path):
    """I9. 'unknown' is recorded, never omitted: a missing key says nobody
    asked, an explicit 'unknown' says we asked and could not tell."""
    m = RunManifest.capture("r1", "p", root=tmp_path)   # not a git checkout
    assert m.git_sha == "unknown"
    assert m.git_dirty is None
    assert "python" in m.package_versions


def test_git_provenance_returns_a_sha_in_this_repo():
    sha, dirty = git_provenance(".")
    assert sha != "" and isinstance(dirty, (bool, type(None)))


def test_package_versions_names_the_libraries_that_move_numbers():
    v = package_versions()
    assert "numpy" in v and "pandas" in v


# --------------------------------------------------------------------------- #
#  Comparability — what the manifest is FOR
# --------------------------------------------------------------------------- #
def _m(**kw):
    base = dict(run_id="r", profile="p", dataset_hash="d",
                apparatus_hash="a", spec_hash="")
    base.update(kw)
    return RunManifest(**base)


def test_two_identical_runs_are_comparable():
    ok, reasons = _m().comparable_with(_m())
    assert ok and reasons == []


def test_a_changed_apparatus_blocks_comparison_and_says_why():
    """I2. This is the check that did not exist before R-04."""
    ok, reasons = _m().comparable_with(_m(apparatus_hash="other"))
    assert not ok
    assert any("apparatus" in r for r in reasons)


def test_a_changed_dataset_blocks_comparison():
    ok, reasons = _m().comparable_with(_m(dataset_hash="other"))
    assert not ok
    assert any("dataset" in r for r in reasons)


def test_an_aborted_run_is_not_rankable_against_a_complete_one():
    """A budget-aborted run holds real rows but an incomplete matrix; ranking
    it against a complete run compares different amounts of work."""
    ok, reasons = _m().comparable_with(_m(aborted=True))
    assert not ok
    assert any("aborted" in r for r in reasons)


def test_reasons_accumulate_rather_than_short_circuiting():
    """A report should be able to print everything that is wrong at once."""
    ok, reasons = _m().comparable_with(
        _m(apparatus_hash="x", dataset_hash="y", aborted=True))
    assert not ok and len(reasons) == 3
