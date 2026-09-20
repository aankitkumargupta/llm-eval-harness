"""
ARC-Easy: a second spec over the existing `ARCAdapter`, offline, against the
committed 20-row fixture (the first 20 rows of the ARC-Easy test split in the
exact shape the fetcher caches).

No new adapter is involved; what this file pins is that the spec is distinct
from `arc_challenge` in the ways that matter (config, checksum, spec_hash) and
identical in the ways that must not drift (chance level, extraction chain).
Expected scores are read off the fixture's `answerKey` by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.bench.adapters.arc import ARCAdapter
from harness.bench.contracts import Failed
from harness.bench.spec import load_spec

FIXTURE = Path(__file__).parent / "fixtures" / "arc_easy" / "arc_easy_test_20.jsonl"
PLACEHOLDER = "sha256:" + "0" * 64


@pytest.fixture(scope="module")
def spec():
    return load_spec("arc_easy")


@pytest.fixture(scope="module")
def adapter(spec):
    return ARCAdapter(spec, data_path=str(FIXTURE))


def _by_id(adapter) -> dict:
    return {it.item_id: it for it in adapter.load()}


# --------------------------------------------------------------------------- #
#  Spec
# --------------------------------------------------------------------------- #
def test_spec_is_valid_and_points_at_the_easy_config(spec):
    assert spec.validate() == []
    assert spec.source.ref == "allenai/ai2_arc"
    assert spec.source.config == "ARC-Easy"
    assert spec.source.split == "test"
    assert spec.source.licence == "CC-BY-SA-4.0" and spec.source.commercial_use
    assert spec.family == "multiple_choice"
    assert spec.scoring.chance_level == pytest.approx(0.25)


def test_spec_pins_a_real_checksum_distinct_from_arc_challenge(spec):
    assert spec.source.checksum.startswith("sha256:")
    assert spec.source.checksum != PLACEHOLDER
    other = load_spec("arc_challenge")
    assert spec.source.checksum != other.source.checksum
    assert spec.spec_hash() != other.spec_hash(), (
        "two specs over different splits must never be comparable")


def test_scoring_rules_are_identical_to_arc_challenge(spec):
    """Same adapter, same chain, same chance: the only variable is the split."""
    other = load_spec("arc_challenge")
    assert list(spec.scoring.chain) == list(other.scoring.chain)
    assert spec.scoring.chance_level == other.scoring.chance_level
    assert spec.scoring.mode == other.scoring.mode


# --------------------------------------------------------------------------- #
#  Fixture
# --------------------------------------------------------------------------- #
def test_fixture_has_twenty_rows_in_the_cached_shape():
    lines = [ln for ln in FIXTURE.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(lines) == 20
    for ln in lines:
        rec = json.loads(ln)
        assert {"id", "question", "choices", "answerKey"} <= set(rec)
        assert len(rec["choices"]["text"]) == len(rec["choices"]["label"])


def test_loads_every_fixture_row(adapter):
    items = adapter.load()
    assert len(items) == 20
    assert len({it.item_id for it in items}) == 20
    assert all(it.gold_answer in it.meta["letters"] for it in items)


# --------------------------------------------------------------------------- #
#  Hand-computed scores: (item_id, gold letter read off answerKey,
#  a model reply, expected accuracy).
# --------------------------------------------------------------------------- #
HAND_COMPUTED: list[tuple[str, str, str, float]] = [
    # row 0: answerKey A (photosynthesis / sunlight)
    ("Mercury_417466", "A", "Answer: A", 1.0),
    # row 2: answerKey D (meiosis occurs in ovary cells)
    ("Mercury_7239733", "D",
     "Germ cells divide in the gonads, so the ovary. Answer: D", 1.0),
    # row 14: labels are 1-4 and answerKey is "2"; the adapter normalises both
    # sides by the same rule, so the gold is B and only B scores.
    ("NYSEDREGENTS_2015_8_28", "B", "Answer: B", 1.0),
    ("NYSEDREGENTS_2015_8_28", "B", "Answer: C", 0.0),
    # row 5: answerKey C (light-year); the model picks the metre
    ("CSZ20679", "C", "Answer: A", 0.0),
    # row 17: answerKey D (the water evaporated), answered in bold only
    ("MCAS_1998_4_11", "D", "**D**", 1.0),
]


@pytest.mark.parametrize("item_id,gold,raw,expected", HAND_COMPUTED)
def test_hand_computed_scores(adapter, item_id, gold, raw, expected):
    item = _by_id(adapter)[item_id]
    assert item.gold_answer == gold
    ext = adapter.extract(raw)
    assert ext.ok
    s = adapter.score(item, ext)
    assert s["accuracy"] == pytest.approx(expected)
    assert s["extraction_failed"] is False


def test_extraction_failure_is_not_a_zero(adapter):
    item = adapter.load(limit=1)[0]
    s = adapter.score(item, Failed("no extractor matched"))
    assert s["accuracy"] is None and s["extraction_failed"] is True


def test_same_seed_and_limit_give_the_same_items(spec):
    a = [it.item_id for it in ARCAdapter(spec, str(FIXTURE)).load(seed=3, limit=7)]
    b = [it.item_id for it in ARCAdapter(spec, str(FIXTURE)).load(seed=3, limit=7)]
    assert a == b and len(a) == 7
