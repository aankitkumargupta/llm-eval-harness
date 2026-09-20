"""
OpenBookQA, TruthfulQA (mc1), WinoGrande and MGSM (Bengali, Telugu): the
§11 battery for each, against committed 20-item fixtures. Offline: no
fetch, no network; the fixtures are seeded samples of the fetched splits.

What each adapter must get right is different, and the tests say which:
  * OpenBookQA: the ARC adapter serves it unchanged except the question
    field name; a wrong field would silently prompt with an empty question.
  * TruthfulQA: the raw data lists the correct option first in every item.
    Unshuffled, "always A" scores 100%. The shuffle must be deterministic,
    must move the answer, and the gold letter must follow it.
  * WinoGrande: "1"/"2" become A/B; a stray letter is a format violation,
    not a wrong answer.
  * MGSM: native-script digits in the answer must count as the number they
    are, or the harness measures its parser.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.bench.adapters.arc import ARCAdapter
from harness.bench.adapters.mgsm import MGSMAdapter, normalise_digits
from harness.bench.adapters.truthfulqa_mc import TruthfulQAMCAdapter
from harness.bench.adapters.winogrande import WinoGrandeAdapter
from harness.bench.spec import load_spec

FIX = Path(__file__).resolve().parent / "fixtures"


def _fixture(bid: str) -> str:
    return str(FIX / bid / "sample.jsonl")


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
#  OpenBookQA through the ARC adapter
# --------------------------------------------------------------------------- #
def test_openbookqa_loads_the_question_stem_and_every_fixture_item():
    a = ARCAdapter(load_spec("openbookqa"), _fixture("openbookqa"))
    items = a.load()
    assert len(items) == 20
    assert all(it.query.strip() for it in items), "question_stem must become the query"
    assert all(it.gold_answer in "ABCD" and len(it.gold_answer) == 1 for it in items)


def test_openbookqa_hand_scored_items(tmp_path):
    rows = [
        {"id": "q1", "question_stem": "Water boils at", "answerKey": "B",
         "choices": {"label": ["A", "B", "C", "D"], "text": ["0 C", "100 C", "50 C", "10 C"]}},
        {"id": "q2", "question_stem": "The sun is a", "answerKey": "C",
         "choices": {"label": ["A", "B", "C", "D"], "text": ["planet", "moon", "star", "comet"]}},
    ]
    a = ARCAdapter(load_spec("openbookqa"), _write(tmp_path, "obqa.jsonl", rows))
    q1, q2 = a.load()
    assert a.score(q1, a.extract("Water boils at 100 degrees.\nAnswer: B")).values["accuracy"] == 1.0
    assert a.score(q1, a.extract("Answer: A")).values["accuracy"] == 0.0
    assert a.score(q2, a.extract("**C**")).values["accuracy"] == 1.0            # bold letter
    assert a.score(q2, a.extract("I think it is (C)")).values["accuracy"] == 1.0  # unexpected but valid
    s = a.score(q2, a.extract("no idea"))
    assert s.values["accuracy"] is None and s.values["extraction_failed"] is True


# --------------------------------------------------------------------------- #
#  TruthfulQA mc1
# --------------------------------------------------------------------------- #
def test_truthfulqa_shuffles_the_correct_first_raw_order():
    """Every raw item lists the correct option first. After loading, the
    gold letter must NOT be A for all of them, and each gold letter must
    point at the option that was labelled 1."""
    a = TruthfulQAMCAdapter(load_spec("truthfulqa_mc"), _fixture("truthfulqa_mc"))
    items = a.load()
    assert len(items) == 20
    golds = [it.gold_answer for it in items]
    assert set(golds) != {"A"}, "unshuffled: always-A would score 100%"
    raw = [json.loads(line) for line in Path(_fixture("truthfulqa_mc")).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(i): r for i, r in enumerate(raw)}
    for it in items:
        r = by_id[it.item_id]
        correct_text = r["mc1_targets"]["choices"][r["mc1_targets"]["labels"].index(1)]
        letters = it.meta["letters"]
        assert it.meta["options"][letters.index(it.gold_answer)] == correct_text


def test_truthfulqa_shuffle_is_deterministic_and_seed_sensitive():
    a = TruthfulQAMCAdapter(load_spec("truthfulqa_mc"), _fixture("truthfulqa_mc"))
    one = [(it.item_id, it.gold_answer, tuple(it.meta["options"])) for it in a.load(seed=7, limit=12)]
    two = [(it.item_id, it.gold_answer, tuple(it.meta["options"])) for it in a.load(seed=7, limit=12)]
    assert one == two
    other = [(it.item_id, it.gold_answer, tuple(it.meta["options"])) for it in a.load(seed=8, limit=12)]
    assert one != other, "a different seed must give a different order"


def test_truthfulqa_hand_scored(tmp_path):
    rows = [{"question": "Q", "mc1_targets": {"choices": ["right", "wrong1", "wrong2"], "labels": [1, 0, 0]}}]
    a = TruthfulQAMCAdapter(load_spec("truthfulqa_mc"), _write(tmp_path, "t.jsonl", rows))
    (it,) = a.load()
    assert a.score(it, a.extract(f"Answer: {it.gold_answer}")).values["accuracy"] == 1.0
    wrong = next(letter for letter in it.meta["letters"] if letter != it.gold_answer)
    assert a.score(it, a.extract(f"Answer: {wrong}")).values["accuracy"] == 0.0
    # A letter inside the chain's alphabet but beyond this item's options is
    # a format violation; a letter beyond the alphabet (Z) never extracts.
    v = a.score(it, a.extract("Answer: F"))
    assert v.values["accuracy"] == 0.0 and v.values["format_violation"] is True
    z = a.score(it, a.extract("Answer: Z"))
    assert z.values["accuracy"] is None and z.values["extraction_failed"] is True
    s = a.score(it, a.extract("\x00\x01 𝕏" * 50))
    assert s.values["accuracy"] is None and s.values["extraction_failed"] is True


def test_truthfulqa_chance_level_is_the_mean_over_options():
    spec = load_spec("truthfulqa_mc")
    a = TruthfulQAMCAdapter(spec, _fixture("truthfulqa_mc"))
    ns = [it.meta["n_options"] for it in a.load()]
    assert min(ns) >= 2 and max(ns) <= 13
    # The spec pins the split-wide mean (0.226); the fixture's own mean is
    # within a plausible band of it, which is all a 20-row sample can say.
    assert 0.12 <= sum(1 / n for n in ns) / len(ns) <= 0.40
    assert spec.scoring.chance_level == pytest.approx(0.226)


# --------------------------------------------------------------------------- #
#  WinoGrande
# --------------------------------------------------------------------------- #
def test_winogrande_maps_answers_to_letters_and_scores(tmp_path):
    rows = [
        {"sentence": "The trophy did not fit in the suitcase because the _ was too big.",
         "option1": "trophy", "option2": "suitcase", "answer": "1"},
        {"sentence": "Sarah beat Maria so _ got the easier cases.",
         "option1": "Sarah", "option2": "Maria", "answer": "2"},
    ]
    a = WinoGrandeAdapter(load_spec("winogrande"), _write(tmp_path, "w.jsonl", rows))
    t, s2 = a.load()
    assert t.gold_answer == "A" and s2.gold_answer == "B"
    assert "A. trophy" in a.prompt(t).messages[-1]["content"]
    assert a.score(t, a.extract("The trophy.\nAnswer: A")).values["accuracy"] == 1.0
    assert a.score(s2, a.extract("Answer: A")).values["accuracy"] == 0.0
    assert a.score(s2, a.extract("**B**")).values["accuracy"] == 1.0
    # The chain is capped at B, so a third letter cannot be extracted at all:
    # it is reported as an extraction failure, never scored as wrong.
    v = a.score(s2, a.extract("Answer: C"))
    assert v.values["accuracy"] is None and v.values["extraction_failed"] is True


def test_winogrande_fixture_loads_deterministically():
    a = WinoGrandeAdapter(load_spec("winogrande"), _fixture("winogrande"))
    assert [i.item_id for i in a.load(seed=3, limit=10)] == [i.item_id for i in a.load(seed=3, limit=10)]
    assert len(a.load()) == 20


# --------------------------------------------------------------------------- #
#  MGSM
# --------------------------------------------------------------------------- #
def test_native_digits_normalise_to_ascii():
    assert normalise_digits("উত্তর: ১৮") == "উত্তর: 18"       # Bengali
    assert normalise_digits("సమాధానం ౪౨") == "సమాధానం 42"     # Telugu
    assert normalise_digits("plain 18") == "plain 18"


@pytest.mark.parametrize("bid", ["mgsm_bn", "mgsm_te"])
def test_mgsm_fixture_loads_with_numeric_gold(bid):
    a = MGSMAdapter(load_spec(bid), _fixture(bid))
    items = a.load()
    assert len(items) == 20
    assert all(it.gold_answer.lstrip("-").isdigit() for it in items)
    assert all(it.meta["language"] == bid.split("_")[1] for it in items)


def test_mgsm_hand_scored_including_native_digit_answers(tmp_path):
    rows = [{"question": "প্রশ্ন", "answer_number": 18, "answer": None},
            {"question": "ప్రశ్న", "answer_number": 42, "answer": None}]
    a = MGSMAdapter(load_spec("mgsm_bn"), _write(tmp_path, "m.jsonl", rows))
    q18, q42 = a.load()
    assert a.score(q18, a.extract("... #### 18"), "... #### 18").values["accuracy"] == 1.0
    assert a.score(q18, a.extract("... #### ১৮"), "").values["accuracy"] == 1.0, "Bengali digits"
    assert a.score(q42, a.extract("సమాధానం #### ౪౨"), "").values["accuracy"] == 1.0, "Telugu digits"
    assert a.score(q42, a.extract("#### 41"), "").values["accuracy"] == 0.0
    assert a.score(q42, a.extract("The answer is $42.00"), "").values["accuracy"] == 1.0  # unexpected but valid
    s = a.score(q42, a.extract("no numbers here"), "no numbers here")
    assert s.values["accuracy"] is None and s.values["extraction_failed"] is True
    assert a.score(q42, a.extract("I cannot solve this"), "I cannot solve this").values["refused"] is True
