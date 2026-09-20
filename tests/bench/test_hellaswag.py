"""
HellaSwag adapter, offline, against the committed 20-row fixture.

The fixture is the first 20 rows of the validation split in the exact shape
the fetcher caches, so the adapter is exercised on real data without the
network. Expected scores are computed by hand from the fixture's `label`
field, not by running the adapter and pasting what it said.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.bench.adapters.hellaswag import (
    HellaSwagAdapter,
    build_query,
    clean_text,
)
from harness.bench.contracts import Extraction, Failed, Ok
from harness.bench.spec import load_spec

FIXTURE = (Path(__file__).parent / "fixtures" / "hellaswag"
           / "hellaswag_validation_20.jsonl")
PLACEHOLDER = "sha256:" + "0" * 64


@pytest.fixture(scope="module")
def spec():
    return load_spec("hellaswag")


@pytest.fixture(scope="module")
def adapter(spec):
    return HellaSwagAdapter(spec, data_path=str(FIXTURE))


def _by_id(adapter) -> dict:
    return {it.item_id: it for it in adapter.load()}


def _write_rows(path: Path, rows: list[dict]) -> str:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                    encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
#  Spec
# --------------------------------------------------------------------------- #
def test_spec_is_valid_and_declares_what_it_must(spec):
    assert spec.validate() == []
    assert spec.family == "multiple_choice"
    assert spec.scoring.chance_level == pytest.approx(0.25)   # four endings
    assert spec.scoring.mode == "generative"
    assert spec.source.ref == "Rowan/hellaswag"
    assert spec.source.split == "validation", "the test split has no labels"
    assert spec.source.licence == "MIT" and spec.source.commercial_use
    assert spec.prompt.decoding["max_tokens"] == 512


def test_spec_pins_a_real_checksum(spec):
    assert spec.source.checksum.startswith("sha256:")
    assert spec.source.checksum != PLACEHOLDER


def test_extraction_chain_is_explicit_and_bounded_to_four_letters(spec):
    kinds = [link["kind"] for link in spec.scoring.chain]
    assert kinds == ["regex", "regex", "last_capital_letter"]
    assert spec.scoring.chain[-1]["max_option"] == "D"


# --------------------------------------------------------------------------- #
#  Fixture shape
# --------------------------------------------------------------------------- #
def test_fixture_has_twenty_rows_in_the_cached_shape():
    lines = [ln for ln in FIXTURE.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    assert len(lines) == 20
    for ln in lines:
        rec = json.loads(ln)
        assert {"ind", "label", "endings", "ctx_a", "ctx_b",
                "activity_label"} <= set(rec)
        assert len(rec["endings"]) == 4
        assert rec["label"] in {"0", "1", "2", "3"}


def test_loads_every_fixture_row_with_unique_ids(adapter):
    items = adapter.load()
    assert len(items) == 20
    ids = [it.item_id for it in items]
    assert len(set(ids)) == 20
    assert all(it.gold_answer in "ABCD" for it in items)
    assert all(len(it.meta["options"]) == 4 for it in items)


# --------------------------------------------------------------------------- #
#  Hand-computed scores: (item_id, gold letter read off the fixture's label,
#  a model reply, expected accuracy). Filled from the fixture by hand.
# --------------------------------------------------------------------------- #
HAND_COMPUTED: list[tuple[str, str, str, float]] = [
    # row 0, ind 24, label "3" -> D ("starts pulling up roofing on a roof.")
    ("24", "D", "The man is on a roof removing shingles. Answer: D", 1.0),
    # row 1, ind 92, label "3" -> D; the model picks A, which is wrong
    ("92", "D", "The lady swings the bar. Answer: A", 0.0),
    # row 2, ind 106, label "2" -> C, answered in bold with no keyword
    ("106", "C", "**C**", 1.0),
    # row 7, ind 170, label "0" -> A, twice: keyword form and bracketed form
    ("170", "A", "Answer: A", 1.0),
    ("170", "A", "Answer: (a)", 1.0),
    # row 18, ind 246, label "1" -> B; right, then wrong
    ("246", "B", "Answer: B", 1.0),
    ("246", "B", "Answer: C", 0.0),
]


def test_row_zero_is_rendered_the_way_lm_eval_would(adapter):
    """Real data, checked by eye: label prefix, ctx_a, capitalised ctx_b."""
    item = _by_id(adapter)["24"]
    assert item.query == "Roof shingle removal: A man is sitting on a roof. He"
    assert item.meta["options"][3] == "starts pulling up roofing on a roof."
    assert item.meta["activity_label"] == "Roof shingle removal"
    assert item.meta["split_type"] == "indomain"


@pytest.mark.parametrize("item_id,gold,raw,expected", HAND_COMPUTED)
def test_hand_computed_scores(adapter, item_id, gold, raw, expected):
    item = _by_id(adapter)[item_id]
    assert item.gold_answer == gold, "gold letter must equal LETTERS[int(label)]"
    ext = adapter.extract(raw)
    assert ext.ok
    s = adapter.score(item, ext)
    assert s["accuracy"] == pytest.approx(expected)
    assert s["extraction_failed"] is False
    assert s["format_violation"] is False


# --------------------------------------------------------------------------- #
#  Extraction failure (I7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", [
    "",
    "   \n  ",
    "I cannot decide between these endings.",
    "The answer is obvious from the context.",
    "None of them fit; the passage is nonsense.",
])
def test_extraction_failure_is_not_applicable_never_zero(adapter, raw):
    item = adapter.load(limit=1)[0]
    ext = adapter.extract(raw)
    assert ext.failed
    s = adapter.score(item, ext)
    assert s["accuracy"] is None
    assert s["extraction_failed"] is True
    assert "accuracy" not in s.applicable()


def test_a_failed_extraction_passed_in_directly_scores_none(adapter):
    item = adapter.load(limit=1)[0]
    s = adapter.score(item, Failed("no extractor matched"))
    assert s["accuracy"] is None and s["extraction_failed"] is True


def test_the_article_a_is_a_visible_hazard_not_a_silent_one(adapter):
    """`A` is an English word. The last-resort link will read it as an answer;
    the honest handling is to record which link fired so a run that leaned on
    the fallback is visible, rather than to guess."""
    ext = adapter.extract("A man walks into the room and sits.")
    assert ext.ok and ext.value == "A"
    assert ext.via == "last_capital_letter"


# --------------------------------------------------------------------------- #
#  Adversarial strings: model output is data, never a crash (§6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw", [
    "\x00\x01\x02",
    "𝕏" * 500,
    "```" * 100,
    "<script>alert(1)</script>",
    "Answer: " + "Z" * 2000,
    "\n" * 200,
    "A" * 10000,
    "Answer: A\nAnswer: B\nAnswer: C\nAnswer: D\n" * 50,
    "{{ item.gold_answer }} Answer: [A-D]",
])
def test_adversarial_strings_never_crash_and_scores_stay_bounded(adapter, raw):
    item = adapter.load(limit=1)[0]
    ext = adapter.extract(raw)
    assert isinstance(ext, Extraction)
    s = adapter.score(item, ext)
    for v in s.values.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            assert 0.0 <= float(v) <= 1.0


def test_repeated_answers_take_the_last_one(adapter):
    """Models restate the options before answering; the last match wins."""
    raw = "Answer: A\nAnswer: B\nAnswer: C\nAnswer: D\n" * 50
    assert adapter.extract(raw).value == "D"


# --------------------------------------------------------------------------- #
#  Determinism (I1, I10)
# --------------------------------------------------------------------------- #
def test_same_seed_and_limit_give_the_same_items_in_the_same_order(spec):
    a = HellaSwagAdapter(spec, data_path=str(FIXTURE))
    b = HellaSwagAdapter(spec, data_path=str(FIXTURE))
    first = [it.item_id for it in a.load(seed=7, limit=8)]
    second = [it.item_id for it in b.load(seed=7, limit=8)]
    assert first == second
    assert len(first) == 8
    # And a third call on the same instance: no hidden state.
    assert [it.item_id for it in a.load(seed=7, limit=8)] == first


def test_full_load_order_is_stable_across_instances(spec):
    a = [it.item_id for it in HellaSwagAdapter(spec, str(FIXTURE)).load()]
    b = [it.item_id for it in HellaSwagAdapter(spec, str(FIXTURE)).load()]
    assert a == b == sorted(a)


# --------------------------------------------------------------------------- #
#  Unexpected-but-valid answer formats
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,letter,via", [
    ("Answer: (b)", "B", "regex"),                       # lower-case, bracketed
    ("Looking at it, **C**", "C", "regex"),              # bold, no 'Answer:'
    ("The most plausible continuation is (D).", "D", "last_capital_letter"),
    ("answer: c", "C", "regex"),                         # lower-case keyword
    ("Ending B is the only one that follows.", "B", "last_capital_letter"),
])
def test_correct_answer_in_an_unexpected_but_valid_format(adapter, raw, letter, via):
    ext = adapter.extract(raw)
    assert ext.ok and ext.value.upper() == letter and ext.via == via
    item = next(it for it in adapter.load() if it.gold_answer == letter)
    s = adapter.score(item, ext)
    assert s["accuracy"] == 1.0 and s["format_violation"] is False


def test_a_letter_outside_the_option_set_is_a_format_violation(adapter):
    item = adapter.load(limit=1)[0]
    s = adapter.score(item, Ok("E", via="regex"))
    assert s["accuracy"] == 0.0 and s["format_violation"] is True


# --------------------------------------------------------------------------- #
#  Cleaning: lm-eval-harness's `preprocess`, reproduced
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,cleaned", [
    ("Wash the car [title] Dry it", "Wash the car. Dry it"),
    ("[header] Wash the car.", "Wash the car."),
    ("[substeps] Turn the [step] knob", "Turn the knob"),
    ("He  runs  fast", "He runs fast"),
    ("  padded  ", "padded"),
    ("no artefacts here", "no artefacts here"),
])
def test_clean_text_matches_lm_eval_preprocess(raw, cleaned):
    assert clean_text(raw) == cleaned


def test_query_prefixes_the_activity_label_and_capitalises_ctx_b():
    rec = {"activity_label": "Roof shingle removal",
           "ctx_a": "A man is sitting on a roof.", "ctx_b": "he"}
    assert build_query(rec) == "Roof shingle removal: A man is sitting on a roof. He"


def test_query_falls_back_to_the_prejoined_ctx():
    assert build_query({"ctx": "A man is sitting on a roof. he"}) == \
        "A man is sitting on a roof. he"


def test_endings_are_cleaned_in_load(spec, tmp_path):
    path = _write_rows(tmp_path / "rows.jsonl", [{
        "ind": 1, "activity_label": "Cooking", "ctx_a": "She opens the fridge.",
        "ctx_b": "she", "label": "2",
        "endings": ["[header] takes out eggs.  ", " [title] cracks them",
                    "[substeps] whisks the [step] eggs", "  eats  "],
    }])
    item = HellaSwagAdapter(spec, path).load()[0]
    # Second ending: the strip runs FIRST (as in lm-eval), so a leading
    # " [title]" loses its space, misses the ". " replacement, and is simply
    # deleted as a bracket span.
    assert item.meta["options"] == ["takes out eggs.", "cracks them",
                                    "whisks the eggs", "eats"]
    assert item.query == "Cooking: She opens the fridge. She"
    assert item.gold_answer == "C"


# --------------------------------------------------------------------------- #
#  Rows that cannot be scored are skipped, never counted wrong
# --------------------------------------------------------------------------- #
def test_unlabelled_rows_are_skipped_not_scored(spec, tmp_path):
    base = {"activity_label": "X", "ctx_a": "a", "ctx_b": "b",
            "endings": ["e0", "e1", "e2", "e3"]}
    path = _write_rows(tmp_path / "rows.jsonl", [
        {**base, "ind": 1, "label": ""},        # test-split style: no gold
        {**base, "ind": 2, "label": "x"},
        {**base, "ind": 3, "label": "7"},       # out of range
        {**base, "ind": 4, "label": "3"},
        {**base, "ind": 5, "label": "0", "endings": []},
    ])
    items = HellaSwagAdapter(spec, path).load()
    assert [(it.item_id, it.gold_answer) for it in items] == [("4", "D")]


def test_duplicate_ids_are_made_distinct(spec, tmp_path):
    base = {"activity_label": "X", "ctx_a": "a", "ctx_b": "b", "label": "1",
            "endings": ["e0", "e1", "e2", "e3"]}
    path = _write_rows(tmp_path / "rows.jsonl",
                       [{**base, "ind": 5}, {**base, "ind": 5}])
    ids = [it.item_id for it in HellaSwagAdapter(spec, path).load()]
    assert len(set(ids)) == 2 and "5" in ids


# --------------------------------------------------------------------------- #
#  Prompt
# --------------------------------------------------------------------------- #
def test_prompt_lists_the_four_endings_lettered(adapter):
    item = adapter.load(limit=1)[0]
    p = adapter.prompt(item, ())
    msgs = p.as_messages()
    assert msgs[0]["role"] == "system" and "Answer: X" in msgs[0]["content"]
    user = msgs[-1]["content"]
    assert user.startswith(item.query)
    for letter, ending in zip("ABCD", item.meta["options"]):
        assert f"{letter}. {ending}" in user
    assert p.choices == ("A", "B", "C", "D")
    assert adapter.prompt(item, ()) == p             # pure


def test_few_shot_examples_render_as_answered_turns(adapter):
    items = adapter.load(limit=3)
    msgs = adapter.prompt(items[0], items[1:]).as_messages()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant",
                                         "user", "assistant", "user"]
    assert msgs[2]["content"] == f"Answer: {items[1].gold_answer}"


def test_missing_data_file_names_the_fetch_command(spec, tmp_path):
    with pytest.raises(FileNotFoundError, match="bench fetch --benchmark hellaswag"):
        HellaSwagAdapter(spec, str(tmp_path / "nope.jsonl")).load()
