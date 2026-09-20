"""
BoolQ adapter — hand-computed scores, extraction edges, determinism.

Everything here runs against the committed 20-row fixture under
`tests/bench/fixtures/boolq/` (CC-BY-SA-3.0; see ATTRIBUTION.txt beside it), so
nothing touches the network. The adapter is built directly rather than through
the registry so this file passes before and after registration.

The fixture is balanced on purpose, ten "yes" and ten "no": a model that
always answers "yes" then lands exactly on the 0.5 chance level, which turns
the chance-adjusted column into a hand-checkable zero.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from harness.bench.adapters.boolq import LABELS, BoolQAdapter
from harness.bench.contracts import (
    BenchmarkAdapter,
    Extraction,
    Failed,
    Ok,
    Prompted,
)
from harness.bench.metrics import chance_adjusted, summarise
from harness.bench.spec import load_spec
from harness.clients.base import GenResult

FIXTURE = "tests/bench/fixtures/boolq/boolq_validation_20.jsonl"
PLACEHOLDER = "sha256:" + "0" * 64


@pytest.fixture(scope="module")
def spec():
    return load_spec("boolq")


@pytest.fixture
def adapter(spec):
    return BoolQAdapter(spec, data_path=FIXTURE)


@pytest.fixture
def items(adapter) -> dict:
    return {it.item_id: it for it in adapter.load()}


def _score(adapter, item, raw: str):
    return adapter.score(item, adapter.extract(raw))


# --------------------------------------------------------------------------- #
#  Spec
# --------------------------------------------------------------------------- #
def test_spec_is_valid_and_pins_the_format(spec):
    assert spec.validate() == []
    assert spec.id == "boolq" and spec.version == 1
    assert spec.family == "classify"
    assert spec.source.kind == "hf"
    assert spec.source.ref == "google/boolq"
    assert spec.source.split == "validation"
    assert spec.source.licence == "CC-BY-SA-3.0"
    assert spec.source.commercial_use is True
    assert spec.scoring.mode == "generative"
    assert spec.scoring.chance_level == pytest.approx(0.5)
    assert "max_tokens" in spec.prompt.decoding
    assert spec.prompt.few_shot.pool_split != spec.source.split


def test_spec_checksum_is_real_not_the_placeholder(spec):
    """A placeholder checksum would let any download through as 'the' dataset."""
    assert spec.source.checksum.startswith("sha256:")
    assert len(spec.source.checksum) == len(PLACEHOLDER)
    assert spec.source.checksum != PLACEHOLDER


def test_labels_are_strings_not_yaml_booleans(spec):
    """YAML 1.1 reads a bare `yes`/`no` as True/False. A boolean label makes
    the `label_set` link raise instead of match, which the runner classifies
    as an adapter bug and uses to stop the whole run."""
    assert spec.label_set == ("yes", "no")
    chain = list(spec.scoring.chain)
    assert [link["kind"] for link in chain] == ["regex", "label_set"]
    assert chain[1]["labels"] == ["yes", "no"]
    assert all(isinstance(lab, str) for lab in chain[1]["labels"])


# --------------------------------------------------------------------------- #
#  Shape and load
# --------------------------------------------------------------------------- #
def test_satisfies_the_adapter_protocol(adapter):
    assert isinstance(adapter, BenchmarkAdapter)
    assert adapter.id == "boolq"


def test_fixture_loads_twenty_balanced_items(adapter):
    items = adapter.load()
    assert len(items) == 20
    golds = [it.gold_answer for it in items]
    assert set(golds) <= set(LABELS)
    assert golds.count("yes") == 10 and golds.count("no") == 10
    assert all(it.meta.get("passage") for it in items)


def test_item_ids_are_unique_and_follow_file_order(adapter):
    ids = [it.item_id for it in adapter.load()]
    assert ids == [f"boolq_{i:05d}" for i in range(20)]
    assert len(set(ids)) == 20


def test_known_items_carry_the_expected_gold(items):
    """Pinned against the fixture rows, so a re-generated fixture that
    changed order or labels is caught here rather than in a score."""
    expected = {
        "boolq_00000": ("does ethanol take more energy make that produces", "no"),
        "boolq_00001": ("is house tax and property tax are same", "yes"),
        "boolq_00005": ("is barq's root beer a pepsi product", "no"),
        "boolq_00006": ("can an odd number be divided by an even number", "yes"),
        "boolq_00019": ("are new balance and nike the same company", "no"),
    }
    for iid, (question, gold) in expected.items():
        assert items[iid].query == question
        assert items[iid].gold_answer == gold


def test_load_is_deterministic_for_the_same_seed_and_limit(adapter):
    """I10 / I1: the same seed and limit twice gives the same items in the
    same order, which is what makes a multi-model run paired."""
    a = [it.item_id for it in adapter.load(seed=7, limit=8)]
    b = [it.item_id for it in adapter.load(seed=7, limit=8)]
    assert a == b
    assert len(a) == 8
    assert a == sorted(a)


def test_spec_seed_is_the_default_seed(adapter, spec):
    explicit = [it.item_id for it in adapter.load(seed=spec.sampling.seed, limit=6)]
    default = [it.item_id for it in adapter.load(limit=6)]
    assert explicit == default


def test_limit_larger_than_the_fixture_returns_everything(adapter):
    assert len(adapter.load(limit=500)) == 20


def test_string_golds_are_normalised_and_junk_gold_raises(tmp_path, spec):
    """A locally prepared file may carry strings; anything unrecognised is a
    data error, because a mislabelled gold scores the model wrong and blames it."""
    ok = tmp_path / "ok.jsonl"
    rows = [{"question": "q1", "passage": "p", "answer": "true"},
            {"question": "q2", "passage": "p", "answer": "False"},
            {"question": "q3", "passage": "p", "answer": "YES"},
            {"question": "q4", "passage": "p", "answer": False}]
    ok.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    golds = [it.gold_answer for it in BoolQAdapter(spec, str(ok)).load()]
    assert golds == ["yes", "no", "yes", "no"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"question": "q", "passage": "p", "answer": "maybe"})
                   + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        BoolQAdapter(spec, str(bad)).load()


def test_a_row_with_a_missing_field_raises_rather_than_being_skipped(tmp_path, spec):
    """Skipping would change `n` silently and unpair the run against a
    complete copy of the same split."""
    p = tmp_path / "hole.jsonl"
    p.write_text(json.dumps({"question": "q", "passage": "p"}) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        BoolQAdapter(spec, str(p)).load()


def test_missing_data_file_names_the_fetch_command(spec):
    with pytest.raises(FileNotFoundError, match="bench fetch"):
        BoolQAdapter(spec, "does/not/exist.jsonl").load()


# --------------------------------------------------------------------------- #
#  Prompt
# --------------------------------------------------------------------------- #
def test_prompt_uses_the_lm_eval_layout(adapter, items):
    item = items["boolq_00000"]
    p = adapter.prompt(item, ())
    assert isinstance(p, Prompted)
    msgs = p.as_messages()
    assert msgs[0]["role"] == "system"
    assert "yes" in msgs[0]["content"].lower() and "no" in msgs[0]["content"].lower()
    user = msgs[-1]
    assert user["role"] == "user"
    assert user["content"].startswith(item.meta["passage"])
    assert f"\nQuestion: {item.query}?\n" in user["content"]
    assert user["content"].endswith("Answer:")
    assert p.choices == ("yes", "no")


def test_prompt_is_pure(adapter, items):
    item = items["boolq_00003"]
    assert adapter.prompt(item, ()) == adapter.prompt(item, ())


def test_prompt_does_not_double_a_question_mark(adapter, items):
    item = items["boolq_00001"]
    already = type(item)(item_id="x", query=item.query + "?",
                         gold_answer=item.gold_answer, meta=item.meta)
    text = adapter.prompt(already, ()).as_messages()[-1]["content"]
    assert "??" not in text
    assert f"Question: {item.query}?\n" in text


def test_few_shot_turns_answer_with_the_shot_gold(adapter, items):
    shot = items["boolq_00005"]
    item = items["boolq_00006"]
    msgs = adapter.prompt(item, (shot,)).as_messages()
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "no"
    assert shot.meta["passage"] in msgs[1]["content"]
    assert item.meta["passage"] in msgs[3]["content"]


# --------------------------------------------------------------------------- #
#  Hand-computed scores for five items
# --------------------------------------------------------------------------- #
HAND_COMPUTED = [
    # item_id,       model output,                                         accuracy
    ("boolq_00000", "No. Corn ethanol returns 1.3 units per unit invested.", 1.0),
    ("boolq_00001", "Yes, the passage treats the two terms as the same.",    1.0),
    ("boolq_00005", "Yes, it is a Pepsi product.",                          0.0),
    ("boolq_00006", "No, an odd number cannot be divided by an even one.",  0.0),
    ("boolq_00019", "Maybe.",                                               None),
]


@pytest.mark.parametrize("iid,raw,expected", HAND_COMPUTED)
def test_hand_computed_item_scores(adapter, items, iid, raw, expected):
    s = _score(adapter, items[iid], raw)
    assert s.get("accuracy") == expected
    if expected is None:
        assert s.get("extraction_failed") is True
        assert "accuracy" not in s.applicable()
    else:
        assert s.get("extraction_failed") is False
        assert s.get("format_violation") is False
        assert s.get("extracted_via") == "regex"


def test_hand_computed_aggregate_over_the_five(adapter, items):
    """Two right, two wrong, one unreadable: accuracy 2/4 = 0.5 over the four
    scoreable items, extraction failure 1/5. The unreadable one is in neither
    the numerator nor the denominator (I7)."""
    scores = [_score(adapter, items[iid], raw) for iid, raw, _ in HAND_COMPUTED]
    scored = [s["accuracy"] for s in scores if s.get("accuracy") is not None]
    assert len(scored) == 4
    assert sum(scored) / len(scored) == pytest.approx(0.5)
    assert sum(1 for s in scores if s.get("extraction_failed")) == 1


def test_an_always_yes_model_scores_exactly_chance(adapter):
    """Ten yes, ten no: always-yes is 10/20 = 0.5 raw, which the chance level
    of 0.5 adjusts to exactly 0.0. The raw number looks like 'half right';
    the adjusted one says what it is."""
    accs = [_score(adapter, it, "Yes.")["accuracy"] for it in adapter.load()]
    assert None not in accs
    raw = sum(accs) / len(accs)
    assert raw == pytest.approx(0.5)
    assert chance_adjusted(raw, 0.5) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
#  Extraction: leading answer, fallback, failure, hazards, adversarial
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,value", [
    ("Yes.", "yes"),
    ("No, the passage says otherwise.", "no"),
    ("**No.** Because the article states it.", "no"),
    ("Answer: Yes", "yes"),
    ("**Answer:** no", "no"),
    ("Answer: **Yes** since the passage says so.", "yes"),
    ("YES!", "yes"),
    ("  no\n\nThe passage explains why.", "no"),
    ("no, but some sources say yes", "no"),
])
def test_leading_answer_is_taken_by_the_regex_link(adapter, raw, value):
    got = adapter.extract(raw)
    assert got.ok and got.via == "regex"
    assert got.value.lower() == value


@pytest.mark.parametrize("raw,value", [
    ("The answer is no.", "no"),
    ("Based on the passage, the answer is yes.", "yes"),
    ("I would say yes here.", "yes"),
])
def test_a_label_anywhere_falls_to_the_label_set_link(adapter, raw, value):
    got = adapter.extract(raw)
    assert got.ok and got.via == "label_set"
    assert got.value == value


@pytest.mark.parametrize("raw", ["", "   ", "\n\n", "Maybe.", "Unclear from the text.",
                                 "42", "It depends.", "Answer: ZZZZ"])
def test_no_label_is_an_extraction_failure_not_a_zero(adapter, items, raw):
    got = adapter.extract(raw)
    assert got.failed
    assert got.reason
    s = adapter.score(items["boolq_00002"], got)
    assert s.get("accuracy") is None
    assert s.get("extraction_failed") is True
    assert s.get("extraction_reason") == got.reason
    assert "accuracy" not in s.applicable()


def test_known_substring_hazard_is_pinned_not_hidden(adapter):
    """`label_set` matches substrings and prefers the longer label, so "not"
    reads as `no`, "eyes" as `yes`, and "not ... yes" as `yes`. This is the
    chain the spec declares; the test records the behaviour so a change to
    it is a visible diff, and `via` shows which link produced the value."""
    hazards = {
        "The passage does not say.": "no",
        "I do not know.": "no",
        "It has eyes.": "yes",
        "Not really, but yes in part.": "yes",
    }
    for raw, value in hazards.items():
        got = adapter.extract(raw)
        assert got.ok and got.value == value and got.via == "label_set", raw


def test_adversarial_strings_never_raise_and_scores_stay_bounded(adapter, items):
    """Model output is untrusted data (§6). A crash here would be recorded
    as an adapter bug and stop the run; a value outside [0, 1] would corrupt
    every mean built on it."""
    item = items["boolq_00004"]
    for raw in ("\x00\x01\x02", "𝕏" * 500, "```" * 100, "<script>alert(1)</script>",
                "-" * 5000, "*" * 3000 + "yes", "yes" * 1000, "\n" * 50,
                "Answer: " * 200, "{\"answer\": true}"):
        got = adapter.extract(raw)
        assert isinstance(got, Extraction)
        s = adapter.score(item, got)
        for v in s.values.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                assert 0.0 <= float(v) <= 1.0


# --------------------------------------------------------------------------- #
#  Unexpected-but-valid formats that are still correct
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("iid,raw", [
    ("boolq_00011", "**No.** The show is fictional according to the passage."),
    ("boolq_00007", "Answer: Yes"),
    ("boolq_00008", "YES!"),
    ("boolq_00013", "The answer is no."),
    ("boolq_00009", "yes\n\nThe passage mentions a third-place play-off."),
    ("boolq_00014", "(No) - the passage says modern cars need no break-in."),
])
def test_correct_answers_in_unexpected_formats_score_one(adapter, items, iid, raw):
    s = _score(adapter, items[iid], raw)
    assert s["accuracy"] == 1.0
    assert s["format_violation"] is False
    assert s["extraction_failed"] is False
    assert s["extracted_via"] in ("regex", "label_set")


# --------------------------------------------------------------------------- #
#  Score: I7 and format violation
# --------------------------------------------------------------------------- #
def test_extraction_failure_never_becomes_a_zero(adapter, items):
    s = adapter.score(items["boolq_00010"], Failed("no extractor matched"))
    assert s.get("accuracy") is None
    assert s.get("extraction_failed") is True
    assert s.get("format_violation") is None


def test_an_answer_outside_the_label_alphabet_is_a_format_violation(adapter, items):
    """Reported beside the score, never inside it: the item stays in the
    denominator as the other adapters' off-alphabet letters do, and the
    flag is what the report turns into `format_violation_rate`."""
    s = adapter.score(items["boolq_00010"], Ok("maybe", via="regex"))
    assert s["accuracy"] == 0.0
    assert s["format_violation"] is True
    assert s["extraction_failed"] is False


def test_wrapping_punctuation_is_not_a_format_violation(adapter, items):
    item = items["boolq_00012"]             # gold: yes
    for value in ("Yes.", "**yes**", "YES!", "(yes)", "'yes'"):
        s = adapter.score(item, Ok(value, via="regex"))
        assert s["accuracy"] == 1.0 and s["format_violation"] is False, value


# --------------------------------------------------------------------------- #
#  Through the real run loop, with a scripted client
# --------------------------------------------------------------------------- #
class _ScriptedClient:
    """Answers from the question text alone; no network, no clock."""

    def __init__(self, reply):
        self._reply = reply

    def generate(self, model, messages, **kw):
        text = self._reply(messages[-1]["content"])
        return GenResult(text=text, prompt_tokens=50, completion_tokens=3,
                         latency_ms=1.0, finish_reason="stop", model=model)


def test_scoreset_keys_land_on_tracerow_columns_via_the_runner(adapter, spec):
    """`run_benchmark` raises on a ScoreSet key with no TraceRow column, so
    driving the real loop proves every key this adapter emits is stored.
    The oracle answers from the passage-free question, one wrong on purpose."""
    from harness.bench.runner import run_benchmark

    gold = {it.query: it.gold_answer for it in adapter.load()}

    def oracle(user_text: str) -> str:
        q = user_text.split("\nQuestion: ")[-1].split("?\nAnswer:")[0]
        if q.startswith("are new balance"):
            return "Maybe."                     # one extraction failure
        if q.startswith("is barq"):
            return "Yes, it is."                # one wrong answer
        return gold[q].capitalize() + "."

    rep = run_benchmark(adapter=adapter, spec=spec, models=["scripted"],
                        client=_ScriptedClient(oracle), run_id="t_boolq")
    assert rep.errors == 0 and not rep.aborted
    assert len(rep.rows) == 20
    df = pd.DataFrame([r.to_dict() for r in rep.rows])
    assert set(df["benchmark"]) == {"boolq"}
    assert df["spec_hash"].nunique() == 1

    summary = summarise(df, "boolq", chance_level=spec.scoring.chance_level)
    row = summary.iloc[0]
    assert row["n_items"] == 20 and row["n_scored"] == 19
    assert row["accuracy"] == pytest.approx(18 / 19)
    assert row["accuracy_chance_adjusted"] == pytest.approx((18 / 19 - 0.5) / 0.5)
    assert row["extraction_failure_rate"] == pytest.approx(1 / 20)
    assert row["format_violation_rate"] == pytest.approx(0.0)
