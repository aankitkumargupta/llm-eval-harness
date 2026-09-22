"""
Your own benchmark, declared in YAML rather than programmed (I11).

The claim under test is narrow and load-bearing: *someone who has their own
labelled data can turn it into a benchmark with a spec file and nothing else*,
and that benchmark behaves exactly like a shipped one, same extraction chain,
same failure accounting (I7), same determinism (I10), same pairing (I1).

The seam test at the bottom is the real proof: it writes a benchmark that has
never existed inside the test file and runs it end to end with zero edits to
the library.
"""

from __future__ import annotations

import json

import pytest

from harness.bench import registry
from harness.bench.adapters.custom import CustomAdapter, CustomBenchmarkError
from harness.bench.contracts import BenchmarkAdapter, Failed
from harness.bench.spec import BenchmarkSpec, load_spec

FIXTURE = "tests/bench/fixtures/custom_demo/spec.yaml"


# --------------------------------------------------------------------------- #
#  Helpers: build a benchmark from nothing, the way a user would
# --------------------------------------------------------------------------- #
def make_benchmark(tmp_path, rows, **spec_overrides):
    """Write a JSONL and a spec, return the built adapter. No library edits."""
    data = tmp_path / "items.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    raw = {
        "id": spec_overrides.pop("id", "my_own_set"),
        "version": 1,
        "family": spec_overrides.pop("family", "multiple_choice"),
        "task": "direct",
        "adapter": "custom",
        "source": {"kind": "local", "ref": str(data), "licence": "in-house"},
        "sampling": {"seed": 1729},
        "prompt": {"decoding": {"temperature": 0.0, "max_tokens": 512}},
        "scoring": {
            "mode": "generative",
            "extraction": {"chain": spec_overrides.pop("chain", [
                {"kind": "regex", "pattern": r"Answer:\s*\(?([A-J])\)?"},
                {"kind": "last_capital_letter"},
            ])},
            "chance_level": spec_overrides.pop("chance_level", 0.25),
        },
    }
    raw.update(spec_overrides)
    spec = BenchmarkSpec.from_dict(raw)
    return registry.build(spec)


MC_ROWS = [
    {"id": "a", "question": "Capital of France?",
     "options": ["Berlin", "Paris", "Madrid", "Rome"], "answer": "B"},
    {"id": "b", "question": "Two plus two?",
     "options": ["3", "4", "5", "6"], "answer": 2},
    {"id": "c", "question": "Largest ocean?",
     "options": ["Atlantic", "Indian", "Pacific", "Arctic"], "answer": "Pacific"},
]


# --------------------------------------------------------------------------- #
#  The fixture benchmark is a benchmark
# --------------------------------------------------------------------------- #
def test_the_fixture_spec_builds_the_generic_adapter():
    spec = load_spec(FIXTURE)
    adapter = registry.build(spec)
    assert isinstance(adapter, CustomAdapter)
    assert isinstance(adapter, BenchmarkAdapter)
    assert spec.adapter == "custom"


def test_the_spec_and_the_registry_agree_on_which_adapters_exist():
    """`spec.py` validates `adapter:` without importing an adapter, so the two
    lists are written twice; if they drift, a valid spec becomes unbuildable."""
    from harness.bench.spec import _VALID_ADAPTERS

    assert sorted(_VALID_ADAPTERS) == registry.generic_known()


# --------------------------------------------------------------------------- #
#  Answers written three different ways all resolve to the same letter
# --------------------------------------------------------------------------- #
def test_letter_index_and_option_text_all_resolve(tmp_path):
    items = {it.item_id: it.gold_answer for it in make_benchmark(tmp_path, MC_ROWS).load()}
    assert items == {"a": "B", "b": "B", "c": "C"}, (
        "a private set writes its answer as a letter, a 1-based number or the "
        "option text; all three must land on the same letter")


def test_an_answer_that_names_no_option_is_refused(tmp_path):
    rows = [{"id": "x", "question": "Q?", "options": ["one", "two"], "answer": "three"}]
    with pytest.raises(CustomBenchmarkError, match="not one of the options"):
        make_benchmark(tmp_path, rows).load()


# --------------------------------------------------------------------------- #
#  Hand-computed scores
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("raw", "expected"), [
    ("Answer: B", 1.0),                                   # the declared format
    ("The capital is Paris.\n\nAnswer: B", 1.0),          # reasoning first
    ("Answer: (B)", 1.0),                                 # parenthesised
    ("Answer: A", 0.0),                                   # wrong, readable
    ("B", 1.0),                                           # bare letter, 2nd link
])
def test_hand_computed_scores(tmp_path, raw, expected):
    adapter = make_benchmark(tmp_path, MC_ROWS)
    item = [it for it in adapter.load() if it.item_id == "a"][0]
    assert adapter.score(item, adapter.extract(raw)).get("accuracy") == expected


def test_a_correct_answer_in_an_unexpected_format_is_not_scored_wrong(tmp_path):
    """'I'd go with B' has no 'Answer:' line. The fallback link reads it, and
    the row records which link fired, so nobody mistakes a guess for a match."""
    adapter = make_benchmark(tmp_path, MC_ROWS)
    item = [it for it in adapter.load() if it.item_id == "a"][0]
    s = adapter.score(item, adapter.extract("I'd go with B here."))
    assert s.get("accuracy") == 1.0
    assert s.get("extracted_via") == "last_capital_letter"


def test_an_unreadable_answer_is_a_failure_not_a_zero(tmp_path):
    """I7. The whole point of the generic path is that it inherits this."""
    adapter = make_benchmark(tmp_path, MC_ROWS)
    item = adapter.load()[0]
    s = adapter.score(item, Failed("no extractor matched"))
    assert s.get("accuracy") is None
    assert s.get("extraction_failed") is True
    assert "accuracy" not in s.applicable()


def test_an_off_menu_letter_is_a_format_violation_not_just_a_miss(tmp_path):
    """"Answer: F" on a four-option item is readable and wrong, which is not
    the same as unreadable: the report separates the two."""
    adapter = make_benchmark(tmp_path, MC_ROWS)
    item = [it for it in adapter.load() if it.item_id == "a"][0]
    s = adapter.score(item, adapter.extract("Answer: F"))
    assert s.get("accuracy") == 0.0
    assert s.get("format_violation") is True
    assert s.get("extraction_failed") is False


def test_a_letter_outside_the_extraction_alphabet_is_a_failure(tmp_path):
    """The chain looks for A-J. "Z" matches no link, so the row is an
    extraction failure, not a wrong answer (I7)."""
    adapter = make_benchmark(tmp_path, MC_ROWS)
    item = [it for it in adapter.load() if it.item_id == "a"][0]
    s = adapter.score(item, adapter.extract("Answer: Z"))
    assert s.get("accuracy") is None
    assert s.get("extraction_failed") is True


# --------------------------------------------------------------------------- #
#  The other three shapes
# --------------------------------------------------------------------------- #
def test_math_scores_by_numeric_equivalence(tmp_path):
    """1,000 and 1000 and 1000.0 are one answer; formatting must not decide
    correctness, which is the single biggest source of fake maths scores."""
    rows = [{"id": "m1", "question": "900 + 100?", "answer": "1000"}]
    adapter = make_benchmark(tmp_path, rows, family="math", chance_level=0.0,
                             chain=[{"kind": "numeric"}])
    item = adapter.load()[0]
    for raw in ("The total is 1,000.", "Answer: 1000", "= 1000.0"):
        assert adapter.score(item, adapter.extract(raw)).get("accuracy") == 1.0
    assert adapter.score(item, adapter.extract("about 999")).get("accuracy") == 0.0


def test_short_answer_normalises_case_and_punctuation_but_nothing_more(tmp_path):
    rows = [{"id": "s1", "question": "Who wrote it?", "answer": "Mahatma Gandhi"}]
    adapter = make_benchmark(tmp_path, rows, family="short_answer",
                             chance_level=0.0, chain=[{"kind": "first_line"}])
    item = adapter.load()[0]
    assert adapter.score(item, adapter.extract("mahatma gandhi.")).get("accuracy") == 1.0
    # Deliberately NOT a fuzzy match: a near miss counted as a hit is the same
    # lie as a parse failure counted as a miss, pointing the other way.
    assert adapter.score(item, adapter.extract("Gandhi")).get("accuracy") == 0.0


def test_classify_needs_its_label_set_and_enforces_it(tmp_path):
    rows = [{"id": "t1", "question": "App crashes on login", "answer": "technical"}]
    with pytest.raises(CustomBenchmarkError, match="label_set"):
        make_benchmark(tmp_path, rows, family="classify", chance_level=0.33,
                       chain=[{"kind": "label_set", "labels": ["billing", "technical"]}])

    adapter = make_benchmark(
        tmp_path, rows, family="classify", chance_level=0.33,
        label_set=["billing", "technical", "other"],
        chain=[{"kind": "label_set", "labels": ["billing", "technical", "other"]}])
    item = adapter.load()[0]
    assert adapter.score(item, adapter.extract("technical")).get("accuracy") == 1.0
    assert adapter.score(item, adapter.extract("billing")).get("accuracy") == 0.0


def test_a_gold_label_outside_the_set_is_refused(tmp_path):
    rows = [{"id": "t1", "question": "x", "answer": "urgent"}]
    with pytest.raises(CustomBenchmarkError, match="not in label_set"):
        make_benchmark(tmp_path, rows, family="classify", chance_level=0.5,
                       label_set=["billing", "technical"],
                       chain=[{"kind": "verbatim"}]).load()


def test_a_family_the_generic_reader_cannot_judge_is_refused(tmp_path):
    """Code, retrieval and agentic sets need their own adapter. Refusing is
    the honest answer; guessing the shape would score the wrong thing."""
    rows = [{"id": "x", "question": "q", "answer": "a"}]
    with pytest.raises(CustomBenchmarkError, match="needs its own"):
        make_benchmark(tmp_path, rows, family="code", chance_level=0.0,
                       chain=[{"kind": "verbatim"}])


# --------------------------------------------------------------------------- #
#  The data file's own mistakes are loud, never silent
# --------------------------------------------------------------------------- #
def test_duplicate_ids_raise_rather_than_collapse(tmp_path):
    rows = MC_ROWS + [dict(MC_ROWS[0])]
    with pytest.raises(CustomBenchmarkError, match="duplicate item ids"):
        make_benchmark(tmp_path, rows).load()


def test_a_row_with_no_answer_raises_rather_than_being_dropped(tmp_path):
    """Dropping it would make the item set depend on the data's mistakes
    instead of on (seed, contents), which breaks pairing the moment two models
    load it separately."""
    rows = [{"id": "a", "question": "Q?", "options": ["x", "y"]}]
    with pytest.raises(CustomBenchmarkError, match="no 'answer'"):
        make_benchmark(tmp_path, rows).load()


def test_a_broken_line_names_the_line(tmp_path):
    data = tmp_path / "items.jsonl"
    data.write_text('{"id": "a", "question": "Q?", "options": ["x","y"], "answer": "A"}\n'
                    "{not json}\n", encoding="utf-8")
    spec = BenchmarkSpec.from_dict({
        "id": "broken", "family": "multiple_choice", "adapter": "custom",
        "source": {"kind": "local", "ref": str(data), "licence": "in-house"},
        "prompt": {"decoding": {"max_tokens": 64}},
        "scoring": {"chance_level": 0.5, "extraction": {"chain": [{"kind": "verbatim"}]}},
    })
    with pytest.raises(CustomBenchmarkError, match="line 2 is not valid JSON"):
        registry.build(spec).load()


def test_a_missing_file_says_so_plainly(tmp_path):
    spec = BenchmarkSpec.from_dict({
        "id": "absent", "family": "short_answer", "adapter": "custom",
        "source": {"kind": "local", "ref": str(tmp_path / "nope.jsonl"), "licence": "in-house"},
        "prompt": {"decoding": {"max_tokens": 64}},
        "scoring": {"extraction": {"chain": [{"kind": "verbatim"}]}},
    })
    with pytest.raises(CustomBenchmarkError, match="not found"):
        registry.build(spec).load()


# --------------------------------------------------------------------------- #
#  Field renaming, so an existing file need not be rewritten
# --------------------------------------------------------------------------- #
def test_existing_column_names_can_be_mapped_instead_of_renamed(tmp_path):
    rows = [{"qid": "r1", "prompt_text": "Capital of France?",
             "choices": ["Berlin", "Paris"], "label": "B"}]
    adapter = make_benchmark(tmp_path, rows, chance_level=0.5, fields={
        "id": "qid", "question": "prompt_text", "options": "choices", "answer": "label"})
    item = adapter.load()[0]
    assert (item.item_id, item.query[:7], item.gold_answer) == ("r1", "Capital", "B")


def test_fields_without_an_adapter_is_rejected_at_load():
    """A `fields:` block with no `adapter:` renames columns nobody reads; the
    spec would look configured and behave as though it were not."""
    from harness.bench.spec import SpecError

    with pytest.raises(SpecError, match="fields"):
        BenchmarkSpec.from_dict({
            "id": "x", "family": "short_answer",
            "fields": {"question": "q"},
            "source": {"kind": "local", "ref": "x.jsonl", "licence": "in-house"},
            "prompt": {"decoding": {"max_tokens": 64}},
            "scoring": {"extraction": {"chain": [{"kind": "verbatim"}]}},
        })


def test_an_unknown_adapter_name_is_rejected_at_load():
    from harness.bench.spec import SpecError

    with pytest.raises(SpecError, match="`adapter` must be one of"):
        BenchmarkSpec.from_dict({
            "id": "x", "family": "short_answer", "adapter": "magic",
            "source": {"kind": "local", "ref": "x.jsonl", "licence": "in-house"},
            "prompt": {"decoding": {"max_tokens": 64}},
            "scoring": {"extraction": {"chain": [{"kind": "verbatim"}]}},
        })


# --------------------------------------------------------------------------- #
#  Determinism and comparability
# --------------------------------------------------------------------------- #
def test_the_item_set_is_a_pure_function_of_seed_and_file(tmp_path):
    rows = [{"id": f"i{n:03d}", "question": f"Q{n}?", "options": ["a", "b"],
             "answer": "A"} for n in range(30)]
    a = make_benchmark(tmp_path, rows)
    first = [it.item_id for it in a.load(seed=11, limit=8)]
    second = [it.item_id for it in a.load(seed=11, limit=8)]
    assert first == second
    assert first == sorted(first), "items must be ordered, not just selected"


def test_declaring_a_generic_adapter_changes_the_spec_hash(tmp_path):
    """Two specs that read the same file differently are not comparable, so
    the hash the reporter refuses to compare across must reflect it."""
    base = {
        "id": "h", "family": "short_answer",
        "source": {"kind": "local", "ref": "x.jsonl", "licence": "in-house"},
        "prompt": {"decoding": {"max_tokens": 64}},
        "scoring": {"extraction": {"chain": [{"kind": "verbatim"}]}},
    }
    plain = BenchmarkSpec.from_dict(dict(base, id="h1"))
    with_adapter = BenchmarkSpec.from_dict(dict(base, id="h1", adapter="custom"))
    renamed = BenchmarkSpec.from_dict(
        dict(base, id="h1", adapter="custom", fields={"question": "q"}))
    assert plain.spec_hash() != with_adapter.spec_hash()
    assert with_adapter.spec_hash() != renamed.spec_hash()


def test_shipped_spec_hashes_did_not_move():
    """The generic-adapter fields join the hash only when used.

    If this fails, every benchmark number recorded before the change became
    incomparable with every number recorded after it, silently. The values are
    golden on purpose: that is the alarm.
    """
    assert load_spec("gsm8k").spec_hash() == (
        "c022692c184163bef80c524291c95718f3eeb110cd0ebf6cdb81df3ff5052a6b")
    assert load_spec("mmlu_pro").spec_hash() == (
        "06a045e048a211397c09d8e9d9d0308953415c37cded0b2de643ca2bee7e60f0")
    assert load_spec("arc_easy").spec_hash() == (
        "46a98d42f2fd705ae143ba9c59e2d801bb82e6c72cfd2ce553c09e9348a19b80")


# --------------------------------------------------------------------------- #
#  The seam (§5): a brand-new benchmark, defined here, runs end to end
# --------------------------------------------------------------------------- #
def test_a_benchmark_invented_in_this_test_file_participates_fully(tmp_path):
    """No library edit, no registry line, no adapter module: a JSONL and a
    spec. If this ever needs a code change to pass, the seam has moved."""
    rows = [
        {"id": "z1", "question": "Which body issues a birth certificate?",
         "options": ["Municipal Corporation", "High Court", "Reserve Bank"],
         "answer": "A"},
        {"id": "z2", "question": "Which body hears a second RTI appeal?",
         "options": ["Gram Panchayat", "State Information Commission", "Police"],
         "answer": "B"},
    ]
    adapter = make_benchmark(tmp_path, rows, id="brand_new_set", chance_level=1 / 3)

    items = adapter.load()
    assert [it.item_id for it in items] == ["z1", "z2"]

    prompted = adapter.prompt(items[0], ())
    messages = prompted.as_messages()
    assert messages[-1]["role"] == "user"
    assert "Municipal Corporation" in messages[-1]["content"]
    assert prompted.choices == ("A", "B", "C")

    scored = [adapter.score(it, adapter.extract(f"Answer: {it.gold_answer}"))
              for it in items]
    assert [s.get("accuracy") for s in scored] == [1.0, 1.0]
    assert all(s.get("extraction_failed") is False for s in scored)
