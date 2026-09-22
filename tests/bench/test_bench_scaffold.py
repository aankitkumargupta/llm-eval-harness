"""
The browser path for declaring your own benchmark.

`tests/bench/test_custom_benchmark.py` proves the generic adapter is honest.
This file proves the *screen* that writes specs for it produces something that
actually loads and runs: the spec it previews is the spec it writes, the rows
it accepts are the rows the adapter can read, and the two files it puts on disk
are a working benchmark with no third step.
"""

from __future__ import annotations

import json

import pytest

from harness.bench import registry
from harness.bench.spec import BenchmarkSpec, load_spec
from harness.web import bench_scaffold as bs
from harness.web.api import ApiError

SHAPES = ["multiple_choice", "short_answer", "math", "classify"]


# --------------------------------------------------------------------------- #
#  What the screen shows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_describes_itself(shape):
    d = bs.describe(shape)
    assert [s["id"] for s in d["shapes"]] == SHAPES
    assert d["samples"] and d["fields"] and d["notes"]
    names = {f["name"] for f in d["fields"]}
    assert {"question", "answer"} <= names


def test_an_unknown_shape_is_refused():
    with pytest.raises(ApiError, match="Unknown shape"):
        bs.describe("interpretive_dance")


@pytest.mark.parametrize("shape", SHAPES)
def test_the_sample_rows_it_shows_pass_the_check_it_applies(shape):
    """The samples are the format's own documentation. If they did not
    validate, every new user's first upload would fail for a reason that is
    the screen's fault, not theirs."""
    d = bs.describe(shape)
    labels = d["defaults"].get("labels")
    got = bs.validate_jsonl(shape, d["samples_jsonl"], labels=labels)
    assert got["ok"], got["problems"]
    assert got["rows"] == len(d["samples"])


# --------------------------------------------------------------------------- #
#  The spec it writes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shape", SHAPES)
def test_the_previewed_spec_is_valid_and_names_the_generic_adapter(shape):
    opts = {"labels": "a, b, c"} if shape == "classify" else {}
    got = bs.preview("my_own_set", shape, opts)
    assert got["errors"] == []
    assert got["spec_hash"]
    spec = BenchmarkSpec.from_dict(bs.spec_dict("my_own_set", shape, opts))
    assert spec.adapter == "custom"
    assert spec.family == shape


def test_chance_level_follows_the_options_declared():
    """25% on four options is not '25% good'. The spec must carry the right
    chance level or accuracy above guessing is wrong by a wide margin."""
    four = BenchmarkSpec.from_dict(bs.spec_dict("s", "multiple_choice", {"options_per_item": 4}))
    ten = BenchmarkSpec.from_dict(bs.spec_dict("s", "multiple_choice", {"options_per_item": 10}))
    assert four.scoring.chance_level == 0.25
    assert ten.scoring.chance_level == 0.1


@pytest.mark.parametrize("labels", ["billing, technical, other",
                                   ["billing", "technical", "other"]])
def test_labels_are_read_the_same_whether_typed_or_listed(labels):
    """The screen sends one comma-separated string, a script sends a list.
    Iterating the string would make every label one character long and every
    row's gold answer unreachable."""
    spec = BenchmarkSpec.from_dict(bs.spec_dict("s", "classify", {"labels": labels}))
    assert spec.label_set == ("billing", "technical", "other")


def test_classify_labels_reach_both_the_label_set_and_the_extractor():
    """The extractor needs the labels to find one in a reply, and the loader
    needs them to refuse a gold answer outside the set. Writing them in one
    place and not the other is how a label benchmark silently scores zero."""
    spec = BenchmarkSpec.from_dict(
        bs.spec_dict("s", "classify", {"labels": "billing, technical, other"}))
    assert spec.label_set == ("billing", "technical", "other")
    link = [x for x in spec.scoring.chain if x.get("kind") == "label_set"][0]
    assert link["labels"] == ["billing", "technical", "other"]
    assert spec.scoring.chance_level == pytest.approx(1 / 3, abs=1e-6)


def test_a_bad_id_is_refused_before_anything_is_written():
    got = bs.preview("Not A Name", "math", {})
    assert got["errors"] and "Name must be" in got["errors"][0]


def test_the_previewed_yaml_is_the_yaml_it_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previewed = bs.preview("round_trip", "short_answer", {})["yaml"]
    written = bs.scaffold("round_trip", "short_answer", {}, root=tmp_path)["yaml"]
    assert previewed == written


# --------------------------------------------------------------------------- #
#  The rows it accepts
# --------------------------------------------------------------------------- #
def test_an_answer_that_names_no_option_is_caught_before_the_file_is_written():
    rows = '{"id": "x", "question": "Q?", "options": ["one", "two"], "answer": "three"}'
    got = bs.validate_jsonl("multiple_choice", rows)
    assert not got["ok"]
    assert "could never be reached" in got["problems"][0]


def test_a_duplicate_id_is_caught():
    rows = "\n".join([
        json.dumps({"id": "a", "question": "Q1", "answer": "1"}),
        json.dumps({"id": "a", "question": "Q2", "answer": "2"})])
    got = bs.validate_jsonl("math", rows)
    assert any("duplicate id" in p for p in got["problems"])


def test_a_non_numeric_answer_is_caught_for_the_number_shape():
    rows = json.dumps({"id": "m", "question": "How many?", "answer": "a few"})
    got = bs.validate_jsonl("math", rows)
    assert any("is not a number" in p for p in got["problems"])


def test_a_label_outside_the_declared_set_is_caught():
    rows = json.dumps({"id": "t", "question": "text", "answer": "urgent"})
    got = bs.validate_jsonl("classify", rows, labels=["billing", "other"])
    assert any("not one of the labels" in p for p in got["problems"])


def test_broken_json_names_the_line_and_an_empty_file_says_so():
    got = bs.validate_jsonl("math", '{"id": "a", "question": "Q", "answer": 1}\n{oops\n')
    assert any("line 2: invalid JSON" in p for p in got["problems"])
    assert not bs.validate_jsonl("math", "\n \n")["ok"]


def test_languages_are_counted_so_the_screen_can_report_them():
    got = bs.validate_jsonl("short_answer", bs.describe("short_answer")["samples_jsonl"])
    assert got["languages"] == {"en": 2, "mr": 1}


def test_a_declared_benchmark_reports_that_it_has_an_adapter(tmp_path, monkeypatch):
    """The catalogue, `bench list`, `bench validate` and the run preflight all
    ask the same question. Asking `spec.id in known()` would label every
    benchmark someone declared as broken, which is a working feature looking
    like a bug."""
    monkeypatch.chdir(tmp_path)
    bs.scaffold("declared_set", "math", {}, root=tmp_path)
    spec = load_spec("configs/benchmarks/declared_set.yaml")
    assert registry.has_adapter(spec) is True
    assert spec.id not in registry.known()   # no id-keyed adapter, and none needed


# --------------------------------------------------------------------------- #
#  What it writes, and that it runs
# --------------------------------------------------------------------------- #
def test_it_writes_exactly_two_things(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    got = bs.scaffold("dept_set", "multiple_choice", {"options_per_item": 4}, root=tmp_path)
    assert got["written"][0] == "configs/benchmarks/dept_set.yaml"
    assert "data/benchmarks/dept_set/items.jsonl" in got["written"]
    assert (tmp_path / "configs/benchmarks/dept_set.yaml").exists()
    assert (tmp_path / "data/benchmarks/dept_set/items.jsonl").exists()
    # No adapter module, no registry edit: that is the whole claim.
    assert not (tmp_path / "harness").exists()


def test_it_refuses_to_overwrite_unless_asked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bs.scaffold("dept_set", "math", {}, root=tmp_path)
    with pytest.raises(ApiError, match="already exists"):
        bs.scaffold("dept_set", "math", {}, root=tmp_path)
    bs.scaffold("dept_set", "math", {}, root=tmp_path, overwrite=True)


def test_a_file_with_problems_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bs.scaffold("dept_set", "math", {}, root=tmp_path)
    before = (tmp_path / "data/benchmarks/dept_set/items.jsonl").read_text(encoding="utf-8")
    with pytest.raises(ApiError, match="nothing was written"):
        bs.write_data("dept_set", "math", '{"id": "x", "question": "Q", "answer": "lots"}',
                      overwrite=True, root=tmp_path)
    assert (tmp_path / "data/benchmarks/dept_set/items.jsonl").read_text(encoding="utf-8") == before


@pytest.mark.parametrize("shape", SHAPES)
def test_what_the_screen_writes_loads_and_scores_end_to_end(tmp_path, monkeypatch, shape):
    """The whole feature in one assertion: two files on disk, and the harness
    runs them like any shipped benchmark."""
    monkeypatch.chdir(tmp_path)
    opts = {"labels": "water_supply, street_lighting, pension, other"} if shape == "classify" else {}
    bs.scaffold("my_set", shape, opts, root=tmp_path)

    spec = load_spec("configs/benchmarks/my_set.yaml")
    adapter = registry.build(spec)
    items = adapter.load()
    assert len(items) == 3

    for item in items:
        raw = (f"Answer: {item.gold_answer}" if shape in ("multiple_choice", "math")
               else str(item.gold_answer))
        s = adapter.score(item, adapter.extract(raw))
        assert s.get("accuracy") == 1.0, (shape, item.item_id, raw)
        assert s.get("extraction_failed") is False

    messages = adapter.prompt(items[0], ()).as_messages()
    assert messages[-1]["role"] == "user"
    assert items[0].query[:20] in messages[-1]["content"]


def test_an_uploaded_file_replaces_the_samples_and_still_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    bs.scaffold("my_set", "short_answer", {}, root=tmp_path)
    rows = "\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"id": "r1", "question": "Which body issues a ration card?",
         "answer": "The Food and Civil Supplies Department"},
        {"id": "r2", "question": "What is the RTI application fee in rupees?", "answer": "10"},
    ])
    got = bs.write_data("my_set", "short_answer", rows, overwrite=True, root=tmp_path)
    assert got["rows"] == 2

    adapter = registry.build(load_spec("configs/benchmarks/my_set.yaml"))
    items = adapter.load()
    assert [it.item_id for it in items] == ["r1", "r2"]
    # "10" against "10" is a numeric match; the fee row is the one that would
    # break if the shape compared strings only.
    fee = [it for it in items if it.item_id == "r2"][0]
    assert adapter.score(fee, adapter.extract("Answer: 10")).get("accuracy") == 1.0
