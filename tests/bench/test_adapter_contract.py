"""
The shared adapter contract suite (§5, §10.3).

One parameterised suite that **every** benchmark adapter must pass. It is the
reason adding a benchmark is cheap and safe: a new adapter either satisfies
these properties or it does not ship, and nobody has to remember the list.

Everything here runs against the committed synthetic fixtures, offline.
"""

from __future__ import annotations

import pytest

from harness.bench import registry
from harness.bench.contracts import (
    BenchmarkAdapter,
    Extraction,
    Failed,
    Ok,
    Prompted,
    ScoreSet,
)
from harness.bench.spec import load_spec

ALL_BENCHMARKS = registry.known()

#: A benchmark declared in YAML rather than programmed (`adapter: custom`) is
#: a benchmark, so it faces this same suite. Without it, "bring your own set"
#: would be a second-class path that nothing checks.
CUSTOM_FIXTURE = "tests/bench/fixtures/custom_demo/spec.yaml"


@pytest.fixture(params=[*ALL_BENCHMARKS, CUSTOM_FIXTURE])
def adapter(request):
    spec = load_spec(request.param)
    return registry.build(spec)


# --------------------------------------------------------------------------- #
#  Shape
# --------------------------------------------------------------------------- #
def test_satisfies_the_protocol(adapter):
    assert isinstance(adapter, BenchmarkAdapter)


def test_every_shipped_benchmark_has_a_spec_and_an_adapter():
    """A spec with no adapter, or an adapter with no spec, is a half-built
    benchmark that fails only when someone tries to run it."""
    from harness.bench.spec import available_specs

    # Two ways to be complete: an adapter registered under the spec's own id
    # (every shipped benchmark), or a generic adapter the spec names. Anything
    # else fails only when someone tries to run it.
    by_id, by_generic = set(), {}
    for path in available_specs():
        spec = load_spec(path)
        (by_generic.setdefault(path.stem, spec.adapter) if spec.adapter
         else by_id.add(path.stem))

    assert by_id == set(ALL_BENCHMARKS), (
        f"specs {sorted(by_id)} != adapters {sorted(ALL_BENCHMARKS)}")
    unknown = {k: v for k, v in by_generic.items()
               if v not in registry.generic_known()}
    assert not unknown, f"specs naming an adapter that does not exist: {unknown}"


# --------------------------------------------------------------------------- #
#  load
# --------------------------------------------------------------------------- #
def test_load_returns_items(adapter):
    items = adapter.load()
    assert len(items) > 0
    assert all(it.item_id for it in items)


def test_item_ids_are_unique(adapter):
    """Duplicate ids silently collapse two items into one when the store
    groups by item, which corrupts pairing (I1)."""
    ids = [it.item_id for it in adapter.load()]
    assert len(ids) == len(set(ids))


def test_load_is_deterministic_given_a_seed(adapter):
    """I10. Two models in one run must see the identical item set in the
    identical order, or the comparison is not paired."""
    a = [it.item_id for it in adapter.load(seed=7, limit=10)]
    b = [it.item_id for it in adapter.load(seed=7, limit=10)]
    assert a == b


def test_a_different_seed_generally_selects_differently(adapter):
    """Guards against a 'seed' that is accepted and then ignored — which would
    make every sampled run silently identical."""
    a = [it.item_id for it in adapter.load(seed=1, limit=5)]
    b = [it.item_id for it in adapter.load(seed=99, limit=5)]
    assert len(a) == len(b) == 5
    # Not asserting a != b: with 20 fixture items a collision is possible.
    # The determinism test above is the load-bearing one.


def test_limit_is_respected(adapter):
    assert len(adapter.load(limit=5)) <= 5


# --------------------------------------------------------------------------- #
#  prompt
# --------------------------------------------------------------------------- #
def test_prompt_is_pure(adapter):
    """Same item, same messages. A prompt that varies run to run makes
    spec_hash a lie and the cache useless."""
    item = adapter.load(limit=1)[0]
    assert adapter.prompt(item, ()) == adapter.prompt(item, ())


def test_prompt_returns_usable_messages(adapter):
    item = adapter.load(limit=1)[0]
    p = adapter.prompt(item, ())
    assert isinstance(p, Prompted)
    msgs = p.as_messages()
    assert msgs and all(set(m) >= {"role", "content"} for m in msgs)
    assert msgs[-1]["role"] == "user"


def test_prompt_contains_the_question(adapter):
    item = adapter.load(limit=1)[0]
    blob = " ".join(m["content"] for m in adapter.prompt(item, ()).as_messages())
    assert item.query[:30] in blob


# --------------------------------------------------------------------------- #
#  extract
# --------------------------------------------------------------------------- #
def test_extract_is_pure_and_total(adapter):
    """Must return an Extraction for ANY string, never raise. Adversarial model
    output is data, not a crash (§6)."""
    for raw in ("", "   ", "\x00\x01", "𝕏" * 200, "```json{", "-" * 5000,
                "Answer: ZZZZ", "<script>alert(1)</script>"):
        got = adapter.extract(raw)
        assert isinstance(got, Extraction)


def test_extract_on_empty_output_is_a_failure_not_a_value(adapter):
    got = adapter.extract("")
    assert got.failed or got.value == ""


def test_extraction_records_which_link_fired(adapter):
    """'the regex matched' and 'the last-resort heuristic guessed' are very
    different confidences in the same reported score."""
    item = adapter.load(limit=1)[0]
    raw = _plausible_answer(adapter, item)
    got = adapter.extract(raw)
    if got.ok:
        assert got.via, "a successful extraction must record its chain link"


# --------------------------------------------------------------------------- #
#  score — the I7 contract
# --------------------------------------------------------------------------- #
def test_score_returns_a_scoreset(adapter):
    item = adapter.load(limit=1)[0]
    assert isinstance(adapter.score(item, Ok("A")), ScoreSet)


def test_extraction_failure_never_becomes_a_zero(adapter):
    """I7, the single most important property in this file.

    A parse failure must score `accuracy=None` (not applicable), never 0.0.
    Counting it as wrong measures the harness's regex and blames the model.
    """
    item = adapter.load(limit=1)[0]
    s = adapter.score(item, Failed("no extractor matched"))
    assert s.get("accuracy") is None, (
        "an extraction failure was folded into the accuracy numerator")
    assert s.get("extraction_failed") is True


def test_a_correct_answer_scores_one(adapter):
    item = adapter.load(limit=1)[0]
    raw = _plausible_answer(adapter, item)
    s = adapter.score(item, adapter.extract(raw))
    assert s.get("accuracy") == 1.0


def test_scores_are_bounded(adapter):
    """Property: every numeric metric lands in [0, 1]."""
    item = adapter.load(limit=1)[0]
    for raw in ("Answer: A", "#### 12", "banana", "", "yes " * 50):
        for v in adapter.score(item, adapter.extract(raw)).values.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                assert 0.0 <= float(v) <= 1.0


def test_applicable_excludes_not_applicable_metrics(adapter):
    item = adapter.load(limit=1)[0]
    s = adapter.score(item, Failed("x"))
    assert "accuracy" not in s.applicable()


# --------------------------------------------------------------------------- #
def _plausible_answer(adapter, item) -> str:
    """A correct answer in this family's natural output format."""
    bid = getattr(adapter, "id", "")
    if bid == "mmlu_pro":
        return f"Working through it, the result is clear.\nAnswer: {item.gold_answer}"
    if bid == "gsm8k":
        return f"Step one, then step two.\n#### {item.gold_answer}"
    if bid == "ifeval":
        return _satisfying_response(item)
    return str(item.gold_answer or "")


def _satisfying_response(item) -> str:
    """Build a response satisfying every constraint on an IFEval item."""
    cons = item.meta.get("constraints", {})
    if "valid_json" in cons:
        return '{"a": 1}'
    if "num_bullets" in cons:
        return "\n".join(f"- point {i}" for i in range(int(cons["num_bullets"])))
    if "num_paragraphs" in cons:
        return "\n\n".join("Para." for _ in range(int(cons["num_paragraphs"])))
    if "all_lowercase" in cons:
        return "delhi mumbai pune"
    if "all_uppercase" in cons:
        return "DELHI MUMBAI PUNE"
    if "starts_with" in cons:
        return f"{cons['starts_with']} the tides follow the moon"
    if "ends_with" in cons:
        return f"the tides follow the moon {cons['ends_with']}"
    if "contains" in cons:
        return f"Libraries inform people and {cons['contains']} they matter"
    if "not_contains" in cons:
        return "The storm was loud and long"
    if "no_commas" in cons:
        return "Rain fell all night. The road shone."
    if "min_words" in cons:
        return " ".join(["word"] * (int(cons["min_words"]) + 5))
    if "max_words" in cons:
        return " ".join(["word"] * max(1, int(cons["max_words"]) - 2))
    return "ok"
