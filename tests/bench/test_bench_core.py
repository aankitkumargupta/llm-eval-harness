"""
Spec validation, extraction, and benchmark metrics.

These are the pure parts of the benchmark subsystem, so the tests are cheap and
there is no excuse for thin coverage. Most of them encode a specific way a
published benchmark number becomes misleading (§10.1).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from harness.bench.extract import numeric_equal, run_chain
from harness.bench.metrics import chance_adjusted, pass_at_k, suite_composite
from harness.bench.spec import BenchmarkSpec, SpecError, load_spec


# --------------------------------------------------------------------------- #
#  Spec validation
# --------------------------------------------------------------------------- #
def _base(**over) -> dict:
    raw = {
        "id": "demo", "version": 1, "family": "multiple_choice", "task": "direct",
        "source": {"kind": "local", "ref": "x.jsonl", "licence": "MIT",
                   "split": "test", "commercial_use": True},
        "sampling": {"seed": 1, "samples_per_item": 1},
        "prompt": {"few_shot": {"n": 0, "selection": "fixed", "pool_split": "dev"},
                   "decoding": {"max_tokens": 512}},
        "scoring": {"mode": "generative", "chance_level": 0.25,
                    "extraction": {"chain": [{"kind": "last_capital_letter"}]}},
    }
    raw.update(over)
    return raw


def test_a_valid_spec_loads():
    assert BenchmarkSpec.from_dict(_base()).id == "demo"


def test_every_shipped_spec_is_valid():
    for bid in ("mmlu_pro", "gsm8k", "ifeval"):
        assert load_spec(bid).validate() == []


def test_few_shot_drawn_from_the_scored_split_is_rejected():
    """The easiest contamination to introduce by accident, and the hardest to
    see afterwards: the run looks normal and the score is simply wrong."""
    raw = _base()
    raw["prompt"]["few_shot"] = {"n": 5, "selection": "fixed", "pool_split": "test"}
    with pytest.raises(SpecError, match="IS the scored split"):
        BenchmarkSpec.from_dict(raw)


def test_multiple_choice_without_a_chance_level_is_rejected():
    """25% on 4-way MC is not '25% good'. §10.4 makes the adjustment mandatory,
    which means the input to it cannot be left at zero."""
    raw = _base()
    raw["scoring"]["chance_level"] = 0.0
    with pytest.raises(SpecError, match="chance_level"):
        BenchmarkSpec.from_dict(raw)


def test_generative_scoring_without_an_extraction_chain_is_rejected():
    """Without a chain every answer is an extraction failure, which a careless
    reader sees as 0% accuracy."""
    raw = _base()
    raw["scoring"]["extraction"] = {"chain": []}
    with pytest.raises(SpecError, match="chain"):
        BenchmarkSpec.from_dict(raw)


def test_missing_max_tokens_is_rejected():
    """The support_triage profile taught this the hard way: an undeclared token
    budget silently truncates reasoning models and reads as a quality finding."""
    raw = _base()
    raw["prompt"]["decoding"] = {"temperature": 0.0}
    with pytest.raises(SpecError, match="max_tokens"):
        BenchmarkSpec.from_dict(raw)


def test_a_fetched_source_needs_a_checksum():
    raw = _base()
    raw["source"] = {"kind": "hf", "ref": "org/ds", "licence": "MIT", "split": "test"}
    with pytest.raises(SpecError, match="checksum"):
        BenchmarkSpec.from_dict(raw)


def test_a_missing_licence_is_rejected():
    raw = _base()
    raw["source"]["licence"] = ""
    with pytest.raises(SpecError, match="licence"):
        BenchmarkSpec.from_dict(raw)


def test_validation_reports_every_problem_at_once():
    """One error at a time turns a five-minute fix into five round trips."""
    raw = _base()
    raw["source"]["licence"] = ""
    raw["scoring"]["chance_level"] = 0.0
    raw["prompt"]["decoding"] = {}
    problems = BenchmarkSpec.from_dict(_base()).validate()
    assert problems == []
    spec = BenchmarkSpec(id="x", family="multiple_choice")
    assert len(spec.validate()) >= 3


def test_non_commercial_licence_requires_acknowledgement():
    raw = _base()
    raw["source"]["commercial_use"] = False
    assert BenchmarkSpec.from_dict(raw).requires_licence_ack()


# --------------------------------------------------------------------------- #
#  spec_hash — the comparability mechanism
# --------------------------------------------------------------------------- #
def test_spec_hash_is_stable():
    assert BenchmarkSpec.from_dict(_base()).spec_hash() == \
           BenchmarkSpec.from_dict(_base()).spec_hash()


def test_changing_the_few_shot_count_changes_the_spec_hash():
    """Format is part of the identity (§10.1): the same benchmark at 0-shot and
    5-shot produces different numbers and must not be compared."""
    a = BenchmarkSpec.from_dict(_base()).spec_hash()
    raw = _base()
    raw["prompt"]["few_shot"] = {"n": 5, "selection": "fixed", "pool_split": "dev"}
    assert BenchmarkSpec.from_dict(raw).spec_hash() != a


def test_changing_the_extraction_chain_changes_the_spec_hash():
    a = BenchmarkSpec.from_dict(_base()).spec_hash()
    raw = _base()
    raw["scoring"]["extraction"] = {"chain": [{"kind": "regex", "pattern": "([A-D])"}]}
    assert BenchmarkSpec.from_dict(raw).spec_hash() != a


def test_changing_the_scoring_mode_changes_the_spec_hash():
    """Log-likelihood and generative MC scoring are not interchangeable."""
    a = BenchmarkSpec.from_dict(_base()).spec_hash()
    raw = _base()
    raw["scoring"]["mode"] = "loglikelihood"
    assert BenchmarkSpec.from_dict(raw).spec_hash() != a


# --------------------------------------------------------------------------- #
#  Extraction
# --------------------------------------------------------------------------- #
MC_CHAIN = [{"kind": "regex", "pattern": r"Answer:\s*\(?([A-J])\)?"},
            {"kind": "last_capital_letter"}]


def test_first_matching_link_wins_and_is_recorded():
    got = run_chain("Thinking... Answer: C", MC_CHAIN)
    assert got.ok and got.value == "C" and got.via == "regex"


def test_falls_through_to_the_next_link():
    got = run_chain("I believe it is B", MC_CHAIN)
    assert got.ok and got.value == "B" and got.via == "last_capital_letter"


def test_no_match_is_a_failure_not_a_value():
    got = run_chain("no idea", [{"kind": "regex", "pattern": r"Answer:\s*([A-J])"}])
    assert got.failed and "no extractor matched" in got.reason


def test_empty_output_fails_cleanly():
    assert run_chain("", MC_CHAIN).failed


def test_the_last_match_wins_not_the_first():
    """Models restate the question before answering, so the first match is
    frequently the prompt echoed back rather than the answer."""
    raw = "The options were A, B, C.\nAfter working it out, Answer: D"
    assert run_chain(raw, MC_CHAIN).value == "D"


def test_a_bad_regex_does_not_crash_the_chain():
    got = run_chain("Answer: C", [{"kind": "regex", "pattern": "([unclosed"},
                                  {"kind": "last_capital_letter"}])
    assert got.ok and got.value == "C"


def test_an_unknown_extractor_is_skipped_not_fatal():
    got = run_chain("Answer: C", [{"kind": "nonsense"},
                                  {"kind": "regex", "pattern": r"Answer:\s*([A-J])"}])
    assert got.ok and got.value == "C"


def test_judge_link_is_not_silently_resolved_on_the_pure_path():
    """An adapter must not call a model. The judge link is declared in the spec
    and resolved by the caller through the pinned Judge capability (I2)."""
    got = run_chain("hmm", [{"kind": "judge", "prompt": "t.j2"}])
    assert got.failed and "judge(skipped)" in got.reason


NUM_CHAIN = [{"kind": "regex", "pattern": r"####\s*(-?[\d,\.]+)"},
             {"kind": "boxed"}, {"kind": "numeric"}]


@pytest.mark.parametrize("raw,expected", [
    ("The answer is 1,234", "1234"),
    ("It costs $1234.", "1234"),
    ("#### 42", "42"),
    (r"so \boxed{99} is it", "99"),
    ("After all that, 1234.00", "1234"),
    ("a total of -5 rupees", "-5"),
])
def test_numeric_extraction_normalises_formatting(raw, expected):
    """§10.4: numeric equivalence 'is the whole game'. A harness that scores
    $1,234.00 differently from 1234 is reporting its own formatting opinions."""
    assert run_chain(raw, NUM_CHAIN).value == expected


@pytest.mark.parametrize("a,b", [
    ("1,234", "1234"), ("$18", "18"), ("42.0", "42"), ("42.", "42"),
])
def test_numeric_equal_accepts_equivalent_formats(a, b):
    assert numeric_equal(a, b)


def test_numeric_equal_rejects_genuinely_different_numbers():
    assert not numeric_equal("42", "43")


def test_numeric_equal_falls_back_to_string_compare():
    """Rather than declaring a mismatch it cannot actually judge."""
    assert numeric_equal("Paris", "paris")
    assert not numeric_equal("Paris", "Lyon")


def test_label_set_prefers_the_longest_match():
    """`not_urgent` must beat `urgent` when both appear as substrings — the
    same rule the classify profile uses, kept identical so they cannot drift."""
    chain = [{"kind": "label_set", "labels": ["urgent", "not_urgent"]}]
    assert run_chain("This is not_urgent", chain).value == "not_urgent"


def test_extraction_survives_adversarial_strings():
    """Model output is untrusted data (§6); it must never crash the parser."""
    for raw in ("\x00\x01\x02", "𝕏" * 500, "```" * 100, "<script>x</script>"):
        assert run_chain(raw, MC_CHAIN) is not None


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def test_chance_adjusted_zero_at_chance():
    assert chance_adjusted(0.25, 0.25) == pytest.approx(0.0)


def test_chance_adjusted_one_at_perfect():
    assert chance_adjusted(1.0, 0.25) == pytest.approx(1.0)


def test_chance_adjusted_midpoint():
    assert chance_adjusted(0.625, 0.25) == pytest.approx(0.5)


def test_below_chance_is_reported_negative_not_clamped():
    """A model below chance is a real finding — usually an extraction bug or an
    inverted label. Clamping to zero hides exactly that."""
    assert chance_adjusted(0.10, 0.25) < 0


def test_chance_adjusted_is_a_no_op_without_a_chance_level():
    assert chance_adjusted(0.73, 0.0) == pytest.approx(0.73)


@pytest.mark.parametrize("n,c,k,expected", [
    (10, 0, 1, 0.0),
    (10, 10, 1, 1.0),
    (10, 5, 1, 0.5),
    (5, 1, 5, 1.0),
    (1, 1, 1, 1.0),
])
def test_pass_at_k_known_values(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_matches_the_closed_form():
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k). Chen et al. (2021) §2.1."""
    n, c, k = 20, 7, 5
    expected = 1.0 - math.comb(n - c, k) / math.comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_pass_at_k_exceeds_naive_pass_at_1():
    """The whole point: more draws means a higher chance at least one passes."""
    assert pass_at_k(10, 3, 3) > pass_at_k(10, 3, 1)


def test_pass_at_k_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        pass_at_k(5, 9, 1)          # more correct than sampled
    with pytest.raises(ValueError):
        pass_at_k(5, 1, 0)          # k must be >= 1


# --------------------------------------------------------------------------- #
#  Aggregation
# --------------------------------------------------------------------------- #
def test_suite_composite_requires_declared_weights():
    """§10.7.2: 'Averaging benchmarks hides the weighting.' An implicit equal
    weighting is the most effective way to hide it — nobody questions a mean."""
    per = pd.DataFrame([{"model": "m", "benchmark": "gsm8k", "accuracy": 0.8},
                        {"model": "m", "benchmark": "ifeval", "accuracy": 0.6}])
    with pytest.raises(ValueError, match="No declared weight"):
        suite_composite(per, {"gsm8k": 1.0})


def test_suite_composite_prints_the_weighting_beside_the_number():
    per = pd.DataFrame([{"model": "m", "benchmark": "gsm8k", "accuracy": 0.8},
                        {"model": "m", "benchmark": "ifeval", "accuracy": 0.6}])
    out = suite_composite(per, {"gsm8k": 3.0, "ifeval": 1.0})
    assert out.iloc[0]["composite"] == pytest.approx((3 * 0.8 + 1 * 0.6) / 4)
    assert "gsm8k=3" in out.iloc[0]["weighting"]
