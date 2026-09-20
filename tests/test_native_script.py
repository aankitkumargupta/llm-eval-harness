"""
native_script_ratio: the one new metric the Indian-government use cases add.

A translation into Hindi that comes back half in Latin script has not done
the job, however faithful the judge finds it. The metric is a pure function
of the answer text (bounds, idempotence, adversarial input), a scorer that
only applies when a profile names a target script, and a column that does
not exist when it does not.
"""

from __future__ import annotations

import pytest

from harness.eval.metrics import SCRIPT_RANGES, native_script_ratio

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


# --------------------------------------------------------------------------- #
#  Hand-computed
# --------------------------------------------------------------------------- #
def test_hand_computed_ratios():
    assert native_script_ratio("जल आपूर्ति कल बाधित रहेगी", "devanagari") == 1.0
    assert native_script_ratio("Water supply tomorrow", "devanagari") == 0.0
    # "Water" (5 Latin letters) + "जल" (2 Devanagari letters) -> 2/7
    assert native_script_ratio("Water जल", "devanagari") == pytest.approx(2 / 7)
    # Digits, punctuation and matras do not count either way.
    assert native_script_ratio("12, 34 !!", "devanagari") is None
    assert native_script_ratio("", "devanagari") is None
    assert native_script_ratio("తెలుగు", "telugu") == 1.0
    assert native_script_ratio("বাংলা", "bengali") == 1.0
    assert native_script_ratio("Hello", "latin") == 1.0


def test_an_unknown_script_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="unknown script"):
        native_script_ratio("x", "klingon")
    assert "devanagari" in SCRIPT_RANGES and "telugu" in SCRIPT_RANGES


# --------------------------------------------------------------------------- #
#  Properties
# --------------------------------------------------------------------------- #
@settings(max_examples=300, deadline=None)
@given(st.text(min_size=0, max_size=200))
def test_bounded_and_never_crashes(text):
    r = native_script_ratio(text, "devanagari")
    assert r is None or 0.0 <= r <= 1.0


@settings(max_examples=200, deadline=None)
@given(st.text(min_size=1, max_size=120), st.sampled_from(sorted(SCRIPT_RANGES)))
def test_invariant_under_whitespace_and_punctuation(text, script):
    a = native_script_ratio(text, script)
    b = native_script_ratio("  " + text.replace(" ", "\n\t ") + " ..., !!", script)
    assert a == b


@settings(max_examples=100, deadline=None)
@given(st.text(alphabet=st.characters(min_codepoint=0x0905, max_codepoint=0x0939), min_size=1, max_size=40))
def test_pure_devanagari_letters_score_one(text):
    assert native_script_ratio(text, "devanagari") == 1.0


# --------------------------------------------------------------------------- #
#  The seam: scorer applies only when a profile names a script
# --------------------------------------------------------------------------- #
def test_scorer_applies_only_with_a_target_script():
    from harness.eval.scoring import NativeScriptScorer, ScoringContext
    from harness.store.schema import EvalItem

    it = EvalItem(item_id="i", query="q")
    off = ScoringContext(item=it, model="m", answer="नमस्ते", active={"native_script_ratio"})
    assert not NativeScriptScorer().applies(off)
    on = ScoringContext(item=it, model="m", answer="नमस्ते", active={"native_script_ratio"},
                        target_script="devanagari")
    assert NativeScriptScorer().applies(on)
    assert NativeScriptScorer().score(on) == {"native_script_ratio": 1.0}
    inactive = ScoringContext(item=it, model="m", answer="नमस्ते", active={"accuracy"},
                              target_script="devanagari")
    assert not NativeScriptScorer().applies(inactive)


def test_a_profile_with_an_unknown_script_fails_validation():
    from harness.profiles.profile import Profile, ProfileError

    base = {"name": "t", "task": "direct", "evalset_path": "x.jsonl",
            "active_metrics": ["accuracy", "native_script_ratio"],
            "metric_weights": {"accuracy": 0.8, "native_script_ratio": 0.2}}
    assert Profile.from_dict({**base, "target_script": "devanagari"}).target_script == "devanagari"
    with pytest.raises(ProfileError, match="target_script"):
        Profile.from_dict({**base, "target_script": "klingon"})


def test_the_trace_row_column_is_nullable_and_present():
    from harness.store.schema import ItemType, Pass, TraceRow

    row = TraceRow(run_id="r", item_id="i", model="m", profile="p",
                   pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE)
    assert row.native_script_ratio is None
