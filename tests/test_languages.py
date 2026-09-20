"""
Languages as first-class features: Marathi, Bengali, Gujarati, Kannada,
Telugu and Hinglish beside Hindi and English. One table drives the native
script metric's aliases, numeric extraction across Indic digits, and the
Evaluate screen's language choice. Offline; no model is run.
"""

from __future__ import annotations

import pytest
import yaml

from harness.eval import languages as L
from harness.eval.metrics import SCRIPT_RANGES, native_script_ratio
from harness.profiles.profile import Profile, ProfileError
from harness.web import scaffold as S


def test_every_language_points_at_a_known_script_and_the_four_new_ones_exist():
    for code, lang in L.LANGUAGES.items():
        assert lang.code == code
        assert lang.script in SCRIPT_RANGES, (code, lang.script)
    for code, name in (("mr", "Marathi"), ("bn", "Bengali"), ("gu", "Gujarati"), ("kn", "Kannada")):
        assert L.LANGUAGES[code].name == name
    assert L.LANGUAGES["mr"].script == "devanagari" == L.LANGUAGES["hi"].script


def test_resolve_script_accepts_scripts_codes_and_names():
    assert L.resolve_script("devanagari") == "devanagari"
    assert L.resolve_script("mr") == "devanagari"
    assert L.resolve_script("Marathi") == "devanagari"
    assert L.resolve_script("kn") == "kannada"
    assert L.resolve_script("Gujarati") == "gujarati"
    assert L.resolve_script("bengali") == "bengali"
    with pytest.raises(ValueError, match="unknown script or language"):
        L.resolve_script("klingon")


def test_native_script_ratio_on_the_new_scripts():
    assert native_script_ratio("पाणी तीन दिवसांपासून आलेले नाही", "devanagari") == 1.0   # Marathi
    assert native_script_ratio("আমার এলাকায় জল নেই", "bengali") == 1.0
    assert native_script_ratio("પાણી આવતું નથી", "gujarati") == 1.0
    assert native_script_ratio("ನೀರು ಬರುತ್ತಿಲ್ಲ", "kannada") == 1.0
    # Mixed: vowel signs are marks, not letters, so "ನೀರು" has 2 letters; 2 of 2 + 5 Latin.
    assert native_script_ratio("ನೀರು water", "kannada") == pytest.approx(2 / 7)


def test_indic_digits_normalise_in_every_listed_script():
    assert L.normalise_digits("उत्तर: १८") == "उत्तर: 18"         # Devanagari (Hindi, Marathi)
    assert L.normalise_digits("উত্তর: ১৮") == "উত্তর: 18"         # Bengali
    assert L.normalise_digits("જવાબ: ૪૨") == "જવાબ: 42"           # Gujarati
    assert L.normalise_digits("ಉತ್ತರ: ೪೨") == "ಉತ್ತರ: 42"         # Kannada
    assert L.normalise_digits("సమాధానం ౪౨") == "సమాధానం 42"       # Telugu
    assert L.normalise_digits("plain 18") == "plain 18"
    from harness.bench.adapters.mgsm import normalise_digits as mgsm_digits
    assert mgsm_digits is L.normalise_digits, "one digit table, shared with the MGSM adapter"


def test_a_profile_may_name_the_language_instead_of_the_script():
    base = {"name": "t", "task": "direct", "evalset_path": "x.jsonl",
            "active_metrics": ["accuracy", "native_script_ratio"],
            "metric_weights": {"accuracy": 0.8, "native_script_ratio": 0.2}}
    for alias in ("marathi", "mr", "kannada", "gu", "Bengali"):
        assert Profile.from_dict({**base, "target_script": alias}).validate() == []
    with pytest.raises(ProfileError, match="target_script"):
        Profile.from_dict({**base, "target_script": "klingon"})


def test_the_scorer_resolves_the_alias():
    from harness.eval.scoring import NativeScriptScorer, ScoringContext
    from harness.store.schema import EvalItem

    ctx = ScoringContext(item=EvalItem(item_id="i", query="q"), model="m",
                         answer="ನೀರು ಬರುತ್ತಿಲ್ಲ", active={"native_script_ratio"}, target_script="Kannada")
    assert NativeScriptScorer().score(ctx) == {"native_script_ratio": 1.0}


@pytest.mark.parametrize("code,script", [("mr", "devanagari"), ("bn", "bengali"), ("gu", "gujarati"), ("kn", "kannada")])
def test_scaffold_language_option_sets_the_script_metric_for_direct_tasks(code, script):
    text = S.profile_yaml("lang_eval", "direct", {"language": code})
    p = Profile.from_dict(yaml.safe_load(text))
    assert p.validate() == []
    assert p.target_script == script
    assert "native_script_ratio" in p.active_metrics
    assert p.metric_weights["native_script_ratio"] == pytest.approx(0.2)
    assert all(w < 0 for m, w in p.metric_weights.items() if m in ("cost_usd", "latency_ms"))


def test_scaffold_language_option_picks_the_multilingual_embedder_for_rag_and_is_listed():
    rag = Profile.from_dict(yaml.safe_load(S.profile_yaml("lang_rag", "rag", {"language": "kn"})))
    assert "multilingual" in rag.embedding_model
    en = Profile.from_dict(yaml.safe_load(S.profile_yaml("lang_rag", "rag", {"language": "en"})))
    assert "multilingual" not in en.embedding_model
    with pytest.raises(Exception, match="language must be one of"):
        S.profile_yaml("lang_rag", "rag", {"language": "xx"})
    d = S.describe("direct")
    codes = {row["code"] for row in d["languages"]}
    assert {"mr", "bn", "gu", "kn", "hi", "en", "te", "hinglish"} <= codes
    kn = next(r for r in d["languages"] if r["code"] == "kn")
    assert kn["embedder_listed"] is False and "unmeasured" in kn["note"]


def test_the_ui_knows_the_language_names():
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for name in ("Marathi", "Bengali", "Gujarati", "Kannada"):
        assert name in js
    assert "Language of the answers" in js
