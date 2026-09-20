"""
Languages the harness knows by name: code, script, digits, and whether the
pinned multilingual embedder documents them.

A language is not a script. Hindi and Marathi share Devanagari; Hinglish is
Hindi in Latin letters. Profiles name a `target_script` for the native
script metric, evalsets tag items with `meta.language` for the per-language
reading, and the Evaluate screen offers the list below. This table is the
one place those three agree.

`embedder_listed` records whether the language appears in the documented
training set of the pinned multilingual embedder
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). It is a
claim about the model card, not a measurement: a language not listed may
still retrieve acceptably through the shared vocabulary, and one that is
listed has still not been measured here. Hindi to English retrieval was
checked on this machine (cosine 0.73 on a paired sentence); the others
were not, and the Evaluate screen says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import SCRIPT_RANGES


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    script: str            # a key of SCRIPT_RANGES
    digit_zero: int | None  # code point of the script's zero, for digit normalisation
    embedder_listed: bool   # named on the pinned multilingual embedder's model card
    note: str = ""


LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "latin", None, True),
    "hi": Language("hi", "Hindi", "devanagari", 0x0966, True,
                   "Hindi to English retrieval checked locally with the multilingual embedder."),
    "mr": Language("mr", "Marathi", "devanagari", 0x0966, True,
                   "Shares Devanagari with Hindi; the native script metric cannot tell them apart."),
    "bn": Language("bn", "Bengali", "bengali", 0x09E6, False,
                   "Not on the multilingual embedder's documented list; retrieval quality unmeasured."),
    "gu": Language("gu", "Gujarati", "gujarati", 0x0AE6, True),
    "kn": Language("kn", "Kannada", "kannada", 0x0CE6, False,
                   "Not on the multilingual embedder's documented list; retrieval quality unmeasured."),
    "te": Language("te", "Telugu", "telugu", 0x0C66, False,
                   "Not on the multilingual embedder's documented list; retrieval quality unmeasured."),
    "hinglish": Language("hinglish", "Hinglish (Hindi in Latin letters)", "latin", None, True,
                         "Scored as Latin script; a Devanagari target would mark it as off-script."),
}

#: Digit blocks of every script in SCRIPT_RANGES that has one, for numeric
#: extraction across languages (MGSM and any numeric-scored profile).
DIGIT_ZEROS: dict[str, int] = {
    "devanagari": 0x0966, "bengali": 0x09E6, "gurmukhi": 0x0A66, "gujarati": 0x0AE6,
    "odia": 0x0B66, "tamil": 0x0BE6, "telugu": 0x0C66, "kannada": 0x0CE6, "malayalam": 0x0D66,
}

_DIGIT_MAP: dict[int, int] = {}
for _zero in DIGIT_ZEROS.values():
    for _d in range(10):
        _DIGIT_MAP[_zero + _d] = ord("0") + _d


def normalise_digits(text: str) -> str:
    """Indic-script digits to ASCII; everything else untouched. Pure."""
    return text.translate(_DIGIT_MAP)


def resolve_script(name: str) -> str:
    """A script key from a script name, a language code or a language name.

    "devanagari", "hi", "marathi" and "Marathi" all resolve to "devanagari".
    Raises ValueError, naming what is known, for anything else, so a
    misspelt profile field fails at validation rather than scoring 0.
    """
    key = (name or "").strip().lower()
    if key in SCRIPT_RANGES:
        return key
    if key in LANGUAGES:
        return LANGUAGES[key].script
    by_name = {lang.name.lower(): lang.script for lang in LANGUAGES.values()}
    if key in by_name:
        return by_name[key]
    raise ValueError(
        f"unknown script or language {name!r}; scripts: {sorted(SCRIPT_RANGES)}; "
        f"languages: {sorted(LANGUAGES)}")


def language_label(code: str) -> str:
    lang = LANGUAGES.get((code or "").strip().lower())
    return lang.name if lang else code


def describe_languages() -> list[dict]:
    """The table the Evaluate screen shows: one row per language."""
    return [{"code": lang.code, "name": lang.name, "script": lang.script,
             "embedder_listed": lang.embedder_listed, "note": lang.note}
            for lang in LANGUAGES.values()]
