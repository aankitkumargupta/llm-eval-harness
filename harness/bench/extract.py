"""
Answer extraction, the part of a benchmark that is quietly the whole benchmark.

For a maths or multiple-choice set, the difference between a reported 0.61 and
a reported 0.78 is usually not the model. It is whether the harness could find
the answer in the model's prose. That makes extraction the highest-leverage and
least-examined code in any eval harness, so this module has three rules:

1. **Explicit chain, first match wins.** The spec lists the extractors in
   order. No hidden fallbacks, no "try a bit harder if it looks like maths".
   Which link fired is recorded on the `Extraction`, because "the regex matched"
   and "the last-resort heuristic guessed" are different confidences in the
   same number.

2. **Failure is a result, not a zero.** When no link matches, the answer is
   `Failed(reason)` and the item scores `accuracy=None`. It counts toward
   `extraction_failure_rate` and toward nothing else (I7). Counting it as
   wrong measures the regex and blames the model.

3. **Pure.** No clock, no RNG, no network, no provider. The `judge` link is
   declared in the spec but resolved by the caller through the pinned `Judge`
   capability (§10.3), because an adapter that called a model directly would
   make the apparatus vary with the benchmark and break I2.

The numeric extractor carries the most scar tissue, and deliberately so: §10.4
names numeric equivalence as "the whole game" for GSM8K/MATH. `1,234`, `1234`,
`$1234`, `1234.`, `1234.00` and `\\boxed{1234}` are the same answer, and a
harness that scores them differently is reporting its own formatting opinions.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from .contracts import Extraction, Failed, Ok

#: Marker a chain link returns when the caller must resolve it (the `judge`
#: link). Kept out of `Extraction` so the pure path has no third state.
JUDGE_PENDING = "__judge__"


def _regex(raw: str, link: dict) -> str | None:
    pattern = link.get("pattern", "")
    flags = re.IGNORECASE if link.get("ignore_case", True) else 0
    if link.get("dotall"):
        flags |= re.DOTALL
    matches = re.findall(pattern, raw, flags)
    if not matches:
        return None
    # LAST match, not first: models restate the question before answering, so
    # the first match is frequently part of the prompt echoed back.
    m = matches[-1]
    return (m[0] if isinstance(m, tuple) else m).strip() or None


def _last_capital_letter(raw: str, link: dict) -> str | None:
    """For MC sets whose options are A-J. Last, for the same echo reason.

    **Known hazard, deliberately not "fixed".** The English pronoun "I" is also
    a valid option letter, so "I would rather not say." extracts `I` and scores
    as a confident answer to option nine. Excluding it would be worse: a model
    that genuinely answers "I" would then silently become an extraction
    failure, and a benchmark that cannot express one of its own options is
    broken in a way that is much harder to notice.

    The honest handling is to make it visible rather than to guess, which is
    why this is a *last-resort* link, why the `Extraction` records which link
    fired, and why the extraction playground in the web UI exists. A spec whose
    models refuse often should put an explicit refusal check ahead of this
    link rather than relying on it.
    """
    hi = link.get("max_option", "J").upper()
    found = re.findall(rf"\b([A-{hi}])\b", raw)
    return found[-1] if found else None


_NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _numeric(raw: str, link: dict) -> str | None:
    """Normalised numeric answer, or None.

    Returns a canonical string so that `1,234`, `$1234` and `1234.00` all
    compare equal downstream. Normalising here rather than at comparison time
    means every adapter gets the same answer to "is this the same number?"
    """
    text = raw
    boxed = re.findall(r"\\boxed\{([^}]*)\}", raw)
    if boxed:
        text = boxed[-1]
    # "#### 42" is GSM8K's own answer delimiter; when present it is definitive.
    hashed = re.findall(r"####\s*(-?[\d,\.]+)", raw)
    if hashed:
        text = hashed[-1]

    nums = _NUM.findall(text)
    if not nums:
        return None
    return _canonical_number(nums[-1])


def _canonical_number(tok: str) -> str | None:
    tok = tok.replace(",", "").replace("$", "").strip().rstrip(".")
    if not tok or tok in ("-", "."):
        return None
    try:
        val = float(tok)
    except ValueError:
        return None
    # Integers render without a trailing ".0" so 42 and 42.0 are one answer.
    if val == int(val) and abs(val) < 1e15:
        return str(int(val))
    return repr(round(val, 6))


def _boxed(raw: str, link: dict) -> str | None:
    found = re.findall(r"\\boxed\{([^}]*)\}", raw)
    return found[-1].strip() if found else None


def _first_line(raw: str, link: dict) -> str | None:
    for line in raw.strip().splitlines():
        if line.strip():
            return line.strip()
    return None


def _verbatim(raw: str, link: dict) -> str | None:
    return raw.strip() or None


def _label_set(raw: str, link: dict) -> str | None:
    """Longest matching label from a declared set.

    Longest wins so `not_urgent` beats `urgent` when both appear as substrings
, the same rule the classify profile already uses, kept identical here so
    the two paths cannot disagree.
    """
    labels = link.get("labels") or ()
    low = raw.lower()
    hits = [lab for lab in labels if lab.lower() in low]
    return max(hits, key=len) if hits else None


EXTRACTORS: dict[str, Callable[[str, dict], str | None]] = {
    "regex": _regex,
    "last_capital_letter": _last_capital_letter,
    "numeric": _numeric,
    "boxed": _boxed,
    "first_line": _first_line,
    "verbatim": _verbatim,
    "label_set": _label_set,
}


def run_chain(raw: str, chain: Sequence[dict],
              allow_judge: bool = False) -> Extraction:
    """Apply an extraction chain. First match wins; no match is a failure.

    `allow_judge=False` (the default for the pure path) treats a `judge` link
    as unreachable rather than pretending it resolved, the caller with access
    to the pinned Judge re-runs the chain with it enabled.
    """
    if not raw or not raw.strip():
        return Failed("empty model output")

    tried: list[str] = []
    for link in chain:
        kind = (link or {}).get("kind", "")
        if kind == "judge":
            if not allow_judge:
                tried.append("judge(skipped)")
                continue
            return Extraction(value=JUDGE_PENDING, via="judge")
        fn = EXTRACTORS.get(kind)
        if fn is None:
            tried.append(f"{kind}(unknown)")
            continue
        try:
            got = fn(raw, link or {})
        except re.error as e:
            tried.append(f"{kind}(bad pattern: {e})")
            continue
        if got is not None:
            return Ok(got, via=kind)
        tried.append(kind)

    return Failed(f"no extractor matched (tried: {', '.join(tried) or 'none'})")


def numeric_equal(a: str | None, b: str | None, tol: float = 1e-6) -> bool:
    """Numeric equivalence, so formatting never decides correctness.

    §10.4: "Numeric-equivalence extraction (fractions, units, LaTeX, trailing
    periods) is the whole game." Falls back to a normalised string compare when
    either side is not a number, rather than declaring a mismatch it cannot
    actually judge.
    """
    ca, cb = _canonical_number(str(a or "")), _canonical_number(str(b or ""))
    if ca is None or cb is None:
        return str(a or "").strip().lower() == str(b or "").strip().lower()
    try:
        return abs(float(ca) - float(cb)) <= tol
    except ValueError:
        return ca == cb
