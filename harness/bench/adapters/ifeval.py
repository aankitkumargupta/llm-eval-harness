"""
IFEval, instruction following, checked programmatically.

§10.4 calls this "the cheapest high-signal family" and says it belongs in the
default suite. Both halves are true and they are connected: the constraints are
verifiable by code, word counts, required phrases, forbidden words, casing,
JSON validity, so there is **no judge**, which means no judge cost, no judge
bias, and no judge disagreement to explain away. A benchmark whose scoring is a
pure function is worth disproportionately more than its size suggests.

Because every constraint is checked independently, this adapter reports
`constraint_satisfaction_rate` with a per-constraint breakdown as well as
strict all-or-nothing accuracy. The breakdown is the useful half: "0.62" tells
you a model follows instructions badly, while "it fails `max_words` 80% of the
time and everything else never" tells you what to do about it.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Ok, Prompted, ScoreSet
from ..spec import BenchmarkSpec

# --------------------------------------------------------------------------- #
#  Constraint checkers, each pure, each returning a plain bool.
# --------------------------------------------------------------------------- #


def _c_min_words(text: str, arg) -> bool:
    return len(text.split()) >= int(arg)


def _c_max_words(text: str, arg) -> bool:
    return len(text.split()) <= int(arg)


def _c_contains(text: str, arg) -> bool:
    return str(arg).lower() in text.lower()


def _c_not_contains(text: str, arg) -> bool:
    return str(arg).lower() not in text.lower()


def _c_starts_with(text: str, arg) -> bool:
    return text.strip().lower().startswith(str(arg).lower())


def _c_ends_with(text: str, arg) -> bool:
    return text.strip().lower().endswith(str(arg).lower())


def _c_all_lowercase(text: str, arg) -> bool:
    return text == text.lower()


def _c_all_uppercase(text: str, arg) -> bool:
    return text == text.upper()


def _c_valid_json(text: str, arg) -> bool:
    stripped = text.strip()
    # Models habitually fence JSON. Rejecting that would measure fencing
    # habits, not instruction following.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    try:
        json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return True


def _c_num_bullets(text: str, arg) -> bool:
    bullets = re.findall(r"^\s*[-*•]\s+", text, re.MULTILINE)
    return len(bullets) == int(arg)


def _c_num_paragraphs(text: str, arg) -> bool:
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return len(paras) == int(arg)


def _c_no_commas(text: str, arg) -> bool:
    return "," not in text


CHECKERS: dict[str, Callable[[str, object], bool]] = {
    "min_words": _c_min_words,
    "max_words": _c_max_words,
    "contains": _c_contains,
    "not_contains": _c_not_contains,
    "starts_with": _c_starts_with,
    "ends_with": _c_ends_with,
    "all_lowercase": _c_all_lowercase,
    "all_uppercase": _c_all_uppercase,
    "valid_json": _c_valid_json,
    "num_bullets": _c_num_bullets,
    "num_paragraphs": _c_num_paragraphs,
    "no_commas": _c_no_commas,
}


class IFEvalAdapter:
    """§10.3 adapter for programmatically-checked instruction following."""

    id = "ifeval"

    def __init__(self, spec: BenchmarkSpec, data_path: str | None = None):
        self.spec = spec
        self.data_path = data_path or spec.source.ref

    # ------------------------------------------------------------------ #
    def load(self, *, seed: int | None = None, limit: int | None = None
             ) -> Sequence[EvalItem]:
        seed = self.spec.sampling.seed if seed is None else seed
        limit = self.spec.sampling.limit if limit is None else limit

        items: list[EvalItem] = []
        for i, rec in enumerate(_read_jsonl(self.data_path)):
            constraints = rec.get("constraints", {}) or {}
            unknown = sorted(set(constraints) - set(CHECKERS))
            if unknown:
                raise ValueError(
                    f"{self.data_path}: item {rec.get('item_id', i)} declares "
                    f"unknown constraint(s) {unknown}. Known: "
                    f"{sorted(CHECKERS)}. An unchecked constraint would be "
                    f"silently satisfied, inflating the score.")
            items.append(EvalItem(
                item_id=str(rec.get("item_id", i)),
                query=str(rec.get("prompt", rec.get("instruction", ""))),
                item_type=ItemType.ANSWERABLE,
                gold_answer=None,              # nothing to match; only to check
                meta={"constraints": constraints},
            ))

        items.sort(key=lambda it: it.item_id)
        if limit is not None and limit < len(items):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(items)), limit))
            items = [items[i] for i in idx]
        return items

    # ------------------------------------------------------------------ #
    def prompt(self, item: EvalItem, shots: Sequence[EvalItem] = ()) -> Prompted:
        messages: list[dict] = []
        if self.spec.prompt.system:
            messages.append({"role": "system", "content": self.spec.prompt.system})
        # No few-shot by default: demonstrating the format teaches the model to
        # imitate the examples rather than follow the stated instruction, which
        # is the one thing this benchmark is trying to measure.
        messages.append({"role": "user", "content": item.query})
        return Prompted(messages=tuple(messages))

    # ------------------------------------------------------------------ #
    def extract(self, raw: str) -> Extraction:
        """The whole response IS the answer, there is nothing to parse out.

        Returning `Ok(raw)` rather than inventing an extraction step keeps
        `extraction_failure_rate` honestly zero for this family instead of
        reporting a statistic that cannot apply.
        """
        return Ok(raw, via="verbatim")

    # ------------------------------------------------------------------ #
    def score(self, item: EvalItem, extraction: Extraction) -> ScoreSet:
        s = ScoreSet()

        # I7. This family extracts verbatim, so a failure here means the model
        # returned nothing at all, which is a non-answer, not a wrong answer.
        # An earlier version of this method skipped the check and scored an
        # empty response 0.0 on every constraint; the shared adapter contract
        # suite caught it, which is what that suite is for.
        if extraction.failed:
            s.set("accuracy", None)
            s.set("extraction_failed", True)
            s.set("extraction_reason", extraction.reason)
            return s

        text = extraction.value or ""
        constraints: dict = item.meta.get("constraints", {}) or {}

        if not constraints:
            s.set("accuracy", None)          # nothing to check: not applicable
            s.set("extraction_failed", False)
            return s

        results = {name: bool(CHECKERS[name](text, arg))
                   for name, arg in constraints.items()}

        s.set("accuracy", 1.0 if all(results.values()) else 0.0)  # strict
        s.set("constraint_satisfaction_rate",
              sum(results.values()) / len(results))               # partial
        s.set("constraints_failed",
              ",".join(sorted(n for n, ok in results.items() if not ok)))
        s.set("extraction_failed", False)
        s.set("format_violation", not all(results.values()))
        return s


def _read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"IFEval data not found at {path}. Point `source.ref` at a local "
            f"JSONL file, or add your own constraint set, this family works "
            f"well as a private in-house benchmark (§10.4).")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
