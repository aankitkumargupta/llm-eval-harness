"""
ARC-Challenge, grade-school science, multiple choice.

A separate adapter from MMLU-Pro rather than a flag on it, because the two
differ in the one place that matters: ARC ships its options as
`{"text": [...], "label": [...]}` with labels that are sometimes `A-D` and
sometimes `1-4`, and MMLU-Pro ships a bare list. Bending one adapter to cover
both would put a `if dataset == ...` branch inside the scoring path, which is
exactly the leak §10.3 forbids.

Two details that decide whether a number here is comparable to a published one:

* **Chance is 0.25**, not 0.1, four options, not ten. Carrying MMLU-Pro's
  chance level across would under-adjust by fifteen points.
* **The numeric labels are real.** A meaningful fraction of ARC rows label
  their choices `1,2,3,4` rather than `A,B,C,D`. They are normalised to
  letters here so the extraction chain has one alphabet to find, and the gold
  answer is normalised the same way, normalising one and not the other is a
  quiet way to score every numeric-labelled item wrong.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import run_chain
from ..spec import BenchmarkSpec

LETTERS = "ABCDEFGH"


def _as_letter(label: str, index: int) -> str:
    """Normalise a choice label to a letter.

    `1` -> `A`, `A` -> `A`, anything unrecognised falls back to position.
    """
    lab = str(label).strip().upper()
    # `len(lab) == 1` is load-bearing: `"" in "ABCDEFGH"` is True in Python,
    # and so is `"AB" in "ABCDEFGH"`. Without the length check an empty or
    # multi-character label returns itself, matches no gold answer, and scores
    # every such row wrong with nothing in the output to say why.
    if len(lab) == 1 and lab in LETTERS:
        return lab
    if lab.isdigit():
        i = int(lab) - 1
        if 0 <= i < len(LETTERS):
            return LETTERS[i]
    return LETTERS[index] if index < len(LETTERS) else "A"


class ARCAdapter:
    """§10.3 adapter for four-way multiple choice."""

    id = "arc_challenge"

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
            choices = rec.get("choices") or {}
            texts = list(choices.get("text") or [])
            labels = list(choices.get("label") or [])
            if not texts:
                continue

            letters = [_as_letter(labels[j] if j < len(labels) else "", j)
                       for j in range(len(texts))]
            gold_raw = str(rec.get("answerKey", "")).strip()
            # Map the gold through the SAME normalisation as the options, not
            # a parallel one: a row labelled 1-4 with a gold of "2" must land
            # on "B" by the identical rule, or every numeric row scores wrong.
            gold = ""
            for j, lab in enumerate(labels):
                if str(lab).strip().upper() == gold_raw.upper():
                    gold = letters[j]
                    break
            if not gold:
                gold = _as_letter(gold_raw, 0)

            items.append(EvalItem(
                item_id=str(rec.get("id", i)),
                # OpenBookQA ships the identical choices/answerKey shape but
                # names the question `question_stem`; accepting both lets a
                # spec alone add it (§10.3: never the runner or the store).
                query=str(rec.get("question") or rec.get("question_stem") or ""),
                item_type=ItemType.ANSWERABLE,
                gold_answer=gold,
                meta={"options": texts, "letters": letters},
            ))

        items.sort(key=lambda it: it.item_id)
        if limit is not None and limit < len(items):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(items)), limit))
            items = [items[i] for i in idx]
        return items

    # ------------------------------------------------------------------ #
    def prompt(self, item: EvalItem, shots: Sequence[EvalItem] = ()) -> Prompted:
        system = self.spec.prompt.system or (
            "Answer the multiple-choice science question. Reply with the "
            "option letter, ending your response with 'Answer: X'.")
        messages: list[dict] = [{"role": "system", "content": system}]
        for shot in shots:
            messages.append({"role": "user", "content": _render(shot)})
            messages.append({"role": "assistant",
                             "content": f"Answer: {shot.gold_answer}"})
        messages.append({"role": "user", "content": _render(item)})
        return Prompted(messages=tuple(messages),
                        choices=tuple(item.meta.get("letters", [])))

    # ------------------------------------------------------------------ #
    def extract(self, raw: str) -> Extraction:
        return run_chain(raw, self.spec.scoring.chain)

    # ------------------------------------------------------------------ #
    def score(self, item: EvalItem, extraction: Extraction) -> ScoreSet:
        s = ScoreSet()
        if extraction.failed:
            # accuracy=None, not 0.0 (I7): unreadable is not wrong.
            s.set("accuracy", None)
            s.set("extraction_failed", True)
            s.set("extraction_reason", extraction.reason)
            return s

        got = (extraction.value or "").strip().upper()[:1]
        s.set("accuracy", 1.0 if got == item.gold_answer else 0.0)
        s.set("extraction_failed", False)
        s.set("extracted_via", extraction.via)
        s.set("format_violation", got not in item.meta.get("letters", LETTERS))
        return s


def _render(item: EvalItem) -> str:
    letters = item.meta.get("letters", [])
    opts = item.meta.get("options", [])
    lines = [item.query, ""]
    lines += [f"{letters[i] if i < len(letters) else LETTERS[i]}. {o}"
              for i, o in enumerate(opts)]
    return "\n".join(lines)


def _read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ARC data not found at {path}. Fetch it with "
            f"`python main.py bench fetch arc_challenge`.")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
