"""
WinoGrande: a sentence with one blank and two candidate fillers, built so
that the lexical shortcuts which solved the original Winograd Schema
Challenge no longer work.

Research note (§11):
  * Source: allenai/winogrande on HuggingFace, config `winogrande_debiased`
    (the paper's reported set), split `validation`, 1,267 items. The dataset
    card lists CC-BY (Sakaguchi et al., 2020, "WinoGrande: An Adversarial
    Winograd Schema Challenge at Scale").
  * Fields: `sentence` with a single `_`, `option1`, `option2`, `answer`
    ("1" or "2").
  * The paper scores by log-likelihood of the sentence with each option
    substituted; lm-eval-harness does the same. This adapter is generative
    (choose A or B), so its numbers are not comparable to published ones.
  * Chance is 0.5 exactly. Any accuracy below about 0.6 on a modern model
    is more likely an extraction problem than a reasoning one; the
    format-violation rate is reported beside accuracy for that reason.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import run_chain
from ..spec import BenchmarkSpec

LETTERS = "AB"


class WinoGrandeAdapter:
    """§10.3 adapter for two-option fill-in-the-blank."""

    id = "winogrande"

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
            sentence = str(rec.get("sentence", ""))
            o1, o2 = str(rec.get("option1", "")), str(rec.get("option2", ""))
            ans = str(rec.get("answer", "")).strip()
            if "_" not in sentence or ans not in ("1", "2") or not (o1 and o2):
                continue
            items.append(EvalItem(
                item_id=str(rec.get("item_id", i)),
                query=sentence,
                item_type=ItemType.ANSWERABLE,
                gold_answer="A" if ans == "1" else "B",
                meta={"options": [o1, o2], "letters": ["A", "B"]},
            ))

        items.sort(key=lambda it: int(it.item_id) if it.item_id.isdigit()
                   else it.item_id)
        if limit is not None and limit < len(items):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(items)), limit))
            items = [items[i] for i in idx]
        return items

    # ------------------------------------------------------------------ #
    def prompt(self, item: EvalItem, shots: Sequence[EvalItem] = ()) -> Prompted:
        system = self.spec.prompt.system or (
            "Fill in the blank marked _ with the option that makes the sentence "
            "make sense. Reply with the option letter, ending your response "
            "with 'Answer: X'.")
        messages: list[dict] = [{"role": "system", "content": system}]
        for shot in shots:
            messages.append({"role": "user", "content": _render(shot)})
            messages.append({"role": "assistant",
                             "content": f"Answer: {shot.gold_answer}"})
        messages.append({"role": "user", "content": _render(item)})
        return Prompted(messages=tuple(messages), choices=("A", "B"))

    # ------------------------------------------------------------------ #
    def extract(self, raw: str) -> Extraction:
        return run_chain(raw, self.spec.scoring.chain)

    # ------------------------------------------------------------------ #
    def score(self, item: EvalItem, extraction: Extraction) -> ScoreSet:
        s = ScoreSet()
        if extraction.failed:
            s.set("accuracy", None)          # I7: unreadable is not wrong
            s.set("extraction_failed", True)
            s.set("extraction_reason", extraction.reason)
            return s
        got = (extraction.value or "").strip().upper()[:1]
        s.set("accuracy", 1.0 if got == item.gold_answer else 0.0)
        s.set("extraction_failed", False)
        s.set("extracted_via", extraction.via)
        s.set("format_violation", got not in LETTERS)
        return s


def _render(item: EvalItem) -> str:
    o1, o2 = item.meta.get("options", ["", ""])
    return f"{item.query}\n\nA. {o1}\nB. {o2}"


def _read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"WinoGrande data not found at {path}. Fetch it with "
            f"`python main.py bench fetch --benchmark winogrande`.")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
