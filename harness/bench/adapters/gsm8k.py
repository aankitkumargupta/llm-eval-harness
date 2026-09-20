"""
GSM8K, grade-school word problems, scored on the final number.

§10.4: "Numeric-equivalence extraction (fractions, units, LaTeX, trailing
periods) is the whole game; report extraction failure loudly."

That is not an exaggeration. The reasoning gap between two modern models on
GSM8K is often smaller than the gap introduced by whether the harness accepts
`$1,234.00` as equal to `1234`. So this adapter does two things carefully and
almost nothing else:

* extraction normalises aggressively (see `harness/bench/extract.py`), and
* comparison is numeric, never string equality.

A model that writes "So Janet makes $18 every day." is correct. A harness that
marks it wrong because the gold is `18` has measured its own parser.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import numeric_equal, run_chain
from ..spec import BenchmarkSpec

#: Phrases that mean "I will not answer" rather than "here is a wrong answer".
#: Tracked separately (I7) because a refusal is a behaviour finding, not a
#: reasoning failure, and averaging it into accuracy conflates the two.
_REFUSALS = ("i cannot", "i can't", "i'm unable", "i am unable",
             "as an ai", "i won't", "i will not")


class GSM8KAdapter:
    """§10.3 adapter for numeric short-answer maths."""

    id = "gsm8k"

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
            answer = str(rec.get("answer", ""))
            # GSM8K ships the worked solution and the final answer in one
            # field, separated by "####". Keep the rationale: it is the
            # few-shot content, and dropping it would silently change the
            # prompt format and therefore the score.
            rationale, _, final = answer.partition("####")
            items.append(EvalItem(
                item_id=str(rec.get("item_id", i)),
                query=str(rec.get("question", "")),
                item_type=ItemType.ANSWERABLE,
                gold_answer=(final or answer).strip(),
                meta={"rationale": rationale.strip()},
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
            "Solve the problem. Show your working, then give the final numeric "
            "answer on its own last line as '#### <number>'.")
        messages: list[dict] = [{"role": "system", "content": system}]

        for shot in shots:
            messages.append({"role": "user", "content": shot.query})
            body = shot.meta.get("rationale", "")
            messages.append({"role": "assistant",
                             "content": f"{body}\n#### {shot.gold_answer}".strip()})

        messages.append({"role": "user", "content": item.query})
        return Prompted(messages=tuple(messages), stop=())

    # ------------------------------------------------------------------ #
    def extract(self, raw: str) -> Extraction:
        return run_chain(raw, self.spec.scoring.chain)

    # ------------------------------------------------------------------ #
    def score(self, item: EvalItem, extraction: Extraction,
              raw: str = "") -> ScoreSet:
        s = ScoreSet()
        refused = any(p in raw.lower() for p in _REFUSALS) if raw else False
        s.set("refused", refused)

        if extraction.failed:
            s.set("accuracy", None)          # I7: not a wrong answer
            s.set("extraction_failed", True)
            s.set("extraction_reason", extraction.reason)
            return s

        s.set("accuracy", 1.0 if numeric_equal(extraction.value,
                                               item.gold_answer) else 0.0)
        s.set("extraction_failed", False)
        s.set("extracted_via", extraction.via)
        return s


def _read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"GSM8K data not found at {path}. Fetch it with "
            f"`python main.py bench fetch gsm8k`, or point `source.ref` at a "
            f"local JSONL file.")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
