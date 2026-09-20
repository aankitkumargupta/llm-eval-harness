"""
MGSM: GSM8K's 250 test problems, professionally translated into ten
languages. Two of them are Indic (Bengali, Telugu), which is why this
adapter exists: the harness's own multilingual work needs a public,
uncontaminated, numerically-scored Indic set, and MGSM is the one with a
citation.

Research note (§11):
  * Source: juletxara/mgsm on HuggingFace, configs `bn` and `te`, split
    `test`, 250 items each. Licence CC-BY-4.0 (Shi et al., 2022, "Language
    Models are Multilingual Chain-of-Thought Reasoners").
  * Fields: `question` (translated problem), `answer_number` (int gold),
    `answer` and `equation_solution` (null in the test split).
  * The paper reports exact-match on the final number after chain-of-thought
    in the target language; lm-eval-harness scores the same way with a
    regex on the last number. Published numbers are generative.
  * Pitfall this adapter handles: models answering in the native script use
    native digits (Bengali ০-৯, Telugu ౦-౯). A numeric comparison that does
    not normalise them marks every such answer wrong and measures the
    parser, not the model. Digits are mapped to ASCII before extraction.
  * Pitfall left visible: the "####" convention in the prompt is English
    scaffolding; the harness asks for it explicitly so extraction is not a
    guess, and reports extraction failure separately (I7) when a model
    ignores it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

#: Native-script digits to ASCII, for every Indic script the harness knows
#: (Devanagari, Bengali, Gujarati, Kannada, Telugu, ...), shared with the
#: language table so a new language is added in one place.
from ...eval.languages import normalise_digits  # noqa: E402  (re-exported)
from ..contracts import EvalItem, Extraction, ItemType
from ..extract import run_chain
from .gsm8k import GSM8KAdapter, _read_jsonl

__all__ = ["MGSMAdapter", "normalise_digits"]


class MGSMAdapter(GSM8KAdapter):
    """§10.3 adapter for MGSM (Bengali and Telugu configs)."""

    id = "mgsm"

    def load(self, *, seed: int | None = None, limit: int | None = None
             ) -> Sequence[EvalItem]:
        seed = self.spec.sampling.seed if seed is None else seed
        limit = self.spec.sampling.limit if limit is None else limit

        items: list[EvalItem] = []
        for i, rec in enumerate(_read_jsonl(self.data_path)):
            gold = rec.get("answer_number")
            if gold is None:
                continue          # no gold, no item: never score a guess
            items.append(EvalItem(
                item_id=str(rec.get("item_id", i)),
                query=str(rec.get("question", "")),
                item_type=ItemType.ANSWERABLE,
                gold_answer=str(gold),
                meta={"rationale": "", "language": self.spec.source.config},
            ))

        items.sort(key=lambda it: int(it.item_id) if it.item_id.isdigit()
                   else it.item_id)
        if limit is not None and limit < len(items):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(items)), limit))
            items = [items[i] for i in idx]
        return items

    def extract(self, raw: str) -> Extraction:
        # Native digits become ASCII before the chain runs, so "#### ১৮"
        # extracts as 18 rather than failing on a non-ASCII token.
        return run_chain(normalise_digits(raw), self.spec.scoring.chain)
