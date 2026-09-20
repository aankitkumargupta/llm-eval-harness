"""
MMLU-Pro, multiple choice, ten options (A-J).

Scoring notes that decide whether a number is comparable to a published one:

* **Chance is 0.10, not 0.25.** MMLU-Pro extends the classic four options to
  ten. A harness that carried MMLU's chance level across would over-adjust by
  15 points.
* **Generative, not log-likelihood.** Published MMLU-Pro numbers come from both
  modes and they are not interchangeable; the spec records which was used and
  the reporter refuses to compare across a `spec_hash` boundary (§10.1).
* **Option order is content.** Shuffling options is a robustness probe and a
  contamination probe at once (§10.4), so it is a declared perturbation rather
  than something the loader does on its own.

`load`, `prompt`, `extract` and `score` are pure. Sampling is a function of
`(seed, dataset_hash)` only, so two runs with the same seed see the same items
in the same order, which is what makes the comparison paired (I1).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import run_chain
from ..spec import BenchmarkSpec

LETTERS = "ABCDEFGHIJ"


class MMLUProAdapter:
    """§10.3 adapter for ten-way multiple choice."""

    id = "mmlu_pro"

    def __init__(self, spec: BenchmarkSpec, data_path: str | None = None):
        self.spec = spec
        self.data_path = data_path or spec.source.ref

    # ------------------------------------------------------------------ #
    def load(self, *, seed: int | None = None, limit: int | None = None
             ) -> Sequence[EvalItem]:
        seed = self.spec.sampling.seed if seed is None else seed
        limit = self.spec.sampling.limit if limit is None else limit

        items: list[EvalItem] = []
        for rec in _read_jsonl(self.data_path):
            options = list(rec.get("options", []))
            if not options:
                continue
            gold = rec.get("answer", "")
            # Accept either a letter ("C") or an index (2); a dataset that
            # switches convention between splits is common enough to handle.
            if isinstance(gold, int):
                gold = LETTERS[gold] if 0 <= gold < len(LETTERS) else ""
            items.append(EvalItem(
                item_id=str(rec.get("question_id", rec.get("item_id", len(items)))),
                query=str(rec.get("question", "")),
                item_type=ItemType.ANSWERABLE,
                gold_answer=str(gold).strip().upper(),
                meta={"options": options,
                          "subject": rec.get("category", rec.get("subject", ""))},
            ))

        items.sort(key=lambda i: i.item_id)      # stable before sampling
        if limit is not None and limit < len(items):
            items = _stratified_sample(items, limit, seed,
                                       self.spec.sampling.stratify_by)
        return items

    # ------------------------------------------------------------------ #
    def prompt(self, item: EvalItem, shots: Sequence[EvalItem] = ()) -> Prompted:
        messages: list[dict] = []
        system = self.spec.prompt.system or (
            "Answer the multiple-choice question. Think briefly if you need to, "
            "then end your reply with 'Answer: X' where X is the option letter.")
        messages.append({"role": "system", "content": system})

        for shot in shots:
            messages.append({"role": "user", "content": _render(shot)})
            messages.append({"role": "assistant",
                             "content": f"Answer: {shot.gold_answer}"})

        messages.append({"role": "user", "content": _render(item)})
        return Prompted(messages=tuple(messages),
                        choices=tuple(LETTERS[:len(item.meta.get("options", []))]))

    # ------------------------------------------------------------------ #
    def extract(self, raw: str) -> Extraction:
        return run_chain(raw, self.spec.scoring.chain)

    # ------------------------------------------------------------------ #
    def score(self, item: EvalItem, extraction: Extraction) -> ScoreSet:
        s = ScoreSet()
        if extraction.failed:
            # accuracy=None, NOT 0.0 (I7): the model may well have been right,
            # and we simply could not tell. It counts as an extraction failure
            # and is absent from the accuracy denominator.
            s.set("accuracy", None)
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
    opts = item.meta.get("options", [])
    lines = [item.query, ""]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)]
    return "\n".join(lines)


def _read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"MMLU-Pro data not found at {path}. Fetch it with "
            f"`python main.py bench fetch mmlu_pro`, or point `source.ref` at "
            f"a local JSONL file.")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _stratified_sample(items, limit: int, seed: int, stratify_by: str):
    """Deterministic sample, proportional within strata when one is declared.

    Proportional rather than equal-sized: a subject that is 2% of the benchmark
    should be 2% of the sample, or the limited run answers a different question
    from the full one.
    """
    rng = random.Random(seed)
    if not stratify_by:
        idx = sorted(rng.sample(range(len(items)), limit))
        return [items[i] for i in idx]

    strata: dict[str, list] = {}
    for it in items:
        strata.setdefault(str(it.meta.get(stratify_by, "")), []).append(it)

    out = []
    for key in sorted(strata):
        group = strata[key]
        take = min(max(1, round(limit * len(group) / len(items))), len(group))
        idx = sorted(rng.sample(range(len(group)), take))
        out.extend(group[i] for i in idx)

    # Proportional rounding rarely lands on `limit` exactly, four strata and a
    # limit of five gives four items. Top up (or trim) deterministically so the
    # caller gets the count it asked for; a `--limit 200` that quietly returns
    # 197 makes two runs unpaired for no reason the user can see.
    if len(out) < limit:
        chosen = {it.item_id for it in out}
        spare = [it for it in items if it.item_id not in chosen]
        spare.sort(key=lambda i: i.item_id)
        need = min(limit - len(out), len(spare))
        if need:
            idx = sorted(rng.sample(range(len(spare)), need))
            out.extend(spare[i] for i in idx)

    out.sort(key=lambda i: i.item_id)
    return out[:limit]
