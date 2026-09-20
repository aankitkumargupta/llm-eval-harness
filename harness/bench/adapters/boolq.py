"""
BoolQ, yes/no reading comprehension over a Wikipedia passage.

Research note
-------------
* **Canonical source.** Clark et al., 2019, "BoolQ: Exploring the Surprising
  Difficulty of Natural Yes/No Questions" (NAACL 2019, arXiv:1905.10044).
  Data: github.com/google-research-datasets/boolean-questions, served on the
  Hub as `google/boolq` (config `default`; `train` 9,427 rows, `validation`
  3,270 rows). The scored split is `validation`: the SuperGLUE test labels are
  hidden, so every open number, including lm-eval-harness's, is on validation.
* **Licence.** CC-BY-SA-3.0. Commercial use is permitted; ShareAlike binds
  redistributed copies, so the 20-row fixture under
  `tests/bench/fixtures/boolq/` carries the same licence and an attribution
  note beside it.
* **Prompt format.** The paper fine-tunes BERT on `[CLS] question [SEP]
  passage` and reports accuracy; there is no prompt. lm-eval-harness (`boolq`,
  in the `super_glue` group) renders `{passage}\\nQuestion: {question}?\\nAnswer:`
  and scores by log-likelihood of the continuations `no` / `yes`
  (`doc_to_choice: ["no", "yes"]`), metric `acc`. This adapter renders the
  same user text and adds a system line asking for a leading yes/no, because
  the generative path has to find the answer inside prose.
* **Metric.** Accuracy over all items; chance is 0.5. The split is ~62% "yes",
  so the majority-class baseline is ~0.62 raw and ~0.24 chance-adjusted. Read
  the adjusted column, not the raw one.
* **Log-likelihood or generative?** Published numbers are either fine-tuned
  (BERT-large 80.4 in the paper; RoBERTa / T5 in the SuperGLUE era) or
  lm-eval log-likelihood zero-shot (the GPT-3 and Llama family reports). A
  generative number from this adapter is comparable to neither, which is why
  `scoring.mode` is part of `spec_hash`.
* **Scoring pitfalls.**
  - Questions in the dataset carry no trailing "?"; lm-eval appends one and so
    does this adapter, since the wording is part of the format.
  - The second chain link, `label_set`, matches substrings and prefers the
    longer label. So "not", "know" and "cannot" read as `no`, "eyes" reads as
    `yes`, and a reply that says "no ... yes" extracts as `yes`. The
    leading-answer regex fires first precisely so the substring link stays a
    fallback. The tests pin this as a known hazard rather than hide it.
  - A hedge ("cannot be determined from the passage") therefore scores as
    `no`. It is not a refusal in the gsm8k sense and is not tracked as one.
  - Two labels means a wrong extraction is right half the time. Extraction
    failures are reported as their own rate (I7) and never as a zero, and
    `extracted_via` records which link fired so a run that leaned on the
    substring fallback is visible in the trace.

`load`, `prompt`, `extract` and `score` are pure. Sampling is a function of
`(seed, dataset_hash)` only, so every model in a run sees the same items in
the same order (I1).
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import run_chain
from ..spec import BenchmarkSpec

#: The label alphabet, in the order the spec's `label_set` declares it.
LABELS = ("yes", "no")

#: Punctuation a model wraps a one-word answer in: "Yes.", "**no**", "yes!".
_WRAP = " \t\r\n.,!?:;\"'`*()[]"


def _as_label(answer: object) -> str:
    """Normalise the dataset's boolean gold to the label alphabet.

    The Hub serves a real `bool`; a locally prepared file may carry
    "true" / "false" / "yes" / "no" strings. Anything else raises: a silently
    mislabelled gold scores the model wrong and blames it.
    """
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    text = str(answer).strip().lower()
    if text in ("true", "yes", "1"):
        return "yes"
    if text in ("false", "no", "0"):
        return "no"
    raise ValueError(f"BoolQ gold answer must be a boolean, got {answer!r}")


class BoolQAdapter:
    """§10.3 adapter for two-way yes/no reading comprehension."""

    id = "boolq"

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
            question = str(rec.get("question", "")).strip()
            passage = str(rec.get("passage", "")).strip()
            # A hole in a checksummed file is a data error, not a row to skip:
            # skipping would change `n` quietly and make the run unpaired
            # against one from a complete copy.
            if not question or not passage or "answer" not in rec:
                raise ValueError(
                    f"{self.data_path}: row {i} is missing question, passage "
                    f"or answer. BoolQ rows carry all three.")
            items.append(EvalItem(
                # The dataset ships no id. Position in the checksummed file is
                # stable, and zero-padding keeps lexical order equal to file
                # order so the sort below is a no-op rather than a reshuffle.
                item_id=f"boolq_{i:05d}",
                query=question,
                item_type=ItemType.ANSWERABLE,
                gold_answer=_as_label(rec["answer"]),
                meta={"passage": passage},
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
            "Read the passage and answer the question. Begin your reply with "
            "the single word 'yes' or 'no'. You may add a brief reason after it.")
        messages: list[dict] = [{"role": "system", "content": system}]
        for shot in shots:
            messages.append({"role": "user", "content": _render(shot)})
            messages.append({"role": "assistant",
                             "content": str(shot.gold_answer or "")})
        messages.append({"role": "user", "content": _render(item)})
        return Prompted(messages=tuple(messages), choices=LABELS)

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

        got = _normalise(extraction.value)
        s.set("accuracy", 1.0 if got == item.gold_answer else 0.0)
        s.set("extraction_failed", False)
        s.set("extracted_via", extraction.via)
        # Neither label: an answer, so it stays in the denominator like the
        # other adapters' off-alphabet letters, and it is flagged beside the
        # score rather than folded into it.
        s.set("format_violation", got not in LABELS)
        return s


def _normalise(value: str | None) -> str:
    return (value or "").strip(_WRAP).lower()


def _render(item: EvalItem) -> str:
    """lm-eval-harness's BoolQ layout: passage, `Question: ...?`, `Answer:`."""
    q = item.query.rstrip()
    if not q.endswith("?"):
        q += "?"
    return f"{item.meta.get('passage', '')}\nQuestion: {q}\nAnswer:"


def _read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"BoolQ data not found at {path}. Fetch it with "
            f"`python main.py bench fetch --benchmark boolq`, or point "
            f"`source.ref` at a local JSONL file.")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
