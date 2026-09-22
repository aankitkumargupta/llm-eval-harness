"""
The generic adapter: a benchmark you declare rather than program.

Every other adapter in this directory exists because a published dataset has a
shape nobody chose, ARC's `{"text": [...], "label": [...]}`, GSM8K's `#### 42`,
IFEval's constraint objects. Those need code. Your *own* benchmark does not:
you decide the file, so you can write the shape the harness already reads, and
then the only thing separating "a JSONL on disk" from "a benchmark with
chance-adjusted accuracy, paired significance and metered cost" is a YAML file.

CLAUDE.md §11 (I11) is the reason this module exists:

    "Adding a profile, model, provider, benchmark, or weight is YAML. If it
     needs Python, the seam is wrong, fix the seam."

Before this adapter, adding a benchmark of a *new* shape meant one YAML spec
**and** one adapter module **and** a registry line. For a shape the harness
already understands that was Python for nothing, so the seam was wrong. A spec
now names `adapter: custom`, and the registry hands it this class.

What it covers, and what it does not
------------------------------------
Four shapes, chosen because they are what a private evaluation set almost
always is:

* `multiple_choice` , a question and a list of options, scored on the letter.
* `short_answer` ,   a question and a reference string.
* `math` ,           a question and a number, scored by numeric equivalence so
                     `1,000`, `1000` and `1000.0` are one answer.
* `classify` ,       a text and exactly one label from a declared set.

Anything with structure beyond that (code execution, retrieval, multi-turn,
programmatic constraints) still needs its own adapter, and should: those are
the cases where a generic reader would quietly score the wrong thing. This
module would rather refuse a shape than guess it.

Two properties it inherits by construction, not by care:

* **Failure is not wrongness (I7).** Extraction runs the spec's own chain and
  a miss returns `accuracy=None` with `extraction_failed=True`, never a zero.
* **Determinism (I10).** `load()` sorts by item id and samples with
  `random.Random(seed)`, so the item set is a pure function of
  `(seed, file contents)` and two models in one run see the identical items in
  the identical order (I1).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import numeric_equal, run_chain
from ..spec import BenchmarkSpec

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: The shapes this adapter will read. A spec whose family is not here is a
#: spec that needs its own adapter, and `validate()` says so rather than
#: scoring it under a shape nobody chose.
SUPPORTED_FAMILIES = ("multiple_choice", "short_answer", "math",
                      "multilingual", "classify")

#: Default field names. Every one is overridable from the spec's `fields:`
#: block, so an existing file does not have to be rewritten to be evaluated.
DEFAULT_FIELDS = {
    "id": "id",
    "question": "question",
    "options": "options",
    "answer": "answer",
}

DEFAULT_SYSTEM = {
    "multiple_choice": ("Answer the multiple-choice question. Reply with the "
                        "option letter, ending your response with 'Answer: X'."),
    "short_answer": ("Answer the question directly and briefly. Give the answer "
                     "only, with no explanation and no preamble."),
    "math": ("Solve the problem. Reason if you need to, then end your response "
             "with the final number on its own line as 'Answer: <number>'."),
    "multilingual": ("Answer the question in the language it is asked in. Give "
                     "the answer only, with no explanation."),
    "classify": ("Classify the text into exactly one label from the allowed "
                 "list. Output only the label, in lowercase, with nothing else."),
}


class CustomBenchmarkError(ValueError):
    """The declared shape and the data file disagree.

    Raised at load, loudly and early, rather than at score time: a row that
    cannot be read is a broken benchmark, and silently dropping it would make
    the item set depend on the data's mistakes instead of on `(seed, file)`,
    which breaks pairing (I1) the moment two models load it separately.
    """


class CustomAdapter:
    """§10.3 adapter driven entirely by its spec.

    Holds no benchmark knowledge of its own: every decision (which fields to
    read, how to prompt, how to extract, what chance level to report) comes
    from the spec, which is what makes one class enough for every private set
    of these shapes.
    """

    id = "custom"

    def __init__(self, spec: BenchmarkSpec, data_path: str | None = None):
        self.spec = spec
        self.data_path = data_path or spec.source.ref
        self.family = spec.family
        if self.family not in SUPPORTED_FAMILIES:
            raise CustomBenchmarkError(
                f"the generic adapter reads {list(SUPPORTED_FAMILIES)}, not "
                f"{self.family!r}. A benchmark of that family needs its own "
                f"adapter in harness/bench/adapters/ so nothing is guessed.")
        self.fields = {**DEFAULT_FIELDS, **(spec.fields or {})}
        self.labels = tuple(spec.label_set)
        if self.family == "classify" and not self.labels:
            raise CustomBenchmarkError(
                "a classify benchmark needs `label_set` in its spec; without "
                "the allowed labels a reply cannot be told from a wrong one.")

    # ------------------------------------------------------------------ #
    def load(self, *, seed: int | None = None, limit: int | None = None
             ) -> Sequence[EvalItem]:
        seed = self.spec.sampling.seed if seed is None else seed
        limit = self.spec.sampling.limit if limit is None else limit

        f = self.fields
        items: list[EvalItem] = []
        for i, rec in enumerate(_read_jsonl(self.data_path)):
            query = str(rec.get(f["question"], "") or "").strip()
            if not query:
                raise CustomBenchmarkError(
                    f"row {i + 1} of {self.data_path} has no "
                    f"{f['question']!r}. Every item needs its question; set "
                    f"`fields.question` in the spec if yours is named "
                    f"differently.")
            raw_answer = rec.get(f["answer"])
            if raw_answer is None or str(raw_answer).strip() == "":
                raise CustomBenchmarkError(
                    f"row {i + 1} of {self.data_path} has no "
                    f"{f['answer']!r}. An item with no reference answer cannot "
                    f"be scored, and dropping it silently would change the "
                    f"item set behind your back.")

            item_id = str(rec.get(f["id"], "") or f"item_{i + 1:05d}")
            meta = {k: v for k, v in (rec.get("meta") or {}).items()}

            if self.family == "multiple_choice":
                options = _options(rec.get(f["options"]), i, self.data_path,
                                   f["options"])
                letters = [LETTERS[j] for j in range(len(options))]
                gold = _gold_letter(raw_answer, options, letters, i)
                meta.update({"options": options, "letters": letters})
            elif self.family == "classify":
                gold = str(raw_answer).strip()
                if gold not in self.labels:
                    raise CustomBenchmarkError(
                        f"row {i + 1} of {self.data_path} has answer {gold!r}, "
                        f"which is not in label_set {list(self.labels)}. A gold "
                        f"answer outside the set can never be reached, so every "
                        f"such row would score zero for every model.")
                meta.setdefault("labels", list(self.labels))
            else:
                gold = str(raw_answer).strip()

            items.append(EvalItem(
                item_id=item_id,
                query=query,
                item_type=ItemType.ANSWERABLE,
                gold_answer=gold,
                meta=meta,
            ))

        if not items:
            raise CustomBenchmarkError(
                f"no rows read from {self.data_path}. A benchmark with no "
                f"items is not an empty result, it is a broken path.")

        ids = [it.item_id for it in items]
        if len(set(ids)) != len(ids):
            dupes = sorted({x for x in ids if ids.count(x) > 1})[:5]
            raise CustomBenchmarkError(
                f"duplicate item ids in {self.data_path}: {dupes}. Ids key the "
                f"paired comparison, so a duplicate silently pairs one model's "
                f"answer with another model's item.")

        # Sorted first, sampled second: the item set must be a pure function of
        # (seed, contents), never of the order someone happened to write rows.
        items.sort(key=lambda it: it.item_id)
        if limit is not None and limit < len(items):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(items)), limit))
            items = [items[j] for j in idx]
        return items

    # ------------------------------------------------------------------ #
    def prompt(self, item: EvalItem, shots: Sequence[EvalItem] = ()) -> Prompted:
        system = self.spec.prompt.system or DEFAULT_SYSTEM.get(
            self.family, DEFAULT_SYSTEM["short_answer"])
        if self.family == "classify":
            system = f"{system} Allowed labels: {', '.join(self.labels)}."

        messages: list[dict] = [{"role": "system", "content": system}]
        for shot in shots:
            messages.append({"role": "user", "content": self._render(shot)})
            messages.append({"role": "assistant",
                             "content": self._shot_answer(shot)})
        messages.append({"role": "user", "content": self._render(item)})
        choices = tuple(item.meta.get("letters", ())) if \
            self.family == "multiple_choice" else ()
        return Prompted(messages=tuple(messages), choices=choices)

    # ------------------------------------------------------------------ #
    def extract(self, raw: str) -> Extraction:
        return run_chain(raw, self.spec.scoring.chain)

    # ------------------------------------------------------------------ #
    def score(self, item: EvalItem, extraction: Extraction) -> ScoreSet:
        s = ScoreSet()
        if extraction.failed:
            # accuracy=None, never 0.0 (I7): unreadable is not wrong, and the
            # report prints the failure rate beside the score rather than
            # inside it.
            s.set("accuracy", None)
            s.set("extraction_failed", True)
            s.set("extraction_reason", extraction.reason)
            return s

        got = (extraction.value or "").strip()
        s.set("extraction_failed", False)
        s.set("extracted_via", extraction.via)

        if self.family == "multiple_choice":
            letters = item.meta.get("letters", [])
            letter = got.upper()[:1]
            s.set("accuracy", 1.0 if letter == item.gold_answer else 0.0)
            s.set("format_violation", letter not in letters)
        elif self.family == "math":
            s.set("accuracy", 1.0 if numeric_equal(got, item.gold_answer) else 0.0)
            s.set("format_violation", _canonical(got) == "" )
        elif self.family == "classify":
            label = got.strip().lower()
            allowed = [x.lower() for x in self.labels]
            s.set("accuracy", 1.0 if label == item.gold_answer.lower() else 0.0)
            s.set("format_violation", label not in allowed)
        else:
            # short_answer / multilingual: a number on either side is compared
            # as a number, otherwise case, surrounding punctuation and runs of
            # whitespace are normalised away. Nothing fuzzier: a near-miss
            # scored as a hit is the same lie as a parse failure scored as a
            # miss, in the other direction.
            hit = numeric_equal(got, item.gold_answer) or \
                _canonical(got) == _canonical(item.gold_answer)
            s.set("accuracy", 1.0 if hit else 0.0)
            s.set("format_violation", False)
        return s

    # ------------------------------------------------------------------ #
    def _render(self, item: EvalItem) -> str:
        if self.family == "multiple_choice":
            letters = item.meta.get("letters", [])
            options = item.meta.get("options", [])
            lines = [item.query, ""]
            lines += [f"{letters[i] if i < len(letters) else LETTERS[i]}. {o}"
                      for i, o in enumerate(options)]
            return "\n".join(lines)
        return item.query

    def _shot_answer(self, shot: EvalItem) -> str:
        if self.family in ("multiple_choice", "math"):
            return f"Answer: {shot.gold_answer}"
        return str(shot.gold_answer)


# --------------------------------------------------------------------------- #
#  Helpers, pure
# --------------------------------------------------------------------------- #
def _canonical(text: str) -> str:
    """Case, outer punctuation and whitespace runs removed. Nothing else."""
    t = re.sub(r"\s+", " ", str(text or "")).strip().casefold()
    return t.strip(" .,:;!?\"'()[]")


def _options(raw, row: int, path: str, name: str) -> list[str]:
    """Read the option list, accepting the two shapes people actually write."""
    if isinstance(raw, dict):                      # {"A": "...", "B": "..."}
        return [str(raw[k]) for k in sorted(raw)]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    raise CustomBenchmarkError(
        f"row {row + 1} of {path} has no usable {name!r}: a multiple-choice "
        f"item needs a list of options (or an object keyed by letter).")


def _gold_letter(raw_answer, options: list[str], letters: list[str],
                 row: int) -> str:
    """Resolve the gold answer to a letter, whichever way it was written.

    A private set writes its answer as a letter, as a 0- or 1-based index, or
    as the option text itself. All three are accepted and normalised the same
    way the options are, because normalising one side and not the other is a
    quiet way to score correct answers wrong.
    """
    ans = str(raw_answer).strip()
    if len(ans) == 1 and ans.upper() in letters:
        return ans.upper()
    if ans.isdigit():
        i = int(ans)
        # 1-based first, since a human-written set almost always counts from 1;
        # 0-based is accepted only when 1-based would fall off the end.
        if 1 <= i <= len(letters):
            return letters[i - 1]
        if 0 <= i < len(letters):
            return letters[i]
    for j, opt in enumerate(options):
        if _canonical(opt) == _canonical(ans):
            return letters[j]
    raise CustomBenchmarkError(
        f"row {row + 1}: answer {ans!r} is not one of the options, an option "
        f"letter, or an option number. Every item's answer must name an "
        f"option, or it can never be reached and scores zero for every model.")


def _read_jsonl(path: str) -> Iterator[dict]:
    p = Path(path)
    if not p.exists():
        raise CustomBenchmarkError(
            f"benchmark data not found at {path}. `source.kind: local` means "
            f"the file is yours: put it there, or fix `source.ref` in the spec.")
    with open(p, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise CustomBenchmarkError(
                    f"{path} line {n} is not valid JSON: {e}. Each line is one "
                    f"complete JSON object (JSONL), not a pretty-printed file."
                ) from e
            if not isinstance(rec, dict):
                raise CustomBenchmarkError(
                    f"{path} line {n} is a {type(rec).__name__}, not an object.")
            yield rec
