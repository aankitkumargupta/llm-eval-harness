"""
HellaSwag: commonsense sentence completion, four-way multiple choice.

Research note
=============

**Canonical source and version.** Zellers, Holtzman, Bisk, Farhadi and Choi,
"HellaSwag: Can a Machine Really Finish Your Sentence?", ACL 2019
(https://arxiv.org/abs/1905.07830). Data: https://github.com/rowanz/hellaswag
(`hellaswag_train.jsonl`, `hellaswag_val.jsonl`, `hellaswag_test.jsonl`).
There is one public release of the data. The HuggingFace mirror used here is
`Rowan/hellaswag` (https://huggingface.co/datasets/Rowan/hellaswag), splits
train 39,905 / validation 10,042 / test 10,003. The **test split ships with
empty labels** (held out for the original leaderboard), so the scored split is
`validation`, exactly as lm-eval-harness configures it (`test_split:
validation`). The spec pins the checksum of the first 1,000 validation rows
as served (`bench fetch --benchmark hellaswag --limit 1000`; the spec file
says why), so a run here scores a 1,000-row prefix, not the full 10,042.

**Licence.** MIT, on both the repository and the dataset card. Commercial use
permitted; redistribution of the 20-row fixture under `tests/bench/fixtures/`
is permitted with the notice retained.

**Prompt format and metric as the authors define them.** Given a context
(a caption from ActivityNet or a WikiHow passage), pick which of four endings
is the real continuation; the three distractors are machine-generated and
adversarially filtered. The metric is accuracy over the four endings, chance
0.25. The paper evaluates fine-tuned classifiers (BERT, GPT and others), so
its own numbers are neither log-likelihood nor generative in the LLM sense.

**Prompt format and metric as lm-eval-harness defines them** (`tasks/hellaswag`).
The query is `activity_label + ": " + ctx_a + " " + ctx_b.capitalize()`; the
query and every ending pass through `preprocess`: strip, replace `" [title]"`
with `". "`, delete every `[...]` span, collapse one level of double spaces.
Scoring is log-likelihood of each ending given the query, reported as `acc`
and `acc_norm` (log-likelihood divided by the ending's byte length). Zero-shot
by default; the Open LLM Leaderboard v1 used 10-shot `acc_norm`.

**Known scoring pitfalls.**

* `acc` and `acc_norm` differ by several points on the same model. A quoted
  number without the suffix is ambiguous.
* Bracket artefacts. The WikiHow half of the data carries `[title]`,
  `[header]`, `[step]` and `[substeps]` tokens plus double spaces, and `ctx_b`
  starts lower-case because it was split mid-sentence. Left in, they leak into
  the prompt and the endings. `load()` applies lm-eval-harness's cleaning
  (`clean_text` below) so the text a model sees here is the text the
  published numbers were computed on. One deliberate addition: each cleaned
  ending is stripped again, because a `[header]` at the start of an ending
  leaves a leading space that lm-eval concatenates onto the context but this
  adapter renders after a letter label.
* `label` is a **string** (`"0"`..`"3"`) in the HF rows and the empty string on
  test rows. A row whose label is not a valid index is skipped, never scored
  as wrong. The validation split has a label on every row.
* Generative letter choice (this spec) is a different measurement from
  log-likelihood ranking. It needs no logprobs, but a model that does not
  follow the `Answer: X` format becomes an extraction failure, reported as its
  own rate (I7), and the length prior that `acc_norm` corrects for does not
  exist here. The mode is recorded in the spec and therefore in `spec_hash`;
  the reporter refuses to compare across it (§10.1).
* `A` is also the English article, so the last-resort `last_capital_letter`
  link can read an article as an answer. `extracted_via` records which link
  fired so a run that leaned on the fallback is visible in the report.
* Contamination. The validation split is widely present in web crawls.
  Option shuffling is declared as a perturbation; the delta is evidence, never
  a verdict (§10.6).

**Are published numbers log-likelihood or generative?** Log-likelihood.
GPT-3's HellaSwag figures (Brown et al., 2020, Table 3.3: 78.9 zero-shot,
79.3 few-shot) use length-normalised likelihood over the endings; the Open LLM
Leaderboard v1 figures are 10-shot `acc_norm`. Nothing this adapter emits in
generative mode is comparable to them.

Purity
======
`load`, `prompt`, `extract` and `score` are pure: no network (the fetcher owns
that), no clock, no global RNG. Sampling is a function of `(seed, limit)` over
the id-sorted item list, so two models in one run see the same items in the
same order (I1, I10).
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

from ..contracts import EvalItem, Extraction, ItemType, Prompted, ScoreSet
from ..extract import run_chain
from ..spec import BenchmarkSpec

LETTERS = "ABCD"

_BRACKET_SPAN = re.compile(r"\[.*?\]")


def clean_text(text: str) -> str:
    """lm-eval-harness's `preprocess` for HellaSwag, plus a final strip.

    Step for step: strip; `" [title]"` becomes `". "` (the WikiHow title marker
    ends the previous sentence); every `[...]` span is deleted (`[header]`,
    `[step]`, `[substeps]` and any other artefact); one level of double space
    collapses to one. The trailing strip is this adapter's addition, explained
    in the module docstring.
    """
    text = str(text).strip()
    text = text.replace(" [title]", ". ")
    text = _BRACKET_SPAN.sub("", text)
    text = text.replace("  ", " ")
    return text.strip()


def build_query(rec: dict) -> str:
    """The context a model is asked to continue, in lm-eval-harness's form.

    `activity_label: ctx_a Ctx_b`. Falls back to the dataset's pre-joined `ctx`
    when the split fields are absent, so a hand-written fixture row that only
    carries `ctx` still loads.
    """
    ctx_a = rec.get("ctx_a")
    ctx_b = rec.get("ctx_b")
    if ctx_a is None and ctx_b is None:
        ctx = str(rec.get("ctx", ""))
    else:
        ctx = f"{ctx_a or ''} {str(ctx_b or '').capitalize()}".strip()
    label = str(rec.get("activity_label", "") or "").strip()
    return clean_text(f"{label}: {ctx}" if label else ctx)


class HellaSwagAdapter:
    """§10.3 adapter for HellaSwag's four-way ending choice."""

    id = "hellaswag"

    def __init__(self, spec: BenchmarkSpec, data_path: str | None = None):
        self.spec = spec
        self.data_path = data_path or spec.source.ref

    # ------------------------------------------------------------------ #
    def load(self, *, seed: int | None = None, limit: int | None = None
             ) -> Sequence[EvalItem]:
        seed = self.spec.sampling.seed if seed is None else seed
        limit = self.spec.sampling.limit if limit is None else limit

        items: list[EvalItem] = []
        seen: set[str] = set()
        for i, rec in enumerate(_read_jsonl(self.data_path)):
            endings = [clean_text(e) for e in (rec.get("endings") or [])]
            if not endings:
                continue
            letters = list(LETTERS[:len(endings)])

            # The gold is an INDEX carried as a string. Anything that is not a
            # valid index (the empty string on test rows, a stray value) means
            # the row cannot be scored, and an unscorable row is skipped rather
            # than folded into the wrong-answer count.
            raw_label = str(rec.get("label", "") if rec.get("label") is not None
                            else "").strip()
            if not raw_label.isdigit() or not 0 <= int(raw_label) < len(letters):
                continue
            gold = letters[int(raw_label)]

            item_id = str(rec.get("ind", i))
            if item_id in seen:
                # Duplicate ids collapse two items into one downstream and
                # corrupt pairing (I1); make the second one distinct and visible.
                item_id = f"{item_id}-{i}"
            seen.add(item_id)

            items.append(EvalItem(
                item_id=item_id,
                query=build_query(rec),
                item_type=ItemType.ANSWERABLE,
                gold_answer=gold,
                meta={
                    "options": endings,
                    "letters": letters,
                    "activity_label": str(rec.get("activity_label", "") or ""),
                    "split_type": str(rec.get("split_type", "") or ""),
                },
            ))

        items.sort(key=lambda it: it.item_id)      # stable before sampling
        if limit is not None and limit < len(items):
            rng = random.Random(seed)
            idx = sorted(rng.sample(range(len(items)), limit))
            items = [items[i] for i in idx]
        return items

    # ------------------------------------------------------------------ #
    def prompt(self, item: EvalItem, shots: Sequence[EvalItem] = ()) -> Prompted:
        system = self.spec.prompt.system or (
            "Choose the ending that most plausibly continues the text. Reply "
            "with the option letter, ending your response with 'Answer: X'.")
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
            f"HellaSwag data not found at {path}. Fetch it with "
            f"`python main.py bench fetch --benchmark hellaswag`.")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
