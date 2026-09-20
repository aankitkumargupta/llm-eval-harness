"""
LLM-as-judge subsystem.

Scores the qualities gold-matching can't capture, faithfulness, relevance,
completeness, and runs the pairwise comparisons behind the Elo leaderboard.

Integrity rules kept from the original design:
  * A fixed judge model at temperature 0.
  * The judge never grades its own family (self-preference bias); the caller
    passes the model under test so this can be enforced.
  * Strict JSON out, so scores parse deterministically.

What's new, and why each matters:

**Caching.** A judge verdict is a pure function of (judge model, rubric,
question, context, answer, gold). It was re-billed on every re-run of an
otherwise fully cached evaluation, and since the judge is a large model called
up to twice per item, it is typically the single largest line in the bill.

**An ensemble.** One judge is a single point of failure with its own biases. A
panel of judges lets you take the median and, more usefully, *measure their
disagreement*. High spread on an item is the honest signal that the item is
ambiguous rather than that the model failed, that's a dataset problem, and it
should surface as one.

**Position-bias-corrected pairwise.** Judges favour whichever answer they see
first. The original left correcting for this as a caller's note; here `pairwise`
runs both orders itself and reports a win only when the verdict survives the
swap. Inconsistent verdicts become ties, which is what they actually are.

**Judged abstention.** Substring matching on "I don't know" both misses real
refusals and fires on answers that merely contain the phrase. Available as an
opt-in second opinion where the abstention metric carries real weight.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass

FAITHFULNESS_RUBRIC = (
    "You are a strict evaluator. Given a QUESTION, the CONTEXT passages, and an "
    "ANSWER, score how FAITHFUL the answer is to the context. A faithful answer "
    "makes only claims supported by the context and invents nothing. "
    "Return ONLY JSON: {\"faithfulness\": <0.0-1.0>, \"reason\": \"...\"}."
)

QUALITY_RUBRIC = (
    "You are a strict evaluator. Given a QUESTION, the CONTEXT passages, the "
    "ANSWER, and the reference GOLD answer, score three things from 0.0 to 1.0: "
    "answer_relevance (does the answer address the question), completeness (does "
    "it cover all relevant information), and accuracy (does it agree with the "
    "gold answer). Return ONLY JSON: {\"answer_relevance\": x, "
    "\"completeness\": y, \"accuracy\": z, \"reason\": \"...\"}."
)

ABSTENTION_RUBRIC = (
    "Decide whether the ANSWER is a refusal / statement of inability to answer "
    "(for example 'I don't know', 'the context does not say'), as opposed to an "
    "actual attempt to answer the question. An answer that states a fact AND "
    "notes some detail is missing is NOT a refusal. "
    "Return ONLY JSON: {\"abstained\": true|false}."
)

PAIRWISE_RUBRIC = (
    "Compare two answers to the same question given the same context. Decide "
    "which is better grounded and more correct. Judge only on substance, never "
    "on length or ordering. Return ONLY JSON: "
    "{\"winner\": \"A\"|\"B\"|\"tie\", \"reason\": \"...\"}."
)


def _family(model: str) -> str:
    """Coarse model family: 'meta-llama/Llama-3-70b' -> 'meta-llama'.

    Also strips a `provider:` routing prefix so 'anthropic:claude-opus-5' and a
    bare 'claude-opus-5' resolve to the same family, otherwise the
    self-judging guard is trivially defeated by how the model happens to be
    addressed in the config.
    """
    m = model.split(":", 1)[1] if ":" in model and "/" not in model.split(":", 1)[0] else model
    if "/" in m:
        return m.split("/")[0].lower()
    # Bare ids ('claude-opus-5', 'gpt-4o') have no vendor prefix, so use the
    # leading token as a family proxy.
    return m.split("-")[0].lower()


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a judge reply (strips code fences etc.)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group()
    return json.loads(text)


def _as_score(value) -> float | None:
    """Coerce a judge's field to a 0-1 float, or None if it isn't usable.

    Judges sometimes answer "0.8/1.0", "80%", or true/false. Clamping to [0,1]
    matters because an out-of-range score silently skews every mean built on it.
    """
    if value is None or isinstance(value, bool):
        return 1.0 if value is True else (0.0 if value is False else None)
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        s = value.strip().rstrip("%")
        m = re.match(r"^-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        v = float(m.group())
        if "%" in value:
            v /= 100.0
        elif v > 1.0:
            v = v / 10.0 if v <= 10.0 else 1.0  # "8/10" style
        return max(0.0, min(1.0, v))
    return None


@dataclass
class JudgeScores:
    faithfulness: float | None = None
    answer_relevance: float | None = None
    completeness: float | None = None
    accuracy: float | None = None
    abstained: bool | None = None
    # Spread across ensemble members. High disagreement means the *item* is
    # ambiguous, not that the model did badly, worth surfacing separately.
    disagreement: float | None = None
    n_judges: int = 1
    parse_failures: int = 0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Judge:
    """Pointwise and pairwise LLM grading, cached and optionally ensembled."""

    def __init__(self, client, judge_model: str, cache=None,
                 ensemble: list[str] | None = None,
                 max_tokens: int = 400, enforce_family_guard: bool = True):
        self.client = client
        self.judge_model = judge_model
        self.cache = cache
        # An ensemble of one is just the single judge, same code path, no
        # special-casing anywhere downstream.
        self.ensemble = list(ensemble) if ensemble else [judge_model]
        self.max_tokens = max_tokens
        self.enforce_family_guard = enforce_family_guard

    # ------------------------------------------------------------------ #
    def _guard_self_judging(self, model_under_test: str) -> None:
        if not self.enforce_family_guard:
            return
        mut = _family(model_under_test)
        for jm in self.ensemble:
            if _family(jm) == mut:
                raise ValueError(
                    f"Judge '{jm}' shares a family with model under test "
                    f"'{model_under_test}'. Pick a different judge to avoid "
                    f"self-preference bias, or drop that model from the panel."
                )

    def _ask(self, judge_model: str, rubric: str, payload: str,
             question: str, context: str, answer: str,
             gold: str | None) -> dict | None:
        """One cached judge call. Returns the parsed dict, or None if unusable."""
        key = None
        if self.cache is not None:
            key = self.cache.key_for_judge(judge_model, rubric, question,
                                           context, answer, gold)
            hit = self.cache.get(key)
            if hit is not None:
                return hit.get("parsed")

        messages = [{"role": "system", "content": rubric},
                    {"role": "user", "content": payload}]
        try:
            r = self.client.judge(judge_model, messages, max_tokens=self.max_tokens)
            parsed = _extract_json(r.text)
        except Exception:  # noqa: BLE001, an unusable verdict is "not measured"
            parsed = None

        if self.cache is not None and key is not None and parsed is not None:
            # Only successes are cached. Caching a parse failure would make a
            # transient bad reply permanent for the life of the cache.
            self.cache.set(key, {"parsed": parsed})
        return parsed

    # ------------------------------------------------------------------ #
    def score(self, model_under_test: str, question: str, context: str,
              answer: str, gold: str | None,
              want_abstention: bool = False) -> JudgeScores:
        """Pointwise scoring across the judge panel.

        Faithfulness is asked without the gold answer on purpose: it measures
        grounding in the retrieved context, which is a different question from
        agreement with a reference, and mixing them lets a model that guessed
        the right answer from parametric memory score as faithful.
        """
        self._guard_self_judging(model_under_test)
        scores = JudgeScores(n_judges=len(self.ensemble))

        faith_vals: list[float] = []
        rel_vals: list[float] = []
        comp_vals: list[float] = []
        acc_vals: list[float] = []
        failures = 0

        for jm in self.ensemble:
            f_payload = (f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                         f"ANSWER:\n{answer}")
            d = self._ask(jm, FAITHFULNESS_RUBRIC, f_payload, question, context,
                          answer, None)
            if d is None:
                failures += 1
            else:
                v = _as_score(d.get("faithfulness"))
                if v is not None:
                    faith_vals.append(v)

            if gold:
                q_payload = (f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                             f"ANSWER:\n{answer}\n\nGOLD:\n{gold}")
                d = self._ask(jm, QUALITY_RUBRIC, q_payload, question, context,
                              answer, gold)
                if d is None:
                    failures += 1
                else:
                    for key, bucket in (("answer_relevance", rel_vals),
                                        ("completeness", comp_vals),
                                        ("accuracy", acc_vals)):
                        v = _as_score(d.get(key))
                        if v is not None:
                            bucket.append(v)

        # Median, not mean: one judge returning a wild value shouldn't drag the
        # panel's verdict with it.
        scores.faithfulness = statistics.median(faith_vals) if faith_vals else None
        scores.answer_relevance = statistics.median(rel_vals) if rel_vals else None
        scores.completeness = statistics.median(comp_vals) if comp_vals else None
        scores.accuracy = statistics.median(acc_vals) if acc_vals else None
        scores.parse_failures = failures

        if len(acc_vals) > 1:
            scores.disagreement = statistics.pstdev(acc_vals)
        elif len(faith_vals) > 1:
            scores.disagreement = statistics.pstdev(faith_vals)

        if want_abstention:
            scores.abstained = self.judge_abstention(question, answer)
        return scores

    def judge_abstention(self, question: str, answer: str) -> bool | None:
        """Ask the judge whether the answer is a refusal.

        More reliable than substring matching, which both misses paraphrased
        refusals and false-positives on answers that merely mention uncertainty.
        Costs a call, so it's opt-in via the `abstention_judge` profile flag.
        """
        payload = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        d = self._ask(self.ensemble[0], ABSTENTION_RUBRIC, payload, question,
                      "", answer, None)
        if d is None:
            return None
        val = d.get("abstained")
        return bool(val) if isinstance(val, (bool, int)) else None

    # ------------------------------------------------------------------ #
    def pairwise(self, model_a: str, model_b: str, question: str, context: str,
                 answer_a: str, answer_b: str,
                 correct_position_bias: bool = True) -> str | None:
        """Compare two answers. Returns "A", "B", "tie", or None if unusable.

        With `correct_position_bias` the comparison runs in both orders and a
        win counts only if it survives the swap. Judges have a measurable
        preference for whichever answer appears first; without this correction
        an Elo table built from single-order verdicts partly ranks *argument
        position*, and it does so invisibly.
        """
        self._guard_self_judging(model_a)
        self._guard_self_judging(model_b)

        first = self._compare(question, context, answer_a, answer_b)
        if not correct_position_bias:
            return first
        # Swap the operands; a consistent judge must now say the mirror image.
        second = self._compare(question, context, answer_b, answer_a)
        if first is None or second is None:
            return None
        mirrored = {"A": "B", "B": "A", "tie": "tie"}[second]
        return first if first == mirrored else "tie"

    def _compare(self, question: str, context: str, first: str,
                 second: str) -> str | None:
        payload = (f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                   f"ANSWER A:\n{first}\n\nANSWER B:\n{second}")
        d = self._ask(self.ensemble[0], PAIRWISE_RUBRIC, payload, question,
                      context, f"{first}\x00{second}", None)
        if d is None:
            return None
        w = d.get("winner", "tie")
        return w if w in ("A", "B", "tie") else "tie"
