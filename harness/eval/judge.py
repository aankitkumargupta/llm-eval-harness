"""
LLM-as-judge subsystem.

Scores the qualities gold-matching can't capture: faithfulness (grounded in
context, no fabrication), answer relevance, and completeness. Also supports
pairwise comparison, which feeds the Elo / Bradley-Terry leaderboard.

Integrity rules baked in:
  * ONE fixed judge model, temperature 0.
  * The judge must never grade its own family's output (self-preference bias);
    the caller passes the model-under-test so we can enforce this.
  * The judge is asked to return strict JSON so scores parse deterministically.

Judge latency is NOT a reported metric, so judge calls belong in the
high-throughput lane and are ideal for batching.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from ..clients.together_client import TogetherClient


# The judge rubric. Kept explicit so a profile can override it. Scores are 0-1.
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


def _family(model: str) -> str:
    """Coarse model family, e.g. 'meta-llama/Llama-3-70b' -> 'meta-llama'.
    Used to prevent a judge from grading its own family."""
    return model.split("/")[0].lower() if "/" in model else model.lower()


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a judge reply (strips code fences etc.)."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    # grab the first {...} block if there's surrounding prose
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        text = m.group()
    return json.loads(text)


@dataclass
class JudgeScores:
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    completeness: Optional[float] = None
    accuracy: Optional[float] = None


class Judge:
    def __init__(self, client: TogetherClient, judge_model: str):
        self.client = client
        self.judge_model = judge_model

    def _guard_self_judging(self, model_under_test: str) -> None:
        if _family(model_under_test) == _family(self.judge_model):
            raise ValueError(
                f"Judge '{self.judge_model}' shares a family with model under "
                f"test '{model_under_test}'. Pick a different judge to avoid "
                f"self-preference bias."
            )

    def score(self, model_under_test: str, question: str, context: str,
              answer: str, gold: Optional[str]) -> JudgeScores:
        """Pointwise scoring. Two judge calls: faithfulness (no gold needed) and
        the gold-referenced quality bundle (only if a gold answer exists)."""
        self._guard_self_judging(model_under_test)
        scores = JudgeScores()

        # 1) Faithfulness — grounded-ness, independent of gold.
        f_msg = [
            {"role": "system", "content": FAITHFULNESS_RUBRIC},
            {"role": "user", "content":
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\nANSWER:\n{answer}"},
        ]
        try:
            r = self.client.judge(self.judge_model, f_msg, max_tokens=300)
            scores.faithfulness = float(_extract_json(r.text).get("faithfulness"))
        except (ValueError, KeyError, json.JSONDecodeError, TypeError):
            scores.faithfulness = None  # unparseable judge reply -> leave unmeasured

        # 2) Quality bundle — needs a gold reference.
        if gold:
            q_msg = [
                {"role": "system", "content": QUALITY_RUBRIC},
                {"role": "user", "content":
                    f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                    f"ANSWER:\n{answer}\n\nGOLD:\n{gold}"},
            ]
            try:
                r = self.client.judge(self.judge_model, q_msg, max_tokens=400)
                d = _extract_json(r.text)
                scores.answer_relevance = float(d.get("answer_relevance"))
                scores.completeness = float(d.get("completeness"))
                scores.accuracy = float(d.get("accuracy"))
            except (ValueError, KeyError, json.JSONDecodeError, TypeError):
                pass
        return scores

    def pairwise(self, model_a: str, model_b: str, question: str, context: str,
                 answer_a: str, answer_b: str) -> Optional[str]:
        """
        Pairwise comparison for Elo. Returns "A", "B", or "tie". To cancel
        position bias, the caller should run each pair twice with answers
        swapped and only count a win if it's consistent across both orders.
        """
        self._guard_self_judging(model_a)
        self._guard_self_judging(model_b)
        msg = [
            {"role": "system", "content":
                "Compare two answers to the same question given the same context. "
                "Decide which is better grounded and more correct. Return ONLY "
                "JSON: {\"winner\": \"A\"|\"B\"|\"tie\", \"reason\": \"...\"}."},
            {"role": "user", "content":
                f"QUESTION:\n{question}\n\nCONTEXT:\n{context}\n\n"
                f"ANSWER A:\n{answer_a}\n\nANSWER B:\n{answer_b}"},
        ]
        try:
            r = self.client.judge(self.judge_model, msg, max_tokens=300)
            w = _extract_json(r.text).get("winner", "tie")
            return w if w in ("A", "B", "tie") else "tie"
        except (ValueError, KeyError, json.JSONDecodeError, TypeError):
            return None
