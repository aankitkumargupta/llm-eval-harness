"""
A deterministic offline provider, for CI and for smoke-testing the run path.

CLAUDE.md asks for this twice: §5 wants "a full fake-provider matrix run, real
Parquet, then reported, gated, exported", and §12's standing gate runs
`harness run --profile fake_smoke` before any phase may be declared done. Both
are impossible without a provider that needs no key and no network.

It is a *provider*, not a test double, and it lives in `harness/clients/`
alongside the real adapters for a reason: if it lived in `tests/`, the code
path CI exercises would differ from the code path a user runs, and the thing
most worth testing, routing, metering, retry, the trace write, would be the
part the test bypassed.

**It is honest about being fake.** Every response carries real usage numbers
(counted, not invented) so the cost meter works end to end, and `provider`
reads `fake` on every row, so a run against it can never be mistaken in the
store for a run against a real model. Pricing for `fake:*` is zero because the
calls genuinely cost nothing, the one case where a zero cost is a measurement
rather than a missing usage block (I3).

Answer quality is a deterministic function of `(model, prompt)`. Different
fake models score differently and reproducibly, which is what makes a
significance test over a fake matrix meaningful rather than a formality.
"""

from __future__ import annotations

import hashlib
import re
import time

from .base import Capability, EmbedResult, GenResult, ProviderInfo

#: Chars per token for the *reported* usage. This is a real count of the text
#: the fake produced, so `usage_estimated` stays False: nothing is missing, the
#: provider simply is the tokenizer.
_CHARS_PER_TOKEN = 4


class FakeClient:
    """Offline provider. `fake:<name>` routes here.

    `accuracy` controls how often it answers correctly, as a fraction. The
    decision is per-item and deterministic: `sha256(model, prompt)` mapped
    into [0, 1) and compared against the threshold, so the same model gets the
    same items right on every run, and two fake models with different names
    disagree on a stable, arbitrary subset. That is exactly the structure a
    paired significance test needs to have something to find.
    """

    def __init__(self, provider: str = "fake", *, accuracy: float = 0.75,
                 latency_ms: float = 4.0, meter=None, pricing=None,
                 truncate_after: int | None = None, **_ignored):
        self.provider = provider
        self.accuracy = accuracy
        self.latency_ms = latency_ms
        self.meter = meter
        self.pricing = pricing
        self.truncate_after = truncate_after
        self.info = ProviderInfo(name="fake", base_url="local://fake",
                                 supports_rerank=False, supports_embeddings=True,
                                 supports_seed=True)
        self.calls = 0

    # ------------------------------------------------------------------ #
    def supports(self, capability: Capability) -> bool:
        return capability in (Capability.GENERATE, Capability.EMBED)

    def preflight(self, models, embedding_model="", rerank_model=""):
        """Nothing to check: no key, no network, no model catalogue."""
        return []

    # ------------------------------------------------------------------ #
    def generate(self, model: str, messages: list[dict], **kw) -> GenResult:
        t0 = time.perf_counter()
        self.calls += 1
        prompt = "\n".join(str(m.get("content", "")) for m in messages)

        text = self._answer(model, prompt, messages)
        finish = "stop"
        max_tokens = kw.get("max_tokens")
        cap = self.truncate_after or (
            max_tokens * _CHARS_PER_TOKEN if max_tokens else None)
        if cap is not None and len(text) > cap:
            text = text[:cap]
            finish = "length"

        p_tok = max(1, len(prompt) // _CHARS_PER_TOKEN)
        c_tok = max(1, len(text) // _CHARS_PER_TOKEN)

        if self.meter is not None:
            usd = 0.0
            if self.pricing is not None:
                try:
                    usd = self.pricing.generation_cost(
                        f"{self.provider}:{model}", p_tok, c_tok)
                except (KeyError, Exception):   # noqa: BLE001 - unpriced fake
                    usd = 0.0
            record = getattr(self.meter, "record_generation", None)
            if record:
                record(usd, p_tok, c_tok)

        return GenResult(
            text=text, prompt_tokens=p_tok, completion_tokens=c_tok,
            latency_ms=(time.perf_counter() - t0) * 1000.0 + self.latency_ms,
            finish_reason=finish, model=model,
            usage_estimated=False,      # counted, not guessed
        )

    def judge(self, model: str, messages: list[dict], **kw) -> GenResult:
        return self.generate(model, messages, **kw)

    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        """Deterministic 16-dim vectors from a hash. Enough for the run path to
        work; not enough for retrieval quality to mean anything, which is why
        no fake RAG number should ever be reported as a result."""
        vectors = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            vectors.append([b / 255.0 for b in h[:16]])
        tokens = max(1, sum(len(t) for t in texts) // _CHARS_PER_TOKEN)
        return EmbedResult(vectors=vectors, prompt_tokens=tokens,
                           usage_estimated=False)

    # ------------------------------------------------------------------ #
    def _roll(self, model: str, prompt: str) -> float:
        h = hashlib.sha256(f"{model}||{prompt}".encode()).digest()
        return int.from_bytes(h[:8], "big") / float(1 << 64)

    def _answer(self, model: str, prompt: str, messages: list[dict]) -> str:
        """Answer in the format the prompt implies, right `accuracy` of the time.

        Format sniffing is deliberately crude, it exists so the fake exercises
        each adapter's extraction chain, not so it simulates a model.
        """
        correct = self._roll(model, prompt) < self.accuracy

        # Multiple choice: options rendered as "A. ..." lines.
        opts = re.findall(r"^([A-J])\.\s", prompt, re.MULTILINE)
        if opts:
            gold = self._mc_gold(prompt, opts)
            pick = gold if correct else self._other(opts, gold, model, prompt)
            return f"Considering each option in turn.\nAnswer: {pick}"

        # Numeric / maths: the prompt asks "how many".
        if re.search(r"####|how many|what is \d+", prompt, re.IGNORECASE):
            n = self._arith(prompt)
            if n is not None:
                return (f"Working it through step by step.\n"
                        f"#### {n if correct else n + 7}")

        # Instruction following: obey, or break one constraint.
        return self._instruction_answer(prompt, correct)

    def _mc_gold(self, prompt: str, opts: list[str]) -> str:
        """The fake cannot see gold labels, so it picks a stable pseudo-gold.

        Scores land near `accuracy` in aggregate rather than exactly, which is
        realistic and keeps anyone from mistaking a fake run for a real result.
        """
        h = hashlib.sha256(prompt.encode()).digest()
        return opts[h[0] % len(opts)]

    def _other(self, opts: list[str], gold: str, model: str, prompt: str) -> str:
        alts = [o for o in opts if o != gold] or opts
        h = hashlib.sha256(f"{model}{prompt}wrong".encode()).digest()
        return alts[h[0] % len(alts)]

    def _arith(self, prompt: str) -> int | None:
        m = re.search(r"saves (\d+) rupees each day for (\d+) days.*?finds (\d+)",
                      prompt, re.DOTALL)
        if m:
            a, b, c = (int(x) for x in m.groups())
            return a * b + c
        m = re.search(r"what is (\d+) multiplied by (\d+)", prompt, re.IGNORECASE)
        if m:
            return int(m.group(1)) * int(m.group(2))
        return None

    def _instruction_answer(self, prompt: str, correct: bool) -> str:
        low = prompt.lower()
        if "json" in low:
            return '{"a": 1}' if correct else "Sure! Here is the object: a=1"
        if "lowercase" in low:
            return "delhi mumbai pune" if correct else "Delhi Mumbai Pune"
        if "bullet" in low:
            n = 3 if correct else 2
            return "\n".join(f"- benefit {i + 1}" for i in range(n))
        if "no commas" in low:
            return ("Rain fell all night. The road shone." if correct
                    else "Rain fell, and the road shone.")
        if "at most 20 words" in low:
            return "The sea moves." if correct else " ".join(["word"] * 40)
        if "at least 30 words" in low:
            return " ".join(["word"] * (40 if correct else 5))
        if "two paragraphs" in low:
            return "One.\n\nTwo." if correct else "Only one paragraph here."
        if "therefore" in low:
            return ("Libraries inform people and therefore they matter."
                    if correct else "Libraries inform people and they matter.")
        if "without using the word" in low or "not_contains" in low:
            return "The storm was loud." if correct else "The storm was very loud."
        if "begin your reply with the word yes" in low:
            return "Yes, tides follow the moon." if correct else "Tides follow the moon."
        return "An answer." if correct else ""
