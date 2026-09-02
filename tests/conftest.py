"""
Shared pytest fixtures, and the import path fix.

The suite previously required `PYTHONPATH=.` to import `harness` at all — so
`pytest tests/` from a clean checkout failed with ModuleNotFoundError, and the
documented `python tests/test_core.py` was the only way to run it. Putting the
repo root on `sys.path` here makes the ordinary invocation work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.clients.base import EmbedResult, GenResult, ProviderInfo  # noqa: E402
from harness.store.schema import EvalItem  # noqa: E402


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Tests read configs/ by relative path; pin the cwd so they don't depend on
    where pytest was invoked from."""
    monkeypatch.chdir(ROOT)


class FakeClient:
    """A deterministic stand-in for a provider.

    Exists so the orchestration path — the part that actually spends money — can
    be tested end to end. Without it, everything above `run_item` is only ever
    exercised against a live API, which means in practice it is never tested.
    """

    def __init__(self, answers: dict[str, str] | None = None,
                 latency_ms: float = 10.0, fail_on: set[str] | None = None,
                 embed_dim: int = 8, meter=None, pricing=None):
        self.info = ProviderInfo(name="fake")
        self.answers = answers or {}
        self.latency_ms = latency_ms
        self.fail_on = fail_on or set()
        self.embed_dim = embed_dim
        # Real adapters record every billed call on the shared meter — that is
        # what the budget guard watches. A fake that skips it would make the
        # budget appear broken (or, worse, appear to work when it doesn't).
        self.meter = meter
        self.pricing = pricing
        self.calls: list[tuple[str, str]] = []
        self.judge_calls = 0

    def _bill(self, model, prompt_tokens, completion_tokens, is_judge=False):
        if self.meter is None:
            return
        usd = 0.0
        if self.pricing is not None:
            try:
                usd = self.pricing.generation_cost(model, prompt_tokens,
                                                   completion_tokens)
            except KeyError:
                usd = 0.0
        self.meter.record_generation(usd, prompt_tokens, completion_tokens,
                                     is_judge=is_judge)

    def generate(self, model, messages, temperature=0.0, max_tokens=1024,
                 seed=0, stream=False, **kw):
        prompt = messages[-1]["content"]
        self.calls.append((model, prompt))
        if model in self.fail_on:
            raise RuntimeError("Rate limit exceeded")
        text = self.answers.get(prompt, f"answer from {model}")
        result = GenResult(text=text, prompt_tokens=100, completion_tokens=20,
                           latency_ms=self.latency_ms,
                           ttft_ms=self.latency_ms * 0.3 if stream else None,
                           finish_reason="stop", model=model)
        self._bill(model, result.prompt_tokens, result.completion_tokens)
        return result

    def judge(self, model, messages, **kw):
        self.judge_calls += 1
        self._bill(model, 200, 30, is_judge=True)
        return GenResult(
            text='{"faithfulness": 0.9, "answer_relevance": 0.85, '
                 '"completeness": 0.8, "accuracy": 1.0}',
            prompt_tokens=200, completion_tokens=30, latency_ms=5.0,
            finish_reason="stop", model=model)

    def embed(self, model, texts):
        # Deterministic pseudo-vectors. The MODEL is part of the vector, because
        # two different embedders must never produce the same vector — that is
        # precisely the confusion the retriever thread-safety fix prevents.
        vectors = [
            [((hash((model, t, i)) % 1000) / 1000.0) for i in range(self.embed_dim)]
            for t in texts
        ]
        if self.meter is not None:
            self.meter.record_embedding(0.0, len(texts) * 10)
        return EmbedResult(vectors=vectors, prompt_tokens=len(texts) * 10)

    def rerank(self, model, query, documents, top_n=None):
        order = list(range(len(documents)))[::-1]
        return [(i, 1.0 - n * 0.01) for n, i in enumerate(order[:top_n or len(order)])]


@pytest.fixture
def fake_client():
    return FakeClient()


@pytest.fixture
def eval_items():
    return [EvalItem(item_id=f"q{i}", query=f"question {i}",
                     gold_answer=f"answer {i}",
                     gold_passage_ids=[f"doc{i}#0"])
            for i in range(6)]
