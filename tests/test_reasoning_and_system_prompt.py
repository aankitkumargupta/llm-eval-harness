"""
Two seams the Indian-government use cases forced open.

1. Hidden reasoning. The models on the account think before they answer;
   the thinking is billed inside completion_tokens and returned in its own
   field. A 24-token label that cost 1,000 tokens of thinking, or an empty
   answer whose whole budget went on reasoning, must be legible in the
   trace (I3: reasoning tokens where reported) and in the truncation
   diagnostic, or it reads as a model-quality finding.

2. Baseline system prompt. The tuning knobs' system_prompts are candidates
   for the ADAPTED pass only; the baseline used the task default, so a
   direct-task profile could not say what the task was (a translation
   profile got "answer the question", and the models summarised). A profile
   may now set `system_prompt`, the baseline and latency lanes honour it,
   and profiles without one keep their config hash.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from harness.clients.base import GenResult
from harness.clients.openai_compatible import OpenAICompatibleClient, extract_reasoning


# --------------------------------------------------------------------------- #
#  extract_reasoning
# --------------------------------------------------------------------------- #
def test_count_from_completion_tokens_details_and_text_from_message():
    usage = NS(completion_tokens_details=NS(reasoning_tokens=48), reasoning_tokens=None)
    msg = NS(reasoning_content="The user asks...", content="", model_extra={})
    assert extract_reasoning(usage, msg) == (48, len("The user asks..."))


def test_top_level_count_is_used_when_details_are_absent():
    usage = NS(completion_tokens_details=None, reasoning_tokens=12)
    msg = NS(model_extra={"reasoning": "Thinking Process:"})
    assert extract_reasoning(usage, msg) == (12, len("Thinking Process:"))


def test_zero_beside_visible_reasoning_is_unknown_not_zero():
    # One provider reports reasoning_tokens=0 while returning reasoning text.
    usage = NS(completion_tokens_details=None, reasoning_tokens=0)
    msg = NS(reasoning="Thinking Process: ...", model_extra={})
    tokens, chars = extract_reasoning(usage, msg)
    assert tokens is None and chars > 0


def test_no_reasoning_at_all_is_none_and_zero():
    assert extract_reasoning(NS(completion_tokens_details=None), NS(content="water", model_extra={})) == (None, 0)
    assert extract_reasoning(None, None) == (None, 0)


# --------------------------------------------------------------------------- #
#  The blocking path carries it into GenResult
# --------------------------------------------------------------------------- #
def _bare_client():
    c = OpenAICompatibleClient.__new__(OpenAICompatibleClient)
    c.provider = "fake"
    c.allow_estimated_usage = False
    return c


def test_blocking_generation_records_reasoning_and_keeps_cost_unchanged():
    c = _bare_client()
    resp = NS(model="m", choices=[NS(finish_reason="length",
                                    message=NS(content="", reasoning_content="x" * 30, model_extra={}))],
              usage=NS(prompt_tokens=40, completion_tokens=48,
                       completion_tokens_details=NS(reasoning_tokens=48)))
    c._client = NS(chat=NS(completions=NS(create=lambda **kw: resp)))
    g = c._generate_blocking({"model": "m", "messages": [{"role": "user", "content": "q"}]})
    assert isinstance(g, GenResult)
    assert g.text == "" and g.truncated
    assert g.completion_tokens == 48, "reasoning is billed inside completion tokens; never double count"
    assert g.reasoning_tokens == 48 and g.reasoning_chars == 30


def test_streaming_generation_counts_reasoning_deltas_and_ttft_on_visible_text():
    c = _bare_client()
    events = [
        NS(model="m", choices=[NS(delta=NS(content=None, reasoning_content="think "), finish_reason=None)], usage=None),
        NS(model="m", choices=[NS(delta=NS(content="water", reasoning_content=None), finish_reason=None)], usage=None),
        NS(model="m", choices=[NS(delta=NS(content=None, reasoning_content=None), finish_reason="stop")],
           usage=NS(prompt_tokens=10, completion_tokens=9, completion_tokens_details=NS(reasoning_tokens=6))),
    ]
    c._client = NS(chat=NS(completions=NS(create=lambda **kw: iter(events))))
    g = c._generate_streaming({"model": "m", "messages": [{"role": "user", "content": "q"}]})
    assert g.text == "water" and g.ttft_ms is not None
    assert g.reasoning_chars == len("think ") and g.reasoning_tokens == 6


# --------------------------------------------------------------------------- #
#  The runner carries it to the row and through the cache
# --------------------------------------------------------------------------- #
def test_the_row_and_the_cache_carry_reasoning_fields(tmp_path, fake_client):
    from harness.cache.cache import Cache
    from harness.clients.cost import CostMeter
    from harness.orchestration.runner import RunContext, run_item
    from harness.rag.prompt import PromptConfig
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import EvalItem, Pass, RetrievalMode, TaskType

    class _Pricing:
        def generation_cost(self, model, a, b):
            return 0.0

        def embedding_cost(self, model, t):
            return 0.0

    calls = {"n": 0}

    def generate(model, messages, **kw):
        calls["n"] += 1
        return GenResult(text="yes", prompt_tokens=10, completion_tokens=60, latency_ms=1.0,
                         finish_reason="stop", reasoning_tokens=55, reasoning_chars=240)

    fake_client.generate = generate
    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(), judge=None,
                     cache=Cache(str(tmp_path / "c")), meter=CostMeter(),
                     embedding_model="", task=TaskType.DIRECT)
    it = EvalItem(item_id="i", query="q", gold_answer="yes")
    args = (it, "fake:a", "p", Pass.BASELINE, RetrievalConfig(mode=RetrievalMode.DENSE),
            PromptConfig(), ctx, "t")
    first = run_item(*args, active_metrics=["accuracy"], accuracy_scorer="exact")
    second = run_item(*args, active_metrics=["accuracy"], accuracy_scorer="exact")
    assert calls["n"] == 1 and second.cache_hit
    for r in (first, second):
        assert r.reasoning_tokens == 55 and r.reasoning_chars == 240


def test_truncation_report_shows_the_reasoning_split():
    import pandas as pd

    from harness.report.aggregate import truncation_report

    df = pd.DataFrame([
        {"model": "a", "truncated": True, "raw_output": "", "reasoning_chars": 900, "reasoning_tokens": 300},
        {"model": "a", "truncated": False, "raw_output": "water", "reasoning_chars": 100, "reasoning_tokens": 40},
        {"model": "b", "truncated": False, "raw_output": "roads", "reasoning_chars": 0, "reasoning_tokens": None},
    ])
    t = truncation_report(df).set_index("model")
    assert t.loc["a", "truncation_rate"] == pytest.approx(0.5)
    assert t.loc["a", "reasoning_rate"] == pytest.approx(1.0)
    assert t.loc["a", "reasoning_tokens_mean"] == pytest.approx(170.0)
    assert t.loc["a", "empty_answer_rate"] == pytest.approx(0.5)
    assert t.loc["b", "reasoning_rate"] == pytest.approx(0.0)
    assert pd.isna(t.loc["b", "reasoning_tokens_mean"])


# --------------------------------------------------------------------------- #
#  Baseline system prompt
# --------------------------------------------------------------------------- #
def test_a_profile_without_a_system_prompt_keeps_its_hash():
    from harness.profiles.profile import Profile

    base = {"name": "t", "task": "direct", "evalset_path": "x.jsonl",
            "active_metrics": ["accuracy"], "metric_weights": {"accuracy": 1.0}}
    plain = Profile.from_dict(base).config_hash_parts()
    assert "" not in plain[-1:] and plain == Profile.from_dict(dict(base)).config_hash_parts()
    custom = Profile.from_dict({**base, "system_prompt": "Translate into Hindi."}).config_hash_parts()
    assert custom != plain and custom[-1] == "Translate into Hindi."


def test_the_baseline_pass_sends_the_profile_system_prompt(tmp_path, fake_client):
    import json

    from harness.clients.cost import CostMeter
    from harness.orchestration.orchestrator import Orchestrator
    from harness.profiles.loaders import load_evalset
    from harness.profiles.profile import Profile
    from harness.store.store import TraceStore

    class _Pricing:
        def generation_cost(self, model, a, b):
            return 0.0

        def embedding_cost(self, model, t):
            return 0.0

    evalset = tmp_path / "e.jsonl"
    evalset.write_text(json.dumps({"item_id": "q1", "query": "Notice text", "gold_answer": "x"}) + "\n",
                       encoding="utf-8")
    profile = Profile.from_dict({
        "name": "t", "task": "direct", "evalset_path": str(evalset),
        "active_metrics": ["accuracy"], "metric_weights": {"accuracy": 1.0},
        "accuracy_scorer": "none", "system_prompt": "Translate the notice into Hindi.",
        "knobs": {"system_prompts": ["A candidate for the adapted pass only"]},
    })
    seen = []

    def generate(model, messages, **kw):
        seen.append(messages)
        return GenResult(text="नमस्ते", prompt_tokens=5, completion_tokens=2, latency_ms=1.0,
                         finish_reason="stop")

    fake_client.generate = generate
    orch = Orchestrator(client=fake_client, qdrant=None, pricing=_Pricing(),
                        store=TraceStore(str(tmp_path / "traces")),
                        cache_dir=str(tmp_path / "cache"), meter=CostMeter(),
                        max_workers=1, checkpoint_every=1)
    orch.run_baseline(profile, ["fake:a"], load_evalset(profile.evalset_path), "run1")
    assert seen and seen[0][0]["role"] == "system"
    assert seen[0][0]["content"] == "Translate the notice into Hindi."
    assert "adapted pass only" not in seen[0][0]["content"]
