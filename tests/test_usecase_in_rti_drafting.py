"""
in_rti_drafting: the profile validates, every reference draft carries the
elements the README promises (office, numbered points, period, fee, the
30-day clock, transfer, exemption and appeal), the builder is idempotent,
and a direct run through the real runner leaves accuracy absent rather than
zero when no judge is configured (I7). Offline throughout.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness.profiles.loaders import load_evalset
from harness.profiles.profile import Profile

PROFILE = "configs/profiles/in_rti_drafting.yaml"
DATA = Path("data/in_rti_drafting/evalset.jsonl")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

REQUIRED = ["Public Information Officer", "Section 6(1)", "Period:", "Rs 10",
            "Section 7(1)", "30 days", "Section 6(3)", "Section 8", "Section 19"]


def test_the_profile_validates_and_weights_cost_negatively():
    p = Profile.from_yaml(PROFILE)
    assert p.task.value == "direct"
    assert p.validate() == [], p.validate()
    assert p.accuracy_scorer == "judge"
    assert {"accuracy", "completeness", "answer_relevance"} <= set(p.active_metrics)
    assert p.metric_weights["cost_usd"] < 0
    assert p.max_tokens >= 600, "a complete draft must not be truncated"


def test_every_reference_draft_has_the_required_elements():
    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)
    assert len(items) >= 45
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids))
    for it in items:
        gold = it.gold_answer or ""
        for needle in REQUIRED:
            assert needle in gold, f"{it.item_id}: reference lacks {needle!r}"
        points = re.findall(r"^\d+\. ", gold, flags=re.M)
        assert 2 <= len(points) <= 4, f"{it.item_id}: {len(points)} points"
        assert it.meta["office"] in gold
    langs = {it.meta["language"] for it in items}
    assert langs == {"en", "hi", "hinglish"}
    hindi = [it for it in items if it.meta["language"] == "hi"]
    assert all(DEVANAGARI.search(it.query) for it in hindi)
    assert sum(it.meta["first_appeal"] for it in items) == 1


def test_the_builder_is_idempotent():
    before = DATA.read_text(encoding="utf-8")
    import runpy
    runpy.run_path("data/in_rti_drafting/build_dataset.py", run_name="__main__")
    assert DATA.read_text(encoding="utf-8") == before


def test_without_a_judge_accuracy_is_absent_not_zero(tmp_path, fake_client):
    from harness.cache.cache import Cache
    from harness.clients.base import GenResult
    from harness.clients.cost import CostMeter
    from harness.orchestration.runner import RunContext, run_item
    from harness.rag.prompt import PromptConfig
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import Pass, RetrievalMode, TaskType

    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)[:2]

    class _Pricing:
        def generation_cost(self, model, a, b):
            return 0.0

        def embedding_cost(self, model, t):
            return 0.0

    def generate(model, messages, **kw):
        return GenResult(text=items[0].gold_answer, prompt_tokens=200,
                        completion_tokens=250, latency_ms=1.0, finish_reason="stop")

    fake_client.generate = generate
    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=None, cache=Cache(str(tmp_path / "c")), meter=CostMeter(),
                     embedding_model="", task=TaskType.DIRECT)
    rows = [run_item(it, "fake:a", p.name, Pass.BASELINE,
                     RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
                     ctx, "t", active_metrics=list(p.active_metrics),
                     accuracy_scorer=p.accuracy_scorer) for it in items]
    assert all(r.error is None for r in rows)
    assert all(r.accuracy is None and r.completeness is None for r in rows)
    assert all(r.finish_reason == "stop" for r in rows)
