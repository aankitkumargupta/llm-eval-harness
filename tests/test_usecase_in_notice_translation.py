"""
in_notice_translation: the profile validates, the data is what the README
claims (English in, Devanagari reference out, numbers preserved), the
builder is idempotent, and a direct run through the real runner scores the
script ratio on its own so that "fluent English" and "wrong Hindi" are
different failures. Offline: the fake provider stands in for a model.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness.profiles.loaders import load_evalset
from harness.profiles.profile import Profile

PROFILE = "configs/profiles/in_notice_translation.yaml"
DATA = Path("data/in_notice_translation/evalset.jsonl")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
NUMBER = re.compile(r"\d+")


def test_the_profile_validates_and_names_the_script():
    p = Profile.from_yaml(PROFILE)
    assert p.task.value == "direct"
    assert p.validate() == [], p.validate()
    assert p.target_script == "devanagari"
    assert "native_script_ratio" in p.active_metrics
    assert p.metric_weights["cost_usd"] < 0


def test_the_evalset_is_what_the_readme_claims():
    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)
    assert len(items) >= 40
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids)), "duplicate ids break the paired tests"
    for it in items:
        assert not DEVANAGARI.search(it.query), f"{it.item_id}: source must be English"
        assert DEVANAGARI.search(it.gold_answer), f"{it.item_id}: reference must be Hindi"
        assert 30 <= len(it.query.split()) <= 130
        # Every number in the notice survives into the reference: that is the
        # fidelity the judge is asked to check, so the references must have it.
        for n in NUMBER.findall(it.query):
            assert n in it.gold_answer, f"{it.item_id}: number {n} missing from reference"


def test_the_builder_is_idempotent():
    before = DATA.read_text(encoding="utf-8")
    import runpy
    runpy.run_path("data/in_notice_translation/build_dataset.py", run_name="__main__")
    assert DATA.read_text(encoding="utf-8") == before


def test_script_ratio_is_scored_apart_from_accuracy(tmp_path, fake_client):
    from harness.cache.cache import Cache
    from harness.clients.base import GenResult
    from harness.clients.cost import CostMeter
    from harness.orchestration.runner import RunContext, run_item
    from harness.rag.prompt import PromptConfig
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import Pass, RetrievalMode, TaskType

    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)[:3]

    class _Pricing:
        def generation_cost(self, model, a, b):
            return 0.0

        def embedding_cost(self, model, t):
            return 0.0

    # Three shapes of answer: the reference itself, an English paraphrase,
    # and Hindi in Latin script. All three "mean" the notice; only one is
    # publishable.
    replies = {
        items[0].item_id: items[0].gold_answer,
        items[1].item_id: "Water supply will be suspended on Thursday from 9 am to 5 pm.",
        items[2].item_id: "Pariksha 15 July se 22 July tak hogi. Admit card zaroor layein.",
    }
    current = {"id": None}

    def generate(model, messages, **kw):
        return GenResult(text=replies[current["id"]], prompt_tokens=80,
                        completion_tokens=60, latency_ms=1.0, finish_reason="stop")

    fake_client.generate = generate
    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=None, cache=Cache(str(tmp_path / "c")), meter=CostMeter(),
                     embedding_model="", task=TaskType.DIRECT,
                     target_script=p.target_script)
    rows = []
    for it in items:
        current["id"] = it.item_id
        rows.append(run_item(it, "fake:a", p.name, Pass.BASELINE,
                             RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
                             ctx, "t", active_metrics=list(p.active_metrics),
                             accuracy_scorer="none"))
    assert all(r.error is None for r in rows)
    ratios = [r.native_script_ratio for r in rows]
    assert ratios[0] == 1.0
    assert ratios[1] == 0.0
    assert ratios[2] == 0.0
    # Without a judge, accuracy is simply absent, never a zero (I7).
    assert all(r.accuracy is None for r in rows)
