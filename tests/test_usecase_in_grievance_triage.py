"""
in_grievance_triage: the profile is runnable, the data is what it claims,
and a classify run through the real runner scores the way the case study
says (a label is scored exact; a non-label is a format failure, not a wrong
department). Offline throughout: the fake provider stands in for a model.
"""

from __future__ import annotations

import re
from pathlib import Path

from harness.profiles.loaders import load_evalset
from harness.profiles.profile import Profile

PROFILE = "configs/profiles/in_grievance_triage.yaml"
DATA = Path("data/in_grievance_triage/evalset.jsonl")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def test_the_profile_validates_with_no_problems():
    p = Profile.from_yaml(PROFILE)
    assert p.task.value == "classify"
    assert p.validate() == [], p.validate()
    assert p.metric_weights["cost_usd"] < 0 and p.metric_weights["latency_ms"] < 0


def test_the_evalset_is_what_the_readme_claims():
    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)
    assert len(items) >= 90
    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids)), "duplicate ids break the paired tests"
    labels = {it.gold_answer for it in items}
    assert labels <= set(p.label_set), labels - set(p.label_set)
    assert len(labels) == len(p.label_set), "every label must have items"
    # Language mix: a real share of Devanagari, a real share without it.
    hindi = sum(bool(DEVANAGARI.search(it.query)) for it in items)
    assert 0.2 <= hindi / len(items) <= 0.5, f"{hindi}/{len(items)} Devanagari items"
    # No item is empty, and nothing is suspiciously short.
    assert all(len(it.query.split()) >= 6 for it in items)


def test_the_builder_is_idempotent(tmp_path):
    before = DATA.read_text(encoding="utf-8")
    import runpy
    runpy.run_path("data/in_grievance_triage/build_dataset.py", run_name="__main__")
    assert DATA.read_text(encoding="utf-8") == before


def test_a_classify_run_scores_labels_exact_and_non_labels_as_format_failures(tmp_path, fake_client):
    from harness.cache.cache import Cache
    from harness.clients.base import GenResult
    from harness.clients.cost import CostMeter
    from harness.orchestration.runner import RunContext, run_item
    from harness.rag.prompt import PromptConfig
    from harness.rag.retrieve import RetrievalConfig
    from harness.store.schema import Pass, RetrievalMode, TaskType

    p = Profile.from_yaml(PROFILE)
    items = load_evalset(p.evalset_path)[:6]

    class _Pricing:
        def generation_cost(self, model, a, b):
            return 0.0

        def embedding_cost(self, model, t):
            return 0.0

    # The fake answers with the gold label for five items and prose for one.
    replies = {it.item_id: it.gold_answer for it in items}
    replies[items[-1].item_id] = "I think this belongs to the water department."
    current = {"id": None}

    def generate(model, messages, **kw):
        return GenResult(text=replies[current["id"]], prompt_tokens=20,
                        completion_tokens=3, latency_ms=1.0, finish_reason="stop")

    fake_client.generate = generate
    ctx = RunContext(client=fake_client, retriever=None, pricing=_Pricing(),
                     judge=None, cache=Cache(str(tmp_path / "c")), meter=CostMeter(),
                     embedding_model="", task=TaskType.CLASSIFY, label_set=list(p.label_set))
    rows = []
    for it in items:
        current["id"] = it.item_id
        rows.append(run_item(it, "fake:a", p.name, Pass.BASELINE,
                             RetrievalConfig(mode=RetrievalMode.DENSE), PromptConfig(),
                             ctx, "t", active_metrics=list(p.active_metrics),
                             accuracy_scorer=p.accuracy_scorer))
    assert all(r.error is None for r in rows)
    assert [r.accuracy for r in rows[:5]] == [1.0] * 5
    last = rows[-1]
    # Prose is not a label: it must not be scored as a wrong department.
    assert last.accuracy in (None, 0.0)
    assert last.predicted_label in (None, "") or last.predicted_label not in p.label_set
