"""
Resume is idempotent for every lane, the latency lane included.

Found live: resuming a run to retry 13 rate-limited items also re-ran the
whole latency lane, because LatencyPass swallowed `done_keys` in `**_`
while BaselinePass honoured it. The retried items cost $0.003; the
duplicated latency lane cost $0.066 and doubled the latency samples. The
CLAUDE.md contract (Phase 3, resume keyed on item content, row-set equality
with an uninterrupted run) does not carve out a lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.clients.cost import CostMeter
from harness.orchestration.orchestrator import Orchestrator
from harness.profiles.loaders import load_evalset
from harness.profiles.profile import Profile
from harness.store.store import TraceStore


class _Pricing:
    def generation_cost(self, model, p, c):
        return 0.0

    def embedding_cost(self, model, tokens):
        return 0.0

    def rerank_cost(self, model, tokens=0, n_docs=0):
        return 0.0


@pytest.fixture
def setup(tmp_path, fake_client):
    evalset = tmp_path / "evalset.jsonl"
    evalset.write_text("\n".join(json.dumps({"item_id": f"q{i}", "query": f"question {i}",
                                             "gold_answer": "yes"}) for i in range(5)),
                       encoding="utf-8")
    profile = Profile.from_dict({
        "name": "t", "task": "classify", "evalset_path": str(evalset),
        "label_set": ["yes", "no"], "accuracy_scorer": "exact",
        "active_metrics": ["accuracy", "cost_usd"],
        "metric_weights": {"accuracy": 1.0, "cost_usd": -0.1},
    })
    store = TraceStore(str(tmp_path / "traces"))
    orch = Orchestrator(client=fake_client, qdrant=None, pricing=_Pricing(), store=store,
                        cache_dir=str(tmp_path / "cache"), meter=CostMeter(),
                        max_workers=1, checkpoint_every=1)
    return profile, store, orch, fake_client, load_evalset(profile.evalset_path)


def test_resuming_the_latency_lane_skips_rows_already_recorded(setup):
    profile, store, orch, client, items = setup
    orch.run_latency(profile, ["good"], items[:4], "lat", warmup=0)
    n_rows = len(store.load_run("lat"))
    n_calls = len(client.calls)
    assert n_rows == 4

    orch.run_latency(profile, ["good"], items[:4], "lat", warmup=0, resume_from="lat")
    assert len(client.calls) == n_calls, "a resumed latency lane must not re-time recorded items"
    assert len(store.load_run("lat")) == n_rows


def test_resuming_the_latency_lane_completes_only_the_missing_items(setup):
    profile, store, orch, client, items = setup
    orch.run_latency(profile, ["good"], items[:2], "lat", warmup=0)
    before = len(client.calls)
    orch.run_latency(profile, ["good"], items[:4], "lat", warmup=0, resume_from="lat")
    assert len(client.calls) == before + 2
    df = store.load_run("lat")
    assert sorted(df["item_id"]) == ["q0", "q1", "q2", "q3"]
    assert not df.duplicated(["item_id", "model", "pass_"]).any()


def test_the_cli_passes_resume_to_every_lane():
    """The CLI wires --resume into the baseline lane; it must do the same for
    the latency lane, or the flag silently means different things per lane."""
    src = Path("main.py").read_text(encoding="utf-8")
    call = src[src.index("orch.run_latency("):]
    call = call[:call.index(")") + 1]
    assert "resume_from=args.resume" in call, call
