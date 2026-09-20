"""
The Evaluate screen's library half: a new user picks a task, sees the data
format with sample rows, gets a profile that validates, uploads a JSONL
that is checked line by line, and has the files written where the harness
looks for them. Never over an existing file, never outside the project,
never under an unsafe name. Offline throughout.
"""

from __future__ import annotations

import json

import pytest
import yaml

from harness.profiles.loaders import load_corpus, load_evalset
from harness.profiles.profile import LOWER_IS_BETTER, Profile
from harness.rag.ingest import chunk_text
from harness.web import scaffold as S
from harness.web.api import ApiError


@pytest.mark.parametrize("task", ["classify", "direct", "rag"])
def test_the_description_has_fields_samples_and_notes(task):
    d = S.describe(task)
    assert d["task"] == task
    names = [f["name"] for f in d["fields"]]
    assert "item_id" in names and "query" in names
    assert ("gold_passage_ids" in names) == (task == "rag")
    assert 2 <= len(d["samples"]) <= 4
    for row in d["samples"]:
        assert "item_id" in row and "query" in row
    assert d["samples_jsonl"].count("\n") == len(d["samples"])
    assert d["notes"] and d["scorers"]
    if task == "rag":
        assert d["corpus_samples"] and d["corpus_fields"][0]["name"] == "doc_id"


@pytest.mark.parametrize("task", ["classify", "direct", "rag"])
def test_the_generated_profile_validates_with_negative_cost_weights(task):
    text = S.profile_yaml("my_eval", task, {})
    p = Profile.from_dict(yaml.safe_load(text))
    assert p.validate() == []
    assert p.name == "my_eval" and p.task.value == task
    assert p.system_prompt, "the baseline instruction must be set, not the generic default"
    for m, w in p.metric_weights.items():
        if m in LOWER_IS_BETTER:
            assert w < 0, (m, w)


def test_options_reach_the_profile():
    text = S.profile_yaml("triage", "classify",
                          {"labels": "yes, no, maybe", "scorer": "exact", "max_tokens": 512,
                           "system_prompt": "Reply with one label."})
    p = Profile.from_dict(yaml.safe_load(text))
    assert p.label_set == ["yes", "no", "maybe"] and p.max_tokens == 512
    assert p.system_prompt == "Reply with one label."
    rag = Profile.from_dict(yaml.safe_load(S.profile_yaml("hi_rag", "rag", {"multilingual": True})))
    assert "multilingual" in rag.embedding_model
    direct = Profile.from_dict(yaml.safe_load(S.profile_yaml("direct_x", "direct", {"scorer": "token_f1"})))
    assert "completeness" not in direct.active_metrics, "no judge, no judge-only metrics"


def test_bad_inputs_are_named_not_silently_fixed():
    with pytest.raises(ApiError, match="Name must be"):
        S.profile_yaml("My Eval", "classify")
    with pytest.raises(ApiError, match="Name must be"):
        S.profile_yaml("../etc", "classify")
    with pytest.raises(ApiError, match="Unknown task"):
        S.profile_yaml("ok_name", "agentic")
    with pytest.raises(ApiError, match="at least two labels"):
        S.profile_yaml("ok_name", "classify", {"labels": "only"})
    with pytest.raises(ApiError, match="scorer must be"):
        S.profile_yaml("ok_name", "direct", {"scorer": "vibes"})


def test_sample_data_loads_and_rag_gold_ids_resolve(tmp_path):
    out = S.scaffold("my_rag", "rag", {}, root=tmp_path)
    assert out["written"][0] == "configs/profiles/my_rag.yaml"
    items = load_evalset(str(tmp_path / "data/my_rag/evalset.jsonl"))
    docs = list(load_corpus(str(tmp_path / "data/my_rag/corpus.jsonl")))
    chunk_ids = {f"{d.doc_id}#{i}" for d in docs for i in range(len(chunk_text(d.text, 200, 40)))}
    for it in items:
        for gid in it.gold_passage_ids:
            assert gid in chunk_ids, gid
    assert any(it.item_type.value == "unanswerable" for it in items)
    prof = Profile.from_yaml(str(tmp_path / "configs/profiles/my_rag.yaml"))
    assert prof.validate() == []


def test_scaffold_refuses_to_overwrite_and_stays_inside_the_project(tmp_path):
    S.scaffold("my_cls", "classify", {}, root=tmp_path)
    with pytest.raises(ApiError, match="already exists"):
        S.scaffold("my_cls", "classify", {}, root=tmp_path)
    S.scaffold("my_cls", "classify", {"labels": "a, b"}, root=tmp_path, overwrite=True)
    assert "a" in (tmp_path / "configs/profiles/my_cls.yaml").read_text(encoding="utf-8")
    assert not (tmp_path.parent / "configs").exists()


def test_uploads_are_checked_line_by_line_and_written_only_when_clean(tmp_path):
    bad = '{"item_id":"a","query":"x","gold_answer":"y"}\n{"item_id":"a","query":""}\nnot json\n'
    check = S.validate_jsonl("evalset", bad)
    assert not check["ok"] and check["rows"] == 2
    assert any("line 2: duplicate item_id" in p for p in check["problems"])
    assert any("line 3: invalid JSON" in p for p in check["problems"])
    with pytest.raises(ApiError, match="nothing was written"):
        S.write_dataset("upl_test", "evalset", bad, root=tmp_path)
    assert not (tmp_path / "data/upl_test/evalset.jsonl").exists()

    good = "\n".join(json.dumps(r) for r in S.SAMPLES["classify"])
    out = S.write_dataset("upl_test", "evalset", good, root=tmp_path, task="classify",
                          labels=["billing", "technical", "other"])
    assert out["rows"] == 3 and (tmp_path / "data/upl_test/evalset.jsonl").exists()
    with pytest.raises(ApiError, match="already exists"):
        S.write_dataset("upl_test", "evalset", good, root=tmp_path)
    # A label outside the set is a problem, not a silent zero later.
    off = json.dumps({"item_id": "z", "query": "q", "gold_answer": "refunds"})
    assert not S.validate_jsonl("evalset", off, labels=["billing"])["ok"]
    # RAG items without gold passages cannot score retrieval.
    assert not S.validate_jsonl("evalset", json.dumps({"item_id": "r", "query": "q", "gold_answer": "a"}),
                                task="rag")["ok"]
    corpus_bad = json.dumps({"doc_id": "d"})
    assert any("'text'" in p for p in S.validate_jsonl("corpus", corpus_bad)["problems"])


def test_the_screen_and_routes_are_wired():
    import inspect

    from harness.web import server
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '["evaluate", "Evaluate"' in js and "evaluate: viewEvaluate" in js
    src = inspect.getsource(server._Handler)
    for route in ("/api/scaffold-spec", "/api/scaffold-preview", "/api/scaffold", "/api/upload-dataset"):
        assert route in src, route
    # The landing page no longer carries an errors tile.
    overview = js[js.index("function viewOverview"):js.index("function viewPreflight")]
    assert '[String(errs), "errors"' not in overview and 'const errs' not in overview
    assert "—" not in js
