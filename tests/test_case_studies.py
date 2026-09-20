"""
Case studies: every write-up under docs/case-studies pairs with a profile
that validates, follows the fixed shape (use case, metrics table, caveats,
results), parses into blocks the UI can render without an HTML parser,
carries no em dash, and is reachable from the task bar through an API route.
Offline throughout: the store used is empty, so every case study reports
"not run yet" and the code path that handles that is the one exercised.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness.web import case_studies as CS

DOCS = Path("docs/case-studies")
REQUIRED_HEADINGS = ("The use case", "Who would run this", "What a wrong answer costs",
                     "The metrics, and why", "Results")


def _docs():
    return sorted(p for p in DOCS.glob("*.md") if p.stem.lower() != "readme")


def test_there_are_at_least_five_case_studies_and_each_has_a_valid_profile():
    from harness.profiles.profile import Profile

    docs = _docs()
    assert len(docs) >= 5
    for md in docs:
        p = Profile.from_yaml(f"configs/profiles/{md.stem}.yaml")
        assert p.validate() == [], (md.stem, p.validate())
        assert p.name == md.stem


@pytest.mark.parametrize("md", _docs(), ids=lambda p: p.stem)
def test_each_case_study_has_the_fixed_shape_and_no_em_dash(md):
    text = md.read_text(encoding="utf-8")
    assert "—" not in text, f"{md.name}: em dash"
    for h in REQUIRED_HEADINGS:
        assert f"## {h}" in text, f"{md.name}: missing section {h!r}"
    blocks = CS._blocks(text)
    kinds = {b["type"] for b in blocks}
    assert {"h2", "p", "table"} <= kinds
    table = next(b for b in blocks if b["type"] == "table")
    assert table["header"][0] == "metric"
    # Every weighted metric in the profile appears in the write-up's table,
    # with the same weight: a case study that quotes a weight the profile
    # does not use is a second definition of the composite.
    from harness.profiles.profile import Profile
    p = Profile.from_yaml(f"configs/profiles/{md.stem}.yaml")
    by_metric = {r[0]: r[1] for r in table["rows"]}
    for metric, w in p.metric_weights.items():
        assert metric in by_metric, f"{md.name}: {metric} missing from the table"
        assert float(by_metric[metric]) == pytest.approx(w), (md.name, metric)


def test_blocks_parse_bullets_tables_and_paragraphs():
    text = ("# Case study: x (p)\n\n## A\n\nfirst line\nsecond line\n\n- one\n  continued\n"
            "- two\n\n| metric | w |\n|---|---|\n| a | 1 |\n\nlast\n")
    b = CS._blocks(text)
    assert [x["type"] for x in b] == ["h2", "p", "ul", "table", "p"]
    assert b[1]["text"] == "first line second line"
    assert b[2]["items"] == ["one continued", "two"]
    assert b[3] == {"type": "table", "header": ["metric", "w"], "rows": [["a", "1"]]}
    assert CS._title(text, "f") == "X", "prefix and profile id stripped, first letter capitalised"


def test_listing_against_an_empty_store_reports_not_run_yet(tmp_path):
    out = CS.list_case_studies(str(tmp_path / "nowhere"))["case_studies"]
    assert {c["id"] for c in out} == {p.stem for p in _docs()}
    for c in out:
        assert c["run"] is None and c["other_runs"] == []
        assert c["profile"] and "invalid" not in c["profile"]
        assert c["profile"]["n_items"] > 0
        assert isinstance(c["results_filled"], bool)
        assert c["title"] and "Case study:" not in c["title"]
    with pytest.raises(Exception, match="No case study"):
        CS.get_case_study(str(tmp_path / "nowhere"), "does_not_exist")


def test_the_screen_is_wired_end_to_end():
    import inspect

    from harness.web import server
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert '["casestudies", "Case studies"' in js
    assert "casestudies: viewCaseStudies" in js
    assert "/api/case-studies" in inspect.getsource(server._Handler)
    # The screen draws the Profile report's payload and computes nothing.
    view = js[js.index("function viewCaseStudies"):js.index("async function loadCaseStudies")]
    assert "/api/profile-results" in js[js.index("async function toggleCaseStudy"):]
    assert not re.search(r"\.reduce\(|Math\.", view)
    assert "—" not in js
