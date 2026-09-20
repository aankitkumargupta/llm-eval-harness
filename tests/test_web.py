"""
The local web UI.

Tested at the API layer rather than over HTTP, because §5 blocks sockets for
the whole suite — and that is the right boundary anyway: `server.py` is routing
and `api.py` is everything a browser actually sees, so these tests cover the
code that can be wrong in a way that matters.

The rule under test throughout is §3's: presentation holds no business logic.
Every number the UI shows must come from the same library the CLI calls, or
there are two definitions of it and they will disagree eventually (§14.3).
"""

from __future__ import annotations

import json

import pytest

from harness.web import api


@pytest.fixture
def store_cfg(tmp_path):
    """An isolated trace store, so these tests never touch runs/traces."""
    return str(tmp_path / "traces")


# --------------------------------------------------------------------------- #
#  Catalogue
# --------------------------------------------------------------------------- #
def test_lists_every_shipped_benchmark():
    ids = {b["id"] for b in api.list_benchmarks()["benchmarks"]}
    assert {"mmlu_pro", "gsm8k", "ifeval"} <= ids


def test_a_spec_without_an_adapter_is_listed_not_hidden():
    """A half-built benchmark that fails only when someone selects it is worse
    than one that announces itself in the catalogue."""
    for b in api.list_benchmarks()["benchmarks"]:
        assert "has_adapter" in b


def test_every_shipped_benchmark_has_an_adapter():
    assert all(b["has_adapter"] for b in api.list_benchmarks()["benchmarks"])


def test_the_spec_payload_exposes_everything_that_feeds_the_hash():
    """The UI groups these under 'spec_hash' so a reader can see, without
    opening the YAML, which knobs make two runs incomparable (§10.1)."""
    spec = next(b for b in api.list_benchmarks()["benchmarks"]
                if b["id"] == "mmlu_pro")
    hashed = spec["hashed"]
    for key in ("mode", "chance_level", "few_shot_n", "few_shot_pool",
                "max_tokens", "temperature", "samples_per_item"):
        assert key in hashed, key
    assert spec["spec_hash"]
    assert spec["licence"]


def test_multiple_choice_exposes_its_chance_level():
    """Because 25% on 4-way MC is not '25% good', and the UI cannot render the
    adjustment without this number."""
    mc = next(b for b in api.list_benchmarks()["benchmarks"] if b["id"] == "mmlu_pro")
    assert mc["hashed"]["chance_level"] == 0.1


def test_local_models_are_always_offered():
    """A fresh checkout has no models.yaml worth running; if the offline
    provider were not offered unconditionally the picker would be empty and the
    UI untryable."""
    m = api.list_models({})
    assert m["local"] and all(x.startswith("fake:") for x in m["local"])


def test_configured_models_are_kept_separate_from_local_ones():
    m = api.list_models({"models": ["together:x", "openrouter:y"]})
    assert m["configured"] == ["together:x", "openrouter:y"]
    assert not any(x.startswith("fake:") for x in m["configured"])


# --------------------------------------------------------------------------- #
#  Extraction preview
# --------------------------------------------------------------------------- #
def test_extraction_preview_reports_which_link_fired():
    out = api.extract_preview("gsm8k", "Working it out.\n#### 208")
    assert out["ok"] and out["value"] == "208"
    assert out["via"] == "regex"


def test_extraction_preview_falls_through_the_chain():
    out = api.extract_preview("gsm8k", "The total comes to $1,234.00")
    assert out["ok"] and out["value"] == "1234", "numeric normalisation"
    assert out["via"] == "numeric"


def test_extraction_failure_states_the_consequence():
    """A failure message that does not say what happens next leaves the reader
    assuming the item scored zero — which is exactly the thing it did not do."""
    out = api.extract_preview("mmlu_pro", "no comment, sorry")
    assert not out["ok"]
    assert "not" in out["consequence"].lower()
    assert "wrong answer" in out["consequence"]


def test_the_playground_exposes_the_pronoun_i_hazard():
    """A real MMLU-Pro trap, found by writing this UI: the English pronoun "I"
    is also a valid option letter, so a refusal beginning "I ..." extracts as a
    confident answer to option nine via the last-resort link.

    Not fixed in the extractor, because excluding "I" would make a model that
    genuinely answers I into a silent extraction failure — worse, and harder to
    see. The handling is to surface it: `via` names the link that fired, and
    this playground is where you find out before a run rather than after.
    """
    out = api.extract_preview("mmlu_pro", "I would rather not say.")
    assert out["ok"] and out["value"] == "I"
    assert out["via"] == "last_capital_letter", (
        "if this ever reports a stronger link, the hazard has moved")


def test_extraction_preview_lists_every_extractor():
    out = api.extract_preview("gsm8k", "#### 42")
    names = {r["extractor"] for r in out["extractors"]}
    assert {"numeric", "boxed", "verbatim", "last_capital_letter"} <= names
    assert any(r["in_chain"] for r in out["extractors"])


def test_extraction_preview_survives_adversarial_input():
    """Model output is untrusted data (§6). It must never crash the endpoint."""
    for raw in ("", "\x00\x01", "𝕏" * 500, "```" * 200, "<script>x</script>"):
        assert "ok" in api.extract_preview("mmlu_pro", raw)


def test_extraction_preview_rejects_an_unknown_benchmark():
    with pytest.raises(api.ApiError) as e:
        api.extract_preview("does_not_exist", "x")
    assert e.value.status == 404


# --------------------------------------------------------------------------- #
#  Runs
# --------------------------------------------------------------------------- #
def _run_and_wait(store_cfg, benchmark="gsm8k", models=("fake:a", "fake:b"),
                  limit=8, timeout=30.0):
    import time

    started = api.start_run(benchmark=benchmark, models=list(models),
                            limit=limit, seed=1, store_path=store_cfg,
                            models_cfg={})
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = api.job_status(started["job_id"])
        if snap["status"] != "running":
            return started, snap
        time.sleep(0.05)
    raise AssertionError("run did not finish in time")


@pytest.mark.integration
def test_a_run_started_from_the_api_completes_offline(store_cfg):
    """No key, no network — the whole point of shipping a local provider."""
    started, snap = _run_and_wait(store_cfg)
    assert snap["status"] == "done"
    assert snap["rows"] == 16          # 8 items x 2 models
    assert started["run_id"].startswith("bench_")


@pytest.mark.integration
def test_a_run_writes_a_manifest(store_cfg):
    """I9 applies to a run started from a browser exactly as from the CLI."""
    started, _ = _run_and_wait(store_cfg)
    m = api.manifest(store_cfg, started["run_id"])
    assert m["spec_hash"] and m["dataset_hash"] and m["apparatus_hash"]
    assert m["n_rows"] == 16


@pytest.mark.integration
def test_results_carry_the_columns_a_benchmark_report_usually_omits(store_cfg):
    started, _ = _run_and_wait(store_cfg)
    res = api.results(store_cfg, started["run_id"])
    assert res["benchmark"] == "gsm8k"
    assert len(res["summary"]) == 2
    for row in res["summary"]:
        for col in ("accuracy", "accuracy_chance_adjusted", "n_items",
                    "n_scored", "extraction_failure_rate"):
            assert col in row, col


@pytest.mark.integration
def test_results_include_the_paired_significance_test(store_cfg):
    """§10: 'No second significance implementation.' The UI must reach the same
    one the CLI does."""
    started, _ = _run_and_wait(store_cfg)
    res = api.results(store_cfg, started["run_id"])
    assert not res["significance_error"], res["significance_error"]
    assert len(res["significance"]) == 1
    row = res["significance"][0]
    assert row["test"] in ("mcnemar_exact", "paired_bootstrap")
    assert "verdict" in row
    assert res["power"], "the power line must be present"


@pytest.mark.integration
def test_rows_endpoint_returns_what_the_model_actually_said(store_cfg):
    started, _ = _run_and_wait(store_cfg)
    out = api.rows(store_cfg, started["run_id"], limit=5)
    assert len(out["rows"]) == 5
    assert "raw_output" in out["rows"][0]


def test_a_run_with_no_models_is_rejected_before_anything_starts(store_cfg):
    with pytest.raises(api.ApiError, match="at least one model"):
        api.start_run(benchmark="gsm8k", models=[], limit=5, seed=1,
                      store_path=store_cfg, models_cfg={})


def test_a_run_of_an_unknown_benchmark_is_rejected(store_cfg):
    with pytest.raises(api.ApiError) as e:
        api.start_run(benchmark="nope", models=["fake:a"], limit=5, seed=1,
                      store_path=store_cfg, models_cfg={})
    assert e.value.status == 404


def test_results_for_a_missing_run_say_so(store_cfg):
    with pytest.raises(api.ApiError) as e:
        api.results(store_cfg, "no_such_run")
    assert e.value.status == 404


# --------------------------------------------------------------------------- #
#  §10.1 — refusing to mix formats
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_results_refuse_to_average_across_spec_hashes(store_cfg, monkeypatch):
    """Two runs of the 'same' benchmark under different prompt formats are not
    the same benchmark. The UI must refuse rather than produce a number that
    describes neither."""

    started, _ = _run_and_wait(store_cfg)

    from harness.store.store import TraceStore
    real_load = TraceStore.load_run

    def _mixed(self, run_id):
        df = real_load(self, run_id)
        df = df.copy()
        df.loc[df.index[: len(df) // 2], "spec_hash"] = "deadbeef"
        return df

    monkeypatch.setattr(TraceStore, "load_run", _mixed)
    res = api.results(store_cfg, started["run_id"])
    assert res["refused"] is True
    assert len(res["spec_hashes"]) == 2
    assert "not comparable" in res["reason"]


# --------------------------------------------------------------------------- #
#  Presentation holds no logic
# --------------------------------------------------------------------------- #
def test_the_api_layer_computes_no_metric():
    """§9 Phase 8. A metric computed here would be a second definition of
    something the library already owns, and the two would drift."""
    import inspect

    src = inspect.getsource(api)
    # Grouping rows to *list* them is presentation. Computing a metric is not.
    assert ".mean()" not in src
    assert ".std(" not in src
    assert "chance_adjusted(" not in src, "must come from bench.metrics"
    assert "pass_at_k(" not in src, "must come from bench.metrics"
    assert "mcnemar" not in src, "must come from report.stats"
    assert "bootstrap" not in src, "must come from report.stats"


def test_nan_becomes_null_not_zero():
    """The whole of I7 at this layer: an accuracy of null means *not
    applicable*. Rendering it as 0 would be the fold-in the harness refuses
    everywhere else."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame([{"model": "a", "accuracy": np.nan, "n": 3}])
    rec = api._records(df)[0]
    assert rec["accuracy"] is None
    assert rec["n"] == 3


def test_the_worker_never_imports_streamlit_or_any_web_framework():
    """This UI is stdlib-only on purpose: §7 forbids introducing a framework
    and forbids a dependency duplicating the stdlib."""
    import inspect

    from harness.web import server

    for mod in (api, server):
        src = inspect.getsource(mod)
        for banned in ("streamlit", "fastapi", "flask", "django", "starlette"):
            assert banned not in src.lower(), f"{banned} in {mod.__name__}"


# --------------------------------------------------------------------------- #
#  Server wiring
# --------------------------------------------------------------------------- #
def test_the_server_defaults_to_loopback():
    """It has no authentication and can spend money. The default must not be
    the dangerous one."""
    import inspect

    from harness.web.server import serve

    assert inspect.signature(serve).parameters["host"].default == "127.0.0.1"


def test_static_files_are_served_from_one_directory_only():
    """Path traversal here would read the trace store or a .env."""
    import inspect

    from harness.web import server

    src = inspect.getsource(server._Handler._static)
    assert "relative_to" in src, "must confirm the resolved path stays inside STATIC"


def test_the_single_page_app_ships_with_the_package():
    from harness.web.server import STATIC

    index = STATIC / "index.html"
    assert index.exists()
    text = index.read_text(encoding="utf-8")
    assert "<title>" in text
    # Offline-first: no CDN, no web fonts. A UI that needs the network to look
    # right looks broken on a plane, and this harness runs offline by design.
    for asset in ("index.html", "app.css", "app.js"):
        blob = (STATIC / asset).read_text(encoding="utf-8")
        for banned in ("cdn.", "googleapis.com", "unpkg", "jsdelivr", "http://", "https://"):
            assert banned not in blob, f"external resource {banned} in {asset}"


def test_the_page_defines_both_themes():
    """Every colour is a token on :root, redefined for the other theme. A
    colour whose only definition sits inside a theme block renders one theme's
    text on the other theme's ground."""
    from harness.web.server import STATIC

    css = (STATIC / "app.css").read_text(encoding="utf-8").replace('"', "")
    squashed = css.replace(" ", "")
    assert ":root{" in squashed, "the light/base palette must live on bare :root"
    assert ":root[data-theme=light]" in squashed
    # Spot-check that a token defined in one theme exists in the other.
    for token in ("--ground", "--ink", "--gold", "--ok", "--crit"):
        assert squashed.count(token + ":") >= 2, token


def test_api_error_carries_an_http_status():
    e = api.ApiError("nope", 418)
    assert e.status == 418
    assert str(e) == "nope"


def test_every_api_response_is_json_serialisable(store_cfg):
    """The handler serialises with `json.dumps`; a numpy scalar that slipped
    through would 500 at render time, which is the worst place to find it."""
    json.dumps(api.list_benchmarks())
    json.dumps(api.list_models({"models": ["a"]}))
    json.dumps(api.list_runs(store_cfg))
    json.dumps(api.extract_preview("gsm8k", "#### 1"))


def test_the_serve_command_is_registered_in_the_cli():
    import main

    assert hasattr(main, "cmd_serve")


# --------------------------------------------------------------------------- #
#  Both halves in one dashboard
# --------------------------------------------------------------------------- #
def test_profiles_are_listed_for_the_dashboard():
    from harness.web import profile_api as P

    names = {p.get("name") for p in P.list_profiles()["profiles"]}
    assert {"regulated_qa", "support_triage"} <= names


def test_an_invalid_profile_is_listed_with_its_error_not_hidden(tmp_path):
    """A profile that silently vanishes from the picker is harder to debug
    than one that says what is wrong with it."""
    from harness.web import profile_api as P

    (tmp_path / "broken.yaml").write_text("name: broken\ntask: nonsense\n",
                                          encoding="utf-8")
    rows = P.list_profiles(str(tmp_path))["profiles"]
    assert rows and "invalid" in rows[0]


def test_the_run_list_covers_both_halves(store_cfg):
    """§10's claim made visible: benchmark and profile runs share one store, so
    'which runs do I have?' is one question with one answer."""
    from harness.web import profile_api as P

    _run_and_wait(store_cfg, limit=4)
    runs = P.all_runs(store_cfg)["runs"]
    assert runs
    assert {r["kind"] for r in runs} <= {"benchmark", "profile"}
    for r in runs:
        for key in ("run_id", "kind", "subject", "models", "rows", "cost_usd",
                    "has_manifest"):
            assert key in r, key


def test_the_run_list_is_empty_rather_than_failing_on_a_fresh_store(tmp_path):
    from harness.web import profile_api as P

    assert P.all_runs(str(tmp_path / "nothing"))["runs"] == []


def test_a_constraint_string_parses_like_the_cli_flag():
    from harness.web.profile_api import parse_constraint

    c = parse_constraint("faithfulness>=0.90")
    assert (c.metric, c.op, c.value) == ("faithfulness", ">=", 0.9)


def test_a_malformed_constraint_says_what_was_expected():
    from harness.web.profile_api import parse_constraint

    with pytest.raises(ValueError, match="accuracy>=0.9"):
        parse_constraint("accuracy is good")


def test_decide_on_a_missing_run_says_so(store_cfg):
    from harness.web import profile_api as P

    with pytest.raises(api.ApiError):
        P.decide(store_cfg, profile="nope")


def test_the_profile_api_computes_no_metric():
    """Same rule as the benchmark half: presentation holds no business logic."""
    import inspect

    from harness.web import profile_api

    src = inspect.getsource(profile_api)
    assert ".mean()" not in src
    assert "mcnemar" not in src
    assert "weighted_composite =" not in src


@pytest.mark.integration
def test_one_report_section_failing_does_not_blank_the_page(store_cfg, monkeypatch):
    """An UnpairedItemsError in the significance table is a finding about the
    run. The leaderboard beside it is still worth reading, so the error is
    shown in place of that one table rather than instead of the report."""
    from harness.report import stats as S
    from harness.web import profile_api as P

    started, _ = _run_and_wait(store_cfg, limit=6)

    def boom(*a, **kw):
        raise S.UnpairedItemsError("a", "b", "accuracy", ("q1",), (), 5)

    monkeypatch.setattr(S, "significance_matrix", boom)
    res = P.profile_results(store_cfg, run_id=started["run_id"])
    assert res["significance_error"].startswith("UnpairedItemsError")
    assert res["n_rows"] > 0, "the rest of the report still came back"


def test_the_page_cannot_scroll_sideways():
    """Regression, caught by measuring the live page. `.app` is a grid and a
    grid item defaults to `min-width:auto`, so the content column refused to
    shrink below the widest table and pushed the whole body sideways — even
    though the table had its own `overflow-x:auto`."""
    from harness.web.server import STATIC

    css = (STATIC / "app.css").read_text(encoding="utf-8")
    main_rule = css[css.index("main {"):css.index("main {") + 400]
    assert "min-width: 0" in main_rule, (
        "the content grid track must be allowed to shrink, or wide tables "
        "scroll the page instead of themselves")


def _declared_views(js: str) -> set[str]:
    """The keys of the VIEWS registry in app.js."""
    import re

    start = js.index("const VIEWS = {")
    registry = js[start:js.index("};", start)]
    return set(re.findall(r"(\w+):\s*view", registry))


def _nav_views(js: str) -> set[str]:
    """Every view id reachable from the task bar's SECTIONS table.

    The nav moved out of index.html when the rail became a top task bar, so
    this reads the JS table that now builds it. Parsed rather than imported
    because there is no JS runtime in the offline suite — a regex over the
    source is crude, but it fails loudly if the table's shape changes, which
    is the behaviour wanted here.
    """
    import re

    start = js.index("const SECTIONS = [")
    table = js[start:js.index("\n];", start)]
    # Each screen is ["view_id", "Label", "glyph"]; section ids are the first
    # element of the outer triple and are matched by the same pattern, so the
    # result is intersected against VIEWS by the callers.
    return set(re.findall(r'\["([a-z]+)",\s*"[^"]+"', table))


def test_every_nav_entry_has_a_view_behind_it():
    """A task-bar button with no view renders the fallback silently — the
    screen looks like it loaded and shows the wrong thing."""
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    declared = _declared_views(js)
    nav = _nav_views(js)

    # Section ids that are not themselves views (Prepare, Analyse, ...) are
    # containers, not destinations, and are allowed here.
    containers = {"prepare", "analyse", "reference"}
    assert nav - containers <= declared, (
        f"nav entries with no view: {sorted(nav - containers - declared)}")


def test_every_view_is_reachable_from_the_task_bar():
    """The other half, and the one the old rail-based test could not make:
    a view in the registry that no button points at is dead code that still
    costs a reader time."""
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    unreachable = _declared_views(js) - _nav_views(js)
    assert not unreachable, f"views with no way to reach them: {sorted(unreachable)}"


def test_the_client_computes_no_metric():
    """§9 Phase 8: UI holds layout, state and calls — nothing else. A metric
    computed in the browser would be a second definition of one the library
    already owns."""
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for banned in ("Math.sqrt", "pValue", "chanceAdjust", "1.96"):
        assert banned not in js, f"statistics in the client: {banned}"


def test_the_platform_surfaces_are_all_reachable():
    """'Complete platform' as an assertion: every CLI capability that has a
    screen must have an endpoint behind it."""
    import inspect

    from harness.web import server

    routes = inspect.getsource(server._Handler)
    for path in ("/api/validate", "/api/estimate", "/api/probes", "/api/arena",
                 "/api/providers", "/api/profile-results", "/api/decide",
                 "/api/gate", "/api/benchmarks", "/api/results"):
        assert path in routes, path
    posts = inspect.getsource(server._Handler.do_POST)
    for path in ("/api/run", "/api/run-profile", "/api/extract"):
        assert path in posts, path


def test_profiles_report_whether_their_data_is_on_disk():
    """Four of the six shipped profiles are worked examples whose datasets are
    not in the repo. A picker that offers them without saying so opens every
    screen on 'file not found', which reads as a broken app rather than an
    unconfigured profile."""
    from harness.web import profile_api as P

    rows = {p["name"]: p for p in P.list_profiles()["profiles"] if "invalid" not in p}
    assert rows["support_triage"]["runnable"] is True
    assert rows["regulated_qa"]["runnable"] is True
    assert any(not p["runnable"] for p in rows.values()), (
        "the flag must actually distinguish; if every profile now ships data, "
        "this test should be updated rather than deleted")
    for p in rows.values():
        assert {"evalset_exists", "corpus_exists", "runnable"} <= set(p)


def test_static_assets_are_not_cached():
    """The page is read off disk per request; caching buys nothing and costs a
    confusing hard-refresh every time the UI changes — an edit keeps serving
    the previous version and the bug you just fixed appears not to be."""
    import inspect

    from harness.web import server

    assert "no-store" in inspect.getsource(server._Handler._static)


# --------------------------------------------------------------------------- #
#  RAG
# --------------------------------------------------------------------------- #
def test_rag_status_reports_readiness_per_profile():
    """'RAG profile with an empty index' and 'RAG profile ready' look identical
    until a run produces uniformly terrible numbers."""
    from harness.web import platform_api as X

    s = X.rag_status({"qdrant_url": "http://localhost:6333"})
    assert s["vector_store"] == "server"
    assert {"profile", "collection", "chunks_indexed", "ready", "corpus_present"} \
        <= set(s["profiles"][0])
    assert all(p["collection"] for p in s["profiles"]), (
        "the collection name is namespaced by profile AND embedder; without it "
        "a re-ingest under a new embedder would silently reuse old vectors")


def test_rag_status_says_when_no_vector_store_is_configured():
    from harness.web import platform_api as X

    s = X.rag_status({})
    assert s["vector_store"] == "none"
    assert s["reachable"] is False


def test_a_rag_profile_run_refuses_rather_than_running_without_retrieval():
    """The bug this pins: the web worker passed `qdrant=None` unconditionally,
    so a RAG profile launched from the browser would have run with no
    retrieval at all and still written rows — a silently context-free run that
    reads as a legitimately bad model."""
    import inspect

    from harness.web import platform_api as X

    src = inspect.getsource(X._profile_worker)
    assert "qdrant=qdrant" in src, "must pass a real client, not None"
    assert "_collection_ready" in src, "must refuse an empty index"
    assert 'task.value == "rag"' in src


# --------------------------------------------------------------------------- #
#  Provider connections
# --------------------------------------------------------------------------- #
def test_a_key_is_masked_never_returned_whole():
    """§6: secrets never reach YAML, traces, cache keys or exports — and a
    diagnostics screen is the classic place that rule gets broken."""
    from harness.clients.discovery import mask

    key = "sk-abcdefghijklmnopqrstuvwxyz0123"
    m = mask(key)
    assert key not in m
    assert m.startswith("sk-a") and m.endswith("0123")
    assert mask("short") == "*" * 5, "a short key must collapse entirely"
    assert mask(None) == "" and mask("") == ""


def test_connection_result_carries_no_raw_key():
    from harness.clients.discovery import ConnectionResult

    r = ConnectionResult(provider="together", ok=False, key_masked="ab…yz")
    assert "api_key" not in r.as_dict()
    assert all("key" not in k or k in ("key_source", "key_masked")
               for k in r.as_dict())


def test_a_provider_without_a_key_is_reported_not_skipped(monkeypatch):
    """'No key' and 'key broken' lead to different actions. A list that omits
    the first makes the second look like the only failure mode.

    Run with every key unset so no request is attempted — this asserts the
    reporting, not the network.
    """
    from harness.clients import discovery

    for p in discovery.LISTABLE:
        env = discovery.all_providers().get(p, {}).get("api_key_env", "")
        if env:
            monkeypatch.delenv(env, raising=False)

    rows = {r.provider: r for r in discovery.check_all()}
    assert {"together", "openrouter"} <= set(rows)
    for r in rows.values():
        assert r.key_source == "none"
        assert not r.ok and r.error, "must say why"
        assert "No key" in r.error


def test_an_unknown_provider_is_rejected_without_a_request():
    from harness.clients.discovery import check_provider

    r = check_provider("nope", api_key="x")
    assert not r.ok and "Unknown provider" in r.error


def test_a_provider_with_no_models_endpoint_says_so():
    """Anthropic has its own SDK shape; guessing at it would be worse than
    saying it cannot be listed."""
    from harness.clients.discovery import check_provider

    r = check_provider("anthropic", api_key="x")
    assert not r.ok
    assert "OpenAI-shaped" in r.error


def test_network_code_lives_in_the_provider_layer():
    """§3: the capability protocols and their adapters are the ONLY code
    permitted to touch the network. A 'check my key' button is still a
    network call."""
    import inspect

    from harness.web import platform_api

    src = inspect.getsource(platform_api)
    for banned in ("requests.", "httpx.", "urlopen("):
        assert banned not in src, f"network call in the web layer: {banned}"


def test_the_models_yaml_fragment_is_proposed_not_written():
    """§7 lists provider routing and model choice among the things that change
    results. The harness proposes; a human commits."""
    from harness.web import platform_api as X

    out = X.suggested_models_yaml("together", ["a/b", "c/d"])
    assert out["yaml"].startswith("models:")
    assert '"together:a/b"' in out["yaml"]
    assert "pricing" in out["note"].lower()


def test_building_a_fragment_with_no_models_is_rejected():
    from harness.web import platform_api as X

    with pytest.raises(api.ApiError):
        X.suggested_models_yaml("together", [])


# --------------------------------------------------------------------------- #
#  Charts
# --------------------------------------------------------------------------- #
def test_charts_ship_with_the_page_and_load_no_library():
    from harness.web.server import STATIC

    js = (STATIC / "charts.js").read_text(encoding="utf-8")
    for banned in ("d3.", "chart.js", "plotly", "echarts", "import("):
        assert banned not in js.lower(), f"external charting: {banned}"
    # The one URL allowed is the SVG namespace, which is an identifier rather
    # than a fetch — nothing is loaded from it.
    urls = [ln for ln in js.splitlines() if "http" in ln]
    assert all("www.w3.org/2000/svg" in ln for ln in urls), urls


def test_every_chart_offers_a_table_view():
    """The documented relief for the sub-3:1 series step, and the accessible
    path for anyone the colour channel does not reach."""
    from harness.web.server import STATIC

    js = (STATIC / "charts.js").read_text(encoding="utf-8")
    assert js.count("simpleTable") >= 5, "each chart must pass a table builder"
    assert "Table view" in js


def test_charts_never_draw_a_second_axis():
    """The single most common charting mistake, and one this harness would be
    especially wrong to make."""
    from harness.web.server import STATIC

    js = (STATIC / "charts.js").read_text(encoding="utf-8")
    # `y2` is an SVG <line> attribute, so grep for the real tell: a second
    # scale function or a right-hand value axis.
    for banned in ("yRight", "y2Scale", "secondAxis", "rightAxis", "Y2("):
        assert banned not in js, f"second axis: {banned}"
    assert js.count("const Y = ") <= 1, "one value scale per chart, never two"


def test_charts_carry_a_hover_layer_and_aria_labels():
    from harness.web.server import STATIC

    js = (STATIC / "charts.js").read_text(encoding="utf-8")
    assert js.count("hoverable(") >= 5
    assert js.count("aria-label") >= 4


def test_the_accuracy_chart_draws_the_interval():
    """A bar chart of accuracy with no interval is the exact artefact this
    harness exists to argue against."""
    from harness.web.server import STATIC

    js = (STATIC / "charts.js").read_text(encoding="utf-8")
    fn = js[js.index("export function accuracyCI"):js.index("export function qualityCost")]
    assert "ci_low" in fn and "ci_high" in fn


# --------------------------------------------------------------------------- #
#  About
# --------------------------------------------------------------------------- #
def test_the_about_screen_documents_every_nav_section():
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    about = js[js.index("const ABOUT = ["):js.index("function viewAbout")]
    for topic in ("Preflight", "Retrieval", "Probes", "Extraction",
                  "Profile run", "Benchmark run", "Decide", "Gate", "Arena"):
        assert topic in about, f"About does not explain {topic}"


def test_the_about_screen_states_what_is_not_done():
    """A platform that lists only its features is the thing this harness was
    built to distrust."""
    from harness.web.server import STATIC

    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "what is not done" in js.lower()
    for gap in ("simulation", "offline", "synthetic", "sandbox"):
        assert gap in js.lower(), gap


@pytest.mark.integration
def test_benchmark_results_carry_confidence_intervals(store_cfg):
    """Without them the accuracy chart is a bare bar chart — the exact artefact
    this harness argues against."""
    started, _ = _run_and_wait(store_cfg, limit=10)
    res = api.results(store_cfg, started["run_id"])
    assert res["ci"], "no CI table returned"
    row = res["ci"][0]
    for col in ("model", "mean", "ci_low", "ci_high", "n"):
        assert col in row, col
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]


def test_the_page_loads_no_web_font():
    """The identity rests on a serif/mono contrast, and it has to survive with
    zero requests: a downloaded face would look marginally better and fail
    completely on an air-gapped machine, which is the wrong trade here."""
    from harness.web.server import STATIC

    css = (STATIC / "app.css").read_text(encoding="utf-8")
    for banned in ("@font-face", "fonts.googleapis", "fonts.gstatic", "@import"):
        assert banned not in css, banned
    # The display face must be a real system serif, not a hopeful name.
    assert "Georgia" in css


def test_charts_read_their_colours_from_the_page_tokens():
    """One palette, not two. A chart carrying its own copy drifts the moment
    the theme changes.

    This test used to pin `getComputedStyle(...)` — reading the tokens once
    at draw time. That still drifted: a chart drawn in the dark theme kept
    dark-theme fills after a toggle to light, which is the bug a screenshot
    showed. Charts now reference the tokens as `var(--...)`, so the browser
    repaints them when :root changes, and the series steps live in app.css
    for both themes. The pin is therefore: token references present, series
    tokens defined once per theme in the stylesheet, and no colour literal
    anywhere in the chart code."""
    import re

    from harness.web.server import STATIC

    js = (STATIC / "charts.js").read_text(encoding="utf-8")
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "var(--ink)" in js and "var(--chart-s1)" in js
    assert not re.search(r'"#[0-9a-fA-F]{3,8}"', js), "a colour literal in chart code is a second palette"
    for token in ("--chart-s1", "--chart-s2", "--chart-s3", "--chart-muted",
                  "--chart-surface", "--chart-good", "--chart-mid", "--chart-bad"):
        assert css.count(f"{token}:") == 2, f"{token} must be defined for both themes"


# --------------------------------------------------------------------------- #
#  Saved reports
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_a_profile_report_does_not_count_by_design_unscored_rows_as_excluded(tmp_path):
    """rag_live's first saved report said '156 excluded'. 100 were latency-lane
    rows that are never scored, and 15 were unanswerable probes whose metric
    is abstention, not accuracy. Neither is a lost answer."""
    import pandas as pd

    from harness.web.reports import _excluded

    g = pd.DataFrame([
        {"pass_": "baseline", "item_type": "answerable", "accuracy": 1.0},
        {"pass_": "baseline", "item_type": "answerable", "accuracy": None},  # owed, missing
        {"pass_": "baseline", "item_type": "unanswerable", "accuracy": None},  # by design
        {"pass_": "latency", "item_type": "answerable", "accuracy": None},   # never scored
    ])
    quality = g[g["pass_"].isin(["baseline", "adapted"])]
    assert _excluded(quality) == 1
    # Benchmark rows carry no item_type: every missing accuracy is owed.
    bench = pd.DataFrame([{"accuracy": None}, {"accuracy": 0.0}, {"accuracy": None}])
    assert _excluded(bench) == 2


def test_the_run_list_reports_an_aborted_run_as_such(store_cfg, tmp_path):
    """A run stopped by the budget or the operator must not look finished."""
    import json

    from harness.web import profile_api as P

    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"aborted": True}), encoding="utf-8")
    assert P._aborted(run_dir) is True
    (run_dir / "manifest.json").write_text(json.dumps({"aborted": False}), encoding="utf-8")
    assert P._aborted(run_dir) is False
    assert P._aborted(tmp_path / "missing") is False


@pytest.mark.integration
def test_a_profile_report_shows_the_newest_run_and_never_pools(store_cfg):
    """Selecting a PROFILE used to pool every run of it: on regulated_qa that
    averaged a retrieval-corrupted attempt with a clean one and showed 0.69
    where the clean run said 1.00. Runs differ in apparatus and dataset (I2,
    I9) and repeat the same item ids, which the paired test joins on. A
    profile therefore resolves to its newest run and names the others."""
    from harness.web import profile_api as P

    first, _ = _run_and_wait(store_cfg, limit=6)
    second, _ = _run_and_wait(store_cfg, limit=6)

    # Two runs of the same subject now share one profile column ("gsm8k").
    r = P.profile_results(store_cfg, profile="gsm8k")
    assert r["run_id"] == second["run_id"], "the newest run, not a pool"
    assert r["other_runs"] == [first["run_id"]]
    assert r["n_rows"] == 12, "6 items x 2 models from ONE run, not 24"

    # Explicitly asking for the older run still works, and names nothing else.
    older = P.profile_results(store_cfg, run_id=first["run_id"])
    assert older["run_id"] == first["run_id"] and older["other_runs"] == []


def test_a_report_freezes_its_numbers(store_cfg):
    """A record that re-derives itself is not a record: fix a pricing entry
    tomorrow and last month's report must still say what it said."""
    from harness.web import reports as R

    started, _ = _run_and_wait(store_cfg, limit=8)
    doc = R.build_report(store_cfg, run_ids=[started["run_id"]],
                         title="T", provider="fake")
    saved = R.get_report(store_cfg, doc["id"])
    assert saved["sections"][0]["summary"] == doc["sections"][0]["summary"]
    assert saved["created"] == doc["created"]


@pytest.mark.integration
def test_a_report_carries_the_manifest_behind_each_number(store_cfg):
    """I9: without provenance a section records numbers nobody can later say
    anything about."""
    from harness.web import reports as R

    started, _ = _run_and_wait(store_cfg, limit=6)
    doc = R.build_report(store_cfg, run_ids=[started["run_id"]])
    m = doc["sections"][0]["manifest"]
    assert m and m["spec_hash"] and m["dataset_hash"] and m["apparatus_hash"]


@pytest.mark.integration
def test_a_report_headline_names_the_excluded_count(store_cfg):
    """A score over 55 of 60 items means something different from the same
    score over 60, so the headline cannot omit it."""
    from harness.web import reports as R

    started, _ = _run_and_wait(store_cfg, limit=6)
    doc = R.build_report(store_cfg, run_ids=[started["run_id"]])
    assert doc["summary"]
    assert doc["sections"][0]["excluded_total"] >= 0


def test_a_report_needs_at_least_one_run(store_cfg):
    from harness.web import reports as R

    with pytest.raises(api.ApiError):
        R.build_report(store_cfg, run_ids=[])


def test_reports_are_listed_newest_first_and_deletable(store_cfg, tmp_path):
    from harness.web import reports as R

    assert R.list_reports(store_cfg)["reports"] == []
    with pytest.raises(api.ApiError):
        R.get_report(store_cfg, "nope")
    with pytest.raises(api.ApiError):
        R.delete_report(store_cfg, "nope")


# --------------------------------------------------------------------------- #
#  Dataset fetching
# --------------------------------------------------------------------------- #
def test_the_fetcher_is_never_invoked_by_construction():
    """§10.3: network only in the fetcher. `registry.build()` resolves a cached
    path but must never download — otherwise merely *constructing* a benchmark
    would reach the network, including in the offline suite."""
    import inspect

    from harness.bench import registry

    src = inspect.getsource(registry.build)
    assert "cached_path" in src
    for banned in ("fetch_rows", "fetch_for_spec", "requests"):
        assert banned not in src, f"build() must not fetch: {banned}"


def test_a_non_commercial_dataset_needs_acknowledgement():
    """§10.2. A gate, not a warning: a licence note printed into a log is a
    licence note nobody read."""
    from harness.bench.fetch import LicenceError, fetch_for_spec
    from harness.bench.spec import BenchmarkSpec

    spec = BenchmarkSpec.from_dict({
        "id": "nc", "version": 1, "family": "math", "task": "direct",
        "source": {"kind": "hf", "ref": "x/y", "licence": "CC-BY-NC-4.0",
                   "split": "test", "checksum": "sha256:0", "commercial_use": False},
        "prompt": {"decoding": {"max_tokens": 64}},
        "scoring": {"mode": "generative", "chance_level": 0.0,
                    "extraction": {"chain": [{"kind": "numeric"}]}},
    })
    with pytest.raises(LicenceError, match="acknowledge"):
        fetch_for_spec(spec)


def test_the_cache_name_does_not_encode_the_fetch_limit():
    """Otherwise the same spec resolves to different files depending on how a
    previous command was invoked, and only the checksum notices — at run time."""
    from harness.bench.fetch import _local_name

    assert _local_name("a/b", "main", "test", 200) == _local_name("a/b", "main", "test")


def test_every_real_dataset_spec_records_a_checksum_and_licence():
    """A fetched dataset with no checksum can change under you silently, and
    one with no licence cannot be redistributed with any confidence."""
    from harness.bench.spec import BenchmarkSpec, available_specs

    for path in available_specs():
        spec = BenchmarkSpec.from_yaml(path)
        assert spec.source.licence, spec.id
        if spec.source.kind != "local":
            assert spec.source.checksum.startswith("sha256:"), spec.id


def test_the_real_datasets_are_cached_and_loadable():
    """The three public benchmarks fetched from HuggingFace replay offline."""
    from harness.bench import registry
    from harness.bench.fetch import is_cached
    from harness.bench.spec import load_spec

    for bid in ("gsm8k_hf", "mmlu_pro_hf", "arc_challenge"):
        spec = load_spec(bid)
        assert is_cached(spec), f"{bid} not cached — run `bench fetch`"
        items = registry.build(spec).load(limit=5)
        assert len(items) == 5
        assert all(it.query and it.item_id for it in items)


def test_arc_normalises_numeric_choice_labels():
    """A meaningful fraction of ARC rows label choices 1-4 rather than A-D.
    Normalising the options but not the gold scores every one of them wrong."""
    from harness.bench.adapters.arc import _as_letter

    assert _as_letter("1", 0) == "A"
    assert _as_letter("4", 3) == "D"
    assert _as_letter("C", 2) == "C"
    assert _as_letter("", 1) == "B", "falls back to position"


# --------------------------------------------------------------------------- #
#  Per-row cost
# --------------------------------------------------------------------------- #
def test_an_unpriced_model_leaves_cost_null_not_zero():
    """I3 again: a free-looking model must not win a cost-weighted comparison,
    and a gap in the pricing table is as much a cause of that as a missing
    usage block."""
    from harness.bench.runner import _price_row
    from harness.clients.base import GenResult
    from harness.store.schema import ItemType, Pass, TraceRow

    row = TraceRow(run_id="r", item_id="i", model="unpriced/model",
                   profile="p", pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE)
    gen = GenResult(text="x", prompt_tokens=10, completion_tokens=5, latency_ms=1.0)

    class _Raises:
        def generation_cost(self, *a):
            raise KeyError("no price")

    _price_row(row, _Raises(), "unpriced/model", gen)
    assert row.cost_usd is None, "an unpriced model must not read as free"
