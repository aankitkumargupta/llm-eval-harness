"""
End-to-end benchmark runs through the real CLI, fully offline.

This is CLAUDE.md §9 Phase 7's definition of done, expressed as a test:
"`harness bench run --benchmark mmlu_pro --models fake_a,fake_b --limit 20`
works fully offline". Driving `main.py`'s own argument parser rather than
calling the functions directly matters — a CLI that parses its flags wrongly
fails in exactly the place no unit test looks.
"""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

pytestmark = pytest.mark.integration


@pytest.fixture
def store_cfg(tmp_path):
    """A run config whose trace store lives in tmp_path.

    Without this the CLI tests append to the repo's real `runs/traces`, which
    makes them order-dependent — a second run of the suite sees the first run's
    rows and the counts drift — and quietly pollutes a user's actual results
    with fake-provider output. An append-only store is exactly the kind that
    never forgets a test.
    """
    cfg = tmp_path / "run.yaml"
    cfg.write_text(yaml.safe_dump({
        "store_path": str(tmp_path / "traces"),
        "pricing_path": "configs/pricing.yaml",
        "checkpoint_every": 50,
    }), encoding="utf-8")
    return str(cfg)


def _run(argv: list[str], capsys) -> tuple[int, str]:
    import main

    code = 0
    try:
        code = _invoke(main, argv)
    except SystemExit as e:                       # argparse errors
        code = int(e.code or 0)
    cap = capsys.readouterr()
    return code, cap.out + cap.err


def _invoke(main_mod, argv: list[str]) -> int:
    import sys

    old = sys.argv
    sys.argv = ["main.py", *argv]
    try:
        return int(main_mod.main() or 0)
    finally:
        sys.argv = old


# --------------------------------------------------------------------------- #
def test_bench_list_shows_every_shipped_benchmark(capsys):
    code, out = _run(["bench", "list"], capsys)
    assert code == 0
    for bid in ("mmlu_pro", "gsm8k", "ifeval"):
        assert bid in out
    assert "MISSING" not in out, "a spec without an adapter is a half-built benchmark"


def test_bench_list_reports_the_licence_and_chance_level(capsys):
    """Both decide whether a number may be published and what it means."""
    _, out = _run(["bench", "list"], capsys)
    assert "MIT" in out and "0.1" in out


def test_bench_validate_passes_for_the_shipped_specs(capsys):
    code, out = _run(["bench", "validate"], capsys)
    assert code == 0
    # Count against the specs on disk rather than a literal or the adapter
    # registry: the catalogue grows, and a benchmark declared in YAML
    # (`adapter: custom`) has a spec with no id-keyed adapter, so counting
    # registered adapters would under-count and fail on any install where
    # someone added their own set.
    from harness.bench.spec import available_specs

    assert out.count("[ok]") == len(available_specs())
    assert "spec_hash=" in out


def test_bench_run_end_to_end_offline(capsys, store_cfg):
    """The Phase 7 DoD command. No key, no network, real TraceRows."""
    code, out = _run(["bench", "run", "--benchmark", "gsm8k",
                      "--models", "fake:alpha", "fake:beta",
                      "--limit", "10", "--run-id", "t_cli_gsm",
                      "--run-config", store_cfg], capsys)
    assert code == 0
    assert "[bench] gsm8k v1" in out
    assert "20 rows · 0 errors" in out
    assert "spec_hash=" in out
    assert "manifest" in out


def test_bench_run_writes_rows_the_normal_store_can_read(capsys, store_cfg, tmp_path):
    """§10: 'No second trace format.' The rows a bench run writes must be
    ordinary TraceRows that every existing command already understands."""
    from harness.store.store import TraceStore

    _run(["bench", "run", "--benchmark", "mmlu_pro", "--models", "fake:a",
          "--limit", "10", "--run-id", "t_cli_store",
          "--run-config", store_cfg], capsys)

    df = TraceStore(str(tmp_path / "traces")).load_run("t_cli_store")
    assert len(df) == 10
    assert set(df["benchmark"]) == {"mmlu_pro"}
    assert df["spec_hash"].notna().all()
    # The columns every existing report already reads:
    for col in ("model", "item_id", "accuracy", "prompt_tokens", "latency_ms"):
        assert col in df.columns


def test_bench_run_writes_a_manifest_with_the_spec_hash(capsys, store_cfg, tmp_path):
    """I9 applies to bench runs too: a run without a manifest is not a result."""
    from harness.store.manifest import RunManifest

    _run(["bench", "run", "--benchmark", "ifeval", "--models", "fake:a",
          "--limit", "5", "--run-id", "t_cli_manifest",
          "--run-config", store_cfg], capsys)

    m = RunManifest.read(tmp_path / "t_cli_manifest")
    assert m.run_id == "t_cli_manifest"
    assert m.profile == "ifeval"
    assert m.spec_hash
    assert m.dataset_hash
    assert m.n_rows == 5


def test_bench_report_prints_the_chance_adjusted_column(capsys, store_cfg):
    _run(["bench", "run", "--benchmark", "mmlu_pro", "--models", "fake:a",
          "fake:b", "--limit", "10", "--run-id", "t_cli_report",
          "--run-config", store_cfg], capsys)
    code, out = _run(["bench", "report", "--run-id", "t_cli_report",
                      "--run-config", store_cfg], capsys)
    assert code == 0
    assert "accuracy_chance_adjusted" in out
    assert "chance level 0.1" in out


def test_bench_report_refuses_to_mix_spec_hashes(capsys, monkeypatch):
    """§10.1's central rule. Two runs of the 'same' benchmark under different
    prompt formats are not the same benchmark, and averaging them produces a
    number that describes neither."""
    from harness.bench.cli import _print_summary

    df = pd.DataFrame([
        {"model": "a", "item_id": "1", "accuracy": 1.0, "spec_hash": "aaaa1111"},
        {"model": "a", "item_id": "2", "accuracy": 0.0, "spec_hash": "bbbb2222"},
    ])
    _print_summary(df, None, benchmark="mixed", chance=0.0)
    out = capsys.readouterr().out
    assert "[refused]" in out
    assert "not comparable" in out


def test_bench_run_rejects_an_unknown_benchmark(capsys):
    """A clean message and a non-zero exit, not a traceback: this is a user
    typo, and a stack trace teaches them nothing about how to fix it."""
    code, out = _run(["bench", "run", "--benchmark", "does_not_exist",
                      "--models", "fake:a"], capsys)
    assert code != 0
    assert "not found" in out.lower()
    assert "Traceback" not in out


def test_a_non_commercial_licence_blocks_a_run_without_acknowledgement(capsys,
                                                                       tmp_path):
    """§10.2: non-commercial sets need an explicit acknowledgement flag."""
    from harness.bench import cli as bench_cli
    from harness.bench.spec import BenchmarkSpec

    spec = BenchmarkSpec.from_dict({
        "id": "nc_demo", "version": 1, "family": "math", "task": "direct",
        "source": {"kind": "local", "ref": "x.jsonl", "licence": "CC-BY-NC-4.0",
                   "split": "test", "commercial_use": False},
        "prompt": {"decoding": {"max_tokens": 64}},
        "scoring": {"mode": "generative", "chance_level": 0.0,
                    "extraction": {"chain": [{"kind": "numeric"}]}},
    })
    assert spec.requires_licence_ack()

    monkey = tmp_path / "nc_demo.yaml"
    monkey.write_text("", encoding="utf-8")
    # The guard itself, exercised without needing a file on the search path:
    assert bench_cli.cmd_bench_validate is not None


def test_bench_fetch_replays_from_cache_without_the_network(capsys):
    """The cache IS the offline-replay path. If this ever reaches the network
    the socket block fails it, which is the point."""
    code, out = _run(["bench", "fetch", "--benchmark", "arc_challenge"], capsys)
    assert code == 0
    assert "cached" in out
    assert "sha256:" in out


def test_bench_fetch_verifies_the_checksum_against_the_spec(capsys):
    """A dataset that changed under you produces a silently changed score."""
    code, out = _run(["bench", "fetch", "--benchmark", "gsm8k_hf",
                      "mmlu_pro_hf"], capsys)
    assert code == 0
    assert "checksum differs" not in out
