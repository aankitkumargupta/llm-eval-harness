"""
Tests for the reliability layer: retries, rate limiting, cost metering, budgets.

These matter more than they look. A retry bug doesn't crash a run — it quietly
turns transient failures into error rows, and the accuracy number drops because
some items never got an answer. That is the worst failure mode an evaluation
harness has: it doesn't stop, it lies.
"""

from __future__ import annotations

import os
import random
import time

import pytest

from harness.clients.cost import (
    BudgetExceeded,
    CostMeter,
    forecast_run,
)
from harness.clients.resilience import (
    RateLimiter,
    RetryExhausted,
    RetryPolicy,
    is_retryable,
    retry_call,
)


class HTTPError(Exception):
    """Stand-in for a provider SDK exception carrying a status code."""

    def __init__(self, code: int, retry_after: str | None = None):
        super().__init__(f"HTTP {code}")
        self.status_code = code
        if retry_after is not None:
            self.response = type("R", (), {"headers": {"retry-after": retry_after}})()


# --------------------------------------------------------------------------- #
#  Classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code,expected", [
    (429, True), (500, True), (502, True), (503, True), (504, True), (408, True),
    (400, False), (401, False), (403, False), (404, False), (422, False),
])
def test_retryable_status_codes(code, expected):
    assert is_retryable(HTTPError(code)) is expected


def test_transient_transport_errors_are_retryable_without_a_status_code():
    # Raw socket failures carry no status; we fall back to message matching.
    assert is_retryable(Exception("Read timed out"))
    assert is_retryable(Exception("Connection reset by peer"))
    assert not is_retryable(Exception("invalid model identifier"))


# --------------------------------------------------------------------------- #
#  Retry behaviour
# --------------------------------------------------------------------------- #
def test_retries_then_succeeds():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise HTTPError(429)
        return "ok"

    result = retry_call(flaky, RetryPolicy(max_attempts=5),
                        sleep=lambda _: None, rng=random.Random(0))
    assert result == "ok"
    assert attempts["n"] == 3


def test_permanent_error_is_not_retried():
    """A 404 is a bad model string. Retrying wastes minutes and hides the bug."""
    attempts = {"n": 0}

    def bad_model():
        attempts["n"] += 1
        raise HTTPError(404)

    with pytest.raises(HTTPError):
        retry_call(bad_model, RetryPolicy(max_attempts=5), sleep=lambda _: None)
    assert attempts["n"] == 1


def test_exhaustion_raises_and_keeps_the_cause():
    def always_fails():
        raise HTTPError(503)

    with pytest.raises(RetryExhausted) as exc:
        retry_call(always_fails, RetryPolicy(max_attempts=3), sleep=lambda _: None)
    assert exc.value.attempts == 3
    assert isinstance(exc.value.last, HTTPError)


def test_retry_after_header_is_honoured():
    """The provider knows its own backoff better than our curve does."""
    slept: list[float] = []
    attempts = {"n": 0}

    def rate_limited():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise HTTPError(429, retry_after="7")
        return "ok"

    retry_call(rate_limited, RetryPolicy(max_attempts=3, max_backoff=30),
               sleep=slept.append, rng=random.Random(0))
    assert slept == [7.0]


def test_retry_after_is_capped_by_max_backoff():
    """A hostile or mistaken Retry-After must not stall the run for an hour."""
    slept: list[float] = []
    attempts = {"n": 0}

    def rate_limited():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise HTTPError(429, retry_after="3600")
        return "ok"

    retry_call(rate_limited, RetryPolicy(max_attempts=3, max_backoff=30),
               sleep=slept.append)
    assert slept == [30.0]


def test_backoff_is_jittered_and_bounded():
    """Full jitter: without it, N workers hitting one 429 retry in lockstep."""
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=10.0)
    rng = random.Random(0)
    delays = [policy.backoff_for(a, rng) for a in range(1, 6)]
    assert all(0.0 <= d <= 10.0 for d in delays)
    assert len(set(delays)) > 1, "jitter should not produce identical delays"


def test_backoff_without_jitter_is_exponential_and_capped():
    policy = RetryPolicy(initial_backoff=1.0, multiplier=2.0, max_backoff=8.0,
                         jitter=False)
    rng = random.Random(0)
    assert [policy.backoff_for(a, rng) for a in range(1, 6)] == [1, 2, 4, 8, 8]


# --------------------------------------------------------------------------- #
#  Rate limiting
# --------------------------------------------------------------------------- #
def test_rate_limiter_disabled_by_default():
    rl = RateLimiter(0)
    assert not rl.enabled
    assert rl.acquire() == 0.0


def test_rate_limiter_paces_requests():
    rl = RateLimiter(rate=100, burst=2)
    t0 = time.perf_counter()
    for _ in range(6):
        rl.acquire()
    elapsed = time.perf_counter() - t0
    # 2 free from the burst, 4 more at 100/s = ~40ms. Generous upper bound so
    # this can't flake on a loaded CI box.
    assert 0.02 < elapsed < 1.0


# --------------------------------------------------------------------------- #
#  Cost metering
# --------------------------------------------------------------------------- #
def test_cost_breakdown_separates_judge_from_generation():
    """The judge is often the majority of the bill; one total would hide it."""
    m = CostMeter()
    m.record_generation(0.001, 1000, 100, is_judge=False)
    m.record_generation(0.004, 1400, 120, is_judge=True)
    m.record_embedding(0.0001, 50)
    m.record_rerank(0.0002)

    assert m.costs.generation == pytest.approx(0.001)
    assert m.costs.judge == pytest.approx(0.004)
    assert m.spent == pytest.approx(0.0053)
    assert m.counts.generation == 1 and m.counts.judge == 1


def test_budget_trips_and_reports_the_overspend():
    m = CostMeter(budget_usd=0.01)
    with pytest.raises(BudgetExceeded) as exc:
        for _ in range(100):
            m.record_generation(0.002, 100, 10)
    assert exc.value.limit == pytest.approx(0.01)
    assert m.tripped
    # Overshoot is bounded by a single call: we can't price a call before its
    # usage block comes back.
    assert m.spent <= 0.012


def test_no_budget_means_no_ceiling():
    m = CostMeter()
    for _ in range(1000):
        m.record_generation(1.0, 10, 10)
    assert m.remaining == float("inf")
    assert not m.tripped


def test_cache_hit_rate():
    m = CostMeter()
    for _ in range(3):
        m.record_generation(0.001)
    for _ in range(7):
        m.record_cache_hit()
    assert m.summary()["cache_hit_rate"] == pytest.approx(0.7)


def test_cost_meter_is_thread_safe():
    """The throughput lane bills from many workers at once."""
    import concurrent.futures as cf

    m = CostMeter()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: m.record_generation(0.001, 10, 1), range(800)))
    assert m.counts.generation == 800
    assert m.spent == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
#  Forecasting
# --------------------------------------------------------------------------- #
class _Pricing:
    models = {"a": {"input": 1.0, "output": 2.0},
              "judge": {"input": 5.0, "output": 5.0}}

    def generation_cost(self, model, p, c):
        if model not in self.models:
            raise KeyError(model)
        return p / 1e6 * self.models[model]["input"] + \
            c / 1e6 * self.models[model]["output"]


def test_forecast_includes_the_judge():
    """The original estimate omitted judge calls entirely - the bigger half."""
    with_judge = forecast_run(_Pricing(), ["a"], n_test=100,
                              judge_model="judge")
    without = forecast_run(_Pricing(), ["a"], n_test=100, judge_model="")
    assert with_judge.judge_calls == 200      # two judge calls per item
    assert without.judge_calls == 0
    assert with_judge.est_usd > without.est_usd


def test_forecast_flags_unpriced_models_rather_than_under_reporting_silently():
    fc = forecast_run(_Pricing(), ["a", "mystery-model"], n_test=10)
    assert "mystery-model" in fc.detail["unpriced_models"]


def test_forecast_scales_with_the_adapted_pass():
    """The adapted pass is ~tuning_budget times bigger - the estimate must say so."""
    base = forecast_run(_Pricing(), ["a"], n_test=100, n_dev=30,
                        tuning_budget=20, do_adapted=False)
    adapted = forecast_run(_Pricing(), ["a"], n_test=100, n_dev=30,
                           tuning_budget=20, do_adapted=True)
    assert adapted.calls > base.calls * 5


# =========================================================================== #
#  .env loading - the documented setup path
# =========================================================================== #
def test_dotenv_file_is_actually_loaded(tmp_path, monkeypatch):
    """The bug this fixes: `python-dotenv` was a dependency and both docs told
    people to create `.env`, but nothing ever called `load_dotenv()`. Following
    the documented setup produced "No API key", with the key sitting in a file
    the app never opened."""
    from harness.env import load_env

    env = tmp_path / ".env"
    env.write_text("TOGETHER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    applied = load_env(env)
    assert applied == ["TOGETHER_API_KEY"]
    assert os.environ["TOGETHER_API_KEY"] == "from-file"


def test_a_real_environment_variable_beats_the_file(tmp_path, monkeypatch):
    """Precedence: a key exported in the shell, injected by Docker or supplied
    by a CI secret store is more specific than a file in the working directory,
    and must not be clobbered by a stale .env someone forgot about."""
    from harness.env import load_env

    env = tmp_path / ".env"
    env.write_text("TOGETHER_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("TOGETHER_API_KEY", "from-shell")

    assert load_env(env) == []
    assert os.environ["TOGETHER_API_KEY"] == "from-shell"
    # ...unless the caller explicitly asks to override.
    load_env(env, override=True)
    assert os.environ["TOGETHER_API_KEY"] == "from-file"


def test_a_missing_env_file_is_not_an_error(tmp_path):
    """Plenty of deployments only ever use real environment variables."""
    from harness.env import load_env

    assert load_env(tmp_path / "does-not-exist") == []


def test_key_file_shapes_people_actually_write(tmp_path, monkeypatch):
    """Comments, blanks, `export` prefixes and quotes - without needing the
    optional python-dotenv package installed."""
    from harness.env import _parse

    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "TOGETHER_API_KEY=plain\n"
        'OPENROUTER_API_KEY="quoted"\n'
        "export GROQ_API_KEY='single'\n"
        "MALFORMED_LINE\n",
        encoding="utf-8")

    values = _parse(env)
    assert values == {"TOGETHER_API_KEY": "plain",
                      "OPENROUTER_API_KEY": "quoted",
                      "GROQ_API_KEY": "single"}


def test_every_entry_point_loads_the_env_file():
    """A key file honoured by the CLI but not the app would be a worse trap than
    one honoured by neither."""
    from pathlib import Path

    for entry in ("app.py", "main.py", "dashboard.py"):
        src = Path(entry).read_text(encoding="utf-8")
        assert "load_env" in src, f"{entry} does not load .env"
