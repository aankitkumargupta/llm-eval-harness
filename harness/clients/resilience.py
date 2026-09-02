"""
Retry, backoff and rate limiting — the layer that makes a long run survivable.

A real benchmark run is thousands of calls against a shared hosted API. Without
this module a single 429 in hour three turns into an error row, and the run's
accuracy number silently drops because some items never got an answer. That is
the worst possible failure mode for an evaluation harness: it doesn't crash, it
*lies*.

Two mechanisms, deliberately separate:

  RateLimiter  — proactive. A token bucket that paces requests so we mostly
                 never hit the provider's limit in the first place.
  retry_call   — reactive. Exponential backoff with full jitter for the errors
                 that get through anyway, honouring `Retry-After` when the
                 provider tells us how long to wait.

Only *transient* failures are retried. A 400 (bad request) or 404 (no such
model) is a bug in your config; retrying it wastes minutes and hides the
problem, so those raise immediately.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
#  Which failures are worth retrying                                           #
# --------------------------------------------------------------------------- #
# Retry: rate limits, server errors, gateway hiccups, and timeouts.
# Do NOT retry: auth, bad request, model-not-found, context-length-exceeded.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Substrings that identify a transient transport failure when no status code is
# available (raw socket errors, DNS blips, connection resets).
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection reset", "connection aborted",
    "connection error", "temporarily unavailable", "remote end closed",
    "bad gateway", "service unavailable", "overloaded", "rate limit",
    "too many requests", "eof occurred",
)


def _status_of(exc: BaseException) -> int | None:
    """Dig an HTTP status code out of whatever exception shape a client raises.

    The `openai` SDK exposes `.status_code`, `requests` nests it under
    `.response.status_code`, and some wrappers only carry `.code`. We check all
    three rather than depending on one SDK's exception hierarchy — this module
    must stay provider-agnostic.
    """
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val
    return None


def _retry_after_of(exc: BaseException) -> float | None:
    """Honour a `Retry-After` header if the provider sent one.

    Providers know their own backoff better than our exponential curve does, so
    when they tell us, we listen (capped by the caller's max_backoff).
    """
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None  # HTTP-date form; fall back to exponential backoff


def is_retryable(exc: BaseException) -> bool:
    """True if `exc` looks transient and is worth another attempt."""
    status = _status_of(exc)
    if status is not None:
        return status in RETRYABLE_STATUS
    text = f"{type(exc).__name__} {exc}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


# --------------------------------------------------------------------------- #
#  Retry policy                                                                #
# --------------------------------------------------------------------------- #
@dataclass
class RetryPolicy:
    """How hard to try before giving up on one call.

    Defaults target a ~2 minute worst case per call: enough to ride out a rate
    limit window, short enough that a genuinely dead endpoint fails the run
    rather than hanging it.
    """
    max_attempts: int = 5
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True

    def backoff_for(self, attempt: int, rng: random.Random) -> float:
        """Delay before attempt `attempt` (1-based, so attempt 2 is the first retry).

        Uses *full* jitter — a uniform draw over [0, capped_delay] rather than
        the capped delay itself. With many worker threads hitting the same 429
        at once, deterministic backoff makes them all retry in lockstep and
        re-trigger the limit; full jitter spreads them out.
        """
        capped = min(self.max_backoff,
                     self.initial_backoff * (self.multiplier ** max(0, attempt - 1)))
        return rng.uniform(0.0, capped) if self.jitter else capped


class RetryExhausted(RuntimeError):
    """Raised when every attempt failed. Carries the last underlying error."""

    def __init__(self, attempts: int, last: BaseException):
        super().__init__(f"gave up after {attempts} attempt(s): "
                         f"{type(last).__name__}: {last}")
        self.attempts = attempts
        self.last = last


def retry_call(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    *,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call `fn`, retrying transient failures with jittered exponential backoff.

    `sleep` and `rng` are injected so tests can run the full retry path without
    actually waiting or depending on random draws.
    """
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    last: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified immediately below
            if not is_retryable(exc):
                raise  # permanent: a config bug, surface it now
            last = exc
            if attempt >= policy.max_attempts:
                break
            delay = _retry_after_of(exc)
            if delay is None:
                delay = policy.backoff_for(attempt, rng)
            delay = min(delay, policy.max_backoff)
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            sleep(delay)

    raise RetryExhausted(policy.max_attempts, last)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
#  Proactive pacing                                                            #
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Thread-safe token bucket, shared by every worker thread of a client.

    `rate` tokens accrue per second up to `burst`. Each request takes one token;
    when the bucket is empty, `acquire()` blocks until one refills. This keeps
    the *average* request rate under the provider's limit while still allowing a
    short burst — which is exactly the shape of a hosted API quota.

    rate <= 0 disables limiting entirely (the default: most users are well under
    their quota and shouldn't pay a latency tax for pacing they don't need).
    """

    def __init__(self, rate: float = 0.0, burst: float | None = None):
        self.rate = float(rate)
        self.burst = float(burst) if burst is not None else max(1.0, float(rate))
        self._tokens = self.burst
        self._last = time.monotonic()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available. Returns how long we waited.

        The sleep happens *outside* the lock: holding it while sleeping would
        serialise every worker behind the first one to find an empty bucket,
        turning a rate limiter into a global mutex.
        """
        if not self.enabled:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst,
                                   self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return waited
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            time.sleep(wait)
            waited += wait
