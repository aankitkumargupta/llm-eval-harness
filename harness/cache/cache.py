"""
Content-addressed cache for API responses.

Why this exists: reporting and evaluators get re-run many times against the same
model outputs. Hosted inference is billed per call and isn't bit-for-bit
deterministic over time, so caching the raw response by a hash of its inputs
buys BOTH cost savings AND reproducibility. The cache, not the seed, is the
real reproducibility guarantee.

What is cached:
  * **Generations**, keyed by (model, messages, params).
  * **Judge verdicts**, keyed by (judge model, rubric, question, context,
    answer, gold). The judge is a large model called up to twice per item and is
    routinely the majority of the bill; a verdict is a pure function of its
    inputs, so it caches perfectly.
  * **Query embeddings**, keyed by (model, text). The same query is otherwise
    embedded once per model, per tuning candidate, per pass.

**Substitutability.** `NullCache` used to subclass `DiskCache`, which broke the
one promise the base class makes: store a value, get it back. Callers that
reasonably assume `set()` then `get()` round-trips were silently wrong whenever
a null cache was passed. It also skipped `super().__init__` and re-assigned
private attributes, so it inherited a *name* rather than any behaviour.

The two are now independent implementations of `KeyValueCache`. Neither is a
subtype of the other, so neither can violate the other's contract; a caller that
needs "really caches" asks for that in its own type, and the latency lane can
hand over a null cache without lying about what it does.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


def stable_hash(*parts: Any) -> str:
    """Deterministic hash of arbitrary JSON-serialisable parts.

    Used for cache keys AND for the config-hash provenance fields on TraceRow.
    """
    h = hashlib.sha256()
    for p in parts:
        # sort_keys makes dict ordering irrelevant; default=str handles enums etc.
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")  # delimiter so ("ab","c") != ("a","bc")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  Key construction, shared by every implementation
# --------------------------------------------------------------------------- #
class CacheKeys:
    """How the harness names the things it caches.

    A mixin rather than free functions so an implementation gets the whole key
    vocabulary by inheriting one thing, and so the key scheme stays identical
    across implementations, two caches that disagreed on keys would silently
    fail to share entries.
    """

    def key_for_generation(self, model: str, messages: list, params: dict) -> str:
        """`params` must exclude anything non-deterministic you don't want in
        the key (request ids, timestamps)."""
        return stable_hash("gen", model, messages, params)

    def key_for_judge(self, judge_model: str, rubric: str, question: str,
                      context: str, answer: str, gold: str | None) -> str:
        """The rubric is part of the key: change the rubric and every prior
        verdict must be re-earned rather than silently reused under new grading
        rules."""
        return stable_hash("judge", judge_model, rubric, question, context,
                           answer, gold or "")

    def key_for_embedding(self, model: str, text: str) -> str:
        return stable_hash("embed", model, text)


@runtime_checkable
class KeyValueCache(Protocol):
    """What the runner depends on. Deliberately tiny."""

    def get(self, key: str) -> dict | None: ...
    def set(self, key: str, value: dict) -> None: ...
    def key_for_generation(self, model: str, messages: list, params: dict) -> str: ...
    def key_for_judge(self, judge_model: str, rubric: str, question: str,
                      context: str, answer: str, gold: str | None) -> str: ...
    def key_for_embedding(self, model: str, text: str) -> str: ...


# --------------------------------------------------------------------------- #
#  Implementations
# --------------------------------------------------------------------------- #
class DiskCache(CacheKeys):
    """An on-disk key -> JSON store, safe for concurrent use.

    Honours the round-trip contract: what you `set` you can `get`.
    """

    def __init__(self, root: str = ".cache", memo: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # A small in-process memo in front of the disk. The tuning search reads
        # the same keys thousands of times in a tight loop; going to the
        # filesystem for every one of those makes the search disk-bound.
        self._memo: dict[str, dict] = {}
        self._use_memo = memo
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        # Shard by the first 2 hex chars: one directory with a million entries
        # is pathological on most filesystems.
        sub = self.root / key[:2]
        sub.mkdir(exist_ok=True)
        return sub / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if self._use_memo:
            with self._lock:
                if key in self._memo:
                    self.hits += 1
                    return self._memo[key]
        p = self._path(key)
        if p.exists():
            try:
                val = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None  # corrupt or unreadable entry -> treat as a miss
            with self._lock:
                self.hits += 1
                if self._use_memo:
                    self._memo[key] = val
            return val
        with self._lock:
            self.misses += 1
        return None

    def set(self, key: str, value: dict) -> None:
        p = self._path(key)
        # Write-then-rename for atomicity. The temp name includes the pid and
        # thread id so two workers writing the same key can't clobber each
        # other's partial file.
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(value), encoding="utf-8")
            os.replace(tmp, p)
        except OSError:
            tmp.unlink(missing_ok=True)
            return  # a cache write failure must never fail the run
        if self._use_memo:
            with self._lock:
                self._memo[key] = value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class NullCache(CacheKeys):
    """A cache that never stores and never returns anything.

    Not a `DiskCache` subclass, because it cannot honour the round-trip
    contract and pretending otherwise is exactly the Liskov violation this
    replaced. It is a peer implementation of the same protocol.

    The latency lane needs real network timings on *every* run, including
    re-runs. A scratch directory is not a bypass, it is a cold cache that warms
    up and then starts serving fabricated timings as if they were measured.
    """

    def __init__(self, *_args, **_kwargs):
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        self.misses += 1
        return None

    def set(self, key: str, value: dict) -> None:
        return None

    @property
    def hit_rate(self) -> float:
        return 0.0


# `Cache` was the old concrete class name and is used throughout the codebase
# and in user scripts. Keeping it as an alias means this refactor is not a
# breaking change for anyone.
Cache = DiskCache
