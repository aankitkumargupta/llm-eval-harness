"""
Content-addressed cache for API responses.

Why this exists: reporting and evaluators get re-run many times against the
same model outputs. Hosted inference is billed per call and isn't bit-for-bit
deterministic over time, so caching the raw response by a hash of its inputs
gives us BOTH cost savings AND reproducibility (the cache is the real
reproducibility guarantee, not the seed).

Keyed by a stable hash of (model, prompt/messages, params). Stored as one JSON
file per key under a cache dir. Small, dependency-free, and safe to delete.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


def stable_hash(*parts: Any) -> str:
    """
    Deterministic hash of arbitrary JSON-serialisable parts.
    Used for cache keys AND for the config-hash provenance fields on TraceRow.
    """
    h = hashlib.sha256()
    for p in parts:
        # sort_keys makes dict ordering irrelevant; default=str handles enums etc.
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
        h.update(b"\x00")  # delimiter so ("ab","c") != ("a","bc")
    return h.hexdigest()


class Cache:
    """A simple on-disk key->JSON store."""

    def __init__(self, root: str = ".cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # shard into subdirs by first 2 hex chars to avoid one giant directory
        sub = self.root / key[:2]
        sub.mkdir(exist_ok=True)
        return sub / f"{key}.json"

    def get(self, key: str) -> Optional[dict]:
        p = self._path(key)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                return None  # corrupt entry -> treat as miss
        return None

    def set(self, key: str, value: dict) -> None:
        p = self._path(key)
        # write-then-rename for atomicity (no half-written cache files)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        os.replace(tmp, p)

    def key_for_generation(self, model: str, messages: list, params: dict) -> str:
        """The canonical generation cache key. params must exclude anything
        non-deterministic you DON'T want in the key (e.g. request id)."""
        return stable_hash("gen", model, messages, params)
