"""
Dataset acquisition, the only place in `bench/` permitted to touch the network.

§10.3 is explicit: "Network only in the fetcher; subprocesses only in the
sandbox." Adapters stay pure so they can be unit-tested offline; everything
that reaches out lives here, behind a cache, so a fetched dataset is fetched
once and replayed forever after.

**Why the datasets-server REST API and not `datasets`.** The `datasets` library
would pull in pyarrow extras, `huggingface_hub`, `fsspec` and a long tail of
transitive dependencies, for a job that is "download some rows as JSON". §7
forbids adding a dependency that duplicates something already vendored, and
`requests` is already a dependency. The REST endpoint serves public datasets
without a token and returns exactly the rows we need.

Three rules the spec insists on, all enforced here:

  * **Checksum or refusal.** A spec with `source.checksum` must match what was
    downloaded, or the fetch fails. A dataset that silently changed under you
    produces a silently changed score, and `spec_hash` covers the checksum
    precisely so that cannot happen quietly.
  * **Licence is a gate, not a note.** `commercial_use: false` requires an
    explicit acknowledgement. Not a warning printed into a log nobody reads.
  * **Cached fetches never re-download.** The cache is the offline-replay path,
    and it is why `tests/` can exercise a real benchmark without the network.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

#: The public datasets-server. Serves rows from any dataset whose viewer is
#: enabled, which covers every benchmark in the catalogue.
ROWS_URL = "https://datasets-server.huggingface.co/rows"
INFO_URL = "https://datasets-server.huggingface.co/info"

#: Rows per request. The server caps a page at 100.
PAGE = 100
DEFAULT_TIMEOUT = 60.0


class FetchError(RuntimeError):
    """A dataset could not be acquired, with a reason the user can act on."""


class LicenceError(FetchError):
    """A non-commercial dataset was fetched without acknowledgement."""


@dataclass
class FetchResult:
    dataset: str
    config: str
    split: str
    n_rows: int
    path: Path
    checksum: str
    cached: bool
    elapsed_s: float

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["path"] = str(self.path)
        return d


def cache_dir(root: str = ".cache/datasets") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _local_name(dataset: str, config: str, split: str,
                limit: int | None = None) -> str:
    """The cache filename. Deliberately does NOT encode the fetch limit.

    The file holds "what we downloaded of this split", and the spec's
    `checksum` identifies those exact bytes. Encoding the limit too would mean
    the same spec resolved to different files depending on how a previous
    command was invoked, and the checksum would then be the only thing that
    noticed, at run time, which is far too late.
    """
    safe = dataset.replace("/", "__")
    return f"{safe}__{config}__{split}.jsonl"


def checksum_file(path: Path) -> str:
    """sha256 of the file, as `sha256:<hex>`, the form a spec records."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def fetch_rows(dataset: str, *, config: str = "default", split: str = "test",
               limit: int | None = None, root: str = ".cache/datasets",
               timeout: float = DEFAULT_TIMEOUT,
               expect_checksum: str = "",
               force: bool = False) -> FetchResult:
    """Download rows to a local JSONL, or replay the cached copy.

    The cache hit is checked *before* any network call, so a second run of the
    same benchmark is offline even though the first was not.
    """
    t0 = time.perf_counter()
    target = cache_dir(root) / _local_name(dataset, config, split, limit)

    if target.exists() and not force:
        got = checksum_file(target)
        if expect_checksum and got != expect_checksum:
            raise FetchError(
                f"Cached {target.name} has checksum {got}, but the spec "
                f"expects {expect_checksum}. The cached copy is stale or the "
                f"spec moved; delete the file to re-fetch, and bump the "
                f"spec's `version` if the dataset genuinely changed.")
        return FetchResult(dataset, config, split, _count_lines(target),
                           target, got, True, time.perf_counter() - t0)

    rows: list[dict] = []
    offset = 0
    while True:
        want = PAGE if limit is None else min(PAGE, limit - len(rows))
        if want <= 0:
            break
        try:
            r = requests.get(ROWS_URL, timeout=timeout, params={
                "dataset": dataset, "config": config, "split": split,
                "offset": offset, "length": want})
        except requests.RequestException as e:
            raise FetchError(
                f"Could not reach the datasets server for {dataset!r}: "
                f"{type(e).__name__}. The fetcher is the only part of the "
                f"benchmark path that needs the network; everything else "
                f"replays from {target.parent}.") from e

        if r.status_code == 404:
            raise FetchError(
                f"{dataset!r} (config {config!r}, split {split!r}) was not "
                f"found. Check the config and split names, many datasets use "
                f"'main' or a subject name rather than 'default'.")
        if r.status_code != 200:
            raise FetchError(f"datasets-server returned {r.status_code} for "
                             f"{dataset!r}: {r.text[:200]}")

        payload = r.json()
        batch = [row.get("row", row) for row in payload.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if limit is not None and len(rows) >= limit:
            break
        if len(batch) < want:
            break                       # split exhausted

    if not rows:
        raise FetchError(f"{dataset!r} returned no rows for split {split!r}.")

    # Written sorted-by-nothing but line-stable: the server returns a stable
    # order, and the checksum is over exactly these bytes.
    with open(target, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    got = checksum_file(target)
    if expect_checksum and got != expect_checksum:
        raise FetchError(
            f"Checksum mismatch for {dataset!r}: got {got}, spec expects "
            f"{expect_checksum}. The upstream dataset has changed. Any score "
            f"computed against it is not comparable to one from before, bump "
            f"the spec's `version` and record the new checksum deliberately.")

    return FetchResult(dataset, config, split, len(rows), target, got, False,
                       time.perf_counter() - t0)


def fetch_for_spec(spec, *, acknowledge_licence: bool = False,
                   limit: int | None = None,
                   root: str = ".cache/datasets") -> FetchResult:
    """Fetch the dataset a `BenchmarkSpec` names, honouring its licence gate."""
    src = spec.source
    if src.kind == "local":
        p = Path(src.ref)
        if not p.exists():
            raise FetchError(f"Local dataset not found: {src.ref}")
        return FetchResult(src.ref, "local", src.split, _count_lines(p), p,
                           checksum_file(p), True, 0.0)

    if src.kind != "hf":
        raise FetchError(f"Unsupported source kind {src.kind!r}. "
                         f"Supported: hf, local.")

    if not src.commercial_use and not acknowledge_licence:
        raise LicenceError(
            f"{spec.id} is licensed {src.licence!r}, which is not marked for "
            f"commercial use. Re-run with --acknowledge-licence to confirm you "
            f"have read it and your use is permitted. This is a gate rather "
            f"than a warning on purpose: a licence note printed into a log is "
            f"a licence note nobody read.")

    config = getattr(src, "config", "") or "default"
    return fetch_rows(src.ref, config=config, split=src.split, limit=limit,
                      root=root, expect_checksum=src.checksum)


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def dataset_info(dataset: str, timeout: float = 20.0) -> dict:
    """Configs and splits a dataset actually offers.

    Worth a call before a fetch: 'default' is wrong for a surprising number of
    benchmarks, and a 404 four layers down is a poor way to learn that.
    """
    try:
        r = requests.get(INFO_URL, params={"dataset": dataset}, timeout=timeout)
    except requests.RequestException as e:
        raise FetchError(f"Could not reach the datasets server: "
                         f"{type(e).__name__}") from e
    if r.status_code != 200:
        raise FetchError(f"datasets-server returned {r.status_code} for "
                         f"{dataset!r}")
    info = (r.json() or {}).get("dataset_info", {})
    out = {}
    for cfg, body in info.items():
        out[cfg] = {
            "splits": sorted((body.get("splits") or {}).keys()),
            "features": sorted((body.get("features") or {}).keys()),
            "license": body.get("license", ""),
        }
    return out


def cached_path(spec, *, limit: int | None = None,
                root: str = ".cache/datasets") -> Path:
    """Where this spec's rows live locally. Does NOT fetch.

    Deliberately pure: `registry.build()` calls it to point an adapter at its
    data, and a build that silently downloaded would put the network inside
    every code path that merely *constructs* a benchmark, including the
    offline test suite.
    """
    src = spec.source
    if src.kind == "local":
        return Path(src.ref)
    config = getattr(src, "config", "") or "default"
    return cache_dir(root) / _local_name(src.ref, config, src.split, limit)


def is_cached(spec, **kw) -> bool:
    return cached_path(spec, **kw).exists()
