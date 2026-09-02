"""
Row collection: buffering, checkpointing, progress and error counting.

Extracted from `Orchestrator`, which had accumulated six unrelated reasons to
change — the three passes, plus buffering, plus progress reporting, plus resume
bookkeeping. Every pass needed the buffering behaviour, so it lived on the
orchestrator and every pass reached into `self._pending`, `self._written`,
`self._errors`, `self._done` and `self._total`. Shared mutable state across
methods is what made the class hard to reason about under concurrency.

`RowCollector` owns exactly one job: take finished rows from many worker
threads, flush them to a sink in batches, and keep the counts. It depends on a
`TraceSink` protocol rather than the concrete `TraceStore`, so a pass can be
tested against an in-memory list.

The checkpointing is the point. Rows used to accumulate in memory until a run
finished, so a crash at item 9,000 of 10,000 lost every paid API call in the
run.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from ..clients.cost import BudgetExceeded
from ..store.schema import TraceRow


@runtime_checkable
class TraceSink(Protocol):
    """Where finished rows go. The narrow half of `TraceStore` that writers need.

    A pass has no business reading the store, so it shouldn't depend on a type
    that can. This is the write side only.
    """

    def write(self, rows: Iterable[TraceRow]) -> int: ...


ProgressCallback = Callable[[int, int, str], None]


class RowCollector:
    """Thread-safe buffer between the worker pool and the trace sink."""

    def __init__(self, sink: TraceSink, checkpoint_every: int = 50,
                 progress_cb: ProgressCallback | None = None):
        self.sink = sink
        self.checkpoint_every = max(1, checkpoint_every)
        self.progress_cb = progress_cb

        self._pending: list[TraceRow] = []
        self._lock = threading.Lock()
        self.written = 0
        self.errors = 0
        self.done = 0
        self.total = 0

    # -- lifecycle -------------------------------------------------------- #
    def expect(self, total: int) -> None:
        """Declare how many rows this phase will produce, for the progress bar."""
        with self._lock:
            self.total = total
            self.done = 0

    def collect(self, row: TraceRow) -> None:
        """Buffer a row, flushing when the batch is full.

        The flush and the progress callback happen *outside* the lock: writing
        Parquet while holding it would serialise every worker behind the one
        doing I/O, and a slow user callback would do the same.
        """
        with self._lock:
            self._pending.append(row)
            if row.error:
                self.errors += 1
            self.done += 1
            should_flush = len(self._pending) >= self.checkpoint_every
            batch = self._pending if should_flush else None
            if should_flush:
                self._pending = []
            done, total = self.done, self.total

        if batch:
            self.written += self.sink.write(batch)
        if self.progress_cb is not None:
            try:
                self.progress_cb(done, total, row.model)
            except Exception:  # noqa: BLE001
                pass  # progress reporting must never break a run

    def flush(self) -> None:
        with self._lock:
            batch, self._pending = self._pending, []
        if batch:
            self.written += self.sink.write(batch)

    def skip(self, n: int = 1) -> None:
        """Count work that completed but produced no row (a discarded warm-up)."""
        with self._lock:
            self.done += n

    # -- draining --------------------------------------------------------- #
    def drain(self, futures) -> BudgetExceeded | None:
        """Collect every finished future, even after the budget trips.

        Abandoning the loop on the first `BudgetExceeded` would discard rows
        from items that had already completed and been paid for — the exact
        opposite of what a budget abort should do. The trip is returned for the
        caller to report.
        """
        tripped: BudgetExceeded | None = None
        for f in cf.as_completed(futures):
            try:
                self.collect(f.result())
            except BudgetExceeded as e:
                # Only budget trips reach here: run_item records every other
                # failure on the row and returns normally.
                tripped = tripped or e
        return tripped

    def drain_tagged(self, futures: dict) -> tuple[dict[int, list[TraceRow]],
                                                   BudgetExceeded | None]:
        """Drain a {future: tag} mapping, grouping rows by tag.

        The tuning search needs rows grouped per candidate while still
        collecting them; doing both in one pass avoids holding every row twice.
        """
        grouped: dict[int, list[TraceRow]] = {}
        tripped: BudgetExceeded | None = None
        for fut in cf.as_completed(futures):
            try:
                row = fut.result()
            except BudgetExceeded as e:
                tripped = tripped or e
                continue
            grouped.setdefault(futures[fut], []).append(row)
            self.collect(row)
        return grouped, tripped
