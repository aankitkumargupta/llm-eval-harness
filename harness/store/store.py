"""
Trace store: persistence for TraceRow records.

The store is the ONLY boundary between the execution half of the system
(orchestration writes rows) and the analysis half (report reads them). Neither
half calls the other directly.

**Why this was rewritten.** The original `write()` did read-all -> concat ->
rewrite-whole-file, because Parquet can't append in place. That's O(n²) in total
bytes written, and the tuning search calls `write()` once per candidate per
model. A 20-candidate search over 4 models is 80 full rewrites of a file that
grows the whole time — by the end, most of the run's wall-clock is Parquet I/O,
not inference. Worse, a crash mid-rewrite could leave a truncated store holding
*nothing*, losing hours of paid API calls.

This version writes each batch as its own Parquet part file under a directory,
which is the standard columnar answer:

    runs/traces/            <- store root
      part-<seq>-<uuid>.parquet

Appends are O(batch), crash-safe (a failed write costs one part, never the
store), and DuckDB reads the whole set through one glob with no concat. A legacy
single-file store is still read transparently, so existing results keep working.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from .schema import TraceRow


class TraceStore:
    def __init__(self, path: str = "runs/traces.parquet"):
        """`path` may be a directory (new layout) or a .parquet file (legacy).

        A path ending in `.parquet` that does not already exist is treated as a
        directory root — new stores get the fast layout while existing files are
        still opened as files.
        """
        p = Path(path)
        self.legacy_file: Path | None = None
        if p.suffix == ".parquet" and p.exists() and p.is_file():
            # An existing single-file store: keep reading it, write new parts
            # into a sibling directory so we never rewrite the old file.
            self.legacy_file = p
            self.root = p.with_suffix("")
        else:
            self.root = p if p.suffix != ".parquet" else p.with_suffix("")
            # Adopt a sibling single-file store if one is there.
            #
            # Configs now point at a DIRECTORY (`workspace/traces`), but anyone
            # who ran the harness before the append-only rewrite has their
            # results in `workspace/traces.parquet` next to it. Without this
            # their entire history silently disappears from the app and every
            # report — the data is intact on disk and simply never read, which
            # is the worst shape a migration bug can take.
            sibling = self.root.with_suffix(".parquet")
            if sibling.exists() and sibling.is_file():
                self.legacy_file = sibling
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0

    # ------------------------------------------------------------------ #
    #  Write                                                              #
    # ------------------------------------------------------------------ #
    def write(self, rows: Iterable[TraceRow]) -> int:
        """Append rows as a new Parquet part. O(batch), never rewrites history."""
        records = [r.to_dict() if isinstance(r, TraceRow) else r for r in rows]
        if not records:
            return 0
        df = pd.DataFrame(records)

        with self._lock:
            self._seq += 1
            seq = self._seq
        # The uuid suffix keeps parts unique across processes writing the same
        # store (the UI thread and a CLI run, say) — a bare counter would collide.
        part = self.root / f"part-{seq:06d}-{uuid.uuid4().hex[:8]}.parquet"
        tmp = part.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(part)  # atomic: readers never observe a partial part
        return len(df)

    # ------------------------------------------------------------------ #
    #  Read                                                               #
    # ------------------------------------------------------------------ #
    def _sources(self) -> list[str]:
        srcs = [str(p) for p in sorted(self.root.glob("part-*.parquet"))]
        if self.legacy_file is not None and self.legacy_file.exists():
            srcs.insert(0, str(self.legacy_file))
        return srcs

    @property
    def exists(self) -> bool:
        return bool(self._sources())

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Run DuckDB SQL against the store, with `traces` wired to the data.

        Pass values via `params` and `?` placeholders rather than formatting
        them into the SQL — the old call sites interpolated a profile name
        straight into a WHERE clause, which breaks on any name containing a
        quote and is an injection vector wherever that name is user-supplied.

            store.query("SELECT * FROM traces WHERE profile = ?", ["regulated_qa"])
        """
        srcs = self._sources()
        if not srcs:
            return pd.DataFrame()
        con = duckdb.connect()
        try:
            # union_by_name tolerates schema drift across parts: a store written
            # before a new metric column existed still reads back cleanly, with
            # the missing column as NULL rather than the whole read failing.
            file_list = ", ".join(f"'{s}'" for s in srcs)
            con.execute(
                f"CREATE VIEW traces AS "
                f"SELECT * FROM read_parquet([{file_list}], union_by_name=true)"
            )
            return con.execute(sql, params or []).fetchdf()
        finally:
            con.close()

    def load_all(self) -> pd.DataFrame:
        srcs = self._sources()
        if not srcs:
            return pd.DataFrame()
        frames = [pd.read_parquet(s) for s in srcs]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    def load_run(self, run_id: str) -> pd.DataFrame:
        return self.query("SELECT * FROM traces WHERE run_id = ?", [run_id])

    def load_profile(self, profile: str) -> pd.DataFrame:
        return self.query("SELECT * FROM traces WHERE profile = ?", [profile])

    def runs(self) -> pd.DataFrame:
        """One row per run: when, which profile, how many rows, how many errors.

        This is what makes run-over-run comparison and the CI regression gate
        possible — you cannot diff two runs you cannot enumerate.
        """
        if not self._sources():
            return pd.DataFrame()
        return self.query(
            "SELECT run_id, "
            "       any_value(profile) AS profile, "
            "       min(ts) AS started, max(ts) AS finished, "
            "       count(*) AS rows, "
            "       count(DISTINCT model) AS models, "
            "       sum(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors "
            "FROM traces GROUP BY run_id ORDER BY started DESC"
        )

    def compact(self) -> int:
        """Fold every part into one file. Purely housekeeping — never required.

        Many small parts are fine for DuckDB but awkward to copy around; this
        makes a store portable as a single artifact.
        """
        df = self.load_all()
        if df.empty:
            return 0
        merged = self.root / "part-000000-compacted.parquet"
        tmp = merged.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        for old in list(self.root.glob("part-*.parquet")):
            old.unlink(missing_ok=True)
        tmp.replace(merged)
        if self.legacy_file is not None and self.legacy_file.exists():
            self.legacy_file.unlink()
            self.legacy_file = None
        return len(df)
