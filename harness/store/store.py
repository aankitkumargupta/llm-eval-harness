"""
Trace store: persistence for TraceRow records.

Writes rows to Parquet (columnar, compresses well, streams from disk) and reads
them back via DuckDB (fast analytical SQL without loading everything into RAM —
important on an 8 GB machine).

The store is the ONLY boundary between the execution half of the system
(orchestration writes rows) and the analysis half (report reads them). Neither
half calls the other directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from .schema import TraceRow


class TraceStore:
    def __init__(self, path: str = "runs/traces.parquet"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows: Iterable[TraceRow]) -> int:
        """
        Append rows to the Parquet dataset. We accumulate a run's rows and write
        them together. Parquet doesn't support in-place append, so if the file
        exists we read + concat + rewrite. For very large stores you'd switch to
        a partitioned directory of Parquet files; for this project's scale
        (hundreds of thousands of rows) a single file is fine.
        """
        new_df = pd.DataFrame([r.to_dict() for r in rows])
        if new_df.empty:
            return 0

        # Ensure list-typed columns survive the Parquet round-trip as objects.
        if self.path.exists():
            existing = pd.read_parquet(self.path)
            df = pd.concat([existing, new_df], ignore_index=True)
        else:
            df = new_df

        df.to_parquet(self.path, index=False)
        return len(new_df)

    def query(self, sql: str) -> pd.DataFrame:
        """
        Run DuckDB SQL against the trace store. Use the placeholder `traces`
        as the table name; it's wired to the Parquet file.

        Example:
            store.query("SELECT model, AVG(accuracy) FROM traces "
                        "WHERE pass_='baseline' GROUP BY model")
        """
        if not self.path.exists():
            return pd.DataFrame()
        con = duckdb.connect()
        con.execute(
            f"CREATE VIEW traces AS SELECT * FROM read_parquet('{self.path}')"
        )
        try:
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def load_all(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        return pd.read_parquet(self.path)
