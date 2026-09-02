"""
Arena: pairwise head-to-head comparison for an Elo ranking.

Separated from the passes because it is not one. The passes *generate* answers;
the arena *reads* answers already in the store and only spends judge calls. It
needs a reader, not a writer, and it produces comparisons rather than trace
rows — a different collaborator set and a different output type, which is
exactly the SRP argument for its own module.

Why an arena at all, given there is already a weighted composite: a composite
scores each model against a rubric, and rubric scores saturate. Once every
strong model lands at 0.9-ish, the ranking is noise. Pairwise comparison stays
discriminative precisely where absolute scoring stops working.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class TraceReader(Protocol):
    """The read half of the trace store. The arena never writes."""

    def load_run(self, run_id: str) -> pd.DataFrame: ...
    def load_profile(self, profile: str) -> pd.DataFrame: ...


class ArenaService:
    """Judges stored answers head to head."""

    def __init__(self, reader: TraceReader, judge):
        self.reader = reader
        self.judge = judge

    def run(self, profile_name: str, models: list[str],
            source_run: str | None = None,
            max_pairs_per_item: int = 3) -> list[tuple[str, str, str]]:
        """Returns [(model_a, model_b, winner)], ready for `elo_from_pairwise`.

        Comparisons are position-bias corrected inside `Judge.pairwise`: each
        pair is judged in both orders and a win counts only if it survives the
        swap. Without that, an Elo table partly ranks argument position, and it
        does so invisibly.
        """
        if self.judge is None:
            raise ValueError("Arena mode needs a judge model.")

        df = (self.reader.load_run(source_run) if source_run
              else self.reader.load_profile(profile_name))
        if df.empty:
            return []

        df = df[df["pass_"].isin(["baseline", "adapted"])]
        if "error" in df.columns:
            df = df[df["error"].isna()]
        if df.empty:
            return []

        results: list[tuple[str, str, str]] = []
        for item_id, group in df.groupby("item_id"):
            by_model = {r["model"]: r for _, r in group.iterrows()
                        if r["model"] in models}
            present = [m for m in models if m in by_model]
            pairs = [(a, b) for i, a in enumerate(present)
                     for b in present[i + 1:]][:max_pairs_per_item]
            for a, b in pairs:
                ra, rb = by_model[a], by_model[b]
                # The assembled prompt stands in for the context the judge
                # needs; truncated because a long-document profile's prompt can
                # be larger than the judge's own window.
                context = str(ra.get("assembled_prompt", ""))[:8000]
                winner = self.judge.pairwise(
                    a, b, str(ra.get("item_id", item_id)), context,
                    str(ra.get("raw_output", "")), str(rb.get("raw_output", "")))
                if winner:
                    results.append((a, b, winner))
        return results
