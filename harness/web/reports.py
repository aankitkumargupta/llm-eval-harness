"""
Saved reports, a run's findings, frozen and kept.

A run's numbers live in the trace store, but the store is a substrate, not a
record: it holds rows, not conclusions, and it keeps changing as more runs land
in it. A report is the other thing, what was found, on which runs, under which
prices, on a given day, written once and not recomputed afterwards.

**Why frozen rather than recomputed on open.** A report that re-derives itself
from the store every time it is opened is not a record of anything: fix a
pricing entry and last month's report silently changes its conclusion. §2's
whole posture is that a result is tied to the conditions that produced it, so a
saved report captures its numbers, the manifests behind them, and the pricing
date, and then stops moving. If you want today's numbers, run it again, that
is a new report, and both are kept.

Stored as JSON beside the runs, one file per report, so they survive the
process, are diffable, and need no database.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .api import ApiError, _records


def reports_dir(store_path: str) -> Path:
    p = Path(store_path).parent / "reports"
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_reports(store_path: str) -> dict:
    """Saved reports, newest first."""
    out = []
    for f in sorted(reports_dir(store_path).glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        out.append({
            "id": d.get("id", f.stem),
            "title": d.get("title", f.stem),
            "created": d.get("created", 0.0),
            "provider": d.get("provider", ""),
            "n_runs": len(d.get("sections", [])),
            "models": d.get("models", []),
            "total_cost_usd": d.get("total_cost_usd", 0.0),
            "summary": d.get("summary", ""),
        })
    out.sort(key=lambda r: r["created"], reverse=True)
    return {"reports": out}


def get_report(store_path: str, report_id: str) -> dict:
    f = reports_dir(store_path) / f"{report_id}.json"
    if not f.exists():
        raise ApiError(f"No saved report {report_id!r}.", 404)
    return json.loads(f.read_text(encoding="utf-8"))


def delete_report(store_path: str, report_id: str) -> dict:
    f = reports_dir(store_path) / f"{report_id}.json"
    if not f.exists():
        raise ApiError(f"No saved report {report_id!r}.", 404)
    f.unlink()
    return {"deleted": report_id}


def build_report(store_path: str, *, run_ids: list[str], title: str = "",
                 provider: str = "", notes: str = "",
                 caveats: list[str] | None = None) -> dict:
    """Freeze a set of runs into one saved report.

    Each run contributes a section carrying its leaderboard, its confidence
    intervals, its paired significance test and its manifest, the manifest
    especially, because without it (I9) the section records numbers whose
    provenance nobody can reconstruct later.
    """
    from ..bench.metrics import summarise
    from ..bench.spec import SpecError, load_spec
    from ..report import aggregate as A
    from ..report import stats as S
    from ..store.manifest import RunManifest
    from ..store.store import TraceStore

    if not run_ids:
        raise ApiError("A report needs at least one run.", 400)

    store = TraceStore(store_path)
    if not store.exists:
        raise ApiError("No trace store yet.", 404)

    sections, models, total_cost = [], set(), 0.0
    for rid in run_ids:
        g = store.load_run(rid)
        if g.empty:
            raise ApiError(f"No rows for run {rid!r}.", 404)

        is_bench = "benchmark" in g.columns and g["benchmark"].notna().any()
        subject = (str(g["benchmark"].dropna().iloc[0]) if is_bench
                   else str(g["profile"].dropna().iloc[0]))
        # A profile run also carries a latency lane whose rows are never
        # scored, by design. Only the quality passes describe the models,
        # the same selection `profile_results` makes, so a saved report
        # reads the same rows the live report does.
        if not is_bench and "pass_" in g.columns:
            quality = g[g["pass_"].isin(["baseline", "adapted"])]
            if not quality.empty:
                g = quality
        chance = 0.0
        if is_bench:
            try:
                chance = load_spec(subject).scoring.chance_level
            except SpecError:
                pass

        table = summarise(g, benchmark=subject, chance_level=chance)
        rows = [] if table.empty else _records(table)
        models.update(r["model"] for r in rows)
        total_cost += sum(r.get("cost_usd") or 0.0 for r in rows)

        def safe(fn, *a, **kw):
            try:
                return fn(*a, **kw), ""
            except Exception as e:                  # noqa: BLE001 - recorded
                return None, f"{type(e).__name__}: {e}"

        ci, _ = safe(A.ci_table, g, "accuracy")
        sig, sig_err = safe(S.significance_matrix, g, metric="accuracy")
        power, _ = safe(S.power_report, g, "accuracy")
        trunc, _ = safe(A.truncation_report, g)
        errs, _ = safe(A.error_attribution, g)

        manifest = None
        mdir = Path(store_path).parent / rid
        if (mdir / "manifest.json").exists():
            manifest = json.loads(RunManifest.read(mdir).to_json())

        sections.append({
            "run_id": rid,
            "kind": "benchmark" if is_bench else "profile",
            "subject": subject,
            "chance_level": chance,
            "n_rows": int(len(g)),
            "summary": rows,
            "ci": [] if ci is None or ci.empty else _records(ci),
            "significance": [] if sig is None or sig.empty else _records(sig),
            "significance_error": sig_err,
            "power": power.summary() if power is not None else "",
            "truncation": [] if trunc is None or trunc.empty else _records(trunc),
            "errors": [] if errs is None or errs.empty else _records(errs),
            "manifest": manifest,
            "excluded_total": _excluded(g),
        })

    report_id = f"rep_{uuid.uuid4().hex[:8]}"
    doc = {
        "id": report_id,
        "title": title or f"Report {time.strftime('%Y-%m-%d %H:%M')}",
        "created": time.time(),
        "created_human": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": provider,
        "models": sorted(models),
        "run_ids": list(run_ids),
        "total_cost_usd": round(total_cost, 6),
        "notes": notes,
        # Caveats travel WITH the numbers. A report whose limitations live in a
        # separate document is a report that will be quoted without them.
        "caveats": list(caveats or []),
        "sections": sections,
        "summary": _headline(sections),
    }
    (reports_dir(store_path) / f"{report_id}.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def _excluded(g) -> int:
    """Rows that owed an accuracy and have none.

    An unanswerable probe scores accuracy=None on purpose, abstention is
    its metric, so it is not an exclusion. Counting it as one made a RAG
    profile with fifteen unanswerable items look like it had lost fifteen
    answers. Benchmark runs have no item_type column and every row owes an
    accuracy, so the old definition survives there unchanged.
    """
    if g.empty or "accuracy" not in g.columns:
        return 0
    owed = g
    if "item_type" in g.columns:
        owed = g[g["item_type"].astype(str) == "answerable"]
    return int(owed["accuracy"].isna().sum())


def _headline(sections) -> str:
    """One sentence that cannot be quoted misleadingly.

    It names the excluded count alongside the score, because a leaderboard
    figure computed over 55 of 60 items means something different from the
    same figure over 60.
    """
    if not sections:
        return ""
    bits = []
    for s in sections:
        rows = [r for r in s["summary"] if r.get("accuracy") is not None]
        if not rows:
            bits.append(f"{s['subject']}: no scoreable items")
            continue
        top = max(rows, key=lambda r: r["accuracy"])
        gap = s["excluded_total"]
        bits.append(
            f"{s['subject']}: {top['model']} {top['accuracy']:.3f}"
            + (f" ({gap} excluded)" if gap else ""))
    return " · ".join(bits)
