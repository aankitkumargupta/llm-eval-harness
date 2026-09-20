"""
Case studies: the worked evaluations under docs/case-studies, paired with
the profile they describe and the newest run of it in the trace store.

A case study is a Markdown document with a fixed shape (the use case, who
would run it, what a wrong answer costs, the metrics and why, caveats, and
a Results section filled from a live run). This module reads those files,
splits them into typed blocks the UI can render without an HTML parser or
a Markdown library, and attaches the profile's summary and the run that
answers it. No number is computed here: the results themselves come from
`profile_results`, the same function the Profile report screen reads, so a
case study and the report can never disagree.

Read-only over the docs directory and the store; no network.
"""

from __future__ import annotations

import re
from pathlib import Path

from .api import ApiError

DOCS_DIR = Path("docs/case-studies")
PROFILES_DIR = Path("configs/profiles")


def _title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            t = line[2:].strip()
            t = re.sub(r"^Case study:\s*", "", t)
            t = re.sub(r"\s*\([a-z0-9_]+\)\s*$", "", t)
            return t[:1].upper() + t[1:]
    return fallback


def _inline(s: str) -> str:
    """Drop the inline markers the write-ups use (emphasis, code spans).

    The UI renders plain text, on purpose: nothing from a document becomes
    markup. So `native_script_ratio` and *who gets it* read as words, not
    as asterisks and backticks.
    """
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", s)
    return s


def _blocks(text: str) -> list[dict]:
    """Split Markdown into typed blocks: h2, p, ul, table.

    Deliberately narrow. The case studies use headings, paragraphs, bullet
    lists and pipe tables and nothing else; anything else is passed through
    as a paragraph so it is still visible rather than silently dropped.
    """
    out: list[dict] = []
    para: list[str] = []
    bullets: list[str] = []
    rows: list[list[str]] = []

    def flush() -> None:
        nonlocal para, bullets, rows
        if para:
            out.append({"type": "p", "text": _inline(" ".join(s.strip() for s in para))})
            para = []
        if bullets:
            out.append({"type": "ul", "items": [_inline(b) for b in bullets]})
            bullets = []
        if rows:
            out.append({"type": "table", "header": [_inline(c) for c in rows[0]],
                        "rows": [[_inline(c) for c in r] for r in rows[1:]]})
            rows = []

    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            flush()
            out.append({"type": "h2", "text": _inline(line[3:].strip())})
        elif line.startswith("|"):
            if para or bullets:
                flush()
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            rows.append(cells)
        elif line.startswith("- "):
            if para or rows:
                flush()
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and bullets:
            bullets[-1] = bullets[-1] + " " + line.strip()
        elif not line:
            flush()
        else:
            if bullets or rows:
                flush()
            para.append(line)
    flush()
    return out


def _profile_summary(name: str) -> dict | None:
    from ..profiles.loaders import load_evalset
    from ..profiles.profile import Profile, ProfileError

    path = PROFILES_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    try:
        p = Profile.from_yaml(str(path))
    except ProfileError as e:
        return {"name": name, "invalid": str(e)}
    n_items, langs = 0, {}
    try:
        items = load_evalset(p.evalset_path)
        n_items = len(items)
        for it in items:
            lang = it.meta.get("language") if isinstance(it.meta, dict) else None
            if lang:
                langs[lang] = langs.get(lang, 0) + 1
    except (OSError, ValueError):
        pass
    return {
        "name": p.name, "task": p.task.value, "description": p.description.strip(),
        "n_items": n_items, "languages": langs,
        "active_metrics": list(p.active_metrics),
        "weights": dict(p.metric_weights),
        "accuracy_scorer": p.accuracy_scorer,
        "embedding_model": p.embedding_model,
        "target_script": p.target_script,
        "path": str(path).replace("\\", "/"),
    }


def list_case_studies(store_path: str, docs_dir: Path | str = DOCS_DIR) -> dict:
    """Every case study, with its profile and the newest run of that profile."""
    from .profile_api import all_runs

    docs = Path(docs_dir)
    if not docs.exists():
        return {"case_studies": []}
    runs = all_runs(store_path)["runs"] if Path(store_path).exists() else []
    out = []
    for md in sorted(docs.glob("*.md")):
        if md.stem.lower() == "readme":
            continue                      # the index of the folder, not a case study
        text = md.read_text(encoding="utf-8")
        cid = md.stem
        blocks = _blocks(text)
        results_filled = True
        for i, b in enumerate(blocks):
            if b["type"] == "h2" and b["text"].lower().startswith("results"):
                nxt = blocks[i + 1]["text"] if i + 1 < len(blocks) and blocks[i + 1]["type"] == "p" else ""
                results_filled = not nxt.lower().startswith("to be filled")
        mine = [r for r in runs if r["kind"] == "profile" and r["subject"] == cid]
        newest = mine[0] if mine else None
        out.append({
            "id": cid, "title": _title(text, cid), "doc": str(md).replace("\\", "/"),
            "profile": _profile_summary(cid),
            "blocks": blocks,
            "results_filled": results_filled,
            "run": newest,
            "other_runs": [r["run_id"] for r in mine[1:]],
        })
    return {"case_studies": out}


def get_case_study(store_path: str, cid: str, docs_dir: Path | str = DOCS_DIR) -> dict:
    for cs in list_case_studies(store_path, docs_dir)["case_studies"]:
        if cs["id"] == cid:
            return cs
    raise ApiError(f"No case study {cid!r}.", 404)
