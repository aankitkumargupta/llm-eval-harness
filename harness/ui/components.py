"""
Non-chart UI pieces.

Every one of these exists because the data's job is *not* a chart:

  stat_tiles     — a handful of headline numbers. A grouped bar chart of five
                   unrelated quantities is a classic way to make simple numbers
                   unreadable.
  hero           — the one answer the page leads with. The recommendation is a
                   name, not a distribution, so it gets a hero figure.
  budget_meter   — a single ratio against a limit. A meter, never a two-slice pie.
  pipeline       — where you are in a linear workflow, and what is blocked.
  verdict_lines  — significance results as sentences, because "p=0.03" is not
                   what a reader needs to act on.

All render through `st.markdown(..., unsafe_allow_html=True)` against the CSS in
`theme.py`, so they inherit the same tokens as the charts and follow the
viewer's light/dark setting.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st


def _esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


# --------------------------------------------------------------------------- #
#  Stat tiles
# --------------------------------------------------------------------------- #
@dataclass
class Tile:
    label: str
    value: str
    note: str = ""
    # "good" | "warning" | "critical" | "" — status only, never decoration.
    status: str = ""


def stat_tiles(tiles: list[Tile]) -> None:
    """A KPI row. Status colour is reserved for actual status."""
    if not tiles:
        return
    cells = "".join(
        f'<div class="hx-tile"><div class="k">{_esc(t.label)}</div>'
        f'<div class="v {_esc(t.status)}">{_esc(t.value)}</div>'
        + (f'<div class="n">{_esc(t.note)}</div>' if t.note else "")
        + "</div>"
        for t in tiles
    )
    st.markdown(f'<div class="hx-tiles">{cells}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Hero
# --------------------------------------------------------------------------- #
def hero(label: str, value: str, note: str = "", ok: bool = True) -> None:
    """The single answer the page exists to give.

    Used for the model recommendation. It is a name, so it is a hero figure
    rather than a chart — and it goes at the top, because a reader who scrolls
    past four tables to find it has been failed by the layout.
    """
    cls = "hx-hero" if ok else "hx-hero none"
    st.markdown(
        f'<div class="{cls}"><div class="k">{_esc(label)}</div>'
        f'<div class="v">{_esc(value)}</div>'
        + (f'<div class="n">{_esc(note)}</div>' if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Budget meter
# --------------------------------------------------------------------------- #
def budget_meter(spent: float, limit: float) -> None:
    """Spend against the ceiling.

    A single ratio against a limit is a meter. It turns colour only as the
    ceiling approaches, so the colour means something when it appears rather
    than being permanent decoration.
    """
    if limit <= 0:
        st.markdown(
            f'<div class="hx-meter"><div class="row"><span>Spend</span>'
            f'<span>${spent:,.4f} · no ceiling set</span></div>'
            f'<div class="track"></div></div>',
            unsafe_allow_html=True)
        return

    frac = min(1.0, spent / limit) if limit else 0.0
    status = "critical" if frac >= 0.9 else ("warning" if frac >= 0.7 else "")
    st.markdown(
        f'<div class="hx-meter"><div class="row"><span>Budget</span>'
        f'<span>${spent:,.4f} / ${limit:,.2f}</span></div>'
        f'<div class="track"><div class="fill {status}" '
        f'style="width:{frac * 100:.1f}%"></div></div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
#  Pipeline stepper
# --------------------------------------------------------------------------- #
def pipeline(steps: list[tuple[str, bool]], active: int = -1) -> None:
    """Where you are in the workflow.

    The app's flow is linear and stateful — you cannot run before ingesting,
    and cannot read results before running. Without this, a disabled button is
    the only signal, and it never says *why* it is disabled.

    `steps` is [(label, done)]; `active` is the index to highlight.
    """
    parts = []
    for i, (label, done) in enumerate(steps):
        cls = "hx-step"
        if i == active:
            cls += " active"
        elif done:
            cls += " done"
        parts.append(f'<span class="{cls}"><span class="dot"></span>'
                     f'{_esc(label)}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="hx-arrow">&rsaquo;</span>')
    st.markdown(f'<div class="hx-steps">{"".join(parts)}</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Verdicts
# --------------------------------------------------------------------------- #
def verdict_lines(rows: list[tuple[bool, str]]) -> None:
    """Significance results as sentences with a status tag.

    The tag is a label, not a bare colour — a reader who cannot distinguish the
    green still reads the word. That is the rule for every status cue here.
    """
    for significant, text in rows:
        tag = "sig" if significant else "ns"
        word = "significant" if significant else "n.s."
        st.markdown(
            f'<div class="hx-verdict"><span class="tag {tag}">{word}</span>'
            f'<span>{_esc(text)}</span></div>',
            unsafe_allow_html=True)


def note(text: str) -> None:
    """A quiet explanatory line. Used heavily: most of this tool's output needs
    a sentence saying what it means before it is safe to act on."""
    st.markdown(f'<div class="hx-note">{_esc(text)}</div>',
                unsafe_allow_html=True)


def section(title: str, explanation: str = "") -> None:
    st.markdown(f"## {title}")
    if explanation:
        note(explanation)


# --------------------------------------------------------------------------- #
#  Feature catalogue rendering
# --------------------------------------------------------------------------- #
def feature_cards(items, columns: int = 2) -> None:
    """A grid of capability cards: name, one line of why it matters.

    Cards rather than a bulleted list because the descriptions are the point —
    a bare list of 27 metric names tells a reader nothing they can act on, and
    a table of them is a wall. Each card carries a name and a sentence.
    """
    if not items:
        return
    rows = "".join(
        f'<div class="hx-fcard">'
        f'<div class="n">{_esc(getattr(i, "name", i))}</div>'
        + (f'<div class="d">{_esc(getattr(i, "note", ""))}</div>'
           if getattr(i, "note", "") else "")
        + (f'<div class="t">{_esc(getattr(i, "detail", ""))}</div>'
           if getattr(i, "detail", "") else "")
        + "</div>"
        for i in items
    )
    st.markdown(
        f'<div class="hx-fgrid" style="--cols:{columns}">{rows}</div>',
        unsafe_allow_html=True)


def chips(labels, tone: str = "") -> None:
    """A compact inline set — for short enumerations like retrieval modes."""
    if not labels:
        return
    body = "".join(f'<span class="hx-chip {_esc(tone)}">{_esc(label)}</span>'
                   for label in labels)
    st.markdown(f'<div class="hx-chips">{body}</div>', unsafe_allow_html=True)


def capability_matrix(providers) -> None:
    """Provider x capability, with the status word beside the mark.

    Never a bare tick colour: a reader who cannot distinguish the green still
    reads "yes" / "no", which is the same rule the charts follow.
    """
    if not providers:
        return
    head = ("<tr><th>provider</th><th>key</th><th>chat</th><th>embeddings</th>"
            "<th>rerank</th><th>hosting</th></tr>")

    def mark(ok: bool) -> str:
        cls = "yes" if ok else "no"
        return f'<td><span class="hx-mark {cls}">{"yes" if ok else "no"}</span></td>'

    body = []
    for p in providers:
        state = p.key_state
        cls = "yes" if state == "ready" else "none"
        key = f'<span class="hx-mark {cls}">{_esc(state)}</span>'
        body.append(
            f"<tr><td><code>{_esc(p.name)}</code></td><td>{key}</td>"
            f"{mark(True)}{mark(p.embeddings)}{mark(p.rerank)}"
            f'<td>{"self-hosted" if p.local else "hosted"}</td></tr>')
    st.markdown(
        f'<div class="hx-scroll"><table class="hx-matrix">{head}'
        f'{"".join(body)}</table></div>', unsafe_allow_html=True)


def kv_rows(pairs) -> None:
    """Label/description rows — for lists where the description carries the weight."""
    if not pairs:
        return
    body = "".join(
        f'<div class="hx-kv"><div class="k">{_esc(k)}</div>'
        f'<div class="v">{_esc(v)}</div></div>' for k, v in pairs)
    st.markdown(body, unsafe_allow_html=True)


def command_list(rows) -> None:
    """CLI commands with what each one answers."""
    if not rows:
        return
    body = "".join(
        f'<div class="hx-cmd"><code>{_esc(cmd)}</code>'
        f'<span>{_esc(desc)}</span></div>' for cmd, desc in rows)
    st.markdown(body, unsafe_allow_html=True)
