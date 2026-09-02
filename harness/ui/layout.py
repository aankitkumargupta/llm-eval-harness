"""
Layout shell: sidebar navigation, cards, page headers, KPI tiles.

The visual language this implements is a layered dark dashboard — a deep page
plane, raised cards, a persistent left rail, and a breadcrumb header — rather
than the flat tab strip the app used before.

The structural change is the nav. Tabs are fine for four sections and fall apart
past that: Streamlit executes every tab body on every rerun, so a ten-tab app
does ten pages of work to show one. A sidebar with a single selected page runs
exactly one page body per interaction.

Responsibilities are kept apart on purpose:

    layout.py   how a page is framed  — chrome, cards, headers, tiles
    pages.py    which pages exist     — the registry
    charts.py   how data is drawn
    theme.py    what things look like — tokens only

so restyling never means touching a page's content, and adding a page never
means touching the chrome.
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from dataclasses import dataclass, field

import streamlit as st

from .theme import Palette


def _esc(x: object) -> str:
    return html.escape("" if x is None else str(x))


# --------------------------------------------------------------------------- #
#  Navigation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NavItem:
    """One entry in the left rail."""
    key: str
    label: str
    icon: str = ""
    group: str = ""
    badge: str = ""
    # A page that cannot run yet is shown but disabled, with the reason. A nav
    # item that silently does nothing is worse than one that says why.
    enabled: bool = True
    blocked_reason: str = ""


@dataclass
class NavState:
    items: list[NavItem] = field(default_factory=list)
    selected: str = ""


def sidebar_nav(items: Iterable[NavItem], palette: Palette,
                state_key: str = "nav_page", default: str = "") -> str:
    """Render the left rail and return the selected page key.

    Uses real Streamlit buttons rather than styled markdown: a nav built from
    anchors would need a page reload to change selection, losing every widget
    value and any running job's progress.

    Grouping is derived from the items themselves, so a new group appears by
    virtue of a page declaring it.
    """
    items = list(items)
    if not items:
        return ""

    selectable = [i.key for i in items if i.enabled]
    current = st.session_state.get(state_key) or default or (
        selectable[0] if selectable else items[0].key)
    # A page can become unavailable between reruns (a dataset gets removed).
    # Falling back beats rendering a page whose preconditions no longer hold.
    if current not in selectable and selectable:
        current = selectable[0]

    seen_groups: set[str] = set()
    for item in items:
        if item.group and item.group not in seen_groups:
            seen_groups.add(item.group)
            st.markdown(f'<div class="hx-navgroup">{_esc(item.group)}</div>',
                        unsafe_allow_html=True)

        active = item.key == current
        label = f"{item.icon}  {item.label}" if item.icon else item.label
        if item.badge:
            label = f"{label}  ·  {item.badge}"

        st.markdown(
            f'<div class="hx-navitem{" active" if active else ""}'
            f'{" disabled" if not item.enabled else ""}">',
            unsafe_allow_html=True)
        clicked = st.button(
            label, key=f"nav_{item.key}", width="stretch",
            disabled=not item.enabled,
            help=item.blocked_reason if not item.enabled else None,
            type="primary" if active else "secondary")
        st.markdown("</div>", unsafe_allow_html=True)

        if clicked and item.enabled:
            st.session_state[state_key] = item.key
            current = item.key

    st.session_state[state_key] = current
    return current


def brand(name: str, subtitle: str = "") -> None:
    """Product mark at the top of the rail."""
    st.markdown(
        f'<div class="hx-brand"><span class="mark"></span>'
        f'<span class="name">{_esc(name)}</span></div>'
        + (f'<div class="hx-brandsub">{_esc(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True)


def nav_profile(name: str, sub: str = "") -> None:
    """The account block that sits at the foot of the rail."""
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "?"
    st.markdown(
        f'<div class="hx-profile"><span class="av">{_esc(initials)}</span>'
        f'<span class="who"><span class="n">{_esc(name)}</span>'
        + (f'<span class="s">{_esc(sub)}</span>' if sub else "")
        + "</span></div>",
        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Page chrome
# --------------------------------------------------------------------------- #
def page_header(breadcrumb: list[str], title: str = "",
                subtitle: str = "") -> None:
    """Breadcrumb + title, matching the dashboard header pattern."""
    crumbs = ' <span class="sep">·</span> '.join(
        f"<span>{_esc(c)}</span>" for c in breadcrumb)
    st.markdown(
        f'<div class="hx-crumbs">{crumbs}</div>'
        + (f'<h1 class="hx-title">{_esc(title)}</h1>' if title else "")
        + (f'<div class="hx-subtitle">{_esc(subtitle)}</div>' if subtitle else ""),
        unsafe_allow_html=True)


class card:
    """A raised panel. Use as a context manager so content nests naturally.

        with card("Total balance", "Revenue minus expenses"):
            st.altair_chart(...)

    Implemented by opening a styled wrapper, yielding to a Streamlit container,
    then closing it. The container is what lets real widgets — charts, tables,
    buttons — live inside the card rather than only static markup.
    """

    def __init__(self, title: str = "", subtitle: str = "",
                 actions: str = "", pad: bool = True):
        self.title = title
        self.subtitle = subtitle
        self.actions = actions
        self.pad = pad
        self._container = None

    def __enter__(self):
        st.markdown(f'<div class="hx-card{"" if self.pad else " flush"}">',
                    unsafe_allow_html=True)
        if self.title or self.actions:
            st.markdown(
                '<div class="hx-cardhead"><div>'
                + (f'<div class="t">{_esc(self.title)}</div>' if self.title else "")
                + (f'<div class="s">{_esc(self.subtitle)}</div>'
                   if self.subtitle else "")
                + "</div>"
                + (f'<div class="a">{self.actions}</div>' if self.actions else "")
                + "</div>",
                unsafe_allow_html=True)
        self._container = st.container()
        return self._container.__enter__()

    def __exit__(self, *exc):
        result = self._container.__exit__(*exc)
        st.markdown("</div>", unsafe_allow_html=True)
        return result


# --------------------------------------------------------------------------- #
#  KPI tiles
# --------------------------------------------------------------------------- #
@dataclass
class Kpi:
    """A headline number with an optional change badge.

    `delta` is a signed fraction (0.12 = +12%). `higher_is_better` decides
    whether a rise is good — for cost and latency it is not, and colouring a
    cost increase green is the kind of small lie that makes a dashboard
    untrustworthy.
    """
    label: str
    value: str
    delta: float | None = None
    higher_is_better: bool = True
    note: str = ""
    icon: str = ""


def kpi_row(items: list[Kpi]) -> None:
    """A row of headline numbers.

    A KPI row rather than a chart: five unrelated quantities on shared axes is
    a classic way to make simple numbers unreadable.
    """
    if not items:
        return
    cells = []
    for k in items:
        badge = ""
        if k.delta is not None:
            good = (k.delta >= 0) == k.higher_is_better
            # The arrow carries the direction and the word carries the
            # judgement, so neither depends on the colour alone.
            arrow = "&#9650;" if k.delta >= 0 else "&#9660;"
            badge = (f'<span class="hx-delta {"up" if good else "down"}">'
                     f'{arrow} {abs(k.delta) * 100:.0f}%</span>')
        cells.append(
            f'<div class="hx-kpi"><div class="v">{_esc(k.value)}{badge}</div>'
            f'<div class="l">{_esc(k.icon)} {_esc(k.label)}</div>'
            + (f'<div class="n">{_esc(k.note)}</div>' if k.note else "")
            + "</div>")
    st.markdown(f'<div class="hx-kpis">{"".join(cells)}</div>',
                unsafe_allow_html=True)


def toolbar(left: str = "", right: str = "") -> None:
    """A thin row of context above a section (filters on the left, actions right)."""
    st.markdown(
        f'<div class="hx-toolbar"><div>{left}</div><div>{right}</div></div>',
        unsafe_allow_html=True)


def pill(text: str, tone: str = "") -> str:
    """An inline status pill. Returns markup so it can compose into a toolbar."""
    return f'<span class="hx-pill {_esc(tone)}">{_esc(text)}</span>'


def empty_state(title: str, body: str = "", hint: str = "") -> None:
    """What a page shows before it has data.

    Named and deliberate: a blank panel makes a reader wonder whether the app is
    broken, and 'no data' alone never says what to do next.
    """
    st.markdown(
        f'<div class="hx-empty"><div class="t">{_esc(title)}</div>'
        + (f'<div class="b">{_esc(body)}</div>' if body else "")
        + (f'<div class="h">{_esc(hint)}</div>' if hint else "")
        + "</div>",
        unsafe_allow_html=True)
