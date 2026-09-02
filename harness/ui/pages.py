"""
The page registry.

A sidebar app needs a mapping from nav entry to rendered content. The obvious
implementation is a chain of `if page == "results": ... elif page == "data":`
in the shell, which makes the shell grow with every screen and puts unrelated
reasons to change in one file.

Instead a page is an object: it declares its own nav entry, decides for itself
whether its preconditions are met, and renders. The shell builds the rail from
the registry and calls exactly one `render`. Adding a screen is a class plus a
registry entry, with nothing existing modified.

Two consequences worth having:

  * **Only the selected page runs.** Streamlit executes every tab body on every
    rerun, so the tabbed version did the work of all five screens to show one.
  * **A blocked page says why.** `available()` returns a reason rather than a
    bool, so the rail can disable an entry and explain it, instead of offering a
    screen that renders an empty panel.

`PageContext` is what a page is allowed to depend on. Pages receive it; they do
not reach for globals, which is what keeps them testable without a browser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from .theme import Palette


@dataclass
class PageContext:
    """Everything a page may use. Passed in, never imported from the shell."""

    palette: Palette
    workspace: Path
    data_dir: Path
    store_path: str
    qdrant_path: str
    cache_dir: str

    models_cfg: dict = field(default_factory=dict)
    run_cfg: dict = field(default_factory=dict)

    api_key: str = ""
    budget: float = 0.0

    # Resolved lazily by the shell and shared across pages so a single rerun
    # does not load the trace store more than once.
    store: object = None
    profile: object = None       # the currently selected Profile, if any
    profile_name: str = ""

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def has_results(self) -> bool:
        return bool(self.store is not None and getattr(self.store, "exists", False))


@dataclass(frozen=True)
class PageMeta:
    """How a page presents itself in the rail and the header."""
    key: str
    label: str
    icon: str = ""
    group: str = ""
    breadcrumb: tuple[str, ...] = ()
    title: str = ""
    subtitle: str = ""


@runtime_checkable
class Page(Protocol):
    """One screen."""

    meta: PageMeta

    def available(self, ctx: PageContext) -> str:
        """Empty string when the page can run, else the reason it cannot.

        A reason rather than a bool, so the rail can disable the entry *and*
        say what to do about it. "Build a dataset first" is actionable;
        a greyed-out button is not.
        """
        ...

    def render(self, ctx: PageContext) -> None: ...


class PageRegistry:
    """Ordered collection of pages, addressable by key."""

    def __init__(self, pages: list[Page] | None = None):
        self._pages: list[Page] = list(pages or [])

    def add(self, page: Page) -> PageRegistry:
        if any(p.meta.key == page.meta.key for p in self._pages):
            raise ValueError(f"duplicate page key: {page.meta.key}")
        self._pages.append(page)
        return self

    def __iter__(self):
        return iter(self._pages)

    def __len__(self) -> int:
        return len(self._pages)

    @property
    def keys(self) -> list[str]:
        return [p.meta.key for p in self._pages]

    def get(self, key: str) -> Page | None:
        return next((p for p in self._pages if p.meta.key == key), None)

    def nav_items(self, ctx: PageContext):
        """Build the rail, asking each page whether it can run."""
        from .layout import NavItem

        items = []
        for page in self._pages:
            reason = page.available(ctx)
            items.append(NavItem(
                key=page.meta.key, label=page.meta.label, icon=page.meta.icon,
                group=page.meta.group, enabled=not reason,
                blocked_reason=reason))
        return items

    def first_available(self, ctx: PageContext) -> str:
        for page in self._pages:
            if not page.available(ctx):
                return page.meta.key
        return self._pages[0].meta.key if self._pages else ""
