"""
The product pages in the left sidebar (capabilities, architecture, trust,
roadmap, deployment): every page is reachable from the sidebar, every
sidebar entry points at a real screen or page, the task bar and its screens
are untouched, the pages compute nothing, and the landing keeps hidden views
hidden. Source-level checks: there is no JS runtime in the offline suite.
"""

from __future__ import annotations

import re

from harness.web.server import STATIC


def _js():
    return (STATIC / "app.js").read_text(encoding="utf-8")


def _block(js, start, end):
    i = js.index(start)
    return js[i:js.index(end, i)]


def test_every_product_page_is_in_the_sidebar_and_every_sidebar_entry_exists():
    js = _js()
    stories = set(re.findall(r"(\w+):\s*view", _block(js, "const STORIES = {", "};")))
    views = set(re.findall(r"(\w+):\s*view", _block(js, "const VIEWS = {", "};")))
    groups = _block(js, "const STORY_GROUPS = [", "\n];")
    entries = set(re.findall(r'\["([a-z]+)", "[^"]+", "[^"]*", "[^"]+"\]', groups))
    assert {"capabilities", "architecture", "trust", "roadmap", "deployment"} <= stories
    assert stories <= entries, f"product pages missing from the sidebar: {sorted(stories - entries)}"
    unknown = entries - views - stories
    assert not unknown, f"sidebar entries with no screen behind them: {sorted(unknown)}"
    # Every task-bar screen is also reachable from the sidebar's Workspace group.
    nav = set(re.findall(r'\["([a-z]+)",\s*"[^"]+",\s*"[^"]+"\]', _block(js, "const SECTIONS = [", "\n];")))
    assert nav <= entries, f"screens absent from the sidebar: {sorted(nav - entries)}"


def test_the_task_bar_is_unchanged_by_the_sidebar():
    js = _js()
    sections = _block(js, "const SECTIONS = [", "\n];")
    # The product pages are not task-bar screens: they live only in STORIES.
    for page in ("capabilities", "architecture", "trust", "roadmap", "deployment"):
        assert f'"{page}"' not in sections
    assert "STORIES[S.view]" in js and "renderSidebar()" in js
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="sidebar"' in html and 'id="sidebtn"' in html and 'id="sections"' in html


def test_product_pages_compute_nothing_and_stay_honest():
    js = _js()
    pages = _block(js, "function viewCapabilities()", "const STORIES = {")
    assert not re.search(r"Math\.|\.reduce\(|fetch\(", pages), "product pages are prose and status only"
    # Roadmap items carry their register numbers; a stated gap is never dressed as built.
    assert "R-30" in pages and "R-34" in pages
    assert 'status("Roadmap")' not in pages  # statuses come from the row data, not hard-coded calls
    assert pages.count('"Roadmap"') >= 3 and pages.count('"Partial"') >= 3
    assert "—" not in pages


def test_sidebar_collapses_to_a_rail_and_remembers_it():
    js = _js()
    side = _block(js, "function buildSidebar()", "function renderSidebar()")
    assert "harness-sidebar-collapsed" in side and 'classList.toggle("collapsed"' in side
    assert 'class: "sidebar-collapse"' in side
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    assert ".sidebar.collapsed" in css and "@media (min-width: 1001px)" in css


def test_sign_out_asks_before_clearing_the_session():
    js = _js()
    user = _block(js, "async function initUser()", "const post = ")
    # The only path to the logout endpoint is the dialog's "confirm" value
    # (or the plain confirm() fallback); the button itself never signs out.
    assert user.count('post("/api/auth/logout"') == 1
    assert 'returnValue === "confirm"' in user and "showModal" in user
    assert "window.confirm(" in user
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert '<dialog id="logoutdlg"' in html and 'value="confirm"' in html and 'value="cancel"' in html


def test_landing_keeps_hidden_views_hidden():
    css = (STATIC / "landing.css").read_text(encoding="utf-8")
    assert ".landing [hidden] { display: none !important; }" in css
