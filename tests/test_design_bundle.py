"""
The design bundle must describe the product, not a fork of it.

`design/build.py` copies `app.css` and `charts.js` out of the shipped UI so
`/design-sync` has self-contained preview pages to upload. A copy is a fork
waiting to happen: someone changes a token in the real stylesheet, the design
system keeps showing the old one, and the pane becomes confidently wrong about
what the product looks like. These tests make that failure loud.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "design" / "build.py"
DIST = ROOT / "design" / "dist"
STATIC = ROOT / "harness" / "web" / "static"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """Run the real builder once, and fail if it cannot run at all."""
    r = subprocess.run([sys.executable, str(BUILD)], capture_output=True,
                       text=True, cwd=ROOT, check=False)
    assert r.returncode == 0, f"design/build.py failed:\n{r.stdout}\n{r.stderr}"
    return DIST


def _build_module():
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("ds_build", BUILD)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_mirrored_files_are_copied_not_stubbed(built):
    """The builder must actually copy the shipped assets, not create empty
    placeholders.

    Note what this does NOT prove: the fixture rebuilds before asserting, so
    it cannot catch a stale copy on disk. Staleness is prevented by
    construction instead — `design/dist/` is generated output, rebuilt from
    source on every run, and never edited by hand. The guard against the
    bundle lying about the product is the class test below, which is the
    failure that actually bites.
    """
    mod = _build_module()
    for name in mod.MIRRORED:
        assert (built / name).read_bytes() == (STATIC / name).read_bytes()
        assert (built / name).stat().st_size > 0


#: Classes belonging to the preview chrome in build.py, not to the product.
_PREVIEW_ONLY = {"ds-h", "ds-sub", "ds-set", "ds-label"}


def test_every_class_a_preview_uses_still_exists_in_the_stylesheet(built):
    """The drift that actually matters.

    A preview naming a class that `app.css` no longer defines renders
    unstyled, and the Design System pane then shows a component the product
    does not have. This is the failure mode a copied stylesheet invites, and
    unlike a byte comparison it survives the rebuild — the preview markup is
    written by hand in build.py, so it can fall behind the real CSS.
    """
    css = (STATIC / "app.css").read_text(encoding="utf-8")
    defined = set(re.findall(r"\.([a-zA-Z][\w-]*)", css))

    used: set[str] = set()
    for p in built.rglob("*.html"):
        for attr in re.findall(r'class="([^"]+)"', p.read_text(encoding="utf-8")):
            used.update(attr.split())

    unknown = sorted(used - defined - _PREVIEW_ONLY)
    assert not unknown, (
        f"previews use classes app.css does not define: {unknown}. "
        f"Either the class was renamed and design/build.py was not updated, "
        f"or the preview invented markup the product does not ship.")


def test_every_preview_declares_its_card_on_the_first_line(built):
    """The Design System pane reads `@dsCard` from line one. A marker on line
    two is a card that silently never appears in the pane."""
    pages = sorted(built.rglob("*.html"))
    assert pages, "the builder produced no preview pages"
    for p in pages:
        first = p.read_text(encoding="utf-8").splitlines()[0]
        assert re.match(r'^<!-- @dsCard group="[^"]+" -->$', first), (
            f"{p.relative_to(built)} line 1 is not a @dsCard marker: {first!r}")


def test_previews_reference_the_bundled_css_and_nothing_remote(built):
    """A preview that fetches from a CDN breaks the offline promise the UI is
    built on, and renders unstyled wherever the network is absent."""
    for p in sorted(built.rglob("*.html")):
        html = p.read_text(encoding="utf-8")
        depth = len(p.relative_to(built).parts) - 1
        assert f'href="{"../" * depth}app.css"' in html, (
            f"{p.relative_to(built)} does not link the bundled app.css")
        remote = re.findall(r'(?:href|src)="(https?://[^"]+)"', html)
        assert not remote, f"{p.relative_to(built)} fetches remotely: {remote}"


def test_the_bundle_covers_the_classes_the_ui_actually_ships(built):
    """A design system that documents six of forty components is a style
    guide nobody trusts. This pins the load-bearing ones rather than every
    class, so adding a utility class does not fail the suite."""
    html = "\n".join(p.read_text(encoding="utf-8")
                     for p in built.rglob("*.html"))
    for cls in ("btn", "card", "kpi", "msg", "tag", "tab", "subtab", "eyebrow",
                "empty", "pick", "failcard", "step", "refuse", "null"):
        assert f'"{cls}' in html or f" {cls}" in html, (
            f"no preview demonstrates .{cls}")


def test_dist_is_not_hand_edited(built):
    """dist/ is generated. A file there that the builder does not know how to
    produce is someone's manual edit, and it will vanish on the next build."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("ds_build", BUILD)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    expected = {p for p, *_ in mod.PAGES} | set(mod.MIRRORED)
    actual = {str(p.relative_to(built)).replace("\\", "/")
              for p in built.rglob("*") if p.is_file()}
    assert actual == expected, (
        f"unexpected files in design/dist: {sorted(actual - expected)}; "
        f"missing: {sorted(expected - actual)}")
