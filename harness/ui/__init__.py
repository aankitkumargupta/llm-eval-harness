"""Shared presentation layer for the Streamlit app and the dashboard.

Kept separate from `harness/report` on purpose: `report` computes numbers and is
importable with no UI dependency, while this package only renders them. The two
surfaces then cannot drift, because they read the same palette, the same chart
builders and the same components.
"""

from .theme import (
    DARK,
    LIGHT,
    MIDNIGHT,
    Palette,
    active_palette,
    page_css,
    register_altair_theme,
)

__all__ = ["LIGHT", "DARK", "MIDNIGHT", "Palette", "active_palette",
           "page_css", "register_altair_theme"]
