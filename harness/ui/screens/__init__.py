"""
The default set of screens, assembled into a registry.

This is the one place that knows which pages exist. Adding a screen is a class
plus a line here, the shell, the nav and every other page are untouched, which
is the Open/Closed property the registry exists to provide.

Order is the rail order, and it follows the journey: what you decided, the
detail behind it, then the workflow that produced it, then the system itself.
Overview sits first because it carries the conclusion.
"""

from __future__ import annotations

from ..pages import PageRegistry
from .analysis import CapabilitiesPage, ReportsPage
from .overview import OverviewPage
from .workflow import DataPage, ProbesPage, RunPage

__all__ = ["OverviewPage", "ReportsPage", "DataPage", "ProbesPage", "RunPage",
           "CapabilitiesPage", "default_registry"]


def default_registry() -> PageRegistry:
    """Every screen the app ships with."""
    return PageRegistry([
        OverviewPage(),
        ReportsPage(),
        DataPage(),
        ProbesPage(),
        RunPage(),
        CapabilitiesPage(),
    ])
