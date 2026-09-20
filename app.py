"""
Multi-Model LLM Evaluation Harness - the app.

    streamlit run app.py

**This file is a shell, not a screen.** It builds the sidebar, assembles the
context a page needs, and renders exactly one page from the registry. Every
screen lives in `harness/ui/screens/`, so adding one is a class plus a registry
entry and nothing here changes.

That split matters for more than tidiness. The previous version used tabs, and
Streamlit executes *every* tab body on *every* rerun, so moving one slider did
the work of all six screens. A rail with a single selected page runs one.

Design choices for a self-contained single-user app:
  * Qdrant runs EMBEDDED (a local folder), so no Docker is required.
  * Evaluation runs on a background thread. The worker only mutates a plain
    `Job` object - it never calls Streamlit, which would fail outside the script
    run context - and a fragment polls that Job.
  * One heavy job at a time, enforced: embedded Qdrant and the provider bill
    both prefer serial runs here.

Every run spends real money, so the Run screen estimates cost *including the
judge*, and a budget ceiling aborts cleanly rather than letting a mis-typed
tuning budget bill for hours.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import yaml

from harness.env import load_env
from harness.store.store import TraceStore
from harness.ui import active_palette, page_css, register_altair_theme
from harness.ui.layout import brand, nav_profile, sidebar_nav
from harness.ui.pages import PageContext
from harness.ui.screens import default_registry

# Before anything reads an API key. Real environment variables still win.
load_env()

st.set_page_config(page_title="LLM Eval Harness", layout="wide",
                   initial_sidebar_state="expanded")

PALETTE = active_palette()
register_altair_theme(PALETTE)
st.markdown(page_css(PALETTE), unsafe_allow_html=True)

WORKSPACE = Path("workspace")
WORKSPACE.mkdir(exist_ok=True)
DATA_DIR = WORKSPACE / "data"
QDRANT_PATH = str(WORKSPACE / "qdrant")
STORE_PATH = str(WORKSPACE / "traces")
CACHE_DIR = str(WORKSPACE / "cache")

PROVIDER_ENV = {
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}


@st.cache_data(show_spinner=False)
def _yaml(path: str, _mtime: float) -> dict:
    """Config, reloaded only when the file actually changes on disk."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _cfg(path: str) -> dict:
    p = Path(path)
    return _yaml(path, p.stat().st_mtime if p.exists() else 0.0)


def _profiles_on_disk() -> list[str]:
    """App-created profiles, discovered from the workspace.

    Read from disk rather than session state so a reload does not lose which
    datasets exist, the tabbed version showed a fresh install on every refresh
    while the data sat right there.
    """
    if not DATA_DIR.exists():
        return []
    return sorted(p.parent.name for p in DATA_DIR.glob("*/profile.yaml"))


def _load_profile(name: str):
    from harness.profiles.profile import Profile, ProfileError

    path = DATA_DIR / name / "profile.yaml"
    if not path.exists():
        return None
    try:
        return Profile.from_yaml(str(path))
    except ProfileError as e:
        st.sidebar.error(str(e))
        return None


# --------------------------------------------------------------------------- #
#  Context
# --------------------------------------------------------------------------- #
models_cfg = _cfg("configs/models.yaml")
run_cfg = _cfg("configs/run.yaml")
default_provider = models_cfg.get("default_provider", "together")
store = TraceStore(STORE_PATH)

available_profiles = _profiles_on_disk()
selected_profile = st.session_state.get("profile_name")
if selected_profile not in available_profiles:
    selected_profile = available_profiles[0] if available_profiles else ""
    st.session_state["profile_name"] = selected_profile

registry = default_registry()

# --------------------------------------------------------------------------- #
#  Sidebar: brand, nav, then setup
# --------------------------------------------------------------------------- #
with st.sidebar:
    brand("Eval Harness", "model evaluation")

    ctx = PageContext(
        palette=PALETTE, workspace=WORKSPACE, data_dir=DATA_DIR,
        store_path=STORE_PATH, qdrant_path=QDRANT_PATH, cache_dir=CACHE_DIR,
        models_cfg=models_cfg, run_cfg=run_cfg,
        api_key=st.session_state.get("api_key", os.environ.get(
            PROVIDER_ENV.get(default_provider, ""), "")),
        budget=float(st.session_state.get("budget",
                                          run_cfg.get("budget_usd", 0.0))),
        store=store,
        profile=_load_profile(selected_profile) if selected_profile else None,
        profile_name=selected_profile,
    )

    current = sidebar_nav(registry.nav_items(ctx), PALETTE,
                          default=registry.first_available(ctx))

    st.divider()
    st.markdown('<div class="hx-navgroup">Setup</div>', unsafe_allow_html=True)

    env_var = PROVIDER_ENV.get(default_provider, "TOGETHER_API_KEY")
    api_key = st.text_input(
        f"{default_provider} key", value=ctx.api_key, type="password",
        help=f"Read from {env_var} if set. Other providers use their own env "
             f"var and are picked up automatically.")
    st.session_state["api_key"] = api_key
    ctx.api_key = api_key

    # Which providers are reachable. Shown up front so a mid-run auth failure
    # becomes a pre-run glance.
    configured = {default_provider: bool(api_key)}
    for name in (models_cfg.get("providers") or {}):
        configured.setdefault(name,
                              bool(os.environ.get(PROVIDER_ENV.get(name, ""))))
    for m in models_cfg.get("models", []):
        head = m.split(":", 1)[0] if ":" in m else ""
        if head in PROVIDER_ENV:
            configured.setdefault(head, bool(os.environ.get(PROVIDER_ENV[head])))

    rows = "".join(
        f'<div class="hx-provrow"><span>{name}</span>'
        f'<span class="{"ok" if ok else "off"}">'
        f'{"connected" if ok else "no key"}</span></div>'
        for name, ok in sorted(configured.items()))
    st.markdown(f'<div class="hx-provlist">{rows}</div>', unsafe_allow_html=True)

    budget = st.number_input(
        "Budget ceiling (USD)", min_value=0.0, value=ctx.budget, step=1.0,
        help="0 = no limit. A run aborts cleanly when measured spend crosses "
             "this; rows already written are kept.")
    st.session_state["budget"] = budget
    ctx.budget = budget

    spent = 0.0
    if store.exists:
        _df = store.load_all()
        if not _df.empty and "cost_usd" in _df.columns:
            spent = float(_df["cost_usd"].sum(skipna=True))
    frac = min(1.0, spent / budget) if budget > 0 else 0.0
    tone = "critical" if frac >= 0.9 else ("warning" if frac >= 0.7 else "")
    meter = (f'<div class="hx-meter"><div class="row"><span>Spend</span>'
             f'<span>${spent:,.4f}')
    meter += (f' / ${budget:,.2f}</span></div>'
              f'<div class="track"><div class="fill {tone}" '
              f'style="width:{frac * 100:.1f}%"></div></div></div>'
              if budget > 0
              else ' · no ceiling</span></div><div class="track"></div></div>')
    st.markdown(meter, unsafe_allow_html=True)

    if available_profiles:
        st.markdown('<div class="hx-navgroup">Dataset</div>',
                    unsafe_allow_html=True)
        chosen = st.selectbox(
            "Active profile", available_profiles,
            index=(available_profiles.index(selected_profile)
                   if selected_profile in available_profiles else 0),
            label_visibility="collapsed")
        if chosen != selected_profile:
            st.session_state["profile_name"] = chosen
            st.rerun()

    st.divider()
    nav_profile("Local workspace", WORKSPACE.resolve().name)

# --------------------------------------------------------------------------- #
#  Render exactly one page
# --------------------------------------------------------------------------- #
page = registry.get(current)
if page is None:
    st.error(f"Unknown page: {current}")
else:
    blocked = page.available(ctx)
    if blocked:
        from harness.ui.layout import empty_state, page_header

        page_header(list(page.meta.breadcrumb), page.meta.title)
        empty_state(blocked,
                    "This screen needs something that is not ready yet.",
                    "Use the workflow screens in the sidebar to get there.")
    else:
        page.render(ctx)
