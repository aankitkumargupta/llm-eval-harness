"""
Design tokens and theming for the Streamlit surfaces.

One palette, defined once, used by every chart and every component in both the
app and the dashboard. The point is that a reader learns the visual language
once: blue always means "the thing under discussion", gray always means
"context", and the status colours only ever mean status.

The palette is not a taste call — it was run through a colour-vision validator
before being written down. The categorical slots clear the CVD separation and
normal-vision floors on both surfaces under the all-pairs rule that scatter and
heatmap forms demand. Dark mode is a **selected** set of steps for the dark
surface, not an automatic inversion of the light one.

Three rules the rest of the UI inherits:

  * **Colour follows the entity, never its rank.** Filtering a model out never
    repaints the survivors.
  * **Status colours are reserved.** good/warning/critical never stand in for
    "series 3", and never carry meaning without a label beside them.
  * **Emphasis over categorical.** When one model is the answer and the rest are
    context, that is one accent hue plus gray — not eight hues.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt


@dataclass(frozen=True)
class Palette:
    """Every colour the UI is allowed to use, for one surface."""

    name: str

    # Surfaces and ink
    surface: str
    page: str
    card: str
    ink: str
    ink_secondary: str
    ink_muted: str
    grid: str
    axis: str
    border: str

    # Categorical slots. Only the first three are used: scatter and heatmap are
    # all-pairs forms, and past three slots no ordering clears the floors.
    series_1: str   # blue   - the subject
    series_2: str   # orange
    series_3: str   # aqua

    # Emphasis: everything that is context rather than subject.
    muted_mark: str

    # Status. Fixed across themes, and never used for identity.
    good: str
    warning: str
    serious: str
    critical: str

    # Sequential ramp (one hue, light -> dark) for magnitude encodings.
    seq: tuple[str, ...]

    # Neutral midpoint for diverging scales. Per-palette rather than a constant:
    # it has to read as "nothing" against *this* surface, and a grey borrowed
    # from a different-hued theme reads as a faint colour instead.
    neutral: str = "#f0efec"

    @property
    def is_dark(self) -> bool:
        return self.name in ("dark", "midnight")

    @property
    def diverging(self) -> tuple[str, str, str]:
        """Blue <-> red with a neutral gray midpoint.

        Warm/cool poles read as opposite; the gray midpoint reads as "nothing",
        which is exactly what "no significant difference" means.
        """
        return (self.series_1, self.grid_mid, self.critical)

    @property
    def grid_mid(self) -> str:
        return self.neutral


LIGHT = Palette(
    name="light",
    surface="#fcfcfb", page="#f9f9f7", card="#ffffff",
    ink="#0b0b0b", ink_secondary="#52514e", ink_muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7", border="rgba(11,11,11,0.10)",
    series_1="#2a78d6", series_2="#eb6834", series_3="#1baf7a",
    muted_mark="#898781",
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
    seq=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
         "#184f95"),
    neutral="#f0efec",
)

DARK = Palette(
    name="dark",
    surface="#1a1a19", page="#0d0d0d", card="#1f1f1e",
    ink="#ffffff", ink_secondary="#c3c2b7", ink_muted="#898781",
    grid="#2c2c2a", axis="#383835", border="rgba(255,255,255,0.10)",
    series_1="#3987e5", series_2="#d95926", series_3="#199e70",
    muted_mark="#898781",
    good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b",
    seq=("#0d366b", "#184f95", "#256abf", "#2a78d6", "#3987e5", "#6da7ec",
         "#9ec5f4"),
    neutral="#383835",
)


# The product palette: a blue-tinted near-black with raised cards, matching the
# dashboard style the app is built to. Distinct from DARK, which is the neutral
# warm-grey dark surface the reference palette was originally stepped for.
#
# The three categorical slots are DARK's, re-validated against this navy surface
# rather than assumed to carry over: all-pairs CVD dE 9.4, normal-vision 20.9,
# every slot >= 3:1 contrast, all inside the dark lightness band. Changing a
# surface changes those results, so it is re-run rather than trusted.
MIDNIGHT = Palette(
    name="midnight",
    # `page` is the deepest layer, `card` sits above it, `surface` is what a
    # chart draws on. Three depths is what makes a card layout read as layered
    # rather than as boxes drawn on a flat background.
    surface="#0f1523", page="#0b1020", card="#141b2d",
    ink="#e8ecf5", ink_secondary="#9aa5bd", ink_muted="#6b7691",
    grid="#1e2740", axis="#2a3550", border="rgba(255,255,255,0.07)",
    series_1="#3987e5", series_2="#d95926", series_3="#199e70",
    muted_mark="#6b7691",
    good="#22c55e", warning="#fab219", serious="#ec835a", critical="#ef4444",
    seq=("#0d2a52", "#123a72", "#1a4f96", "#2464bd", "#3987e5", "#6da7ec",
         "#9ec5f4"),
    neutral="#383a40",
)


def active_palette() -> Palette:
    """The palette to render with.

    Resolution order, and the reason for it:

    1. **`theme.base` from `.streamlit/config.toml`**, when set. Pinning a base
       is an explicit deployment choice, so it wins.
    2. **`st.context.theme.type`**, when no base is pinned. Then the app follows
       the viewer, including a mid-session switch.

    The order is this way round because the two disagree. `st.context.theme`
    reports the *browser's* preference, not the theme Streamlit actually
    applied: with `base = "dark"` in config and a light OS setting it returns
    `"light"` while the chrome renders dark. Trusting it first produced light
    cards inside dark chrome — verified against a running app, not assumed.
    """
    try:
        import streamlit as st
    except Exception:  # noqa: BLE001 - importable without Streamlit installed
        return LIGHT

    configured = None
    try:
        configured = st.get_option("theme.base")
    except Exception:  # noqa: BLE001
        pass
    if configured in ("dark", "light"):
        return MIDNIGHT if configured == "dark" else LIGHT

    live = getattr(getattr(st, "context", None), "theme", None)
    return MIDNIGHT if getattr(live, "type", None) == "dark" else LIGHT


# --------------------------------------------------------------------------- #
#  Altair theme
# --------------------------------------------------------------------------- #
def register_altair_theme(palette: Palette) -> str:
    """Register and enable an Altair theme built from `palette`.

    Centralising the chrome here is what keeps every chart quiet: hairline
    solid gridlines one step off the surface, no chart border, no dashes, ink
    in text tokens rather than series colours, and generous padding. Individual
    charts then only describe their data.
    """
    theme_name = f"harness_{palette.name}"

    @alt.theme.register(theme_name, enable=True)
    def _theme() -> alt.theme.ThemeConfig:  # type: ignore[misc]
        return alt.theme.ThemeConfig(
            {
                "config": {
                    "background": "transparent",
                    "font": 'system-ui, -apple-system, "Segoe UI", sans-serif',
                    "view": {"stroke": None, "continuousWidth": 320,
                             "continuousHeight": 240},
                    "padding": {"top": 8, "bottom": 8, "left": 4, "right": 4},
                    "axis": {
                        "labelColor": palette.ink_muted,
                        "titleColor": palette.ink_secondary,
                        "labelFontSize": 11,
                        "titleFontSize": 11,
                        "titleFontWeight": "normal",
                        "titlePadding": 8,
                        "domainColor": palette.axis,
                        "tickColor": palette.axis,
                        "domainWidth": 1,
                        "tickSize": 4,
                        # Solid hairline: no gridDash key at all, since any
                        # dashing reads as "threshold" when this is only a grid.
                        "gridColor": palette.grid,
                        "gridWidth": 1,
                        "labelPadding": 4,
                    },
                    "legend": {
                        "labelColor": palette.ink_secondary,
                        "titleColor": palette.ink_secondary,
                        "labelFontSize": 11,
                        "titleFontSize": 11,
                        "titleFontWeight": "normal",
                        "symbolType": "circle",
                        "symbolSize": 90,
                        "orient": "top",
                        "direction": "horizontal",
                        "offset": 4,
                    },
                    "title": {
                        "color": palette.ink,
                        "fontSize": 13,
                        "fontWeight": 600,
                        "anchor": "start",
                        "offset": 10,
                        "subtitleColor": palette.ink_muted,
                        "subtitleFontSize": 11,
                    },
                    "bar": {
                        # 4px rounded data-end, square at the baseline.
                        "cornerRadiusEnd": 4,
                        "color": palette.series_1,
                    },
                    "line": {"strokeWidth": 2, "strokeCap": "round",
                             "strokeJoin": "round"},
                    "point": {"size": 90, "filled": True,
                              "strokeWidth": 2, "stroke": palette.surface},
                    "rect": {"stroke": None},
                    "text": {"color": palette.ink_secondary, "fontSize": 11},
                    "range": {
                        "category": [palette.series_1, palette.series_2,
                                     palette.series_3],
                        "heatmap": list(palette.seq),
                        "ramp": list(palette.seq),
                    },
                }
            }
        )

    return theme_name


# --------------------------------------------------------------------------- #
#  Page CSS
# --------------------------------------------------------------------------- #
def page_css(palette: Palette) -> str:
    """Global CSS for the app.

    Deliberately narrow: it tightens Streamlit's default spacing, styles the
    custom components below, and does nothing that fights the framework. It
    does not restyle Streamlit's own widgets, which would break the moment
    Streamlit changes a class name.
    """
    p = palette
    return f"""
<style>
  :root {{
    --surface: {p.surface};
    --page: {p.page};
    --card: {p.card};
    --ink: {p.ink};
    --ink-2: {p.ink_secondary};
    --ink-3: {p.ink_muted};
    --grid: {p.grid};
    --border: {p.border};
    --accent: {p.series_1};
    --good: {p.good};
    --warning: {p.warning};
    --critical: {p.critical};
    --radius: 10px;
  }}

  /* Tighten the default page rhythm; Streamlit ships very airy defaults that
     push the decision-critical content below the fold. */
  .block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
  h1 {{ font-size: 1.55rem !important; font-weight: 650 !important;
        letter-spacing: -0.01em; margin-bottom: .1rem !important; }}
  h2 {{ font-size: 1.1rem !important; font-weight: 620 !important;
        margin-top: 1.6rem !important; }}
  h3 {{ font-size: .95rem !important; font-weight: 600 !important; }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}

  .hx-sub {{ color: var(--ink-3); font-size: .88rem; margin: .1rem 0 1.1rem; }}

  /* --- stat tiles --------------------------------------------------------- */
  .hx-tiles {{ display: grid; gap: .6rem; grid-template-columns:
      repeat(auto-fit, minmax(140px, 1fr)); margin: .2rem 0 1rem; }}
  .hx-tile {{ border: 1px solid var(--border); border-radius: var(--radius);
      padding: .7rem .85rem; background: var(--card); }}
  .hx-tile .k {{ font-size: .68rem; letter-spacing: .05em; text-transform: uppercase;
      color: var(--ink-3); font-weight: 600; }}
  .hx-tile .v {{ font-size: 1.4rem; font-weight: 650; color: var(--ink);
      margin-top: .15rem; line-height: 1.15; }}
  .hx-tile .n {{ font-size: .72rem; color: var(--ink-3); margin-top: .1rem; }}
  .hx-tile .v.good {{ color: var(--good); }}
  .hx-tile .v.warning {{ color: var(--warning); }}
  .hx-tile .v.critical {{ color: var(--critical); }}

  /* --- hero verdict ------------------------------------------------------- */
  .hx-hero {{ border: 1px solid var(--border); border-left: 3px solid var(--accent);
      border-radius: var(--radius); padding: 1rem 1.15rem; background: var(--card);
      margin: .4rem 0 1rem; }}
  .hx-hero .k {{ font-size: .68rem; letter-spacing: .05em; text-transform: uppercase;
      color: var(--ink-3); font-weight: 600; }}
  .hx-hero .v {{ font-size: 1.75rem; font-weight: 660; color: var(--ink);
      margin: .2rem 0 .1rem; line-height: 1.15; word-break: break-word; }}
  .hx-hero .n {{ font-size: .85rem; color: var(--ink-2); }}
  .hx-hero.none {{ border-left-color: var(--warning); }}

  /* --- pipeline stepper --------------------------------------------------- */
  .hx-steps {{ display: flex; flex-wrap: wrap; gap: .35rem; align-items: center;
      margin: .1rem 0 1.2rem; }}
  .hx-step {{ display: inline-flex; align-items: center; gap: .4rem;
      border: 1px solid var(--border); border-radius: 99px; padding: .28rem .7rem;
      font-size: .78rem; color: var(--ink-3); background: var(--card); }}
  .hx-step .dot {{ width: 7px; height: 7px; border-radius: 50%;
      background: var(--grid); flex: none; }}
  .hx-step.done {{ color: var(--ink-2); }}
  .hx-step.done .dot {{ background: var(--good); }}
  .hx-step.active {{ color: var(--ink); border-color: var(--accent);
      font-weight: 600; }}
  .hx-step.active .dot {{ background: var(--accent); }}
  .hx-arrow {{ color: var(--grid); font-size: .8rem; }}

  /* --- budget meter ------------------------------------------------------- */
  .hx-meter {{ margin: .35rem 0 .2rem; }}
  .hx-meter .row {{ display: flex; justify-content: space-between;
      font-size: .75rem; color: var(--ink-3); margin-bottom: .25rem; }}
  .hx-meter .track {{ height: 6px; border-radius: 99px; background: var(--grid);
      overflow: hidden; }}
  .hx-meter .fill {{ height: 100%; border-radius: 99px; background: var(--accent); }}
  .hx-meter .fill.warning {{ background: var(--warning); }}
  .hx-meter .fill.critical {{ background: var(--critical); }}

  /* --- verdict lines ------------------------------------------------------ */
  .hx-verdict {{ display: flex; gap: .55rem; align-items: baseline;
      border: 1px solid var(--border); border-radius: 8px; padding: .5rem .7rem;
      margin-bottom: .35rem; background: var(--card); font-size: .87rem;
      color: var(--ink-2); }}
  .hx-verdict .tag {{ font-size: .66rem; font-weight: 700; letter-spacing: .04em;
      text-transform: uppercase; padding: .1rem .4rem; border-radius: 4px;
      flex: none; }}
  .hx-verdict .tag.sig {{ color: var(--good); border: 1px solid var(--good); }}
  .hx-verdict .tag.ns {{ color: var(--ink-3); border: 1px solid var(--border); }}

  .hx-note {{ color: var(--ink-3); font-size: .8rem; margin: .1rem 0 .6rem; }}

  /* --- feature catalogue -------------------------------------------------- */
  .hx-fgrid {{ display: grid; gap: .55rem; margin: .3rem 0 1rem;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
  .hx-fcard {{ border: 1px solid var(--border); border-radius: 9px;
      padding: .6rem .75rem; background: var(--card); }}
  .hx-fcard .n {{ font-size: .84rem; font-weight: 620; color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .hx-fcard .d {{ font-size: .78rem; color: var(--ink-2); margin-top: .22rem;
      line-height: 1.4; }}
  .hx-fcard .t {{ font-size: .7rem; color: var(--ink-3); margin-top: .25rem; }}

  .hx-chips {{ display: flex; flex-wrap: wrap; gap: .3rem; margin: .2rem 0 .9rem; }}
  .hx-chip {{ border: 1px solid var(--border); border-radius: 99px;
      padding: .16rem .55rem; font-size: .76rem; color: var(--ink-2);
      background: var(--card);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}

  /* --- capability matrix -------------------------------------------------- */
  .hx-scroll {{ overflow-x: auto; }}
  table.hx-matrix {{ border-collapse: collapse; width: 100%; font-size: .82rem;
      margin: .3rem 0 .9rem; }}
  table.hx-matrix th, table.hx-matrix td {{ text-align: left;
      padding: .38rem .55rem; border-bottom: 1px solid var(--border);
      white-space: nowrap; }}
  table.hx-matrix th {{ color: var(--ink-3); font-weight: 600; font-size: .7rem;
      text-transform: uppercase; letter-spacing: .04em; }}
  table.hx-matrix code {{ font-size: .8rem; color: var(--ink); }}
  /* A status word, never a bare colour: the label carries the meaning. */
  .hx-mark {{ font-size: .72rem; font-weight: 600; padding: .08rem .4rem;
      border-radius: 4px; border: 1px solid var(--border); color: var(--ink-3); }}
  .hx-mark.yes {{ color: var(--good); border-color: var(--good); }}
  .hx-mark.no  {{ color: var(--ink-3); }}
  .hx-mark.none {{ color: var(--ink-3); border-style: dashed; }}

  /* --- key/value rows ----------------------------------------------------- */
  .hx-kv {{ display: grid; grid-template-columns: minmax(150px, 210px) 1fr;
      gap: .8rem; padding: .42rem 0; border-bottom: 1px solid var(--border);
      font-size: .82rem; align-items: baseline; }}
  .hx-kv .k {{ font-weight: 600; color: var(--ink); }}
  .hx-kv .v {{ color: var(--ink-2); line-height: 1.45; }}

  /* --- command list ------------------------------------------------------- */
  .hx-cmd {{ display: grid; grid-template-columns: minmax(190px, 280px) 1fr;
      gap: .8rem; padding: .34rem 0; border-bottom: 1px solid var(--border);
      font-size: .8rem; align-items: baseline; }}
  .hx-cmd code {{ color: var(--accent); font-size: .78rem;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .hx-cmd span {{ color: var(--ink-2); }}


  /* --- app shell ---------------------------------------------------------- */
  .stApp {{ background: var(--page); }}
  .block-container {{ padding-top: 1.6rem !important; max-width: 1320px; }}

  /* The sidebar is the nav rail: darker than the cards, hairline separator. */
  section[data-testid="stSidebar"] {{ background: var(--page);
      border-right: 1px solid var(--border); }}
  section[data-testid="stSidebar"] > div {{ padding-top: 1.1rem; }}

  /* Nav buttons. Streamlit buttons are used so selection survives a rerun
     without a page reload; these rules make them read as nav rows. */
  .hx-navitem + div button {{ width: 100%; justify-content: flex-start !important;
      text-align: left !important; border: 1px solid transparent !important;
      background: transparent !important; color: var(--ink-2) !important;
      font-weight: 500 !important; font-size: .86rem !important;
      padding: .42rem .6rem !important; border-radius: 8px !important;
      box-shadow: none !important; }}
  .hx-navitem + div button:hover {{ background: var(--card) !important;
      color: var(--ink) !important; }}
  .hx-navitem.active + div button {{ background: var(--accent) !important;
      color: #fff !important; border-color: var(--accent) !important;
      font-weight: 600 !important; }}
  .hx-navitem.disabled + div button {{ opacity: .4; }}
  .hx-navgroup {{ font-size: .66rem; text-transform: uppercase;
      letter-spacing: .09em; color: var(--ink-3); font-weight: 700;
      margin: 1rem .6rem .3rem; }}

  .hx-brand {{ display: flex; align-items: center; gap: .5rem;
      padding: 0 .5rem .1rem; }}
  .hx-brand .mark {{ width: 22px; height: 22px; border-radius: 6px; flex: none;
      background: linear-gradient(135deg, var(--accent), #7aa9f0); }}
  .hx-brand .name {{ font-size: 1rem; font-weight: 680; color: var(--ink);
      letter-spacing: -.01em; }}
  .hx-brandsub {{ font-size: .7rem; color: var(--ink-3);
      padding: 0 .5rem .7rem 2.2rem; }}

  .hx-profile {{ display: flex; align-items: center; gap: .55rem;
      border: 1px solid var(--border); border-radius: 10px; padding: .5rem .6rem;
      background: var(--card); margin: .5rem 0; }}
  .hx-profile .av {{ width: 28px; height: 28px; border-radius: 50%; flex: none;
      background: var(--accent); color: #fff; font-size: .72rem; font-weight: 700;
      display: flex; align-items: center; justify-content: center; }}
  .hx-profile .who {{ display: flex; flex-direction: column; min-width: 0; }}
  .hx-profile .n {{ font-size: .8rem; font-weight: 600; color: var(--ink);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .hx-profile .s {{ font-size: .68rem; color: var(--ink-3); }}

  /* --- page header -------------------------------------------------------- */
  .hx-crumbs {{ font-size: .78rem; color: var(--ink-3); margin-bottom: .15rem; }}
  .hx-crumbs .sep {{ opacity: .5; margin: 0 .12rem; }}
  .hx-title {{ font-size: 1.4rem !important; font-weight: 660 !important;
      color: var(--ink); margin: .05rem 0 .1rem !important;
      letter-spacing: -.015em; }}
  .hx-subtitle {{ font-size: .84rem; color: var(--ink-3); margin-bottom: 1rem; }}

  /* --- cards -------------------------------------------------------------- */
  .hx-card {{ background: var(--card); border: 1px solid var(--border);
      border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: .9rem; }}
  .hx-card.flush {{ padding: 0; overflow: hidden; }}
  .hx-cardhead {{ display: flex; justify-content: space-between;
      align-items: flex-start; gap: 1rem; margin-bottom: .7rem; }}
  .hx-cardhead .t {{ font-size: .92rem; font-weight: 620; color: var(--ink); }}
  .hx-cardhead .s {{ font-size: .76rem; color: var(--ink-3); margin-top: .1rem; }}
  .hx-cardhead .a {{ font-size: .76rem; color: var(--ink-3); flex: none; }}

  /* Charts and tables sit directly on the card, not on their own panel. */
  .hx-card [data-testid="stVegaLiteChart"],
  .hx-card [data-testid="stDataFrame"] {{ background: transparent; }}

  /* --- KPI row ------------------------------------------------------------ */
  .hx-kpis {{ display: grid; gap: .75rem; margin: .1rem 0 1rem;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
  .hx-kpi {{ background: var(--card); border: 1px solid var(--border);
      border-radius: 14px; padding: .9rem 1rem; }}
  .hx-kpi .v {{ font-size: 1.65rem; font-weight: 680; color: var(--ink);
      line-height: 1.15; letter-spacing: -.02em;
      display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap; }}
  .hx-kpi .l {{ font-size: .78rem; color: var(--ink-3); margin-top: .3rem; }}
  .hx-kpi .n {{ font-size: .7rem; color: var(--ink-3); margin-top: .15rem;
      opacity: .8; }}
  /* Direction is carried by the arrow glyph as well as the colour, so a reader
     who cannot separate the hues still sees which way it moved. */
  .hx-delta {{ font-size: .72rem; font-weight: 650; padding: .1rem .4rem;
      border-radius: 6px; letter-spacing: 0; }}
  .hx-delta.up {{ color: var(--good); background: color-mix(in srgb, var(--good) 14%, transparent); }}
  .hx-delta.down {{ color: var(--critical); background: color-mix(in srgb, var(--critical) 14%, transparent); }}

  /* Provider readiness list in the rail. */
  .hx-provlist {{ margin: .35rem 0 .6rem; }}
  .hx-provrow {{ display: flex; justify-content: space-between;
      font-size: .74rem; padding: .1rem 0; color: var(--ink-3); }}
  .hx-provrow .ok {{ color: var(--good); font-weight: 600; }}
  .hx-provrow .off {{ color: var(--ink-3); }}

  /* --- misc --------------------------------------------------------------- */
  .hx-toolbar {{ display: flex; justify-content: space-between;
      align-items: center; gap: 1rem; margin: .1rem 0 .6rem;
      font-size: .8rem; color: var(--ink-3); flex-wrap: wrap; }}
  .hx-pill {{ display: inline-block; padding: .14rem .5rem; border-radius: 99px;
      font-size: .72rem; font-weight: 600; border: 1px solid var(--border);
      color: var(--ink-2); background: var(--card); }}
  .hx-pill.good {{ color: var(--good); border-color: color-mix(in srgb, var(--good) 45%, transparent); }}
  .hx-pill.warn {{ color: var(--warning); border-color: color-mix(in srgb, var(--warning) 45%, transparent); }}
  .hx-pill.bad {{ color: var(--critical); border-color: color-mix(in srgb, var(--critical) 45%, transparent); }}

  .hx-empty {{ border: 1px dashed var(--border); border-radius: 14px;
      padding: 2rem 1.2rem; text-align: center; background: var(--card); }}
  .hx-empty .t {{ font-size: .95rem; font-weight: 620; color: var(--ink); }}
  .hx-empty .b {{ font-size: .82rem; color: var(--ink-2); margin-top: .35rem;
      max-width: 46ch; margin-inline: auto; line-height: 1.5; }}
  .hx-empty .h {{ font-size: .76rem; color: var(--ink-3); margin-top: .5rem; }}

  /* Streamlit widgets, nudged onto the dark surface. Deliberately minimal:
     restyling framework internals wholesale breaks on the next release. */
  .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid var(--border); }}
  [data-testid="stMetricValue"] {{ font-size: 1.5rem; color: var(--ink); }}
  [data-testid="stDataFrame"] {{ border-radius: 10px; }}
  div[data-testid="stExpander"] {{ border: 1px solid var(--border);
      border-radius: 12px; background: var(--card); }}

  @media (max-width: 640px) {{
    .hx-kv, .hx-cmd {{ grid-template-columns: 1fr; gap: .15rem; }}
  }}
</style>
"""
