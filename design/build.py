"""
Build the design-system bundle from the shipped UI.

`/design-sync` uploads a component library: one self-contained preview page
per component, each carrying a `@dsCard` marker on its first line that becomes
a card in the Design System pane. This repo's UI is not shaped like that, it
is one `app.css` and one `app.js` where components exist only as class names,
so this script produces that shape without forking the styling.

**The copies are generated, never edited.** `app.css` and `charts.js` are
copied verbatim from `harness/web/static/` on every build, so a stale copy
cannot survive a build, that is prevention by construction, not by test.

What a test cannot prevent that way is the preview *markup* falling behind:
the HTML below is written by hand, so renaming a class in `app.css` leaves a
preview rendering unstyled while still claiming to show the component. That
is the drift that matters, and
`tests/test_design_bundle.py::test_every_class_a_preview_uses_still_exists_in_the_stylesheet`
is the guard for it. A design system that lies about the product is worse
than no design system.

Previews render in the dark theme, which is the default. `app.css` defines
both themes on `:root`, so a single page cannot show them side by side, the
colour page therefore lists both palettes as literal swatches instead.

    python design/build.py          # writes design/dist/
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "harness" / "web" / "static"
OUT = Path(__file__).resolve().parent / "dist"

#: Files copied verbatim from the shipped UI. Pinned by a test.
MIRRORED = ("app.css", "charts.js")


def page(title: str, group: str, subtitle: str, body: str, *,
         depth: int = 1, module: str = "") -> str:
    """One preview page.

    The `@dsCard` comment must be the FIRST line: the app's self-check reads
    it to build the pane's card index, and a marker on line two is a card that
    silently never appears.
    """
    up = "../" * depth
    script = f'\n<script type="module">\n{module}\n</script>' if module else ""
    return f"""<!-- @dsCard group="{group}" -->
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="{up}app.css">
<style>
  /* Preview chrome only, never component styling. Anything that changes how
     a component looks belongs in app.css, or this page stops being evidence. */
  body {{ padding: 30px; }}
  .ds-h {{ margin: 0 0 4px; font-family: var(--serif); font-size: 21px;
           letter-spacing: -.02em; }}
  .ds-sub {{ color: var(--ink-3); font-size: 12.5px; margin: 0 0 26px;
             font-family: var(--mono); }}
  .ds-set {{ margin-bottom: 30px; }}
  .ds-set:last-child {{ margin-bottom: 0; }}
  .ds-label {{ font: 9.5px/1 var(--mono); letter-spacing: .18em;
               text-transform: uppercase; color: var(--ink-3);
               margin-bottom: 12px; }}
</style>
</head>
<body>
<h1 class="ds-h">{title}</h1>
<p class="ds-sub">{subtitle}</p>
{body}{script}
</body>
</html>
"""


def group_set(label: str, html: str) -> str:
    return f'<div class="ds-set"><div class="ds-label">{label}</div>{html}</div>'


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
DARK = [
    ("--ground", "#091128", "page ground"),
    ("--surface", "rgba(255,255,255,.022)", "card fill"),
    ("--gold", "#c9a028", "accent"),
    ("--gold-ink", "#e3c873", "accent as text"),
    ("--ink", "#ecf0f9", "primary text"),
    ("--ink-2", "rgba(236,240,249,.74)", "secondary text"),
    ("--ink-3", "rgba(236,240,249,.48)", "muted text"),
    ("--ok", "#4ec9a4", "status: good"),
    ("--warn", "#e0b45c", "status: warning"),
    ("--crit", "#ef7b74", "status: critical"),
]
LIGHT = [
    ("--ground", "#fbf9f3", "page ground"),
    ("--surface", "#ffffff", "card fill"),
    ("--gold", "#8a6a05", "accent"),
    ("--gold-ink", "#6f5504", "accent as text"),
    ("--ink", "#0b1430", "primary text"),
    ("--ink-2", "rgba(11,20,48,.74)", "secondary text"),
    ("--ink-3", "rgba(11,20,48,.5)", "muted text"),
    ("--ok", "#127a5c", "status: good"),
    ("--warn", "#8a6314", "status: warning"),
    ("--crit", "#ad2f28", "status: critical"),
]


def swatches(rows: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div style="border:1px solid var(--hair);border-radius:10px;'
        f'overflow:hidden">'
        f'<div style="height:52px;background:{val}"></div>'
        f'<div style="padding:9px 11px">'
        f'<div class="m" style="font-size:11px">{name}</div>'
        f'<div class="m" style="font-size:10px;color:var(--ink-3)">{val}</div>'
        f'<div style="font-size:10.5px;color:var(--ink-3);margin-top:3px">{note}</div>'
        f"</div></div>"
        for name, val, note in rows)
    return ('<div style="display:grid;gap:12px;'
            f'grid-template-columns:repeat(auto-fill,minmax(150px,1fr))">{cells}</div>')


COLOUR = group_set("dark, default", swatches(DARK)) + group_set(
    "light, a selected design, not an inversion", swatches(LIGHT))

TYPE = """
<div class="ds-set">
  <div class="ds-label">display: Georgia, a system serif</div>
  <div style="font-family:var(--serif);font-size:42px;letter-spacing:-.03em">
    Cheapest model clearing the bar</div>
  <div style="font-family:var(--serif);font-size:21px;letter-spacing:-.02em;
              margin-top:10px">Section heading</div>
</div>
<div class="ds-set">
  <div class="ds-label">body, system sans</div>
  <p style="max-width:70ch;margin:0">Bars are the point estimate; the whiskers
    are the interval. A ranking whose whiskers all overlap is a ranking to
    distrust.</p>
  <p class="note">Muted note text, for the sentence under a number.</p>
</div>
<div class="ds-set">
  <div class="ds-label">mono, eyebrows, numerals, identifiers</div>
  <div class="m">run_id · deepseek-ai/DeepSeek-V4-Flash · 0.978 · $0.00019</div>
  <div style="margin-top:12px"><span class="eyebrow">Analyse</span></div>
</div>
<div class="ds-set">
  <div class="ds-label">why no web font</div>
  <p class="note" style="max-width:70ch">The harness runs offline by design.
    A downloaded face would look marginally better and fail completely on an
    air-gapped machine, so the serif/mono contrast that carries the identity
    is built from system faces and costs zero requests.</p>
</div>
"""

# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
BUTTONS = (
    group_set("variants", """
<div class="row">
  <button class="btn">Run benchmark</button>
  <button class="btn ghost">Cancel</button>
  <button class="btn sm">Small</button>
  <button class="btn ghost sm">Ghost small</button>
  <button class="btn" disabled>Disabled</button>
</div>""")
    + group_set("icon button, task bar utility", """
<div class="row">
  <button class="iconbtn" aria-label="Switch theme">&#9680;</button>
  <button class="iconbtn" aria-label="Menu">&#9776;</button>
</div>""")
    + group_set("pills, multi-select", """
<div class="pills">
  <button class="pick" aria-pressed="true">fake:alpha</button>
  <button class="pick" aria-pressed="false">openai/gpt-oss-120b</button>
  <button class="pick local" aria-pressed="false">zai-org/GLM-5.3-Flash</button>
</div>"""))

CARDS = (
    group_set("card, title, rationale, content", """
<div class="card">
  <h3>Preflight</h3>
  <p class="why">Validates a profile end to end before it costs anything: dataset
    present, every model priced, routing resolvable.</p>
  <p class="note">A missing price stops you here rather than surfacing as a
    zero in the report.</p>
</div>""")
    + group_set("two-column content grid", """
<div class="card"><div class="grid2">
  <div class="screencard"><b>Decide</b>
    <p class="note">Requirements in, a model out, with the cost at your volume.</p></div>
  <div class="screencard"><b>Gate</b>
    <p class="note">Baseline versus candidate, with an exit code for CI.</p></div>
</div></div>""")
    + group_set("empty state", """
<div class="empty"><b>No saved reports yet</b>Run something, then freeze it here.
  <p class="note">A report captures its numbers and stops moving.</p></div>"""))

KPIS = group_set("kpi row, tone carries meaning, never decoration", """
<div class="kpis">
  <div class="kpi"><div class="v">540</div><div class="k">api calls</div></div>
  <div class="kpi ok"><div class="v">0</div><div class="k">errors</div></div>
  <div class="kpi"><div class="v">$0.0777</div><div class="k">metered</div></div>
  <div class="kpi crit"><div class="v">23.3%</div><div class="k">truncation</div></div>
</div>""")

MESSAGES = (
    group_set("messages", """
<div class="msg info">Reading rows only, nothing here opens a connection.</div>
<div class="msg ok" style="margin-top:12px">Manifest written. Run is reproducible.</div>
<div class="msg warn" style="margin-top:12px">Read before quoting these numbers:
  the models were not scored on an identical item set.</div>
<div class="msg crit" style="margin-top:12px">UnpairedItemsError: cannot pair
  'model-a' against 'model-b' on 'accuracy'.</div>""")
    + group_set("tags", """
<div class="row">
  <span class="tag ok">enforced</span>
  <span class="tag warn">convention</span>
  <span class="tag crit">unenforced</span>
  <span class="tag accent">live</span>
  <span class="tag dim">offline</span>
</div>""")
    + group_set("refusal list, a stop, not a bullet", """
<ul class="plain refuse">
  <li>Rank models that were not scored on the same items.</li>
  <li>Declare a winner when the interval straddles zero.</li>
  <li>Report a cost for a model with no pricing entry.</li>
</ul>"""))

TABLES = group_set("table, numerals tabular, nulls explicit", """
<div class="card"><div class="tw"><table>
<thead><tr><th>model</th><th>n</th><th>accuracy</th><th>cost $</th><th>$/correct</th></tr></thead>
<tbody>
<tr><td class="m">zai-org/GLM-5.3-Flash</td><td class="n">58</td><td class="n">0.966</td>
    <td class="n">0.00593</td><td class="n">0.000106</td></tr>
<tr><td class="m">openai/gpt-oss-120b</td><td class="n">60</td><td class="n">0.950</td>
    <td class="n">0.00547</td><td class="n">0.000096</td></tr>
<tr><td class="m">deepseek-ai/DeepSeek-V4</td><td class="n">59</td><td class="n">0.949</td>
    <td class="n"><span class="null">n/a</span></td><td class="n">0.000045</td></tr>
</tbody></table></div></div>""")

NAV = (
    group_set("task bar, tier one, the stage of work", """
<div class="topbar" style="position:static">
  <div class="barrow">
    <span class="brand"><span class="mark"></span>
      <span><b>Eval Harness</b><span>model evaluation</span></span></span>
    <nav class="tabs">
      <button class="tab">Overview</button>
      <button class="tab">Prepare</button>
      <button class="tab" aria-current="true">Analyse</button>
      <button class="tab">Guide</button>
    </nav>
  </div>
  <div class="subbar">
    <button class="subtab" aria-current="true"><span class="ic">&#9638;</span>Profile report</button>
    <button class="subtab"><span class="ic">&#9636;</span>Benchmark results</button>
    <button class="subtab"><span class="ic">&#9670;</span>Decide</button>
  </div>
</div>""")
    + group_set("page head", """
<div class="head"><span class="eyebrow">Analyse</span>
  <h1>Benchmark results</h1>
  <p class="sub">The leaderboard, the failure rates beside it, and the paired
    test that decides whether the order means anything.</p></div>"""))

FORMS = group_set("fields", """
<div class="card">
  <div class="row">
    <div class="f" style="flex:2"><label>Title</label>
      <input type="text" value="Together AI - 3 models"></div>
    <div class="f"><label>Provider</label>
      <select><option>together</option><option>anthropic</option></select></div>
  </div>
  <div class="f" style="margin-top:14px"><label>Caveats, one per line</label>
    <textarea style="min-height:70px">No winner is established here.</textarea></div>
</div>""")

STEPS = group_set("numbered walkthrough", """
<div class="card">
  <div class="step"><div class="stepn">1</div><div class="stepbody">
    <b>Check the catalogue</b>
    <p>See which benchmarks are installed and what licence each carries.</p></div></div>
  <div class="step"><div class="stepn">2</div><div class="stepbody">
    <b>Preflight a profile</b>
    <p>Confirm every model has a price, then read the cost estimate.</p></div></div>
</div>""")

FAILCARDS = group_set("failure-mode card, problem in muted ink, answer behind a gold rule", """
<div class="card"><div class="grid2">
  <div class="failcard"><div class="failhead">Measuring noise</div>
    <p class="note">A three-point gap on 200 items is usually nothing.</p>
    <p class="fix">Every comparison is paired at the item level and corrected
      across the family.</p></div>
  <div class="failcard"><div class="failhead">Under-counted cost</div>
    <p class="note">The judge, embedder and reranker are all billed.</p>
    <p class="fix">Every paid call is attributed to a bucket from the
      provider's own usage block.</p></div>
</div></div>""")

# --------------------------------------------------------------------------- #
# Charts, rendered from the real module with fixture data.
# --------------------------------------------------------------------------- #
CHART_JS = """
import * as Charts from "../charts.js";
const mount = (id, fig) => { if (fig) document.getElementById(id).append(fig); };

mount("ci", Charts.accuracyCI([
  { model: "zai-org/GLM-5.3-Flash", mean: 0.966, ci_low: 0.925, ci_high: 0.994, n: 58 },
  { model: "openai/gpt-oss-120b", mean: 0.950, ci_low: 0.906, ci_high: 0.981, n: 60 },
  { model: "deepseek-ai/DeepSeek-V4", mean: 0.949, ci_low: 0.903, ci_high: 0.980, n: 59 },
], { metric: "accuracy", chance: 0.25 }));

mount("split", Charts.scoredSplit([
  { model: "model-a", n_items: 60, n_scored: 54 },
  { model: "model-b", n_items: 60, n_scored: 46 },
]));

mount("cost", Charts.costBuckets([
  { bucket: "generation", usd: 0.0412 }, { bucket: "judge", usd: 0.0197 },
  { bucket: "embedding", usd: 0.0068 }, { bucket: "rerank", usd: 0.0031 },
]));
"""

CHARTS = """
<div class="ds-set"><div class="ds-label">magnitude with uncertainty</div>
  <div class="card" id="ci"></div></div>
<div class="ds-set"><div class="ds-label">part-to-whole, failure is not wrongness</div>
  <div class="card" id="split"></div></div>
<div class="ds-set"><div class="ds-label">composition, where the money goes</div>
  <div class="card" id="cost"></div></div>
"""

STATUS_JS = """
import * as Charts from "../charts.js";
document.getElementById("inv").append(Charts.invariantStatus([
  { id: "I1", name: "Paired comparison", status: "code", note: "raises rather than intersecting" },
  { id: "I7", name: "Failure is not wrongness", status: "convention", note: "two rates documented, not computed" },
  { id: "I12", name: "Secrets never leak", status: "unenforced", note: "no redaction processor" },
]));
document.getElementById("ver").append(Charts.verification([
  { area: "bench adapters", live: 3, offline: 1, unexercised: 0 },
  { area: "\\u00a710.4 families", live: 2, offline: 1, unexercised: 9 },
]));
"""

STATUS = """
<div class="ds-set"><div class="ds-label">status, glyph and word beside every marker</div>
  <div class="card" id="inv"></div></div>
<div class="ds-set"><div class="ds-label">status, stacked, re-stepped for light mode</div>
  <div class="card" id="ver"></div></div>
<p class="note" style="max-width:74ch">Status is a reserved job: its palette
  never doubles as a series palette. The light-mode middle step is re-stepped
  to #c07c10 for charts, because the page's own warn and crit tokens sit below
  the normal-vision separation floor when placed adjacent in one bar.</p>
"""

PAGES = [
    ("tokens/colour.html", "Colour", "Foundations",
     "Both themes, defined at token level", COLOUR, ""),
    ("tokens/type.html", "Type", "Foundations",
     "Serif display, system sans body, mono for anything you would copy",
     TYPE, ""),
    ("components/buttons.html", "Buttons", "Components",
     "Primary, ghost, small, icon, pills", BUTTONS, ""),
    ("components/cards.html", "Cards", "Components",
     "Card, content grid, empty state", CARDS, ""),
    ("components/kpis.html", "KPIs", "Components",
     "Headline numbers with tone", KPIS, ""),
    ("components/messages.html", "Messages and tags", "Components",
     "Info, ok, warn, critical, tags, refusal list", MESSAGES, ""),
    ("components/tables.html", "Tables", "Components",
     "Tabular numerals, explicit nulls", TABLES, ""),
    ("components/navigation.html", "Navigation", "Components",
     "Two-tier task bar and page head", NAV, ""),
    ("components/forms.html", "Form fields", "Components",
     "Text, select, textarea", FORMS, ""),
    ("components/steps.html", "Steps", "Components",
     "Numbered walkthrough", STEPS, ""),
    ("components/failure-cards.html", "Failure-mode cards", "Components",
     "Problem and answer, visually separated", FAILCARDS, ""),
    ("charts/data.html", "Data charts", "Charts",
     "Hand-drawn SVG, no library, validated palette", CHARTS, CHART_JS),
    ("charts/status.html", "Status charts", "Charts",
     "Reserved palette, never colour alone", STATUS, STATUS_JS),
]


def main() -> None:
    # Clear the CONTENTS rather than the directory itself. Removing the root
    # fails on Windows whenever anything holds it open, a shell sitting in
    # it, an editor, a static server previewing the bundle, and a build that
    # dies because someone is looking at its output is a build people stop
    # running.
    OUT.mkdir(parents=True, exist_ok=True)
    for child in OUT.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    for name in MIRRORED:
        shutil.copyfile(STATIC / name, OUT / name)

    for path, title, group, subtitle, body, module in PAGES:
        dest = OUT / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        depth = len(Path(path).parts) - 1
        dest.write_text(page(title, group, subtitle, body,
                             depth=depth, module=module), encoding="utf-8")

    print(f"design bundle -> {OUT}")
    print(f"  {len(MIRRORED)} mirrored file(s), {len(PAGES)} preview page(s)")
    for path, title, group, *_ in PAGES:
        print(f"    [{group:<12}] {title:<22} {path}")


if __name__ == "__main__":
    main()
