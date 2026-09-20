# Design bundle

The harness UI as a component library, so `/design-sync` has something to push
to a claude.ai/design design-system project.

## Build

```
python design/build.py      # writes design/dist/
```

`design/dist/` is **generated**. Do not edit it, the next build wipes it.
Change [`build.py`](build.py) instead, or `harness/web/static/app.css` for
anything that affects how a component actually looks.

## Why it is generated

The shipped UI is one `app.css` and one `app.js`; components exist only as
class names. A design system wants the opposite shape: one self-contained
preview page per component, each with an `@dsCard` marker on line one.

Generating that shape from the real stylesheet, rather than hand-writing a
parallel copy, is what stops the design system from drifting into a
confident description of a product that no longer exists.

`tests/test_design_bundle.py` pins five properties:

| Test | Catches |
|---|---|
| mirrored files copied, not stubbed | a broken copy step |
| every class a preview uses still exists | **the drift that matters**, a renamed class leaves a preview rendering unstyled while still claiming to show the component |
| `@dsCard` on line one | a card that silently never appears in the pane |
| bundled CSS, nothing remote | a preview that breaks the offline promise |
| dist matches the builder's declared output | a hand edit that will vanish |

## Connecting to claude.ai/design

1. `claude` in a terminal, then `/design-login`, authorizes this machine.
   It cannot be run from the desktop app's Code tab.
2. Enable the `/design-sync` skill.
3. `/design-sync` from a session in this repo, pointing at `design/dist/`.

## Contents

13 previews in three groups: **Foundations** (colour, type), **Components**
(buttons, cards, KPIs, messages and tags, tables, navigation, forms, steps,
failure-mode cards), **Charts** (data, status).

Previews render in the dark theme, which is the default. `app.css` defines
both themes on `:root`, so one page cannot show both side by side, the
colour page lists both palettes as literal swatches instead.
