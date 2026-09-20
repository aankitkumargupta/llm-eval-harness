/* ===========================================================================
   Charts: hand-drawn SVG, no library.

   No CDN is reachable from this page by design, so every mark here is drawn
   from the data. That constraint turns out to be a feature: a charting library
   would happily render the two things this harness exists to prevent (a bare
   leaderboard with no interval, and a dual axis) and none of these forms is
   hard enough to justify the dependency.

   Reporting forms, each chosen for the job rather than for variety:

     accuracyCI      magnitude WITH uncertainty. The one chart that matters, and
                     the reason a plain bar chart would be wrong here: the CI
                     whiskers are the point, not decoration. Overlapping
                     intervals are how you see that a ranking is noise.
     qualityCost     a trade-off, so a scatter. Emphasis rather than categorical:
                     the Pareto frontier is the subject and everything else is
                     context, so dominated models go de-emphasis gray.
     scoredSplit     part-to-whole. Makes I7 visible: how much of each model's
                     denominator was EXCLUDED rather than answered wrong.
     latencyRange    p50 -> p95 per model, which is a before/after shape, so a
                     dumbbell. Two shades of one hue, never two hues.
     failureSplit    part-to-whole again, but by kind: how much of each model's
                     denominator left as extraction failure, truncation,
                     refusal or error. Stacked, one bar per model.
     pairwiseMatrix  every model against every other, from the corrected
                     paired test. A word in every cell, so "separable" is never
                     carried by colour alone.
     costPerCorrect  the procurement number, cost per correct answer, as bars.
                     Lower is better, and the API computed it, not this file.
     deltaDumbbell   baseline to candidate per metric, for gate comparisons.
                     Each row on its own scale, with the tolerance floor drawn.
     volumeCost      projected daily spend at a stated query volume, with a
                     word beside each bar saying whether the model clears the
                     requirements at all.
     winMatrix       arena win rates as a grid. One hue, stepped by opacity,
                     with the number in every cell.
     familyBars      counts per probe family or stratum, n beside each bar.
     spendByRun      metered spend per run, as bars.

   Explanatory and status forms follow (sections 5 and 6). They teach, or
   report on the harness itself, and are labelled as such on the page.

   Palette: gold leads, because gold is the product's accent and the primary
   magnitude is the thing the eye should land on. The steps are validated
   against both surfaces (lightness band, chroma floor, adjacent-pair CVD
   separation, normal-vision floor and 3:1 contrast all pass in both modes,
   with no warnings). Every chart still ships visible value labels and a table
   view, because colour is never the only channel.

   Rules kept throughout: one axis, never two. Bars capped at 24px with a 4px
   rounded data-end. 2px surface gaps between touching marks. Dots carry a 2px
   surface ring. Gridlines are hairline and recessive. Text wears ink tokens,
   never the series colour. Every chart has a hover layer and a table view. A
   legend appears whenever two or more series share a chart. Colours are read
   only through var(--...) tokens; a colour literal here would be a second
   palette. Captions read in plain language, with the metric id in
   parentheses the first time it is named.
   =========================================================================== */
"use strict";

const NS = "http://www.w3.org/2000/svg";

/* Read the live theme tokens so charts follow the page's toggle rather than
   carrying a second, drifting copy of the palette. */
function tok() {
  // Every colour is a CSS variable, not a value read once. A chart drawn in
  // the dark theme used to keep dark-theme fills after a toggle to light:
  // near-white text on a white card. With var() the browser repaints the
  // SVG when :root changes. The variable values per theme are the series
  // steps validated against each surface (OKLab CVD ΔE, lightness band,
  // chroma floor, 3:1 contrast); all checks pass in both modes.
  return {
    ink: "var(--ink)", ink2: "var(--ink-2)", ink3: "var(--ink-3)",
    grid: "var(--hair)", surface: "var(--chart-surface)",
    s1: "var(--chart-s1)",         // gold: primary magnitude
    s1dim: "var(--chart-s1-dim)",  // its second shade, for dumbbells
    s2: "var(--chart-s2)",         // blue: the contrast category
    s3: "var(--chart-s3)",         // green: reserved third
    muted: "var(--chart-muted)",   // de-emphasis
  };
}

const svg = (w, h) => {
  const s = document.createElementNS(NS, "svg");
  s.setAttribute("viewBox", `0 0 ${w} ${h}`);
  s.setAttribute("width", "100%");
  s.setAttribute("role", "img");
  s.style.display = "block";
  s.style.overflow = "visible";
  return s;
};

function node(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined) continue;
    // A var() colour goes on the style object, where every engine resolves
    // custom properties; as a presentation attribute it is engine-dependent.
    if ((k === "fill" || k === "stroke") && String(v).startsWith("var(")) {
      n.style[k] = String(v);
    } else {
      n.setAttribute(k, String(v));
    }
  }
  return n;
}

/* `halo`, when given a surface token, paints a thin surface-coloured stroke
   under the glyphs so a number stays legible on a tinted cell. */
function text(x, y, str, { fill, size = 11, anchor = "start", weight = 400,
                           mono = false, halo = null } = {}) {
  const t = node("text", { x, y, "text-anchor": anchor, fill,
                           "font-size": size, "font-weight": weight });
  if (fill && String(fill).startsWith("var(")) t.style.fill = fill;
  t.style.fontFamily = mono ? "var(--mono)" : "var(--sans)";
  if (mono) t.style.fontVariantNumeric = "tabular-nums";
  if (halo) {
    t.style.paintOrder = "stroke";
    t.style.stroke = halo;
    t.style.strokeWidth = "3px";
    t.style.strokeLinejoin = "round";
  }
  t.textContent = str;
  return t;
}

/* A bar with a rounded data-end and a square baseline end. Drawn as a path
   rather than a rect with a uniform radius, because rounding the baseline too
   detaches the mark from its axis. */
function barPath(x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, h / 2, w));
  return `M${x},${y} H${x + w - rr} Q${x + w},${y} ${x + w},${y + rr}` +
         ` V${y + h - rr} Q${x + w},${y + h} ${x + w - rr},${y + h} H${x} Z`;
}

/* ---------- shared chrome ---------- */
function figure(title, note, svgEl, tableFn) {
  const fig = document.createElement("figure");
  fig.style.margin = "0";
  if (title) {
    const cap = document.createElement("figcaption");
    cap.style.cssText = "margin-bottom:4px";  // face comes from app.css
    cap.textContent = title;
    fig.append(cap);
  }
  if (note) {
    const p = document.createElement("p");
    p.className = "why";
    p.style.marginBottom = "14px";
    p.textContent = note;
    fig.append(p);
  }
  fig.append(svgEl);

  // The table view is not optional decoration: it is the documented relief for
  // the sub-3:1 series step, and the accessible path for anyone the colour
  // channel does not reach.
  if (tableFn) {
    const d = document.createElement("details");
    d.style.marginTop = "12px";
    const s = document.createElement("summary");
    s.style.cssText = "cursor:pointer;color:var(--ink-3);font-size:12.5px";
    s.textContent = "Table view";
    d.append(s, tableFn());
    fig.append(d);
  }
  return fig;
}

function legend(items) {
  const l = document.createElement("div");
  l.style.cssText = "display:flex;gap:16px;flex-wrap:wrap;margin:10px 0 0;" +
                    "font-size:12px;color:var(--ink-2)";
  for (const [label, colour] of items) {
    const i = document.createElement("span");
    i.style.cssText = "display:inline-flex;align-items:center;gap:7px";
    const sw = document.createElement("span");
    sw.style.cssText = `width:10px;height:10px;border-radius:3px;background:${colour};` +
                       "flex:0 0 auto";
    i.append(sw, document.createTextNode(label));
    l.append(i);
  }
  return l;
}

/* A legend for one sequential hue: the same swatch at each opacity step,
   labelled, so a heatmap reader can map a tint back to a number. */
function ramp(colour, stops) {
  const l = document.createElement("div");
  l.style.cssText = "display:flex;gap:12px;flex-wrap:wrap;margin:10px 0 0;" +
                    "font-size:12px;color:var(--ink-2);align-items:center";
  const lead = document.createElement("span");
  lead.textContent = "win rate";
  lead.style.color = "var(--ink-3)";
  l.append(lead);
  for (const [label, opacity] of stops) {
    const i = document.createElement("span");
    i.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    const sw = document.createElement("span");
    sw.style.cssText = `width:14px;height:10px;border-radius:3px;background:${colour};` +
                       `opacity:${opacity};flex:0 0 auto`;
    i.append(sw, document.createTextNode(label));
    l.append(i);
  }
  return l;
}

/* One tooltip element, reused. Positioned from the page, so it is never
   clipped by the SVG's own bounds. */
let TIP = null;
function tip() {
  if (!TIP) {
    TIP = document.createElement("div");
    TIP.className = "tip";   // lets the stylesheet give it the pane treatment
    TIP.style.cssText =
      "position:fixed;z-index:60;pointer-events:none;opacity:0;transition:opacity .1s;" +
      "background:var(--surface-2);border:1px solid var(--hair-2);border-radius:8px;" +
      "padding:8px 11px;font:12px/1.5 var(--sans);color:var(--ink);" +
      "box-shadow:var(--shadow);max-width:280px";
    document.body.append(TIP);
  }
  return TIP;
}
function hoverable(el, html) {
  el.style.cursor = "default";
  el.addEventListener("mouseenter", e => {
    const t = tip();
    t.innerHTML = html;
    t.style.opacity = "1";
    move(e);
  });
  el.addEventListener("mousemove", move);
  el.addEventListener("mouseleave", () => { tip().style.opacity = "0"; });
  function move(e) {
    const t = tip();
    const r = t.getBoundingClientRect();
    t.style.left = Math.min(e.clientX + 14, window.innerWidth - r.width - 12) + "px";
    t.style.top = Math.max(8, e.clientY - r.height - 12) + "px";
  }
}

function simpleTable(cols, rows) {
  const w = document.createElement("div");
  w.className = "tw";
  w.style.marginTop = "10px";
  const t = document.createElement("table");
  const th = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const c of cols) {
    const h = document.createElement("th");
    h.textContent = c;
    hr.append(h);
  }
  th.append(hr);
  const tb = document.createElement("tbody");
  for (const r of rows) {
    const tr = document.createElement("tr");
    for (const [i, v] of r.entries()) {
      const td = document.createElement("td");
      td.className = i === 0 ? "m" : "n";
      td.textContent = v === null || v === undefined ? "n/a" : String(v);
      tr.append(td);
    }
    tb.append(tr);
  }
  t.append(th, tb);
  w.append(t);
  return w;
}

const shorten = (s, n = 30) => {
  s = String(s);
  if (s.length <= n) return s;
  // An org/model id is identified by its tail, so cut from the front.
  if (s.includes("/")) return "…" + s.slice(s.length - (n - 1));
  return s.slice(0, n - 1) + "…";
};

/* A metric id, readable: underscores to spaces, first letter up. The id
   itself still appears in parentheses in every caption, once. */
const pretty = id => {
  const s = String(id).replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
};

/* Number formatting at the display boundary only. Nothing here rounds a
   value that is then reused; tooltips and tables format the raw field. */
function fmt(v) {
  const n = Number(v);
  if (v === null || v === undefined || !Number.isFinite(n)) return "n/a";
  const a = Math.abs(n);
  if (a >= 100) return n.toFixed(0);
  if (a >= 1) return n.toFixed(2);
  if (a >= 0.01 || a === 0) return n.toFixed(3);
  return n.toFixed(5);
}
const usd = (v, d = 5) =>
  Number.isFinite(Number(v)) ? "$" + Number(v).toFixed(d) : "n/a";
const pct = v =>
  Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + "%" : "n/a";
const withCommas = n => String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

/* A round-number axis top: the first of 1, 2, 4, 5, 10 (times a power of
   ten) that clears the largest value, so the quarter gridlines land on
   numbers a reader can say aloud. */
function niceTop(v) {
  if (!(v > 0)) return 1;
  const e = Math.pow(10, Math.floor(Math.log10(v)));
  for (const m of [1, 2, 4, 5, 10, 20]) if (m * e >= v * 1.02) return m * e;
  return 20 * e;
}

/* Horizontal bars, one per item, on a single shared axis from zero. Every
   simple "how much, per model" form below is this helper plus its own
   caption and table, so the bar cap, the surface gap, the tick face and the
   label column stay identical across the page.

   items: [{label, sub, value, colour, valueText, tail, tip}]
     sub        optional second line under the label (e.g. a run's subject)
     valueText  the label at the right; defaults to fmt(value)
     tail       optional status mark {glyph, word, colour} drawn after the
                value; the word is what carries the meaning, not the colour
     tip        tooltip HTML for the row */
function hBars(items, { ariaLabel, tick = fmt, w = 860, padR = 100,
                        tailX = null } = {}) {
  const T = tok();
  const padL = 236, padT = 14, rowH = 32;
  const h = padT + items.length * rowH + 30;
  const plotW = w - padL - padR;
  const finite = items.map(it => Number(it.value)).filter(Number.isFinite);
  const top = niceTop(Math.max(0, ...finite));
  const X = v => padL + (Math.max(0, Math.min(top, v)) / top) * plotW;

  const s = svg(w, h);
  s.setAttribute("aria-label", ariaLabel);

  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    s.append(node("line", { x1: X(v), y1: padT, x2: X(v),
                            y2: padT + items.length * rowH,
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(X(v), padT + items.length * rowH + 16, tick(v),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }

  items.forEach((it, i) => {
    const y = padT + i * rowH, cy = y + rowH / 2;
    const barH = Math.min(24, rowH - 10);

    if (it.sub) {
      s.append(text(padL - 12, cy, shorten(it.label, 26),
                    { fill: T.ink2, size: 11.5, anchor: "end", mono: true }));
      s.append(text(padL - 12, cy + 11, shorten(it.sub, 36),
                    { fill: T.ink3, size: 9, anchor: "end", mono: true }));
    } else {
      s.append(text(padL - 12, cy + 4, shorten(it.label),
                    { fill: T.ink2, size: 12, anchor: "end", mono: true }));
    }

    const v = Number(it.value);
    if (Number.isFinite(v)) {
      s.append(node("path", {
        d: barPath(padL, cy - barH / 2, Math.max(1, X(v) - padL), barH, 4),
        fill: it.colour }));
      s.append(text(w - padR + 8, cy + 4, it.valueText ?? fmt(v),
                    { fill: T.ink, size: 11.5, mono: true, weight: 600 }));
    } else {
      s.append(text(padL + 4, cy + 4, "n/a",
                    { fill: T.ink3, size: 11, mono: true }));
    }

    if (it.tail) {
      // Marker AND word: colour is never the only channel.
      const tx = tailX ?? (w - padR + 96);
      s.append(node("circle", { cx: tx, cy, r: 5, fill: it.tail.colour }));
      s.append(text(tx, cy + 3.5, it.tail.glyph,
                    { fill: T.surface, size: 8, anchor: "middle", weight: 700 }));
      s.append(text(tx + 12, cy + 4, it.tail.word,
                    { fill: it.tail.colour, size: 11, mono: true }));
    }

    const hit = node("rect", { x: padL, y, width: plotW, height: rowH,
                               fill: "transparent" });
    hoverable(hit, it.tip || `<b>${it.label}</b>`);
    s.append(hit);
  });
  return s;
}

/* ===========================================================================
   1. Accuracy with 95% CI
   =========================================================================== */
export function accuracyCI(rows, { metric = "accuracy", chance = 0 } = {}) {
  const T = tok();
  const data = rows.filter(r => r.mean !== null && r.mean !== undefined);
  if (!data.length) return null;

  const padL = 236, padR = 56, padT = 14, rowH = 34;
  const h = padT + data.length * rowH + 34;
  const w = 860;
  const plotW = w - padL - padR;
  const x = v => padL + Math.max(0, Math.min(1, v)) * plotW;

  const s = svg(w, h);
  s.setAttribute("aria-label",
    `${pretty(metric)} per model with 95% confidence intervals`);

  // Gridlines: hairline, solid, recessive, and behind everything.
  for (let g = 0; g <= 1.0001; g += 0.25) {
    s.append(node("line", { x1: x(g), y1: padT, x2: x(g), y2: padT + data.length * rowH,
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(x(g), padT + data.length * rowH + 18, g.toFixed(2),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }

  // Chance level, when the benchmark has one: the honest baseline a bar chart
  // of raw accuracy would let a reader forget.
  if (chance > 0) {
    s.append(node("line", { x1: x(chance), y1: padT - 4,
                            x2: x(chance), y2: padT + data.length * rowH,
                            stroke: T.s2, "stroke-width": 2,
                            "stroke-dasharray": "3 3" }));
    s.append(text(x(chance), padT - 6, "chance",
                  { fill: T.s2, size: 10, anchor: "middle", mono: true }));
  }

  data.forEach((r, i) => {
    const y = padT + i * rowH;
    const barH = Math.min(24, rowH - 12);
    const cy = y + rowH / 2;

    s.append(text(padL - 12, cy + 4, shorten(r.model), {
      fill: T.ink2, size: 12, anchor: "end", mono: true }));

    const bar = node("path", {
      d: barPath(padL, cy - barH / 2, Math.max(1, x(r.mean) - padL), barH, 4),
      fill: T.s1 });
    s.append(bar);

    // The interval, drawn ON the bar. A ranking whose whiskers overlap is a
    // ranking you cannot act on, and that has to be visible at a glance.
    const lo = r.ci_low, hi = r.ci_high;
    if (lo !== null && hi !== null && lo !== undefined && hi !== undefined) {
      const g = node("g", {});
      g.append(node("line", { x1: x(lo), y1: cy, x2: x(hi), y2: cy,
                              stroke: T.ink, "stroke-width": 2,
                              "stroke-linecap": "round", opacity: .85 }));
      for (const v of [lo, hi]) {
        g.append(node("line", { x1: x(v), y1: cy - 6, x2: x(v), y2: cy + 6,
                                stroke: T.ink, "stroke-width": 2,
                                "stroke-linecap": "round", opacity: .85 }));
      }
      s.append(g);
    }

    s.append(text(w - padR + 8, cy + 4, r.mean.toFixed(3),
                  { fill: T.ink, size: 12, mono: true, weight: 600 }));

    const hit = node("rect", { x: padL, y, width: plotW, height: rowH,
                               fill: "transparent" });
    hoverable(hit, `<b>${r.model}</b><br>${metric} <b>${r.mean.toFixed(3)}</b>` +
      (lo !== null && lo !== undefined
        ? `<br>95% CI [${lo.toFixed(3)}, ${hi.toFixed(3)}]` : "") +
      (r.n ? `<br>n = ${r.n}` : ""));
    s.append(hit);
  });

  const fig = figure(
    `${pretty(metric)} (${metric}) with a 95% confidence interval`,
    "The bar is the best estimate; the whiskers show the range the true " +
    "value is likely to fall in (the 95% confidence interval). Two " +
    "overlapping intervals do not by themselves prove two models are tied, " +
    "only the paired test can say that, but a ranking whose whiskers all " +
    "overlap is a ranking to distrust.",
    s,
    () => simpleTable(["model", metric, "ci low", "ci high", "n"],
      data.map(r => [r.model, r.mean?.toFixed(4), r.ci_low?.toFixed(4),
                     r.ci_high?.toFixed(4), r.n])));
  return fig;
}

/* ===========================================================================
   2. Quality vs cost, Pareto frontier emphasised
   =========================================================================== */
export function qualityCost(rows) {
  const T = tok();
  const data = rows.filter(r => r.accuracy !== null && r.cost !== null &&
                                r.accuracy !== undefined && r.cost !== undefined);
  if (data.length < 2) return null;

  const w = 720, h = 342, padL = 62, padR = 24, padT = 30, padB = 52;
  const xs = data.map(d => d.cost), ys = data.map(d => d.accuracy);
  const xMax = Math.max(...xs) * 1.15 || 1, xMin = 0;
  const yMin = Math.max(0, Math.min(...ys) - 0.08);
  const yMax = Math.min(1, Math.max(...ys) + 0.08);

  const X = v => padL + ((v - xMin) / (xMax - xMin || 1)) * (w - padL - padR);
  const Y = v => h - padB - ((v - yMin) / (yMax - yMin || 1)) * (h - padT - padB);

  const s = svg(w, h);
  s.setAttribute("aria-label", "accuracy against cost per query, Pareto frontier emphasised");

  for (let i = 0; i <= 4; i++) {
    const v = yMin + (yMax - yMin) * (i / 4);
    s.append(node("line", { x1: padL, y1: Y(v), x2: w - padR, y2: Y(v),
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(padL - 10, Y(v) + 4, v.toFixed(2),
                  { fill: T.ink3, size: 10, anchor: "end", mono: true }));
  }
  for (let i = 0; i <= 4; i++) {
    const v = xMin + (xMax - xMin) * (i / 4);
    s.append(text(X(v), h - padB + 18, v.toFixed(5),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }
  s.append(text(padL, h - 8, "cost per query (USD) →", { fill: T.ink3, size: 11 }));
  s.append(text(padL, padT - 12, "↑ accuracy", { fill: T.ink3, size: 11 }));

  // Emphasis, not categorical: the frontier is the subject, the rest is
  // context. Dominated models go gray rather than getting a hue of their own.
  for (const d of data) {
    const on = !!d.on_frontier;
    const g = node("g", {});
    // 2px surface ring so overlapping points stay legible.
    g.append(node("circle", { cx: X(d.cost), cy: Y(d.accuracy), r: 8,
                              fill: T.surface }));
    g.append(node("circle", { cx: X(d.cost), cy: Y(d.accuracy), r: 6,
                              fill: on ? T.s1 : T.muted }));
    hoverable(g, `<b>${d.model}</b><br>accuracy <b>${d.accuracy.toFixed(3)}</b>` +
      `<br>cost $${d.cost.toFixed(6)}/query` +
      `<br>${on ? "on the Pareto frontier"
                : "dominated: some other model is both better and cheaper"}`);
    s.append(g);

    if (on) {
      s.append(text(X(d.cost) + 12, Y(d.accuracy) + 4, shorten(d.model, 26),
                    { fill: T.ink2, size: 11, mono: true }));
    }
  }

  const fig = figure(
    "Quality against cost (accuracy versus cost per query)",
    "Each dot is a model. A model that no other model beats on both quality " +
    "and price sits on the frontier (the Pareto frontier) and is labelled; " +
    "the rest are dominated, meaning something else is both better and " +
    "cheaper. When the top models tie on quality, this is where the decision " +
    "actually lives.",
    s,
    () => simpleTable(["model", "accuracy", "cost/query", "frontier"],
      data.map(d => [d.model, d.accuracy?.toFixed(4), d.cost?.toFixed(6),
                     d.on_frontier ? "yes" : "no"])));
  fig.append(legend([["on the frontier", T.s1], ["dominated", T.muted]]));
  return fig;
}

/* ===========================================================================
   3. Scored vs excluded: I7, made visible
   =========================================================================== */
export function scoredSplit(rows) {
  const T = tok();
  const data = rows.filter(r => r.n_items);
  if (!data.length) return null;
  if (!data.some(r => r.n_items - r.n_scored > 0)) return null;  // nothing to say

  const padL = 236, padR = 70, padT = 10, rowH = 30;
  const w = 860, h = padT + data.length * rowH + 10;
  const plotW = w - padL - padR;
  const s = svg(w, h);
  s.setAttribute("aria-label", "items scored versus excluded, per model");

  data.forEach((r, i) => {
    const y = padT + i * rowH;
    const barH = Math.min(24, rowH - 10);
    const cy = y + rowH / 2;
    const frac = r.n_scored / r.n_items;
    const scoredW = Math.max(0, frac * plotW);
    // 2px surface gap between the two segments: the separator is negative
    // space, never a stroke.
    const excW = Math.max(0, plotW - scoredW - 2);

    s.append(text(padL - 12, cy + 4, shorten(r.model), {
      fill: T.ink2, size: 12, anchor: "end", mono: true }));

    const a = node("path", { d: barPath(padL, cy - barH / 2, scoredW, barH, 0),
                             fill: T.s1 });
    hoverable(a, `<b>${r.model}</b><br><b>${r.n_scored}</b> of ${r.n_items} scored`);
    s.append(a);

    if (excW > 0) {
      const b = node("path", {
        d: barPath(padL + scoredW + 2, cy - barH / 2, excW, barH, 4),
        fill: T.s2 });
      const gap = r.n_items - r.n_scored;
      hoverable(b, `<b>${r.model}</b><br><b>${gap}</b> item(s) excluded` +
        "<br>extraction failure or error: left out of the accuracy " +
        "denominator, NOT counted wrong");
      s.append(b);
    }

    s.append(text(w - padR + 8, cy + 4, `${r.n_scored}/${r.n_items}`,
                  { fill: T.ink, size: 11.5, mono: true }));
  });

  const fig = figure(
    "Scored versus excluded (n_scored of n_items)",
    "An item whose answer could not be read out of the response scores " +
    "nothing, not zero: it is left out of both the top and the bottom of the " +
    "accuracy fraction. A wide excluded band means the accuracy beside it " +
    "was computed over far fewer items than the run attempted.",
    s,
    () => simpleTable(["model", "attempted", "scored", "excluded"],
      data.map(r => [r.model, r.n_items, r.n_scored, r.n_items - r.n_scored])));
  fig.append(legend([["scored", T.s1], ["excluded (not wrong)", T.s2]]));
  return fig;
}

/* ===========================================================================
   4. Latency p50 -> p95 (dumbbell)
   =========================================================================== */
export function latencyRange(rows) {
  const T = tok();
  const data = rows.filter(r => r.p50 !== null && r.p50 !== undefined);
  if (!data.length) return null;

  const padL = 236, padR = 74, padT = 14, rowH = 32;
  const w = 860, h = padT + data.length * rowH + 30;
  const plotW = w - padL - padR;
  const max = Math.max(...data.map(r => r.p95 ?? r.p50)) * 1.1 || 1;
  const X = v => padL + (v / max) * plotW;

  const s = svg(w, h);
  s.setAttribute("aria-label", "response time, median to 95th percentile, per model");

  for (let i = 0; i <= 4; i++) {
    const v = (max / 4) * i;
    s.append(node("line", { x1: X(v), y1: padT, x2: X(v),
                            y2: padT + data.length * rowH,
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(X(v), padT + data.length * rowH + 16, Math.round(v),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }

  data.forEach((r, i) => {
    const cy = padT + i * rowH + rowH / 2;
    s.append(text(padL - 12, cy + 4, shorten(r.model),
                  { fill: T.ink2, size: 12, anchor: "end", mono: true }));

    const p95 = r.p95 ?? r.p50;
    // One hue, two shades: never two hues for a before/after.
    s.append(node("line", { x1: X(r.p50), y1: cy, x2: X(p95), y2: cy,
                            stroke: T.s1dim, "stroke-width": 4,
                            "stroke-linecap": "round" }));
    for (const [v, colour] of [[r.p50, T.s1dim], [p95, T.s1]]) {
      s.append(node("circle", { cx: X(v), cy, r: 7, fill: T.surface }));
      s.append(node("circle", { cx: X(v), cy, r: 5, fill: colour }));
    }

    s.append(text(w - padR + 8, cy + 4, `${Math.round(p95)}ms`,
                  { fill: T.ink, size: 11.5, mono: true }));

    const hit = node("rect", { x: padL, y: cy - rowH / 2, width: plotW,
                               height: rowH, fill: "transparent" });
    hoverable(hit, `<b>${r.model}</b><br>p50 <b>${Math.round(r.p50)}ms</b>` +
      `<br>p95 <b>${Math.round(p95)}ms</b>`);
    s.append(hit);
  });

  const fig = figure(
    "Response time, typical to slow (latency p50 to p95)",
    "The left dot is the median response time (p50): half of requests were " +
    "faster than this. The right dot is the time 95 in 100 requests beat " +
    "(p95). The spread is the finding: a model with a quick median and a " +
    "long tail is a model whose slow moments your users will feel, and an " +
    "average would hide it completely.",
    s,
    () => simpleTable(["model", "p50 ms", "p95 ms"],
      data.map(r => [r.model, Math.round(r.p50), Math.round(r.p95 ?? r.p50)])));
  fig.append(legend([["p50 (median)", T.s1dim], ["p95", T.s1]]));
  return fig;
}

/* ===========================================================================
   5. Explanatory forms, for the Guide.

   These teach rather than report. They still obey the same rules (one axis,
   ink-token text, a table view wherever there is data to table) but their
   input is a fixed illustration, not a run, and each is labelled as such on
   the page. A teaching chart that looks like a result is how a mock-up ends
   up quoted in a decision.
   =========================================================================== */

/* The end-to-end path, as boxes. Not a chart but a diagram; it belongs in
   this file because it reads the same theme tokens and would otherwise carry
   a second, drifting copy of the palette. */
export function pipeline() {
  const T = tok();
  const STAGES = [
    ["Dataset", "HF / local\nsha256 pinned", T.s2],
    ["Prompt", "template + shots\nspec_hash", T.s2],
    ["Provider", "the only\nnetwork boundary", T.s1],
    ["Extract", "declared chain\nOk | Failed", T.s1],
    ["Score", "pure over\n(item, response)", T.s1],
    ["TraceRow", "append-only\none row per item", T.s3],
    ["Analyse", "paired test\ncorrected", T.s3],
    ["Decision", "cheapest model\nclearing the bar", T.s3],
  ];
  const w = 980, boxW = 104, gap = 20, h = 132;
  const y = 30, boxH = 60;

  const s = svg(w, h);
  s.setAttribute("aria-label", "The evaluation pipeline, eight stages");

  STAGES.forEach(([name, sub, colour], i) => {
    const x = i * (boxW + gap);

    s.append(node("rect", {
      x, y, width: boxW, height: boxH, rx: 9,
      fill: T.surface, stroke: colour, "stroke-width": 1.5 }));
    s.append(node("rect", {
      x, y, width: 3, height: boxH, rx: 1.5, fill: colour }));

    s.append(text(x + boxW / 2, y + 23, name,
                  { fill: T.ink, size: 12.5, anchor: "middle", weight: 600 }));
    sub.split("\n").forEach((line, j) =>
      s.append(text(x + boxW / 2, y + 38 + j * 12, line,
                    { fill: T.ink3, size: 9.5, anchor: "middle", mono: true })));

    if (i < STAGES.length - 1) {
      const ax = x + boxW, mid = y + boxH / 2;
      s.append(node("line", { x1: ax + 4, y1: mid, x2: ax + gap - 7, y2: mid,
                              stroke: T.grid, "stroke-width": 1.5 }));
      s.append(node("path", {
        d: `M${ax + gap - 7},${mid} l-5,-3.5 v7 z`, fill: T.grid }));
    }
  });

  // The one boundary worth drawing: nothing to the right of TraceRow may open
  // a connection, which is what makes analysis reproducible from the store.
  const bx = 5 * (boxW + gap) - gap / 2;
  s.append(node("line", { x1: bx, y1: 8, x2: bx, y2: h - 26,
                          stroke: T.s3, "stroke-width": 1.5,
                          "stroke-dasharray": "4 4" }));
  s.append(text(bx - 8, h - 12, "runs · may touch the network",
                { fill: T.ink3, size: 10, anchor: "end", mono: true }));
  s.append(text(bx + 8, h - 12, "analysis · reads rows only",
                { fill: T.s3, size: 10, mono: true }));

  return figure("What happens to one item, end to end",
    "Every stage except the provider is a plain function of its input: no " +
    "network, no clock. Once a row is written it is never changed, so any " +
    "number in a report traces back to the item that produced it.", s);
}

/* The denominator trap, drawn from the real live run. Two models, the items
   each actually answered, and the overlap on which they can be compared. */
export function pairingDemo(rows, { n = 60 } = {}) {
  const T = tok();
  const w = 880, padL = 236, padR = 120, rowH = 44, padT = 16;
  const h = padT + rows.length * rowH + 58;
  const plotW = w - padL - padR;
  const x = v => padL + (v / n) * plotW;

  const s = svg(w, h);
  s.setAttribute("aria-label", "Items answered versus items excluded per model");

  for (let g = 0; g <= n; g += 15) {
    s.append(node("line", { x1: x(g), y1: padT, x2: x(g),
                            y2: padT + rows.length * rowH,
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(x(g), padT + rows.length * rowH + 18, String(g),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }

  rows.forEach((r, i) => {
    const y = padT + i * rowH, barH = 22, cy = y + rowH / 2;
    s.append(text(padL - 12, cy + 4, shorten(r.model), {
      fill: T.ink2, size: 12, anchor: "end", mono: true }));

    // Scored, then excluded, with a 2px surface gap between the two fills.
    const scoredW = Math.max(1, x(r.scored) - padL);
    s.append(node("path", { d: barPath(padL, cy - barH / 2, scoredW, barH, 4),
                            fill: T.s1 }));
    const exW = Math.max(0, x(n) - x(r.scored) - 2);
    if (exW > 0) {
      s.append(node("path", {
        d: barPath(padL + scoredW + 2, cy - barH / 2, exW, barH, 4),
        fill: T.muted }));
      s.append(text(padL + scoredW + 2 + exW / 2, cy + 4, String(n - r.scored),
                    { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
    }
    s.append(text(padL + 8, cy + 4, `${r.scored} scored`,
                  { fill: T.surface, size: 10.5, mono: true, weight: 600 }));
    s.append(text(w - padR + 8, cy + 4, r.accuracy.toFixed(3),
                  { fill: T.ink, size: 12, mono: true, weight: 600 }));

    const hit = node("rect", { x: padL, y, width: plotW, height: rowH,
                               fill: "transparent" });
    hoverable(hit, `<b>${r.model}</b><br>accuracy <b>${r.accuracy.toFixed(3)}</b>` +
      `<br>scored ${r.scored} of ${n}<br>excluded ${n - r.scored} (truncated)`);
    s.append(hit);
  });

  const fig = figure("The denominator trap: MMLU-Pro, live run, 60 items",
    "The model with the highest accuracy answered the fewest items. " +
    "Truncation (running out of output room) is not random: a model runs " +
    "out on the items it thinks longest about, so its hardest items leave " +
    "the count its accuracy is divided by, and its score rises. On the 44 " +
    "items the top and bottom models both answered, they disagreed on zero.",
    s,
    () => simpleTable(["model", "accuracy", "scored", "excluded"],
      rows.map(r => [r.model, r.accuracy.toFixed(3), r.scored, n - r.scored])));
  fig.append(legend([["scored (in the accuracy denominator)", T.s1],
                     ["excluded: truncated, not wrong (I7)", T.muted]]));
  return fig;
}

/* Two comparisons, one overlapping and one separated. The point is that an
   ordering is not a finding until the interval says so. */
export function significanceDemo() {
  const T = tok();
  const CASES = [
    ["Observed gap 0.016", 0.950, 0.906, 0.981, 0.966, 0.925, 0.994,
     "more items needed", false],
    ["Observed gap 0.140", 0.780, 0.735, 0.822, 0.920, 0.884, 0.951,
     "holds after correction", true],
  ];
  const w = 760, padL = 40, padR = 250, padT = 30, blockH = 80;
  const h = padT + CASES.length * blockH + 20;
  const plotW = w - padL - padR;
  const lo = 0.65, hi = 1.0;
  const x = v => padL + ((v - lo) / (hi - lo)) * plotW;

  const s = svg(w, h);
  s.setAttribute("aria-label", "Two comparisons, one separable and one not");

  for (let g = 0.7; g <= 1.0001; g += 0.1) {
    s.append(node("line", { x1: x(g), y1: padT - 12, x2: x(g),
                            y2: padT + CASES.length * blockH - 20,
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(x(g), padT - 18, g.toFixed(1),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }

  CASES.forEach(([label, aM, aL, aH, bM, bL, bH, verdict, sep], i) => {
    const top = padT + i * blockH;
    s.append(text(padL, top + 2, label,
                  { fill: T.ink2, size: 11.5, mono: true }));

    [[aM, aL, aH, T.s1, "A"], [bM, bL, bH, T.s2, "B"]].forEach(
      ([m, l, hgh, colour, name], j) => {
        const cy = top + 28 + j * 24;
        s.append(node("line", { x1: x(l), y1: cy, x2: x(hgh), y2: cy,
                                stroke: colour, "stroke-width": 2,
                                "stroke-linecap": "round" }));
        for (const v of [l, hgh])
          s.append(node("line", { x1: x(v), y1: cy - 5, x2: x(v), y2: cy + 5,
                                  stroke: colour, "stroke-width": 2,
                                  "stroke-linecap": "round" }));
        s.append(node("circle", { cx: x(m), cy, r: 4.5, fill: colour,
                                  stroke: T.surface, "stroke-width": 2 }));
        s.append(text(x(m), cy - 11, `${name} ${m.toFixed(3)}`,
                      { fill: T.ink2, size: 10, anchor: "middle", mono: true }));
      });

    s.append(text(w - padR + 16, top + 34,
                  sep ? "separable" : "not separable",
                  { fill: sep ? T.s3 : T.ink3, size: 12.5, weight: 600 }));
    s.append(text(w - padR + 16, top + 51, verdict,
                  { fill: T.ink3, size: 10.5, mono: true }));
  });

  const fig = figure("When a gap is a finding, and when it is noise",
    "Both rows show model B ahead of model A. Only the second is a result. " +
    "The harness refuses to rank the first, and states how many more items " +
    "would be needed to settle it, rather than printing the ordering anyway.",
    s);
  fig.append(legend([["model A", T.s1], ["model B", T.s2]]));
  return fig;
}

/* Where the money actually goes. A run's spend is not one number, and the
   buckets nobody budgets for are the judge and the embedder. */
export function costBuckets(rows) {
  const T = tok();
  const total = rows.reduce((a, r) => a + r.usd, 0);
  if (!total) return null;

  const w = 760, h = 118, barY = 28, barH = 34;
  const s = svg(w, h);
  s.setAttribute("aria-label", "Run spend by bucket");

  const COLOURS = [T.s1, T.s2, T.s3, T.s1dim, T.muted];
  let cx = 0;
  rows.forEach((r, i) => {
    const segW = (r.usd / total) * w - 2;
    if (segW <= 0) return;
    s.append(node("rect", { x: cx, y: barY, width: segW, height: barH, rx: 4,
                            fill: COLOURS[i % COLOURS.length] }));
    const pctOf = (r.usd / total) * 100;
    if (segW > 46)
      s.append(text(cx + segW / 2, barY + barH / 2 + 4, `${pctOf.toFixed(0)}%`,
                    { fill: T.surface, size: 11, anchor: "middle",
                      mono: true, weight: 600 }));
    const hit = node("rect", { x: cx, y: barY - 8, width: segW + 2,
                               height: barH + 16, fill: "transparent" });
    hoverable(hit, `<b>${r.bucket}</b><br>$${r.usd.toFixed(5)} · ` +
                   `${pctOf.toFixed(1)}% of run`);
    s.append(hit);
    cx += segW + 2;
  });

  s.append(text(0, 16, "one run's spend, by bucket",
                { fill: T.ink3, size: 10.5, mono: true }));
  s.append(text(w, 16, `$${total.toFixed(4)} total`,
                { fill: T.ink, size: 11.5, anchor: "end", mono: true,
                  weight: 600 }));

  const fig = figure("Every paid call lands in a bucket (spend by bucket)",
    "Generation is the bucket everyone budgets for. Judge, embedding and " +
    "rerank are the ones that arrive on the invoice unexplained, so they are " +
    "metered from the provider's own usage report, never estimated, and a " +
    "model with no price is a hard failure rather than a free one.", s,
    () => simpleTable(["bucket", "usd", "share"],
      rows.map(r => [r.bucket, r.usd.toFixed(5),
                     ((r.usd / total) * 100).toFixed(1) + "%"])));
  fig.append(legend(rows.map((r, i) => [r.bucket, COLOURS[i % COLOURS.length]])));
  return fig;
}

/* The layer stack, as a stack. Imports point downward only, and the diagram
   should make an upward arrow look as wrong as it is. */
export function layers() {
  const T = tok();
  const L = [
    ["cli · web", "presentation: no business logic", T.muted],
    ["decide · report · arena · gate", "analysis over traces only", T.s3],
    ["store", "TraceRow in, tables out", T.s3],
    ["run · passes · bench", "orchestration", T.s2],
    ["scorers · judges", "pure over (item, response)", T.s2],
    ["providers", "the only network boundary", T.s1],
    ["config · contracts", "depended on by all, depends on nothing", T.muted],
  ];
  const w = 760, rowH = 42, padT = 10;
  const h = padT + L.length * rowH + 8;
  const s = svg(w, h);
  s.setAttribute("aria-label", "The seven layers, imports pointing downward");

  L.forEach(([name, note, colour], i) => {
    const y = padT + i * rowH;
    // Each layer inset a little further: the shape itself says "this depends
    // on what is below it".
    const inset = i * 10;
    s.append(node("rect", { x: inset, y, width: w - inset - 260,
                            height: rowH - 8, rx: 7,
                            fill: T.surface, stroke: colour,
                            "stroke-width": 1.25 }));
    s.append(node("rect", { x: inset, y, width: 3, height: rowH - 8, rx: 1.5,
                            fill: colour }));
    s.append(text(inset + 14, y + 22, name,
                  { fill: T.ink, size: 12, mono: true, weight: 600 }));
    s.append(text(w - 250, y + 22, note, { fill: T.ink3, size: 10.5 }));
  });

  return figure("Seven layers, imports pointing downward only",
    "An upward import is a build failure, not a review comment: it is " +
    "enforced by an import-linter contract. That is what keeps a provider " +
    "quirk out of the scoring code and the network out of analysis.", s);
}

/* ===========================================================================
   6. Status forms, for About.

   These report on the harness itself rather than on a model. Status is a
   reserved job: its palette never doubles as a series palette, and every
   marker ships with a word beside it, because "the green one" is not a
   channel everybody has.
   =========================================================================== */

const STATUS = {
  code: ["enforced in code", "✔"],
  convention: ["convention only", "~"],
  unenforced: ["unenforced", "✕"],
  live: ["verified live", "✔"],
  offline: ["tested offline", "~"],
  unexercised: ["never run", "✕"],
};

/* Status steps for CHARTS, which are not the same as the page's --ok/--warn
   /--crit text tokens and deliberately so.

   In light mode the design system's warn (#8a6314, a dark gold) and crit
   (#ad2f28, a dark red) sit at ΔE 2.5 under deuteranopia and 13.1 even for
   normal vision, below the 15 floor. Beside a paragraph that is fine,
   because each carries its own words; as two adjacent segments of one bar it
   is not, and the rule is explicit that a label does not excuse a
   normal-vision failure. So the middle step is re-stepped lighter and more
   orange for chart use only. Validated on both surfaces:
     dark  #4ec9a4 #e0b45c #ef7b74   CVD ΔE 8.7, normal 15.1, contrast pass
     light #127a5c #c07c10 #ad2f28   CVD ΔE 10.6, normal 18.1, contrast pass
   Every marker still ships a glyph and a word, so colour is never the only
   channel either way. */
function statusSteps() {
  // Same rule as tok(): variables, so the status colours follow the theme
  // live. Values per theme are in app.css (--chart-good/-mid/-bad).
  return { good: "var(--chart-good)", mid: "var(--chart-mid)", bad: "var(--chart-bad)" };
}

function statusColour(C, key) {
  if (key === "code" || key === "live") return C.good;
  if (key === "convention" || key === "offline") return C.mid;
  return C.bad;
}

/* One row per invariant: the marker, the word, and the claim. Sorted by the
   caller, not here; the order is editorial. */
export function invariantStatus(rows) {
  const T = tok();
  const C = statusSteps();

  const w = 760, rowH = 26, padT = 8;
  const h = padT + rows.length * rowH + 6;
  const s = svg(w, h);
  s.setAttribute("aria-label", "Enforcement status of each invariant");

  rows.forEach((r, i) => {
    const y = padT + i * rowH, cy = y + rowH / 2;
    const colour = statusColour(C, r.status);
    const [word, glyph] = STATUS[r.status] || ["unknown", "?"];

    if (i % 2 === 0)
      s.append(node("rect", { x: 0, y, width: w, height: rowH,
                              fill: T.grid, opacity: .25 }));

    s.append(text(6, cy + 4, r.id,
                  { fill: T.ink3, size: 10.5, mono: true }));
    s.append(text(42, cy + 4, r.name, { fill: T.ink, size: 12 }));

    // Marker AND word: colour is never the only channel.
    s.append(node("circle", { cx: 468, cy, r: 5, fill: colour }));
    s.append(text(468, cy + 3.5, glyph,
                  { fill: T.surface, size: 8, anchor: "middle", weight: 700 }));
    s.append(text(482, cy + 4, word, { fill: colour, size: 11, mono: true }));

    const hit = node("rect", { x: 0, y, width: w, height: rowH,
                               fill: "transparent" });
    hoverable(hit, `<b>${r.id}: ${r.name}</b><br>${word}<br>${r.note || ""}`);
    s.append(hit);
  });

  return figure("Which invariants the code actually enforces",
    "An invariant held up only by convention is one that breaks the first " +
    "time someone is in a hurry. This is the register, not the aspiration.",
    s,
    () => simpleTable(["invariant", "name", "status", "note"],
      rows.map(r => [r.id, r.name, (STATUS[r.status] || [])[0], r.note || ""])));
}

/* How much of each subsystem has actually been run, as opposed to read. */
export function verification(rows) {
  const T = tok();
  const S = statusSteps();
  const C = { live: S.good, offline: S.mid, unexercised: S.bad };

  const w = 860, padL = 236, padR = 40, rowH = 30, padT = 8;
  const h = padT + rows.length * rowH + 14;
  const plotW = w - padL - padR;
  const s = svg(w, h);
  s.setAttribute("aria-label", "Verification state per subsystem");

  rows.forEach((r, i) => {
    const y = padT + i * rowH, barH = 18, cy = y + rowH / 2;
    s.append(text(padL - 12, cy + 4, r.area,
                  { fill: T.ink2, size: 12, anchor: "end", mono: true }));

    const total = r.live + r.offline + r.unexercised || 1;
    let cx = padL;
    for (const key of ["live", "offline", "unexercised"]) {
      const v = r[key];
      if (!v) continue;
      const segW = (v / total) * plotW - 2;
      if (segW <= 0) continue;
      const seg = node("rect", { x: cx, y: cy - barH / 2, width: segW,
                                 height: barH, rx: 3, fill: C[key] });
      hoverable(seg, `<b>${r.area}</b><br>${v} · ${STATUS[key][0]}`);
      s.append(seg);
      if (segW > 24)
        s.append(text(cx + segW / 2, cy + 4, String(v),
                      { fill: T.surface, size: 10, anchor: "middle",
                        mono: true, weight: 600 }));
      cx += segW + 2;
    }
  });

  const fig = figure("Verified by running it, or only by reading it",
    "Counted in surfaces: an adapter, a provider, a command path. The " +
    "distinction matters because code that has only been read has never " +
    "failed, which is not the same as working.", s,
    () => simpleTable(["area", "verified live", "tested offline", "never run"],
      rows.map(r => [r.area, r.live, r.offline, r.unexercised])));
  fig.append(legend([["verified live", C.live],
                     ["tested offline", C.offline],
                     ["never run", C.unexercised]]));
  return fig;
}

/* ===========================================================================
   7. Metric profile: small multiples, one panel per metric.

   A RAG profile scores eight things at once (faithfulness, abstention,
   citations, injection resistance, retrieval hit rate…) and a single
   composite hides which of them a model is actually good at. The honest form
   is one small panel per metric with the same models in the same order, so
   the eye reads down a column and across a row. One axis per panel, never a
   radar: a radar's area depends on the arbitrary order of its axes.

   Input is the composite table (per-model means), the numbers the report
   already computed, not a second computation. Cost is drawn on its own
   panel with its own scale and marked lower-is-better; it is never folded
   into the same axis as a quality metric.
   =========================================================================== */
export function metricProfile(rows, metrics, { lowerIsBetter = ["cost_usd"] } = {}) {
  const T = tok();
  const models = rows.map(r => r.model);
  const present = metrics.filter(m => rows.some(r => r[m] !== null && r[m] !== undefined
                                                     && !Number.isNaN(r[m])));
  if (!models.length || !present.length) return null;

  const cols = Math.min(4, present.length);
  const panelW = 235, panelH = 26 * models.length + 46, gapX = 22, gapY = 26;
  const rowsN = Math.ceil(present.length / cols);
  const w = cols * panelW + (cols - 1) * gapX;
  const h = rowsN * panelH + (rowsN - 1) * gapY;
  const labelW = 92;
  const s = svg(w, h);
  s.setAttribute("aria-label", "One panel per metric, same models in each");

  present.forEach((metric, pi) => {
    const px = (pi % cols) * (panelW + gapX);
    const py = Math.floor(pi / cols) * (panelH + gapY);
    const lower = lowerIsBetter.includes(metric);
    const vals = rows.map(r => Number(r[metric]));
    const finite = vals.filter(Number.isFinite);
    // Quality metrics live on [0,1]; cost gets its own honest scale.
    const lo = 0;
    const hi = lower ? Math.max(...finite, 1e-9) : 1;
    const plotL = px + labelW, plotW = panelW - labelW - 8;
    const x = v => plotL + (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * plotW;
    const best = lower ? Math.min(...finite) : Math.max(...finite);

    s.append(text(px, py + 12, metric.replace(/_/g, " "),
                  { fill: T.ink, size: 11.5, weight: 600 }));
    if (lower) s.append(text(px + panelW - 8, py + 12, "lower is better",
                             { fill: T.ink3, size: 9.5, anchor: "end", mono: true }));

    for (const g of [0, 0.5, 1]) {
      const gx = plotL + g * plotW;
      s.append(node("line", { x1: gx, y1: py + 20, x2: gx, y2: py + 20 + models.length * 26,
                              stroke: T.grid, "stroke-width": 1 }));
    }

    rows.forEach((r, i) => {
      const cy = py + 20 + i * 26 + 13;
      const v = Number(r[metric]);
      s.append(text(plotL - 8, cy + 4, shorten(r.model, 13),
                    { fill: T.ink2, size: 10.5, anchor: "end", mono: true }));
      if (!Number.isFinite(v)) {
        s.append(text(plotL + 4, cy + 4, "n/a", { fill: T.ink3, size: 10, mono: true }));
        return;
      }
      const isBest = v === best && finite.length > 1;
      // A thin track, a bar to the value, and the best model's bar in gold.
      s.append(node("rect", { x: plotL, y: cy - 4, width: plotW, height: 8, rx: 4,
                              fill: T.grid, opacity: .5 }));
      s.append(node("path", { d: barPath(plotL, cy - 4, Math.max(1, x(v) - plotL), 8, 4),
                              fill: isBest ? T.s1 : T.s2 }));
      s.append(text(plotL + plotW + 6, cy + 3.5,
                    lower ? "$" + v.toFixed(4) : v.toFixed(2),
                    { fill: T.ink, size: 9.5, mono: true, weight: isBest ? 700 : 400 }));
      const hit = node("rect", { x: px, y: cy - 13, width: panelW, height: 26,
                                 fill: "transparent" });
      hoverable(hit, `<b>${r.model}</b><br>${metric} <b>${lower ? "$" + v.toFixed(5) : v.toFixed(3)}</b>` +
                     (isBest ? "<br>best on this metric" : ""));
      s.append(hit);
    });
  });

  const fig = figure("Metric profile: the composite score, unfolded (one panel per metric)",
    "One panel per active metric, the same models in the same order. The " +
    "gold bar is the best on that panel. A model that wins the composite by " +
    "being adequate everywhere and a model that wins it by dominating one " +
    "heavily-weighted metric look identical in a leaderboard and nothing " +
    "alike here. These are averages only: the interval for the headline " +
    "metric is on the chart above, and the paired test below decides what " +
    "is real.", s,
    () => simpleTable(["model", ...present],
      rows.map(r => [r.model, ...present.map(m => Number.isFinite(Number(r[m]))
        ? Number(r[m]).toFixed(4) : "n/a")])));
  fig.append(legend([["best on this metric", T.s1], ["others", T.s2]]));
  return fig;
}

/* ===========================================================================
   8. Comparison forms.

   Each of these arranges numbers the API already computed. None of them
   derives a metric: a cost per correct answer, a daily projection, an
   adjusted p-value or a win rate arrives as a field and is drawn as it is.
   The only arithmetic here is placement (where a mark goes on the axis) and
   the comparison a caption promises to show (candidate against the
   tolerance floor).
   =========================================================================== */

/* What left the denominator, by kind. The stacked length of each bar is the
   share of a model's attempted items that were never scored; each segment
   is one reason. A rate the API did not report is drawn as zero. */
export function failureSplit(rows) {
  const T = tok();
  const SERIES = [
    ["extraction_failure_rate", "extraction failure", T.s1],
    ["truncation_rate", "truncation", T.s2],
    ["refusal_rate", "refusal", T.s3],
    ["error_rate", "error", T.s1dim],
  ];
  const rate = (r, k) => {
    const v = Number(r[k]);
    return Number.isFinite(v) && v > 0 ? v : 0;
  };
  const data = (rows || []).filter(r => r && r.model).map(r => {
    const parts = SERIES.map(([k]) => rate(r, k));
    return { model: r.model, parts, total: parts.reduce((a, b) => a + b, 0) };
  });
  if (!data.length) return null;

  const padL = 236, padR = 130, padT = 14, rowH = 32;
  const w = 860, h = padT + data.length * rowH + 30;
  const plotW = w - padL - padR;
  const top = niceTop(Math.max(...data.map(d => d.total)));
  const X = v => padL + (v / top) * plotW;

  const s = svg(w, h);
  s.setAttribute("aria-label",
    "share of each model's items excluded from scoring, split by reason");

  for (let i = 0; i <= 4; i++) {
    const v = (top / 4) * i;
    s.append(node("line", { x1: X(v), y1: padT, x2: X(v),
                            y2: padT + data.length * rowH,
                            stroke: T.grid, "stroke-width": 1 }));
    s.append(text(X(v), padT + data.length * rowH + 16, pct(v),
                  { fill: T.ink3, size: 10, anchor: "middle", mono: true }));
  }

  data.forEach((d, i) => {
    const y = padT + i * rowH, cy = y + rowH / 2;
    const barH = Math.min(24, rowH - 10);
    s.append(text(padL - 12, cy + 4, shorten(d.model),
                  { fill: T.ink2, size: 12, anchor: "end", mono: true }));

    const live = SERIES.map((sr, k) => [sr, d.parts[k]]).filter(([, v]) => v > 0);
    let cx = padL;
    live.forEach(([[, name, colour], v], k) => {
      const last = k === live.length - 1;
      // 2px surface gap between touching segments; only the data-end rounds.
      const segW = Math.max(1, (v / top) * plotW - (last ? 0 : 2));
      const seg = node("path", {
        d: barPath(cx, cy - barH / 2, segW, barH, last ? 4 : 0), fill: colour });
      hoverable(seg, `<b>${d.model}</b><br>${name} <b>${pct(v)}</b>` +
        `<br>${pct(d.total)} of attempted items left the denominator in all` +
        "<br>excluded, not counted as wrong (I7)");
      s.append(seg);
      cx += segW + 2;
    });
    if (!live.length) {
      const hit = node("rect", { x: padL, y, width: plotW, height: rowH,
                                 fill: "transparent" });
      hoverable(hit, `<b>${d.model}</b><br>nothing excluded: every attempted item was scored`);
      s.append(hit);
    }

    s.append(text(w - padR + 8, cy + 4, `${pct(d.total)} excluded`,
                  { fill: T.ink, size: 11.5, mono: true, weight: 600 }));
  });

  const fig = figure(
    "What left the denominator, by reason (extraction_failure_rate, " +
    "truncation_rate, refusal_rate, error_rate)",
    "Each bar is the share of a model's attempted items that were never " +
    "scored, split by why: the answer could not be read out of the response " +
    "(extraction failure), the response was cut off (truncation), the model " +
    "declined to answer (refusal), or the call itself failed (error). None " +
    "of these counts as a wrong answer; they are reported beside accuracy, " +
    "never inside it. A rate the API did not report is drawn as zero.",
    s,
    () => simpleTable(
      ["model", "extraction failure", "truncation", "refusal", "error", "excluded in all"],
      data.map(d => [d.model, ...d.parts.map(pct), pct(d.total)])));
  fig.append(legend(SERIES.map(([, name, colour]) => [name, colour])));
  return fig;
}

/* Every model against every other, from the corrected paired test. Rows
   minus columns; a word in every cell, because "the green ones" is not a
   channel everybody has. The mirrored half is filled from the same row with
   the sign flipped, so the grid reads the same from either side. */
export function pairwiseMatrix(sig, models) {
  const T = tok();
  const C = statusSteps();
  const rows = (sig || []).filter(r => r && r.model_a && r.model_b);
  const names = models && models.length
    ? [...models]
    : [...new Set(rows.flatMap(r => [r.model_a, r.model_b]))];
  if (names.length < 2) return null;

  const key = (a, b) => JSON.stringify([a, b]);
  const cell = new Map();
  for (const r of rows) {
    const d = Number(r.diff);
    cell.set(key(r.model_a, r.model_b), { ...r, diff: d });
    if (!cell.has(key(r.model_b, r.model_a))) {
      cell.set(key(r.model_b, r.model_a),
               { ...r, model_a: r.model_b, model_b: r.model_a,
                 diff: Number.isFinite(d) ? -d : d });
    }
  }

  const labelW = 170, cw = 112, ch = 46, headH = 34, padT = 18;
  const w = labelW + names.length * cw;
  const h = padT + headH + names.length * ch + 4;
  const s = svg(w, h);
  s.setAttribute("aria-label",
    "paired difference between every pair of models, marked separable or not");

  s.append(text(0, padT + headH - 10, "row minus column",
                { fill: T.ink3, size: 9.5, mono: true }));
  names.forEach((b, j) => {
    s.append(text(labelW + j * cw + cw / 2, padT + headH - 10, shorten(b, 15),
                  { fill: T.ink2, size: 10.5, anchor: "middle", mono: true }));
  });

  const signed = d => Number.isFinite(d) ? (d >= 0 ? "+" : "") + d.toFixed(3) : "n/a";
  const pAdj = p => Number.isFinite(Number(p)) ? Number(p).toFixed(4) : "n/a";

  names.forEach((a, i) => {
    const y = padT + headH + i * ch;
    s.append(text(labelW - 10, y + ch / 2 + 4, shorten(a, 22),
                  { fill: T.ink2, size: 11, anchor: "end", mono: true }));
    names.forEach((b, j) => {
      if (i === j) return;   // the diagonal is blank: a model against itself
      const x = labelW + j * cw;
      const r = cell.get(key(a, b));
      const g = node("g", {});
      if (!r) {
        g.append(node("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2,
                                rx: 6, fill: T.grid, opacity: .2 }));
        g.append(text(x + cw / 2, y + ch / 2 + 4, "n/a",
                      { fill: T.ink3, size: 10.5, anchor: "middle", mono: true }));
        hoverable(g, `<b>${a}</b> against <b>${b}</b><br>no paired test recorded`);
        s.append(g);
        return;
      }
      const sep = !!r.significant;
      const colour = sep ? C.good : T.muted;
      g.append(node("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2,
                              rx: 6, fill: colour, opacity: sep ? .22 : .4 }));
      if (sep)
        g.append(node("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2,
                                rx: 6, fill: "none", stroke: colour,
                                "stroke-width": 1.5 }));
      g.append(text(x + cw / 2, y + ch / 2 - 1, signed(r.diff),
                    { fill: T.ink, size: 11.5, anchor: "middle", mono: true,
                      weight: 600 }));
      g.append(text(x + cw / 2, y + ch / 2 + 13, sep ? "separable" : "not separable",
                    { fill: T.ink3, size: 9, anchor: "middle", mono: true }));
      hoverable(g, `<b>${a}</b> minus <b>${b}</b>` +
        `<br>difference <b>${signed(r.diff)}</b>` +
        `<br>adjusted p ${pAdj(r.p_adjusted)}` +
        `<br>${sep ? "separable after correction" : "not separable at this n"}` +
        (r.n_pairs ? `<br>n = ${r.n_pairs} paired items` : ""));
      s.append(g);
    });
  });

  const fig = figure(
    "Every model against every other (paired difference, adjusted p-value)",
    "Each cell is the row model's score minus the column model's, on the " +
    "same items (a paired comparison). A cell is marked separable only when " +
    "the difference survives the correction for testing many pairs at once. " +
    "Every other cell is a gap that could be noise at this number of items, " +
    "and the harness will not rank on it.",
    s,
    () => simpleTable(["model a", "model b", "difference", "adjusted p", "n pairs", "verdict"],
      rows.map(r => [r.model_a, r.model_b, signed(Number(r.diff)), pAdj(r.p_adjusted),
                     r.n_pairs, r.significant ? "separable" : "not separable"])));
  fig.append(legend([["separable after correction", C.good],
                     ["not separable", T.muted]]));
  return fig;
}

/* The procurement number: what one correct answer costs. Computed by the
   report from metered spend and the count of correct items; drawn here as it
   arrives. A model with no correct answers has no bar. */
export function costPerCorrect(rows) {
  const T = tok();
  const data = (rows || []).filter(r => r && r.model);
  if (!data.length) return null;
  const finite = data.map(r => Number(r.cost_per_correct_answer)).filter(Number.isFinite);
  const best = finite.length ? Math.min(...finite) : null;

  const items = data.map(r => {
    const v = Number(r.cost_per_correct_answer);
    const ok = Number.isFinite(v);
    const isBest = ok && v === best && finite.length > 1;
    return {
      label: r.model, value: ok ? v : null,
      colour: isBest ? T.s1 : T.s2, valueText: usd(v, 5),
      tip: `<b>${r.model}</b><br>cost per correct answer <b>${usd(v, 5)}</b>` +
           (ok ? "" : "<br>not computable: no correct answers, or no metered cost") +
           (isBest ? "<br>lowest of the models shown" : ""),
    };
  });

  const s = hBars(items, {
    ariaLabel: "cost per correct answer per model, lower is better",
    tick: v => usd(v, 5), padR: 96 });

  const fig = figure(
    "What a correct answer costs (cost_per_correct_answer), lower is better",
    "Metered spend divided by the number of items the model got right, as " +
    "computed by the report; this chart only draws it. It is the number a " +
    "buying decision actually turns on, because a cheap model that is often " +
    "wrong is not cheap. A model with no correct answers has no bar.",
    s,
    () => simpleTable(["model", "cost per correct answer (USD)"],
      data.map(r => [r.model, usd(r.cost_per_correct_answer, 5)])));
  fig.append(legend([["lowest cost per correct answer", T.s1], ["others", T.s2]]));
  return fig;
}

/* Baseline to candidate, one metric per row, for gate comparisons. Metrics
   do not share a unit, so each row has its own scale and a track of its own;
   the reader compares within a row, never across rows. The dashed tick is
   the floor the gate allows (baseline less tolerance, or plus tolerance when
   lower is better). The gate's own verdict wins when the row carries one
   (`regressed`); otherwise the row is marked from that floor. */
export function deltaDumbbell(rows, { baselineLabel = "baseline",
                                      candidateLabel = "candidate" } = {}) {
  const T = tok();
  const C = statusSteps();
  const data = (rows || []).filter(r => r && r.metric &&
    Number.isFinite(Number(r.baseline)) && Number.isFinite(Number(r.candidate)));
  if (!data.length) return null;

  const w = 900, padL = 200, padR = 310, padT = 14, rowH = 36;
  const h = padT + data.length * rowH + 12;
  const plotW = w - padL - padR;
  const s = svg(w, h);
  s.setAttribute("aria-label",
    `${candidateLabel} against ${baselineLabel}, one metric per row, each on its own scale`);

  const verdicts = [];
  data.forEach((r, i) => {
    const y = padT + i * rowH, cy = y + rowH / 2;
    const b = Number(r.baseline), c = Number(r.candidate);
    const tol = Number.isFinite(Number(r.tolerance)) ? Math.abs(Number(r.tolerance)) : 0;
    const lower = !!r.lower_is_better;
    const floor = lower ? b + tol : b - tol;
    const worse = r.regressed !== undefined && r.regressed !== null
      ? !!r.regressed
      : (lower ? c > floor : c < floor);
    verdicts.push(worse);

    const lo = Math.min(0, b, c, floor);
    const hi = Math.max(b, c, floor, lo + 1e-9);
    const X = v => padL + ((v - lo) / (hi - lo)) * (plotW - 16);

    s.append(text(padL - 12, cy + 4, shorten(r.metric, 24),
                  { fill: T.ink2, size: 12, anchor: "end", mono: true }));

    // A thin track for the row's own scale, then the floor, then the pair.
    s.append(node("rect", { x: padL, y: cy - 3, width: plotW, height: 6, rx: 3,
                            fill: T.grid, opacity: .5 }));
    s.append(node("line", { x1: X(floor), y1: cy - 11, x2: X(floor), y2: cy + 11,
                            stroke: T.ink3, "stroke-width": 1.5,
                            "stroke-dasharray": "2 2" }));
    // One hue, two shades: the pale dot is the baseline, the full one the
    // candidate. Status is carried by the word at the right, never the dot.
    s.append(node("line", { x1: X(b), y1: cy, x2: X(c), y2: cy,
                            stroke: T.s1dim, "stroke-width": 4,
                            "stroke-linecap": "round" }));
    for (const [v, colour] of [[b, T.s1dim], [c, T.s1]]) {
      s.append(node("circle", { cx: X(v), cy, r: 7, fill: T.surface }));
      s.append(node("circle", { cx: X(v), cy, r: 5, fill: colour }));
    }

    s.append(text(w - padR + 10, cy + 4, `${fmt(b)} → ${fmt(c)}`,
                  { fill: T.ink, size: 11.5, mono: true, weight: 600 }));
    const mx = w - padR + 170;
    const [glyph, word, colour] = worse
      ? ["✕", "regressed", C.bad] : ["✔", "within tolerance", C.good];
    s.append(node("circle", { cx: mx, cy, r: 5, fill: colour }));
    s.append(text(mx, cy + 3.5, glyph,
                  { fill: T.surface, size: 8, anchor: "middle", weight: 700 }));
    s.append(text(mx + 12, cy + 4, word, { fill: colour, size: 11, mono: true }));

    const hit = node("rect", { x: padL, y, width: w - padL, height: rowH,
                               fill: "transparent" });
    hoverable(hit, `<b>${r.metric}</b>` +
      `<br>${baselineLabel} <b>${fmt(b)}</b><br>${candidateLabel} <b>${fmt(c)}</b>` +
      `<br>tolerance ${fmt(tol)}${lower ? " (lower is better)" : ""}` +
      `<br>${worse ? "regressed: past the floor the gate allows" : "within tolerance"}`);
    s.append(hit);
  });

  const fig = figure(
    `${pretty(candidateLabel)} against ${baselineLabel}, one metric per row (gate comparison)`,
    "Each row is one metric drawn on its own scale, so read across a row and " +
    "never down a column. The paler dot is the baseline run and the darker " +
    "dot the candidate; the small dashed tick is the floor the candidate must " +
    "stay on the right side of, which is the baseline value less the " +
    "tolerance the gate allows (or plus it, for metrics where lower is " +
    "better). A row marked regressed is one the candidate failed.",
    s,
    () => simpleTable(["metric", baselineLabel, candidateLabel, "tolerance", "verdict"],
      data.map((r, i) => [r.metric, fmt(r.baseline), fmt(r.candidate),
                          fmt(r.tolerance), verdicts[i] ? "regressed" : "within tolerance"])));
  fig.append(legend([[baselineLabel, T.s1dim], [candidateLabel, T.s1]]));
  return fig;
}

/* Projected daily spend at a stated query volume, as bars, with a word beside
   each saying whether the model clears the requirements at all. The
   projection is the decide step's; this chart only draws it. */
export function volumeCost(rows, { qpd = null } = {}) {
  const T = tok();
  const C = statusSteps();
  const data = (rows || []).filter(r => r && r.model);
  if (!data.length) return null;

  const finite = data.map(r => Number(r.daily_usd)).filter(Number.isFinite);
  const dec = finite.length && Math.max(...finite) >= 1 ? 2 : 4;
  const items = data.map(r => {
    const v = Number(r.daily_usd), ok = Number.isFinite(v);
    const clears = !!r.clears;
    return {
      label: r.model, value: ok ? v : null,
      colour: clears ? T.s1 : T.muted,
      valueText: ok ? `${usd(v, dec)} / day` : "n/a",
      tail: clears
        ? { glyph: "✔", word: "clears the bar", colour: C.good }
        : { glyph: "✕", word: "does not clear", colour: C.bad },
      tip: `<b>${r.model}</b><br>projected <b>${usd(v, dec)}</b> per day` +
           (qpd ? ` at ${withCommas(qpd)} queries` : "") +
           `<br>${clears ? "clears every requirement" : "does not clear the requirements"}`,
    };
  });

  const w = 900, padR = 250;
  const s = hBars(items, {
    ariaLabel: "projected daily spend per model, marked by whether it clears the requirements",
    tick: v => usd(v, dec), w, padR, tailX: w - padR + 8 + 112 });

  const fig = figure(
    qpd ? `Projected daily spend at ${withCommas(qpd)} queries per day (daily_usd)`
        : "Projected daily spend (daily_usd)",
    "Each bar is what a day would cost at this volume, as the decide step " +
    "projected it from the model's metered cost per query. The mark beside " +
    "each bar says whether the model clears every requirement you set; a " +
    "model that does not clear them is drawn for scale only, however cheap " +
    "it looks.",
    s,
    () => simpleTable(["model", "usd per day", "clears the requirements"],
      data.map(r => [r.model, usd(r.daily_usd, dec), r.clears ? "yes" : "no"])));
  fig.append(legend([["clears the requirements", T.s1],
                     ["does not clear them", T.muted]]));
  return fig;
}

/* Who wins head to head: arena win rates as a grid. One hue, stepped by
   opacity, with the number in every cell so the tint is never the only
   channel. Rows are the model that won; columns the model it beat. */
export function winMatrix(matrix) {
  const T = tok();
  const names = (matrix && Array.isArray(matrix.models)) ? matrix.models : [];
  const rates = (matrix && matrix.rates) || {};
  if (names.length < 2) return null;

  const labelW = 170, cw = 84, ch = 40, headH = 34, padT = 18;
  const w = labelW + names.length * cw;
  const h = padT + headH + names.length * ch + 4;
  const s = svg(w, h);
  s.setAttribute("aria-label",
    "arena win rate of each model (rows) against each other model (columns)");

  // Five opacity steps on one hue. The top step stays a tint rather than a
  // solid so ink on it keeps its contrast; the halo does the rest.
  const step = r => 0.1 + 0.6 * (Math.round(Math.max(0, Math.min(1, r)) * 4) / 4);

  s.append(text(0, padT + headH - 10, "row beat column",
                { fill: T.ink3, size: 9.5, mono: true }));
  names.forEach((b, j) => {
    s.append(text(labelW + j * cw + cw / 2, padT + headH - 10, shorten(b, 11),
                  { fill: T.ink2, size: 10.5, anchor: "middle", mono: true }));
  });

  const at = (a, b) => Number((rates[a] || {})[b]);
  names.forEach((a, i) => {
    const y = padT + headH + i * ch;
    s.append(text(labelW - 10, y + ch / 2 + 4, shorten(a, 22),
                  { fill: T.ink2, size: 11, anchor: "end", mono: true }));
    names.forEach((b, j) => {
      if (i === j) return;   // blank diagonal
      const x = labelW + j * cw;
      const r = at(a, b);
      const g = node("g", {});
      if (!Number.isFinite(r)) {
        g.append(node("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2,
                                rx: 6, fill: T.grid, opacity: .2 }));
        g.append(text(x + cw / 2, y + ch / 2 + 4, "n/a",
                      { fill: T.ink3, size: 10.5, anchor: "middle", mono: true }));
        hoverable(g, `<b>${a}</b> against <b>${b}</b><br>no paired judgements recorded`);
      } else {
        g.append(node("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2,
                                rx: 6, fill: T.s1, opacity: step(r) }));
        g.append(text(x + cw / 2, y + ch / 2 + 4, r.toFixed(2),
                      { fill: T.ink, size: 11.5, anchor: "middle", mono: true,
                        weight: 600, halo: T.surface }));
        hoverable(g, `<b>${a}</b> beat <b>${b}</b> in <b>${pct(r)}</b> of paired judgements`);
      }
      s.append(g);
    });
  });

  const fig = figure(
    "Who wins head to head (arena win rate)",
    "Each cell is how often the row model was preferred over the column " +
    "model, across the paired judgements the arena recorded. A darker cell " +
    "means the row model won more often, and the number in every cell is the " +
    "rate itself. A rate near one half is a tie, which is what most cells " +
    "will show at a small number of judgements.",
    s,
    () => simpleTable(["model", ...names],
      names.map(a => [a, ...names.map(b => a === b ? "" :
        (Number.isFinite(at(a, b)) ? at(a, b).toFixed(2) : "n/a"))])));
  fig.append(ramp(T.s1, [["0%", step(0)], ["25%", step(0.25)], ["50%", step(0.5)],
                         ["75%", step(0.75)], ["100%", step(1)]]));
  return fig;
}

/* Counts (or rates) per probe family or per stratum, with n beside each bar.
   The caption may be overridden by the caller, because the same shape serves
   both a "how many probes of each kind" and a "score in each subject" job. */
export function familyBars(rows, { title = null, note = null } = {}) {
  const T = tok();
  const data = (rows || []).filter(r => r && (r.label !== null && r.label !== undefined));
  if (!data.length) return null;

  const items = data.map(r => {
    const v = Number(r.value), ok = Number.isFinite(v);
    const n = r.n === null || r.n === undefined ? null : r.n;
    return {
      label: String(r.label), value: ok ? v : null, colour: T.s1,
      valueText: ok ? fmt(v) + (n !== null ? `  (n = ${withCommas(n)})` : "") : "n/a",
      tip: `<b>${r.label}</b><br>value <b>${fmt(v)}</b>` +
           (n !== null ? `<br>n = ${withCommas(n)} items` : ""),
    };
  });

  const s = hBars(items, {
    ariaLabel: "one bar per family or stratum, with the number of items beside it",
    padR: 150 });

  const fig = figure(
    title || "Per family or stratum (value), with n",
    note ||
    "One bar per group. The n beside each bar is how many items that group " +
    "holds; a group with a small n carries a wide margin of error whatever " +
    "the bar says, so read the two together.",
    s,
    () => simpleTable(["group", "value", "n"],
      data.map(r => [r.label, fmt(r.value), r.n ?? "n/a"])));
  return fig;
}

/* Metered spend per run, as bars. What each run actually cost, read from the
   providers' own usage reports across every bucket, never estimated. */
export function spendByRun(rows) {
  const T = tok();
  const data = (rows || []).filter(r => r && r.run_id);
  if (!data.length) return null;

  const items = data.map(r => {
    const v = Number(r.cost_usd), ok = Number.isFinite(v);
    return {
      label: String(r.run_id), sub: r.subject ? String(r.subject) : null,
      value: ok ? v : null, colour: T.s1, valueText: usd(v, 5),
      tip: `<b>${r.run_id}</b>` + (r.subject ? `<br>${r.subject}` : "") +
           `<br>metered spend <b>${usd(v, 5)}</b>`,
    };
  });

  const s = hBars(items, {
    ariaLabel: "metered spend per run", tick: v => usd(v, 5), padR: 96 });

  const fig = figure(
    "Metered spend per run (cost_usd)",
    "Each bar is what a run actually cost, read from the providers' own " +
    "usage reports across every bucket (generation, judge, embedding, " +
    "rerank), never estimated. Runs that share a subject can be compared; " +
    "runs that do not are listed for the record.",
    s,
    () => simpleTable(["run", "subject", "usd"],
      data.map(r => [r.run_id, r.subject ?? "", usd(r.cost_usd, 5)])));
  return fig;
}
