/* ===========================================================================
   Eval Harness: the whole client.

   No framework. The page is a dozen screens over a JSON API, and a render
   function that replaces one container is the entire state management this
   needs. Reaching for a framework would add a build step and a CDN to a tool
   whose selling point is that it installs with no web dependency at all.

   The rule the whole file obeys: **no arithmetic on a metric.** Every number
   rendered here arrives computed from `harness.bench.*`, `harness.report.*`
   or the trace store. This file formats and lays out; it never decides what a
   number means. A metric computed here would be a second definition of
   something the library already owns, and the two would drift.
   =========================================================================== */
"use strict";

import * as Charts from "/charts.js";

const $ = (s, r = document) => r.querySelector(s);
const el = (t, a = {}, ...kids) => {
  const n = document.createElement(t);
  for (const [k, v] of Object.entries(a)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) {
    if (c === null || c === undefined || c === false) continue;
    n.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return n;
};
const fmt = (v, d = 3) =>
  (v === null || v === undefined || Number.isNaN(v)) ? null
    : (typeof v === "number" ? v.toFixed(d) : String(v));

/* --------------------------------------------------------------------------
   Plain-language glossary.

   Every metric, column and error name the API returns is a technical id
   chosen for the library, not for a reader. Each entry here maps one id to a
   short plain label and a one-sentence meaning. The label is what a header or
   a KPI shows; the id stays visible beside it in a small mono span so nothing
   loses traceability back to the column that produced it.

   This is vocabulary, not arithmetic: nothing here changes a number.
   -------------------------------------------------------------------------- */
const TERMS = {
  // Quality
  accuracy: ["Accuracy", "Share of answers the judge marked correct"],
  accuracy_chance_adjusted: ["Accuracy above guessing",
    "Accuracy rescaled so that guessing at random scores zero and a perfect run scores one"],
  hit_rate_at_k: ["Right passage retrieved",
    "How often the passage holding the answer was among the retrieved ones"],
  mrr: ["Rank of the right passage",
    "How high the first passage holding the answer was ranked, averaged over items (1 means always first)"],
  ndcg_at_k: ["Ranking quality",
    "How well the retrieved passages were ordered, with relevant ones near the top counting most"],
  context_recall: ["Needed passages found",
    "Share of the passages an item needs that retrieval actually returned"],
  context_precision: ["Retrieved passages that mattered",
    "Share of the retrieved passages that were actually needed"],
  faithfulness: ["Stays within the context",
    "Whether the answer only claims things the retrieved passages support"],
  completeness: ["Covers the whole answer",
    "Whether the answer includes every part of the reference answer"],
  answer_relevance: ["Answers the question asked",
    "Whether the answer addresses the question rather than something nearby"],
  citation_supporting: ["Citations that back the claim",
    "Share of citations pointing at a passage that actually supports the sentence citing it"],
  citation_valid_pointer: ["Citations that exist",
    "Share of citations pointing at a passage that was actually retrieved"],
  citation_density: ["Citations per answer",
    "How many citations the answer carries relative to its length"],
  citation_recall: ["Claims with a citation",
    "Share of the answer's claims that carry a citation"],
  abstention_correct: ["Refuses when it should",
    "Whether the model declined to answer exactly when the context had no answer"],
  abstention: ["Declined to answer",
    "Whether the model refused or said it could not answer"],
  injection_resisted: ["Ignores planted instructions",
    "Whether the model ignored an instruction hidden inside a retrieved passage"],
  consistency: ["Same answer to the same question",
    "Whether paraphrases of one question received matching answers"],
  composite: ["Weighted score",
    "The profile's metrics combined with its declared weights; cost and latency count against a model"],
  metrics_used: ["Metrics in the weighted score",
    "How many weighted metrics had a value for this model"],
  mean: ["Mean", "Average over the scored items"],
  quality: ["Quality", "The quality metric the decision was made on"],

  // Cost and latency
  cost_usd: ["Cost per item (USD)",
    "Metered spend per item, from the provider's own usage counts"],
  cost: ["Cost per item (USD)",
    "Metered spend per item, from the provider's own usage counts"],
  gen_cost_usd: ["Generation cost (USD)",
    "The part of the cost that was the model under test, excluding judge, embedding and rerank"],
  total_cost_usd: ["Total cost (USD)", "Every metered bucket summed for the run"],
  cost_per_correct_answer: ["Cost per correct answer (USD)",
    "Total metered cost divided by the number of correct answers"],
  cost_per_query: ["Cost per query (USD)",
    "Measured serving cost of one query, with judge spend excluded"],
  queries_per_day: ["Queries per day", "The daily volume the projection assumes"],
  daily_usd: ["Cost per day (USD)", "Cost per query multiplied by the daily volume"],
  monthly_usd: ["Cost per month (USD)", "Cost per day multiplied by thirty"],
  annual_usd: ["Cost per year (USD)", "Cost per day multiplied by 365"],
  quality_delta: ["Quality gain over the cheapest",
    "Difference in the quality metric from the cheapest model in the set"],
  monthly_delta_usd: ["Extra cost per month (USD)",
    "Monthly spend above the cheapest model in the set"],
  usd_per_point: ["Cost of one extra point (USD per month)",
    "Monthly cost of each additional point of quality over the cheapest model; negative means worse and dearer"],
  baseline_model: ["Cheapest model", "The model the headroom figures are measured against"],
  latency_ms: ["Response time (ms)",
    "Wall-clock time from request to last token, in milliseconds"],
  latency: ["Response time (ms)",
    "Wall-clock time from request to last token, in milliseconds"],
  latency_p95_ms: ["Slowest 5% of responses (ms)",
    "The response time that 95% of items came in under"],
  latency_p95: ["Slowest 5% of responses (ms)",
    "The response time that 95% of items came in under"],
  ttft_ms: ["Time to first token (ms)", "How long before the first token arrived"],
  ttft: ["Time to first token (ms)", "How long before the first token arrived"],

  // Failure modes, kept beside accuracy and never inside it
  extraction_failure_rate: ["Answers the harness could not read",
    "Share of items where no extractor found an answer in the output; left out of accuracy, not counted wrong"],
  refusal_rate: ["Answers refused", "Share of items where the model declined to answer"],
  native_script_ratio: ["Share of the answer in the expected script",
    "Of the letters in the answer, how many are in the script the profile asks for (Devanagari for Hindi and Marathi, Bengali, Gujarati, Kannada, Telugu); 1 means all of them"],
  reasoning_tokens: ["Hidden reasoning tokens",
    "Tokens the model spent thinking before its visible answer, as the provider counted them; billed inside completion tokens"],
  reasoning_chars: ["Hidden reasoning text",
    "Characters of thinking the provider returned beside the answer; 0 means the model did not think first"],
  reasoning_rate: ["Answers with hidden reasoning",
    "Share of answers where the model thought before answering"],
  reasoning_tokens_mean: ["Hidden reasoning per answer",
    "Average tokens spent thinking before the visible answer, where the provider counted them"],
  empty_answer_rate: ["Empty answers",
    "Share of answers with no visible text at all, usually because the token budget went on hidden reasoning"],
  system_prompt: ["Task instruction",
    "The system prompt every model receives in the baseline pass; the task itself"],
  truncation_rate: ["Answers cut off by the token limit",
    "Share of answers that hit max_tokens before finishing"],
  truncated: ["Answers cut off", "Count of answers that hit the token limit"],
  error_rate: ["Items that errored", "Share of items where the call failed before scoring"],
  errors: ["Errors", "Number of items where the call failed"],
  error_kind: ["Error type", "Which kind of failure the call hit"],
  estimated_rate: ["Token counts estimated",
    "Share of rows whose usage was estimated rather than reported by the provider"],
  estimated: ["Rows with estimated tokens", "Count of rows whose usage was estimated"],
  usage_estimated: ["Token count estimated",
    "Usage came from an estimate, not from the provider's usage block"],
  excluded: ["Items left out of the score",
    "Extraction failures and errors: absent from the accuracy denominator, never folded into it"],
  n_items: ["Items attempted", "Number of items this model was asked"],
  n_scored: ["Items scored",
    "Items that produced a readable answer and entered the accuracy denominator"],
  n: ["Items", "Number of items behind this number"],
  n_rows: ["Rows", "Number of trace rows in the run"],
  n_errors: ["Errored rows", "Number of trace rows that recorded an error"],
  n_baseline: ["Baseline items", "Items scored on the baseline pass"],
  n_adapted: ["Adapted items", "Items scored on the adapted (tuned) pass"],

  // Statistics
  n_pairs: ["Items both models answered",
    "Number of items scored for both models; the paired test runs only over these"],
  diff: ["Difference (A minus B)",
    "Mean of model A's scores minus model B's, over the paired items"],
  mean_a: ["Mean, model A", "Model A's average over the paired items"],
  mean_b: ["Mean, model B", "Model B's average over the paired items"],
  ci_low: ["Interval low", "Lower end of the 95% confidence interval"],
  ci_high: ["Interval high", "Upper end of the 95% confidence interval"],
  p_value: ["p, uncorrected",
    "Chance of a gap this large if the models were really equal, before correcting for multiple comparisons"],
  p_adjusted: ["p, corrected for multiple comparisons",
    "The p-value after Holm-Bonferroni correction across every pair compared"],
  significant: ["Real difference?",
    "Whether the corrected p-value clears the threshold; if not, the order is noise"],
  test: ["Test used",
    "Exact McNemar for yes/no metrics, paired bootstrap for continuous ones; chosen automatically"],
  verdict: ["Verdict", "The comparison in one sentence"],
  model_a: ["Model A", "First model of the pair"],
  model_b: ["Model B", "Second model of the pair"],
  on_frontier: ["On the frontier",
    "Not beaten on every one of accuracy, cost and latency by some other model"],
  elo: ["Elo rating", "Head-to-head rating; a 400-point gap means roughly ten-to-one odds"],
  games: ["Comparisons", "Number of head-to-head judgements this model took part in"],
  wins: ["Wins", "Head-to-head judgements this model won"],
  losses: ["Losses", "Head-to-head judgements this model lost"],
  ties: ["Ties", "Head-to-head judgements the judge could not separate"],
  win_rate: ["Win rate", "Wins plus half of ties, divided by comparisons"],
  chance_level: ["Score from guessing", "The accuracy a random guess would get"],

  // Provenance and configuration
  spec_hash: ["Format fingerprint",
    "Hash of the benchmark's prompt, few-shot, decoding and extraction settings; runs with different fingerprints are never compared"],
  apparatus_hash: ["Apparatus fingerprint",
    "Hash of the pinned embedder, reranker and judge; a change means retrieval or grading moved"],
  dataset_hash: ["Dataset fingerprint", "Hash of the items the run was scored on"],
  git_sha: ["Code version", "Git commit the run was made from"],
  git_dirty: ["Uncommitted changes", "Whether the working tree had edits not in that commit"],
  pricing_as_of: ["Pricing table date", "Which day's prices the cost column used"],
  aborted: ["Stopped early", "Whether the run hit its budget ceiling and stopped"],
  run_id: ["Run", "Identifier of the run"],
  profile: ["Profile", "The workload definition: dataset, retrieval settings, metrics and weights"],
  model: ["Model", "The model under test"],
  models: ["Models", "The models in the run"],
  metric: ["Metric", "The metric this number is about"],
  benchmark: ["Benchmark", "The public or private item set the run used"],
  family: ["Family", "The kind of task: multiple choice, math, instruction following and so on"],
  version: ["Spec version", "Bumped on any change to the benchmark's format"],
  mode: ["Scoring mode", "Generative (read the answer from the text) or log-likelihood"],
  licence: ["Licence", "The dataset's licence"],
  commercial_use: ["Commercial use allowed", "Whether the licence permits commercial use"],
  adapter: ["Adapter present", "Whether code exists to load, prompt, extract and score this benchmark"],
  item_id: ["Item", "Identifier of the evaluated item"],
  extracted_via: ["Extractor that matched", "Which link of the extraction chain found the answer"],
  finish_reason: ["Why the model stopped", "stop means it finished; length means it hit the token limit"],
  raw_output: ["Model output", "What the model actually said, before extraction"],
  baseline: ["Baseline", "The run being compared against"],
  candidate: ["Candidate", "The run under test"],
  delta: ["Change", "Candidate minus baseline"],
  passed: ["Passed", "Whether the check cleared"],
  note: ["Note", "Why the check passed or failed"],
  subject: ["Subject", "The profile or benchmark the run was made on"],
  kind: ["Kind", "Profile run or benchmark run"],
  rows: ["Rows", "Number of trace rows"],
  manifest: ["Manifest", "The provenance record written beside the traces"],
  weights: ["Weights", "How much each metric counts in the weighted score"],
  active_metrics: ["Active metrics", "The metrics the profile scores"],
  task: ["Task", "Direct answering, retrieval-augmented answering or classification"],
  scorer: ["Scorer", "How accuracy is decided: exact match, a judge, or a programmatic check"],
  data: ["Dataset on disk", "Whether the profile's item file is present"],
  max_tokens: ["Token limit", "Longest answer the model may produce"],
  temperature: ["Temperature", "Sampling randomness; zero means deterministic"],
  samples_per_item: ["Samples per item", "How many answers are drawn per item"],
  few_shot: ["Worked examples in the prompt", "How many examples the prompt shows and where they come from"],
};

/* Plain sentences for the error classes the API can surface. The original
   message is always kept underneath in mono: the sentence is the meaning, the
   message is the evidence. */
const ERRORS = {
  UnpairedItemsError: "The models were not scored on the same items, so a fair comparison is not possible.",
  MissingUsageError: "The provider returned no token counts, so this call cannot be priced. The harness will not guess.",
  LicenceError: "This dataset's licence needs an explicit acknowledgement before it can be fetched or run.",
  FetchError: "The dataset could not be fetched or its checksum did not match the spec.",
  SpecError: "The benchmark spec on disk is invalid.",
  DatasetError: "The profile's dataset could not be loaded.",
  ProfileError: "The profile file is invalid.",
  CapabilityError: "This provider does not offer the capability the profile asks for.",
  ProviderError: "The provider call failed.",
  FileNotFoundError: "A file the run needs is not on disk.",
};

/* The plain label for an id, or the id itself when the glossary has no entry.
   Two derived families are resolved by shape: `<metric>_gain` from the tuning
   table and `kind_<error>` from error attribution. */
function term(id) {
  const t = TERMS[id];
  if (t) return t[0];
  if (id.endsWith("_gain") && TERMS[id.slice(0, -5)])
    return "Gain in " + TERMS[id.slice(0, -5)][0].toLowerCase();
  if (id.startsWith("kind_")) return "Errors: " + id.slice(5).replace(/_/g, " ");
  return id;
}
const termWhy = id => (TERMS[id] || [null, null])[1];

/* The technical id behind an info button. A plain label must never hide the
   column it came from, but printing the id under every label repeated the
   same word twice on every tile and header. So the id and the glossary's
   one-line reason sit in a small popover: hover or focus shows it, a click
   or tap pins it (and closes any other), Escape or a click elsewhere clears
   it. The button's hit area is padded to 36px so it works on a phone. The
   `block` argument is kept for callers; it no longer changes the layout. */
/* A small "i" button with a popover. Hover or focus shows it, a click pins
   it (closing any other), Escape or a click elsewhere clears it. */
function infoTip(aria, content) {
  const wrap = el("span", { class: "tid" });
  const btn = el("button", {
    class: "ibtn", type: "button", "aria-label": aria, "aria-expanded": "false",
    onclick: e => {
      e.stopPropagation();
      const open = !wrap.classList.contains("open");
      for (const other of document.querySelectorAll(".tid.open")) other.classList.remove("open");
      wrap.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
      if (open) placeTip(wrap);
    },
    onmouseenter: () => placeTip(wrap),
    onfocus: () => placeTip(wrap),
  }, "i");
  wrap.append(btn, el("span", { class: "tip", role: "tooltip" }, content));
  return wrap;
}

function tid(id, _block = false) {
  const wrap = el("span", { class: "tid" });
  const why = termWhy(id);
  const btn = el("button", {
    class: "ibtn", type: "button", "aria-label": `About ${term(id)}: technical name ${id}`,
    "aria-expanded": "false",
    onclick: e => {
      e.stopPropagation();
      const open = !wrap.classList.contains("open");
      for (const other of document.querySelectorAll(".tid.open")) other.classList.remove("open");
      wrap.classList.toggle("open", open);
      btn.setAttribute("aria-expanded", String(open));
      if (open) placeTip(wrap);
    },
    onmouseenter: () => placeTip(wrap),
    onfocus: () => placeTip(wrap),
  }, "i");
  const tip = el("span", { class: "tip", role: "tooltip" },
    el("code", {}, id),
    why ? el("span", {}, why) : el("span", {}, "Column name in the trace and report tables."));
  wrap.append(btn, tip);
  return wrap;
}

/* Keep a popover inside the viewport: it opens under its button and slides
   left when it would cross the right edge. */
function placeTip(wrap) {
  const tip = wrap.querySelector(".tip");
  if (!tip) return;
  tip.style.left = "0px";
  const r = tip.getBoundingClientRect();
  const over = r.right - (document.documentElement.clientWidth - 16);
  if (over > 0) tip.style.left = `${-over}px`;
}
document.addEventListener("click", () => {
  for (const other of document.querySelectorAll(".tid.open")) {
    other.classList.remove("open");
    other.querySelector(".ibtn")?.setAttribute("aria-expanded", "false");
  }
});
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  for (const other of document.querySelectorAll(".tid.open")) other.classList.remove("open");
});

/* A label made of the plain term plus its info button. For an id the glossary
   does not know (or one that is not an id at all, like "kind" in a runs
   table), the text is returned unchanged. */
function termLabel(id, block = false) {
  if (typeof id !== "string") return id;
  const plain = term(id);
  if (plain === id) return id;
  return el("span", { class: "term" }, plain, tid(id, block));
}

/* An error box for a message from the API. A message naming a known error
   class gets the plain sentence first and the original underneath in mono. */
function errMsg(text) {
  const s = String(text ?? "");
  const hit = Object.keys(ERRORS).find(k => s.includes(k));
  if (!hit) return el("div", { class: "msg crit" }, s);
  return el("div", { class: "msg crit" },
    el("div", {}, ERRORS[hit]),
    el("div", { class: "m", style: "margin-top:6px;font-size:11.5px;opacity:.85" }, s));
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({ error: "HTTP " + r.status }));
  // A session that has expired: the server now serves the landing page at
  // "/", so a reload takes the reader to sign-in with their hash intact.
  if (r.status === 401 && !path.startsWith("/api/auth/")) { location.reload(); }
  if (!r.ok) throw new Error(body.error || ("HTTP " + r.status));
  return body;
}

/* Who is signed in (from the HttpOnly session the server holds), shown in the
   task bar; sign-out clears it server-side and returns to the landing. */
async function initUser() {
  try {
    const me = await api("/api/auth/me");
    const u = me.user;
    $("#userchip").textContent = u ? `${u.name} · ${u.role}` : "";
  } catch { /* shown empty; the next API call redirects if the session is gone */ }
  $("#logoutbtn").addEventListener("click", async () => {
    try { await post("/api/auth/logout", {}); } catch { /* clearing anyway */ }
    location.href = "/";
  });
}
const post = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

/* --------------------------------------------------------------------------
   State. One plain object; views read it, handlers write it, render() redraws.
   -------------------------------------------------------------------------- */
const S = {
  view: "overview",
  // Which screen each section was last on, so returning to a section resumes
  // where you left it rather than resetting to its first tab.
  sectionLast: {},
  benchmarks: [], profiles: [], allRuns: [], benchRuns: [],
  models: { local: [], configured: [] },
  // benchmark run
  benchmark: "", picked: new Set(), limit: 20, seed: 1729,
  job: null, runId: null, results: null, poll: null,
  // extraction
  raw: "", extract: null,
  // profile side
  pProfile: "", pRun: "", pMetric: "accuracy", pResults: null, pUnpaired: false,
  // profile run
  rProfile: "", rPicked: new Set(), rLatency: true, rJob: null, rPoll: null,
  // preflight
  vProfile: "", validation: null, estimation: null,
  // probes
  probeProfile: "", probeSeed: 0, probes: null,
  // decide
  dProfile: "", dOptimise: "cost_usd", dQpd: 10000, dRequire: "", decision: null,
  // gate
  gBase: "", gCand: "", gMetric: "accuracy", gTol: 0.02, gateResult: null,
  // arena / providers
  aRun: "", arena: null, providers: [], ragStatus: null,
  // connections
  // reports
  reports: null, reportOpen: null, repPicked: new Set(), repTitle: "",
  repProvider: "together", repNotes: "", repError: "",
  // case studies
  caseStudies: null, csOpen: {}, csResults: {}, csError: "",
  // evaluate (start here)
  ev: { task: "classify", spec: null, name: "", labels: "", scorer: "", sys: "", maxTokens: "",
        multilingual: false, language: "en", yaml: "", yamlErr: "", created: null, createErr: "", uploads: {},
        uploadErr: "", overwrite: false, bench: "" },
  connections: null, connProvider: "", connKey: "", connDetail: null,
  connPicked: new Set(), modelsYaml: null,
  error: "",
};

/* --------------------------------------------------------------------------
   Building blocks
   -------------------------------------------------------------------------- */
const head = (eyebrow, title, sub) => el("div", { class: "head" },
  el("span", { class: "eyebrow" }, eyebrow),
  el("h1", {}, title),
  el("p", { class: "sub" }, sub));

const card = (title, why, ...kids) => el("div", { class: "card" },
  title && el("h3", {}, title), why && el("p", { class: "why" }, why), ...kids);

/* A KPI label is either a glossary id (rendered as the plain term with the
   id underneath), plain text, or a ready-made node. */
const kpis = items => el("div", { class: "kpis" }, items.map(([v, k, tone]) =>
  el("div", { class: "kpi " + (tone || "") },
    el("div", { class: "v" }, v), el("div", { class: "k" }, termLabel(k, true)))));

/* Column headers are glossary ids where one exists: the plain term is the
   header and the id sits under it, so a reader who knows the column name and
   a reader who does not both find their way. */
function table(cols, rows, render) {
  if (!rows || !rows.length) return el("p", { class: "note" }, "Nothing to show.");
  return el("div", { class: "tw" }, el("table", {},
    el("thead", {}, el("tr", {}, cols.map(c => el("th", {}, termLabel(c, true))))),
    el("tbody", {}, rows.map(r => el("tr", {}, render(r))))));
}

const numTd = (v, d = 3) => {
  const s = fmt(v, d);
  return s === null
    ? el("td", { class: "n" }, el("span", { class: "null" }, "n/a"))
    : el("td", { class: "n" }, s);
};

/* Build a table from whatever columns the library returned. The report layer
   owns which columns exist; hard-coding them here would mean editing the UI
   every time a metric is added, and quietly dropping it until someone did. */
function autoTable(rows) {
  if (!rows || !rows.length) return el("p", { class: "note" }, "Nothing to show.");
  const cols = Object.keys(rows[0]);
  return table(cols, rows, r => cols.map(c => {
    const v = r[c];
    if (typeof v === "number") return numTd(v, Math.abs(v) < 0.01 && v !== 0 ? 6 : 3);
    if (typeof v === "boolean") return el("td", {}, el("span",
      { class: "tag " + (v ? "ok" : "dim") }, v ? "yes" : "no"));
    return v === null
      ? el("td", { class: "m" }, el("span", { class: "null" }, "n/a"))
      : el("td", { class: "m" }, String(v));
  }));
}

const empty = (title, body, hint) => el("div", { class: "empty" },
  el("b", {}, title), body, hint && el("p", { class: "note" }, hint));

/* Append a chart inside a card, or nothing at all. A chart builder returns
   null when the data cannot support the form: an empty axis frame is worse
   than no chart, because it implies there was something to see.

   The title is a small label above the figure's own caption. When it is a
   glossary id (a chart about one metric) it shows the plain term with the id
   beside it, so the chart is traceable to the column it draws. */
const CHART_LABEL_STYLE = "font:10px/1.4 var(--mono);letter-spacing:.12em;" +
  "text-transform:uppercase;color:var(--ink-3);margin:0 0 8px";
function charted(parent, title, fig) {
  if (!fig) return;
  const c = el("div", { class: "card" });
  if (title) c.append(el("div", { class: "chartlabel", style: CHART_LABEL_STYLE },
    termLabel(title)));
  c.append(fig);
  parent.append(c);
}

const spec = () => S.benchmarks.find(b => b.id === S.benchmark) || null;
const profileRuns = () => S.allRuns.filter(r => r.kind === "profile");

/* --------------------------------------------------------------------------
   Overview
   -------------------------------------------------------------------------- */
/* Every past attempt as a timeline: what was tried, on which models, what
   it cost, and whether it is a result (manifest) or merely rows. Newest first,
   because the question on opening the dashboard is "what happened last". */
function attemptsTimeline(runs) {
  const c = card("Past attempts",
    `${runs.length} run(s) in this store. A profile run is a workload of your ` +
    "own; a benchmark run is a public set through the same apparatus.");
  const tl = el("div", { class: "timeline" });
  for (const r of runs) {
    const when = r.ts ? new Date(r.ts * 1000) : null;
    const open = () => r.kind === "benchmark"
      ? loadResults(r.run_id) : loadProfileReport({ run_id: r.run_id });
    tl.append(el("div", { class: "attempt " + r.kind, role: "button", tabindex: 0,
      onclick: open, onkeydown: e => { if (e.key === "Enter") open(); } },
      el("div", {},
        el("div", { class: "when" },
          (when ? when.toLocaleString() : "undated") + " · " + r.kind +
          (r.passes && r.passes.length ? " · " + r.passes.join(" + ") : "")),
        el("div", { class: "what" }, r.subject + "  ·  " + r.run_id),
        el("div", { class: "who" }, r.models.join("  ·  ")),
        el("div", { class: "facts", style: "margin-top:6px" },
          el("span", {}, el("b", {}, String(r.rows)), " rows"),
          el("span", {}, el("b", {}, String(r.models.length)), " models"),
          el("span", {}, el("b", {}, "$" + (r.cost_usd || 0).toFixed(4)), " metered"),
          el("span", {}, el("b", {}, String(r.errors || 0)), " errors"),
          el("span", {}, el("span", { class: "tag " + (r.has_manifest ? "ok" : "crit") },
            r.has_manifest ? "manifest" : "no manifest")),
          r.aborted && el("span", {}, el("span", { class: "tag warn" }, "aborted")))),
      el("button", { class: "btn ghost sm", onclick: e => { e.stopPropagation(); open(); } },
        "open")));
  }
  c.append(tl);
  return c;
}

function viewOverview() {
  const w = el("div", {}, head("Dashboard", "Overview",
    "Both halves of the harness over one trace store. A benchmark run and a " +
    "profile run differ only in where the items came from: they share the " +
    "store, the cost meter and the significance implementation, so they " +
    "belong in one list, not two."));

  const bench = S.allRuns.filter(r => r.kind === "benchmark");
  const prof = profileRuns();
  const cost = S.allRuns.reduce((a, r) => a + (r.cost_usd || 0), 0);
  const noManifest = S.allRuns.filter(r => !r.has_manifest).length;

  w.append(kpis([
    [String(S.allRuns.length), "runs", "accent"],
    [String(prof.length), "profile runs"],
    [String(bench.length), "benchmark runs"],
    [String(S.profiles.length), "profiles"],
    [String(S.benchmarks.length), "benchmarks"],
    ["$" + cost.toFixed(4), "metered spend"],
    [String(noManifest), "no manifest", noManifest ? "crit" : "ok"],
  ]));

  if (!S.allRuns.length) {
    w.append(empty("No runs in the store yet",
      "Start with a benchmark. The fake: models need no key and no network.",
      "Everything on this dashboard works offline against the local provider."));
  } else {
    w.append(attemptsTimeline(S.allRuns));
    charted(w, "Spend per attempt", Charts.spendByRun(S.allRuns.map(r => ({
      run_id: r.run_id, cost_usd: r.cost_usd, subject: r.subject }))));
    w.append(card("Runs", "The same attempts as a table. Open one for its report.",
      table(["kind", "subject", "run_id", "models", "rows", "errors", "total_cost_usd", "manifest", ""],
        S.allRuns, r => [
          el("td", {}, el("span", { class: "tag " + (r.kind === "benchmark" ? "accent" : "dim") }, r.kind)),
          el("td", { class: "m" }, r.subject),
          el("td", { class: "m" }, r.run_id),
          el("td", { class: "m" }, r.models.join(", ")),
          numTd(r.rows, 0), numTd(r.errors, 0), numTd(r.cost_usd, 5),
          el("td", {}, el("span", { class: "tag " + (r.has_manifest ? "ok" : "crit") },
            r.has_manifest ? "yes" : "none")),
          el("td", {}, el("button", {
            class: "btn ghost sm",
            onclick: () => r.kind === "benchmark"
              ? loadResults(r.run_id) : loadProfileReport({ run_id: r.run_id }),
          }, "open")),
        ])));
    if (noManifest) w.append(el("p", { class: "note" },
      "A run with no manifest predates the manifest contract. Under I9 (the " +
      "reproducibility rule: every run records what it was made from) it " +
      "records no git SHA, dataset hash or apparatus hash, so nothing can say " +
      "what it may be compared against."));
  }

  w.append(el("hr", { class: "rule" }));
  w.append(card("Profiles", "Workloads defined in configs/profiles/.",
    table(["profile", "task", "data", "scorer", "active_metrics", "weights"], S.profiles, p =>
      p.invalid
        ? [el("td", { class: "m" }, p.name),
           el("td", {}, el("span", { class: "tag crit" }, "invalid")),
           el("td", { class: "m" }, p.invalid.slice(0, 70)),
           el("td", {}), el("td", {}), el("td", {})]
        : [el("td", { class: "m" }, p.name),
           el("td", {}, el("span", { class: "tag dim" }, p.task)),
           el("td", {}, el("span", { class: "tag " + (p.runnable ? "ok" : "warn") },
             p.runnable ? "present" : "missing")),
           el("td", { class: "m" }, p.accuracy_scorer),
           el("td", { class: "m" }, (p.active_metrics || []).join(", ")),
           el("td", { class: "m" }, Object.entries(p.metric_weights || {})
             .map(([k, v]) => k + "=" + v).join(" "))])));
  const unrunnable = S.profiles.filter(p => !p.invalid && !p.runnable).length;
  if (unrunnable) w.append(el("p", { class: "note" },
    unrunnable + " profile(s) are configured but their dataset is not on disk. " +
    "They are shipped as worked examples; point `evalset_path` at your own data, " +
    "or build one from PDFs with prepare_dataset.py."));
  return w;
}

/* --------------------------------------------------------------------------
   Preflight: validate + estimate, the cheap half of the workflow
   -------------------------------------------------------------------------- */
function viewPreflight() {
  const w = el("div", {}, head("Before you spend", "Preflight",
    "The expensive mistakes in evaluation are made before the first API call. " +
    "Validate checks the profile, the dataset and the gold passages; Estimate " +
    "forecasts the bill including the judge, which on a judge-scored profile " +
    "is routinely the largest line."));

  w.append(card("Profile", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Profile"),
      el("select", { onchange: e => { S.vProfile = e.target.value; runValidate(); } },
        S.profiles.filter(p => !p.invalid).map(p =>
          el("option", { value: p.name, selected: p.name === S.vProfile },
            p.name + (p.runnable ? "" : "  (dataset missing)"))))),
    el("div", { class: "f", style: "flex:0 0 auto" },
      el("label", {}, " "), el("button", { class: "btn", onclick: runValidate }, "Validate")))));

  const V = S.validation;
  if (V) {
    w.append(el("div", { class: "msg " + (V.ok ? "ok" : "crit") },
      V.ok ? `Profile '${V.profile}' is valid (task=${V.task}, scorer=${V.accuracy_scorer})`
           : `Profile '${V.profile}' has ${V.problems.length} problem(s)`));
    for (const p of V.problems) w.append(errMsg(p));
    for (const wn of V.warnings) w.append(el("div", { class: "msg warn" }, wn));
    if (V.dataset) w.append(card("Dataset", null,
      el("pre", { class: "out" }, V.dataset.summary),
      el("div", { class: "kv", style: "margin-top:14px" }, [
        ["items", V.dataset.n_items], ["evalset", V.dataset.evalset_path],
        ["corpus", V.dataset.corpus_path || "none"], ["max_tokens", V.max_tokens],
      ].map(([k, v]) => el("div", {}, el("dt", {}, termLabel(k)), el("dd", {}, String(v)))))));
    if (V.gold_problems && V.gold_problems.length)
      w.append(card("Gold-passage problems",
        "A gold id that matches no chunk makes every retrieval metric for that " +
        "item meaningless.",
        el("pre", { class: "out" }, V.gold_problems.join("\n"))));
  }

  w.append(el("hr", { class: "rule" }));

  const all = [...S.models.local, ...S.models.configured];
  w.append(card("Estimate", "Pick the models you would actually run.",
    el("div", { class: "pills" }, all.map(m => el("button", {
      class: "pick" + (m.startsWith("fake:") ? " local" : ""),
      "aria-pressed": S.picked.has(m),
      onclick: () => { S.picked.has(m) ? S.picked.delete(m) : S.picked.add(m); render(); },
    }, m))),
    el("div", { class: "row", style: "margin-top:14px" },
      el("button", { class: "btn", disabled: !S.picked.size, onclick: runEstimate },
        "Estimate cost"))));

  const E = S.estimation;
  if (E) {
    if (E.error) w.append(errMsg(E.error));
    else {
      w.append(kpis([
        [String(E.models.length), "models"],
        [E.n_test.toLocaleString(), "test items"],
        [E.calls.toLocaleString(), "model calls"],
        [E.judge_calls.toLocaleString(), "judge calls",
          E.judge_calls ? "warn" : ""],
        ["$" + E.est_usd.toFixed(4), "estimated", "accent"],
      ]));
      w.append(el("p", { class: "note" }, E.note));
      if (E.detail && E.detail.unpriced_models && E.detail.unpriced_models.length)
        w.append(el("div", { class: "msg warn" },
          "No pricing for " + E.detail.unpriced_models.join(", ") +
          ". Their cost metric would be blank rather than zero, because a " +
          "free-looking model would win a cost-weighted leaderboard."));
    }
  }
  return w;
}

async function runValidate() {
  if (!S.vProfile) return;
  try { S.validation = await api("/api/validate?profile=" + encodeURIComponent(S.vProfile)); }
  catch (e) { S.validation = { profile: S.vProfile, ok: false, problems: [e.message], warnings: [] }; }
  render();
}

async function runEstimate() {
  const q = new URLSearchParams({ profile: S.vProfile });
  for (const m of S.picked) q.append("model", m);
  try { S.estimation = await api("/api/estimate?" + q); }
  catch (e) { S.estimation = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Probes
   -------------------------------------------------------------------------- */
function viewProbes() {
  const w = el("div", {}, head("Before you spend", "Adversarial probes",
    "A RAG corpus is an attack surface: anything that can put text into your " +
    "index can put instructions in front of your model. These five families " +
    "are derived from questions you already have, so they test your system " +
    "rather than a generic benchmark."));

  w.append(card("Generate", "Deterministic, free, and needs no re-ingest.",
    el("div", { class: "row" },
      el("div", { class: "f" }, el("label", {}, "Profile"),
        el("select", { onchange: e => { S.probeProfile = e.target.value; runProbes(); } },
          S.profiles.filter(p => !p.invalid).map(p =>
            el("option", { value: p.name, selected: p.name === S.probeProfile },
              p.name + (p.runnable ? "" : "  (dataset missing)"))))),
      el("div", { class: "f" }, el("label", {}, "Seed"),
        el("input", { type: "number", min: "0", value: S.probeSeed,
          onchange: e => { S.probeSeed = +e.target.value || 0; runProbes(); } })),
      el("div", { class: "f", style: "flex:0 0 auto" }, el("label", {}, " "),
        el("button", { class: "btn", onclick: runProbes }, "Generate")))));

  const P = S.probes;
  if (!P) return w;
  if (P.error) { w.append(errMsg(P.error)); return w; }

  w.append(kpis([
    [String(P.base_items), "base items"],
    [String(P.generated), "probes generated", "accent"],
    ...Object.entries(P.by_family).map(([k, v]) => [String(v), k]),
  ]));
  w.append(el("p", { class: "note" }, P.note));

  charted(w, "Probes generated per family", Charts.familyBars(
    Object.entries(P.by_family).map(([k, v]) => ({ label: k, value: v, n: v })),
    { title: "Probes generated per family",
      note: "How many adversarial items each family contributed. Every one is scored " +
            "on its own metric, never folded into accuracy." }));
  w.append(card("Families", "What each one catches.",
    el("div", { class: "kv" }, Object.entries(P.families).map(([k, v]) =>
      el("div", { style: "flex-direction:column;align-items:flex-start;gap:5px" },
        el("dt", {}, el("span", { class: "tag accent" }, k)),
        el("dd", { style: "text-align:left;font-family:var(--sans);font-size:12.5px" }, v))))));

  if (P.samples.length) w.append(card("Sample probes",
    "Canary strings are deliberately not shown: they exist as plaintext in the " +
    "item, and printing one into a browser (and its history, and a screenshot) " +
    "is exactly the leak that I12 (secrets and canaries never leak) is about.",
    table(["item", "family", "type", "canary", "query"], P.samples, s => [
      el("td", { class: "m" }, s.item_id),
      el("td", {}, el("span", { class: "tag accent" }, s.family)),
      el("td", { class: "m" }, s.item_type),
      el("td", {}, el("span", { class: "tag " + (s.has_canary ? "warn" : "dim") },
        s.has_canary ? "yes" : "no")),
      el("td", { class: "m" }, s.query),
    ])));
  return w;
}

async function runProbes() {
  if (!S.probeProfile) return;
  const q = new URLSearchParams({ profile: S.probeProfile, seed: String(S.probeSeed) });
  try { S.probes = await api("/api/probes?" + q); }
  catch (e) { S.probes = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Profile run
   -------------------------------------------------------------------------- */
function viewRunProfile() {
  const w = el("div", {}, head("Run", "Profile run",
    "The original apparatus: retrieve, prompt, generate, score, meter. The " +
    "same path the CLI takes, writing the same TraceRows."));

  w.append(card("Configuration", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Profile"),
      el("select", { onchange: e => { S.rProfile = e.target.value; render(); } },
        S.profiles.filter(p => !p.invalid).map(p =>
          el("option", { value: p.name, selected: p.name === S.rProfile },
            p.name + (p.runnable ? "" : "  (dataset missing)"))))),
    el("div", { class: "f", style: "flex:0 0 auto;min-width:0" },
      el("label", {}, "Latency lane"),
      el("button", {
        class: "pick", "aria-pressed": S.rLatency,
        onclick: () => { S.rLatency = !S.rLatency; render(); },
      }, S.rLatency ? "on" : "off")))));

  const all = [...S.models.local, ...S.models.configured];
  w.append(card("Models", S.models.note,
    el("div", { class: "pills" }, all.map(m => el("button", {
      class: "pick" + (m.startsWith("fake:") ? " local" : ""),
      "aria-pressed": S.rPicked.has(m),
      onclick: () => { S.rPicked.has(m) ? S.rPicked.delete(m) : S.rPicked.add(m); render(); },
    }, m)))));

  const real = [...S.rPicked].filter(m => !m.startsWith("fake:"));
  if (real.length) w.append(el("div", { class: "msg warn" },
    `${real.length} model(s) will make real, billed calls: ${real.join(", ")}. ` +
    `Run Preflight first: it forecasts the bill including the judge.`));

  const busy = S.rJob && S.rJob.status === "running";
  const blocked = !S.rPicked.size ? "Pick at least one model."
    : busy ? "A run is in flight." : "";
  w.append(el("div", { class: "row" },
    el("button", { class: "btn", disabled: !!blocked, onclick: startProfileRun },
      busy ? "Running…" : "Run profile"),
    blocked && el("span", { class: "note" }, blocked)));
  if (S.rJob) w.append(jobPanel(S.rJob, () => loadProfileReport({ run_id: S.rRunId })));
  return w;
}

async function startProfileRun() {
  try {
    const d = await post("/api/run-profile", {
      profile: S.rProfile, models: [...S.rPicked],
      passes: { baseline: true, latency: S.rLatency },
    });
    S.rRunId = d.run_id;
    S.rJob = { status: "running", done: 0, total: d.total, frac: 0, elapsed: 0 };
    render();
    clearInterval(S.rPoll);
    S.rPoll = setInterval(async () => {
      try { S.rJob = await api("/api/job?job_id=" + d.job_id); }
      catch (e) { S.rJob = { status: "error", error: e.message }; }
      if (S.rJob.status !== "running") { clearInterval(S.rPoll); await refreshRuns(); }
      if (S.view === "runprofile") render();
    }, 700);
  } catch (e) { S.rJob = { status: "error", error: e.message }; render(); }
}

/* --------------------------------------------------------------------------
   Benchmark run
   -------------------------------------------------------------------------- */
function viewRun() {
  const sp = spec();
  const w = el("div", {}, head("Run", "Benchmark run",
    "Every model sees the identical item set in the identical order, sampled " +
    "deterministically from the seed. That is what makes the comparison paired."));

  w.append(card("Benchmark", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Benchmark"),
      el("select", { onchange: e => { S.benchmark = e.target.value; render(); } },
        S.benchmarks.map(b => el("option", { value: b.id, selected: b.id === S.benchmark },
          b.id + (b.has_adapter ? "" : "  (no adapter)"))))),
    el("div", { class: "f" }, el("label", {}, "Items"),
      el("input", { type: "number", min: "1", max: "10000", value: S.limit,
        oninput: e => { S.limit = Math.max(1, +e.target.value || 1); renderKpis(); } })),
    el("div", { class: "f" }, el("label", {}, "Seed"),
      el("input", { type: "number", min: "0", value: S.seed,
        oninput: e => { S.seed = +e.target.value || 0; } })))));

  if (sp) w.append(specCard(sp));

  const all = [...S.models.local, ...S.models.configured];
  w.append(card("Models", S.models.note,
    el("div", { class: "pills" }, all.map(m => el("button", {
      class: "pick" + (m.startsWith("fake:") ? " local" : ""),
      "aria-pressed": S.picked.has(m),
      onclick: () => { S.picked.has(m) ? S.picked.delete(m) : S.picked.add(m); render(); },
    }, m)))));

  const real = [...S.picked].filter(m => !m.startsWith("fake:"));
  if (real.length) w.append(el("div", { class: "msg warn" },
    `This will make real, billed calls on ${real.join(", ")}.`));

  w.append(el("div", { id: "kpiwrap" }));
  const busy = S.job && S.job.status === "running";
  const blocked = !S.picked.size ? "Pick at least one model."
    : (sp && !sp.has_adapter) ? "That benchmark has no registered adapter."
    : (sp && sp.needs_licence_ack) ? `${sp.id} is ${sp.licence} (non-commercial); use the CLI with --acknowledge-licence.`
    : busy ? "A run is in flight." : "";
  w.append(el("div", { class: "row" },
    el("button", { class: "btn", disabled: !!blocked, onclick: startRun },
      busy ? "Running…" : "Run benchmark"),
    blocked && el("span", { class: "note" }, blocked)));
  if (S.job) w.append(jobPanel(S.job, () => loadResults(S.runId)));
  return w;
}

function specCard(sp) {
  const h = sp.hashed;
  const c = card("Spec",
    "Everything here feeds spec_hash, the format fingerprint. Change any of it " +
    "and the reporter refuses to compare the result against an older run: that " +
    "is the mechanism stopping a re-worded prompt being compared with last " +
    "month's score.",
    el("div", { class: "kv" }, [
      ["family", sp.family], ["version", sp.version], ["mode", h.mode],
      ["chance_level", h.chance_level],
      ["few_shot", h.few_shot_n + " from " + h.few_shot_pool],
      ["max_tokens", h.max_tokens], ["temperature", h.temperature],
      ["samples_per_item", h.samples_per_item],
      ["licence", sp.licence + (sp.commercial_use ? "" : " (non-commercial)")],
      ["spec_hash", sp.spec_hash.slice(0, 16)],
    ].map(([k, v]) => el("div", {}, el("dt", {}, termLabel(k)), el("dd", {}, String(v))))),
    el("p", { class: "note" }, "Extraction chain, in order; first match wins:"),
    el("div", { class: "chain" }, sp.chain.flatMap((lk, i) => [
      i ? el("span", { class: "arrow" }, "→") : null,
      el("span", { class: "link" }, lk.kind),
    ]).filter(Boolean)));
  if (sp.problems && sp.problems.length)
    c.append(errMsg("Invalid spec: " + sp.problems.join("; ")));
  if (h.chance_level > 0) c.append(el("p", { class: "note" },
    `Chance is ${h.chance_level}, so a raw accuracy of ${h.chance_level} is zero ` +
    `information. The chance-adjusted column is the honest one.`));
  return c;
}

function renderKpis() {
  const box = $("#kpiwrap"); if (!box) return;
  const sp = spec();
  box.replaceChildren(kpis([
    [String(S.picked.size), "models"],
    [S.limit.toLocaleString(), "items"],
    [(S.picked.size * S.limit).toLocaleString(), "calls"],
    [sp ? String(sp.hashed.chance_level) : "n/a", "chance_level"],
    [sp ? sp.spec_hash.slice(0, 10) : "n/a", "spec_hash"],
  ]));
}

async function startRun() {
  S.error = "";
  try {
    const d = await post("/api/run", {
      benchmark: S.benchmark, models: [...S.picked], limit: S.limit, seed: S.seed,
    });
    S.runId = d.run_id;
    S.job = { status: "running", done: 0, total: d.total, frac: 0, elapsed: 0 };
    render();
    clearInterval(S.poll);
    S.poll = setInterval(async () => {
      try { S.job = await api("/api/job?job_id=" + d.job_id); }
      catch (e) { S.job = { status: "error", error: e.message }; }
      if (S.job.status !== "running") { clearInterval(S.poll); await refreshRuns(); }
      if (S.view === "run") render();
    }, 700);
  } catch (e) { S.job = { status: "error", error: e.message }; render(); }
}

function jobPanel(j, onOpen) {
  if (j.status === "running") {
    const pct = Math.round((j.frac || 0) * 100);
    return card(null, null,
      el("div", {}, el("span", { class: "spin" }),
        `${(j.done || 0).toLocaleString()} / ${(j.total || 0).toLocaleString()} ` +
        `(${pct}%) · ${Math.round(j.elapsed || 0)}s`),
      el("div", { class: "bar" }, el("i", { style: "width:" + pct + "%" })));
  }
  if (j.status === "done") {
    return el("div", { class: "msg ok" },
      `Finished in ${Math.round(j.elapsed)}s: ${(j.rows || 0).toLocaleString()} rows. ` +
      (j.messages || []).join(" "), " ",
      el("button", { class: "btn ghost sm", style: "margin-left:10px", onclick: onOpen },
        "See results"));
  }
  return errMsg(j.error || "failed");
}

/* --------------------------------------------------------------------------
   Benchmark results
   -------------------------------------------------------------------------- */
function viewResults() {
  const w = el("div", {}, head("Analyse", "Benchmark results",
    "The leaderboard, the failure rates that sit beside it, and the paired " +
    "test that decides whether the order means anything."));

  if (!S.benchRuns.length) {
    w.append(empty("No benchmark runs yet",
      "Run one on the Benchmark run screen; the fake: models need no key."));
    return w;
  }

  w.append(card("Run", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Run"),
      el("select", { onchange: e => loadResults(e.target.value) },
        S.benchRuns.map(r => el("option", { value: r.run_id, selected: r.run_id === S.runId },
          `${r.run_id} · ${r.benchmark} · ${r.rows} rows`)))))));

  const R = S.results;
  if (!R) { w.append(el("p", { class: "note" }, "Loading…")); return w; }
  if (R.refused) {
    w.append(errMsg(R.reason));
    w.append(el("p", { class: "note" }, "Format fingerprints (spec_hash) present: " +
      (R.spec_hashes || []).join(", ")));
    return w;
  }
  if (R.error) { w.append(errMsg(R.error)); return w; }

  const best = R.summary[0] || {};
  w.append(kpis([
    [R.benchmark, "benchmark", "accent"],
    [String(R.summary.length), "models"],
    [fmt(best.accuracy) ?? "n/a", el("span", {}, "best ", termLabel("accuracy", true))],
    [fmt(best.accuracy_chance_adjusted) ?? "n/a",
     el("span", {}, `best above guessing (chance ${R.chance_level})`,
       tid("accuracy_chance_adjusted", true))],
    [String(R.excluded_total), "excluded", R.excluded_total ? "warn" : "ok"],
  ]));

  // Charts first: the shape of the result before its digits. Each is skipped
  // when its data cannot support it, rather than drawn empty.
  // Prefer the bootstrap CI table: the interval is the point of this chart.
  // Fall back to point estimates only when the run has too few items for one.
  const ciRows = (R.ci && R.ci.length) ? R.ci : R.summary.map(r => ({
    model: r.model, mean: r.accuracy, n: r.n_scored,
    ci_low: null, ci_high: null,
  }));
  charted(w, "accuracy", Charts.accuracyCI(ciRows,
    { metric: "accuracy", chance: R.chance_level }));
  charted(w, "excluded", Charts.scoredSplit(R.summary));
  // Why each model's denominator shrank: read errors, cut-off answers,
  // refusals and run errors are four different events and none is a wrong
  // answer. Drawn from the rates the summary already carries.
  charted(w, "Why items were excluded", Charts.failureSplit(R.summary.map(r => ({
    model: r.model,
    extraction_failure_rate: r.extraction_failure_rate,
    truncation_rate: r.truncation_rate,
    refusal_rate: r.refusal_rate,
    error_rate: r.error_rate,
  }))));
  charted(w, "Quality against cost", Charts.qualityCost(R.summary.map(r => ({
    model: r.model, accuracy: r.accuracy, cost: r.cost_usd,
    on_frontier: true,
  }))));
  charted(w, "cost_per_correct_answer", Charts.costPerCorrect(R.summary.map(r => ({
    model: r.model, cost_per_correct_answer: r.cost_per_correct_answer,
  }))));
  if (R.significance && R.significance.length)
    charted(w, "Which pairs are separable", Charts.pairwiseMatrix(R.significance));

  w.append(card("Leaderboard",
    "Excluded items are extraction failures and errors: absent from the " +
    "accuracy denominator, never folded into it. A parse failure is not a " +
    "wrong answer (I7, failure is not wrongness).",
    table(["model", "n_items", "n_scored", "accuracy", "accuracy_chance_adjusted",
           "extraction_failure_rate", "refusal_rate", "truncation_rate", "cost_usd",
           "cost_per_correct_answer"],
      R.summary, r => [
        el("td", { class: "m" }, r.model),
        numTd(r.n_items, 0), numTd(r.n_scored, 0),
        numTd(r.accuracy), numTd(r.accuracy_chance_adjusted),
        numTd(r.extraction_failure_rate), numTd(r.refusal_rate),
        numTd(r.truncation_rate), numTd(r.cost_usd, 5),
        numTd(r.cost_per_correct_answer, 5),
      ])));

  for (const r of R.summary) {
    const gap = r.n_items - r.n_scored;
    if (gap) w.append(el("p", { class: "note" },
      `${r.model}: ${gap} item(s) excluded from accuracy. Counting them wrong ` +
      `would measure the extractor and blame the model.`));
  }
  w.append(significanceCard(R));

  w.append(el("div", { class: "row" },
    el("button", { class: "btn ghost", onclick: () => showExtra("rows") }, "Raw rows"),
    el("button", { class: "btn ghost", onclick: () => showExtra("manifest") }, "Manifest")));
  w.append(el("div", { id: "extra" }));
  return w;
}

function significanceCard(R) {
  const c = card("Is the difference real?",
    "Paired at the item level (every model scored on the same items), " +
    "Holm-corrected for multiple comparisons, exact McNemar when the metric " +
    "is binary. The same implementation the profile path uses; §10 (the " +
    "benchmark subsystem spec) forbids a second.");
  if (R.significance_error) {
    c.append(errMsg(R.significance_error),
      el("p", { class: "note" },
        "An UnpairedItemsError is a finding, not a crash: the models were not " +
        "scored on the same items, so a comparison would be selected by one " +
        "model's failures."));
  } else if (!R.significance || !R.significance.length) {
    c.append(el("p", { class: "note" }, "Needs two or more models."));
  } else {
    c.append(table(["model_a", "model_b", "n_pairs", "diff", "p_adjusted", "significant", "test"],
      R.significance, r => [
        el("td", { class: "m" }, r.model_a), el("td", { class: "m" }, r.model_b),
        numTd(r.n_pairs, 0), numTd(r.diff), numTd(r.p_adjusted),
        el("td", {}, el("span", { class: "tag " + (r.significant ? "ok" : "dim") },
          r.significant ? "yes" : "no")),
        el("td", { class: "m" }, r.test)]));
    for (const r of R.significance)
      c.append(el("p", { class: "note" },
        (r.significant ? "✓ " : "· ") + r.model_a + " vs " + r.model_b + ": " + r.verdict));
    if (R.power) c.append(el("p", { class: "note" }, R.power));
  }
  return c;
}

async function showExtra(kind) {
  const box = $("#extra");
  box.replaceChildren(el("p", { class: "note" }, "Loading…"));
  try {
    if (kind === "rows") {
      const d = await api(`/api/rows?run_id=${encodeURIComponent(S.runId)}&limit=60`);
      box.replaceChildren(card("Raw rows", "What the model actually said.",
        table(["model", "item_id", "accuracy", "extracted_via", "finish_reason", "raw_output"],
          d.rows, r => [
          el("td", { class: "m" }, r.model), el("td", { class: "m" }, r.item_id),
          numTd(r.accuracy, 1), el("td", { class: "m" }, r.extracted_via || "none"),
          el("td", { class: "m" }, r.finish_reason || "n/a"),
          el("td", { class: "m" }, (r.raw_output || "").slice(0, 120))])));
    } else {
      const m = await api(`/api/manifest?run_id=${encodeURIComponent(S.runId)}`);
      const keys = ["run_id", "profile", "git_sha", "git_dirty", "dataset_hash",
        "apparatus_hash", "spec_hash", "pricing_as_of", "n_rows", "n_errors",
        "aborted", "total_cost_usd"];
      box.replaceChildren(card("Run manifest",
        "I9, the reproducibility rule: a run without one of these is not a " +
        "result. It records what the run was, and therefore what it may be " +
        "compared against.",
        el("div", { class: "kv" }, keys.filter(k => k in m).map(k =>
          el("div", {}, el("dt", {}, termLabel(k)), el("dd", {}, String(m[k])))))));
    }
  } catch (e) { box.replaceChildren(errMsg(e.message)); }
}

async function loadResults(runId) {
  S.runId = runId; S.view = "results"; S.results = null; render();
  try { S.results = await api("/api/results?run_id=" + encodeURIComponent(runId)); }
  catch (e) { S.results = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Extraction playground
   -------------------------------------------------------------------------- */
const SAMPLES = {
  mmlu_pro: "Let me consider each option.\n\nAnswer: C",
  gsm8k: "She saves 23 * 8 = 184.\nAdding 24 gives 208.\n#### 208",
  ifeval: "delhi mumbai pune",
};

function viewExtract() {
  const w = el("div", {}, head("Before you spend", "Extraction",
    "Extraction is quietly the whole benchmark: the gap between a reported " +
    "0.61 and a reported 0.78 is usually not the model, it is whether the " +
    "harness could find the answer in the prose. Try yours here first."));

  w.append(card("Chain", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Benchmark"),
      el("select", {
        onchange: e => {
          S.benchmark = e.target.value;
          S.raw = SAMPLES[S.benchmark] || "Answer: C";
          runExtract();
        },
      }, S.benchmarks.map(b =>
        el("option", { value: b.id, selected: b.id === S.benchmark }, b.id)))))));

  w.append(card("Model output", "Paste anything a model might return.",
    el("textarea", { oninput: e => { S.raw = e.target.value; debounceExtract(); } }, S.raw),
    el("p", { class: "note" }, "Runs as you type. Pure: no provider, no cost.")));

  w.append(el("div", { id: "exres" }));
  if (S.extract) queueMicrotask(() => renderExtract());
  return w;
}

let _t = null;
const debounceExtract = () => { clearTimeout(_t); _t = setTimeout(runExtract, 220); };

async function runExtract() {
  if (!S.benchmark) return;
  try { S.extract = await post("/api/extract", { benchmark: S.benchmark, raw: S.raw }); S.error = ""; }
  catch (e) { S.error = e.message; S.extract = null; }
  if ($("#exres")) renderExtract(); else render();
}

function renderExtract() {
  const box = $("#exres"); if (!box) return;
  const X = S.extract;
  if (!X) { box.replaceChildren(errMsg(S.error || "No result.")); return; }

  box.replaceChildren(
    X.ok ? el("div", { class: "msg ok" }, `Extracted "${X.value}" via ${X.via}`)
         : el("div", { class: "msg crit" }, "Extraction failed: " + X.reason),
    el("p", { class: "note" }, X.consequence),
    card("Which link fired", null,
      el("div", { class: "chain" }, X.chain.flatMap((lk, i) => [
        i ? el("span", { class: "arrow" }, "→") : null,
        el("span", { class: "link" + (X.ok && lk.kind === X.via ? " hit" : "") }, lk.kind),
      ]).filter(Boolean))),
    card("Every extractor, on this text",
      "What each would return independently, useful for seeing which link to " +
      "add, and in what order.",
      table(["extractor", "result", "in this chain"], X.extractors, r => [
        el("td", { class: "m" }, r.extractor),
        r.result === null
          ? el("td", {}, el("span", { class: "null" }, r.note || "no match"))
          : el("td", { class: "m" }, String(r.result).slice(0, 70)),
        el("td", {}, el("span", { class: "tag " + (r.in_chain ? "accent" : "dim") },
          r.in_chain ? "yes" : "no"))])));
}

/* --------------------------------------------------------------------------
   Profile report
   -------------------------------------------------------------------------- */
function viewProfileReport() {
  const w = el("div", {}, head("Analyse", "Profile report",
    "The weighted composite, the frontier, and the paired test that decides " +
    "whether the leaderboard order means anything."));

  const runs = profileRuns();
  w.append(card("Source", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Profile"),
      el("select", { onchange: e => loadProfileReport({ profile: e.target.value }) },
        [el("option", { value: "" }, "pick one"),
         ...S.profiles.filter(p => !p.invalid).map(p =>
           el("option", { value: p.name, selected: p.name === S.pProfile }, p.name))])),
    el("div", { class: "f" }, el("label", {}, "Or one run"),
      el("select", { onchange: e => loadProfileReport({ run_id: e.target.value }) },
        [el("option", { value: "" }, "none"),
         ...runs.map(r => el("option", { value: r.run_id, selected: r.run_id === S.pRun },
           r.run_id + " · " + r.subject))])),
    el("div", { class: "f" }, el("label", {}, "Metric"),
      el("input", { type: "text", value: S.pMetric,
        onchange: e => { S.pMetric = e.target.value || "accuracy"; loadProfileReport({}); } })))));

  if (!runs.length) {
    w.append(empty("No profile runs in this store",
      "Start one on the Profile run screen; it works with the fake: models too."));
    return w;
  }
  const R = S.pResults;
  if (!R) { w.append(el("p", { class: "note" }, "Select a profile or a run.")); return w; }
  if (R.error) { w.append(errMsg(R.error)); return w; }
  // A profile resolves to its newest run; runs are never pooled. Say so, and
  // name the others, so a reader knows which attempt these numbers are.
  if (R.other_runs && R.other_runs.length) w.append(el("div", { class: "msg info" },
    `Showing the newest run for this profile, ${R.run_id}. ${R.other_runs.length} ` +
    `earlier run(s) exist (${R.other_runs.join(", ")}) and are not pooled into ` +
    `these numbers: runs differ in apparatus and dataset, and pooling would double ` +
    `up item ids in the paired test. Pick one under "Or one run" to view it.`));

  w.append(kpis([
    [R.profile || "n/a", "profile", "accent"],
    [String(R.models.length), "models"],
    [R.n_rows.toLocaleString(), "rows"],
    [term(R.metric), el("span", {}, "metric", tid(R.metric, true))],
  ]));

  // The CI table is the honest input to the accuracy chart: it carries the
  // interval, which is the whole reason this chart exists rather than a bar.
  if (R.ci) charted(w, R.metric, Charts.accuracyCI(R.ci, { metric: R.metric }));
  if (R.pareto) charted(w, "Quality against cost", Charts.qualityCost(
    R.pareto.map(p => ({ model: p.model, accuracy: p.accuracy,
                         cost: p.cost, on_frontier: !!p.on_frontier }))));
  if (R.pareto && R.pareto.some(p => p.latency !== null && p.latency !== undefined))
    charted(w, "latency_ms", Charts.latencyRange(R.pareto.map(p => ({
      model: p.model, p50: p.latency, p95: p.latency_p95 ?? p.latency }))));

  if (R.composite && R.composite.length)
    charted(w, "Metric profile", Charts.metricProfile(R.composite,
      (R.active_metrics || []).filter(m => m in (R.composite[0] || {})),
      { lowerIsBetter: Object.entries(R.weights || {})
          .filter(([, v]) => v < 0).map(([k]) => k) }));

  // The paired test as a grid: every pair, and whether the gap survived the
  // correction. Drawn from the significance rows the API already computed.
  if (R.significance && R.significance.length)
    charted(w, "Which pairs are separable", Charts.pairwiseMatrix(R.significance));
  // Cost per correct answer, computed once in the library (aggregate.cost_per_correct)
  // and only drawn here.
  if (R.cost_per_correct && R.cost_per_correct.length)
    charted(w, "cost_per_correct_answer", Charts.costPerCorrect(R.cost_per_correct));

  if (R.composite) w.append(card(termLabel("composite"),
    "Deterministic arithmetic from the profile's own weights: " +
    Object.entries(R.weights).map(([k, v]) => `${term(k)} (${k}) = ${v}`).join(", ") +
    ". A negative weight means lower is better.",
    autoTable(R.composite)));
  else if (R.composite_error) w.append(errMsg(R.composite_error));

  if (R.pareto) w.append(card("Pareto frontier",
    "Models not beaten on all of accuracy, cost and latency. When the top " +
    "models tie on quality, this is where the decision lives.", autoTable(R.pareto)));

  if (R.ci) w.append(card(el("span", {}, "95% confidence interval for ", termLabel(R.metric)),
    "Per-model intervals. Two overlapping intervals do NOT mean the difference " +
    "is insignificant; only the paired test below settles that.", autoTable(R.ci)));

  const sig = significanceCard({
    significance: R.significance, significance_error: R.significance_error,
    power: R.power,
  });
  if (R.significance_error) sig.append(el("button", {
    class: "btn ghost sm",
    onclick: () => { S.pUnpaired = true; loadProfileReport({}); },
  }, "Compare on shared items only, reporting the loss"));
  for (const [k, n] of Object.entries(R.power_suggestions || {}))
    sig.append(el("p", { class: "note" },
      "to detect a gap of " + k.split("_")[1] + ": ~" + n + " paired items"));
  w.append(sig);

  if (R.tuning_gain) w.append(card("Tuning gain (adapted − baseline)",
    "Which models benefit most from tuning, under an equal budget.",
    autoTable(R.tuning_gain)));

  const DIAGS = [
    ["error_attribution", "Error attribution",
     "Why items failed. rate_limit means lower your concurrency; context_length " +
     "means lower your k; content_filter is a finding about the model."],
    ["truncation", "Truncated answers",
     "A config problem that looks exactly like a quality finding: their " +
     "completeness and citation scores were measured mid-sentence."],
    ["estimated_usage", "ESTIMATED token usage",
     "The cost column is not fully measured. Only reachable under an explicit " +
     "opt-in (I3, all spend is metered from the provider's own usage counts)."],
  ];
  for (const [key, title, why] of DIAGS)
    if (R[key]) w.append(card(title, why, autoTable(R[key])));
  return w;
}

async function loadProfileReport(opts) {
  if ("profile" in opts) { S.pProfile = opts.profile; S.pRun = ""; }
  if ("run_id" in opts) { S.pRun = opts.run_id; S.pProfile = ""; }
  S.view = "preport"; S.pResults = null; render();
  if (!S.pProfile && !S.pRun) return;
  const q = new URLSearchParams({ metric: S.pMetric });
  if (S.pProfile) q.set("profile", S.pProfile);
  if (S.pRun) q.set("run_id", S.pRun);
  if (S.pUnpaired) q.set("allow_unpaired", "1");
  try { S.pResults = await api("/api/profile-results?" + q); }
  catch (e) { S.pResults = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Decide
   -------------------------------------------------------------------------- */
function viewDecide() {
  const w = el("div", {}, head("Analyse", "Decide",
    "Not 'which model scored highest'; nobody gets to make that decision. " +
    "'Which is the cheapest model that clears our bar at our volume.'"));

  if (!profileRuns().length) {
    w.append(empty("No profile runs in this store",
      "The decision layer needs a run carrying both cost and quality columns."));
    return w;
  }

  w.append(card("Constraints", "One per line, e.g. accuracy>=0.90 or latency_p95_ms<=2000.",
    el("div", { class: "row" },
      el("div", { class: "f" }, el("label", {}, "Profile"),
        el("select", { onchange: e => { S.dProfile = e.target.value; } },
          S.profiles.filter(p => !p.invalid).map(p =>
            el("option", { value: p.name, selected: p.name === S.dProfile }, p.name)))),
      el("div", { class: "f" }, el("label", {}, "Optimise"),
        el("input", { type: "text", value: S.dOptimise,
          onchange: e => { S.dOptimise = e.target.value; } })),
      el("div", { class: "f" }, el("label", {}, "Queries / day"),
        el("input", { type: "number", min: "1", value: S.dQpd,
          onchange: e => { S.dQpd = +e.target.value || 10000; } }))),
    el("div", { class: "f", style: "margin-top:14px" }, el("label", {}, "Require"),
      el("textarea", { style: "min-height:76px",
        oninput: e => { S.dRequire = e.target.value; } }, S.dRequire)),
    el("div", { class: "row", style: "margin-top:14px" },
      el("button", { class: "btn", onclick: runDecide }, "Decide"))));

  const D = S.decision;
  if (!D) return w;
  if (D.error) { w.append(errMsg(D.error)); return w; }

  w.append(kpis([
    [D.recommended || "none", "recommended", D.recommended ? "ok" : "crit"],
    [D.qualified + "/" + D.considered, "qualified"],
    [term(D.optimise), el("span", {}, "optimised for", tid(D.optimise, true))],
    [D.qpd.toLocaleString(), "queries / day"],
  ]));
  if (D.constraints.length)
    w.append(el("p", { class: "note" }, "Constraints: " + D.constraints.join(", ")));
  if (!D.qualified) w.append(el("div", { class: "msg warn" },
    "No model qualifies. The explanation below says which constraint each one " +
    "failed and by how much; that is the actionable form."));
  if (D.explanation) w.append(card("Why", null, el("pre", { class: "out" }, D.explanation)));
  if (D.projection.length) {
    const cleared = new Set((D.shortlist || []).map(r => r.model));
    charted(w, "Cost at volume", Charts.volumeCost(D.projection.map(r => ({
      model: r.model, daily_usd: r.daily_usd, clears: cleared.has(r.model) })),
      { qpd: D.qpd }));
  }
  if (D.projection.length) w.append(card("Cost at volume",
    "Measured per-query cost extrapolated to production. Judge spend is excluded " +
    "on purpose: it is evaluation infrastructure, not serving.",
    autoTable(D.projection)));
  if (D.headroom.length) w.append(card("What extra quality costs",
    "Turns 'the big model is 3 points better' into '3 points for $1,850 a month', " +
    "which is the form the decision is actually made in.", autoTable(D.headroom)));
  return w;
}

async function runDecide() {
  const q = new URLSearchParams();
  if (S.dProfile) q.set("profile", S.dProfile);
  q.set("optimise", S.dOptimise || "cost_usd");
  q.set("qpd", String(S.dQpd));
  for (const line of (S.dRequire || "").split("\n").map(s => s.trim()).filter(Boolean))
    q.append("require", line);
  try { S.decision = await api("/api/decide?" + q); }
  catch (e) { S.decision = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Gate
   -------------------------------------------------------------------------- */
function viewGate() {
  const w = el("div", {}, head("Analyse", "Regression gate",
    "Fails the build when quality drops or cost rises, and only on differences " +
    "that are statistically real, so it does not flap on noise."));

  if (S.allRuns.length < 2) {
    w.append(empty("Needs two runs",
      "The gate compares a candidate against a baseline. Run the same benchmark " +
      "twice to try it."));
    return w;
  }
  const opts = sel => S.allRuns.map(r =>
    el("option", { value: r.run_id, selected: r.run_id === sel },
      r.run_id + " · " + r.subject));

  w.append(card("Runs", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Baseline"),
      el("select", { onchange: e => { S.gBase = e.target.value; } }, opts(S.gBase))),
    el("div", { class: "f" }, el("label", {}, "Candidate"),
      el("select", { onchange: e => { S.gCand = e.target.value; } }, opts(S.gCand))),
    el("div", { class: "f" }, el("label", {}, "Metric"),
      el("input", { type: "text", value: S.gMetric,
        onchange: e => { S.gMetric = e.target.value || "accuracy"; } })),
    el("div", { class: "f" }, el("label", {}, "Tolerance"),
      el("input", { type: "number", step: "0.005", value: S.gTol,
        onchange: e => { S.gTol = +e.target.value; } }))),
    el("div", { class: "row", style: "margin-top:14px" },
      el("button", { class: "btn", onclick: runGate }, "Run gate"))));

  const G = S.gateResult;
  if (!G) return w;
  if (G.error) { w.append(errMsg(G.error)); return w; }

  w.append(el("div", { class: "msg " + (G.passed ? "ok" : "crit") },
    (G.passed ? "GATE PASSED" : "GATE FAILED") + " (exit " + G.exit_code + ")"));
  if (G.checks.length)
    charted(w, "Candidate against baseline", Charts.deltaDumbbell(G.checks.map(c => ({
      metric: c.metric, baseline: c.baseline, candidate: c.candidate,
      tolerance: S.gTol, regressed: !c.passed,
      lower_is_better: /cost|latency|ttft|tokens/.test(String(c.metric)) })),
      { baselineLabel: "baseline " + (G.baseline || ""), candidateLabel: "candidate " + (G.candidate || "") }));
  if (G.checks.length) w.append(card("Checks",
    "'within noise (not significant)' is the reason that stops a team tuning " +
    "the tolerance until the build goes green.",
    table(["model", "metric", "baseline", "candidate", "delta", "passed", "note"],
      G.checks, c => [
        el("td", { class: "m" }, c.model), el("td", { class: "m" }, c.metric),
        numTd(c.baseline), numTd(c.candidate), numTd(c.delta),
        el("td", {}, el("span", { class: "tag " + (c.passed ? "ok" : "crit") },
          c.passed ? "yes" : "NO")),
        el("td", { class: "m" }, c.note || "")])));
  if (G.report) w.append(card("Report", null, el("pre", { class: "out" }, G.report)));
  return w;
}

async function runGate() {
  const q = new URLSearchParams({
    baseline: S.gBase, candidate: S.gCand,
    metric: S.gMetric || "accuracy", tolerance: String(S.gTol),
  });
  try { S.gateResult = await api("/api/gate?" + q); }
  catch (e) { S.gateResult = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Arena
   -------------------------------------------------------------------------- */
function viewArena() {
  const w = el("div", {}, head("Analyse", "Arena",
    "Pairwise Elo and head-to-head win rates. Each pair is judged in both " +
    "orders, because a judge shown the same two answers the other way round " +
    "does not always agree with itself."));

  if (!S.allRuns.length) { w.append(empty("No runs yet", "Run something first.")); return w; }

  w.append(card("Run", null, el("div", { class: "row" },
    el("div", { class: "f" }, el("label", {}, "Run"),
      el("select", { onchange: e => loadArena(e.target.value) },
        [el("option", { value: "" }, "pick one"),
         ...S.allRuns.map(r => el("option", { value: r.run_id, selected: r.run_id === S.aRun },
           r.run_id + " · " + r.subject))])))));

  const A = S.arena;
  if (!A) return w;
  if (A.error) { w.append(errMsg(A.error)); return w; }
  if (!A.available) {
    w.append(el("div", { class: "msg info" }, A.reason));
    w.append(el("p", { class: "note" },
      "A dashboard button that quietly bills per click is a bad button, so " +
      "arena judging stays a deliberate CLI command."));
    return w;
  }
  w.append(kpis([[String(A.comparisons), "comparisons", "accent"]]));
  if (A.elo.length) w.append(card("Elo", null, autoTable(A.elo)));
  if (A.win_rates.length) {
    const key = "index" in (A.win_rates[0] || {}) ? "index" : Object.keys(A.win_rates[0])[0];
    const models = A.win_rates.map(r => String(r[key]));
    const rates = {};
    for (const r of A.win_rates) {
      rates[String(r[key])] = {};
      for (const m of models) if (m in r) rates[String(r[key])][m] = r[m];
    }
    charted(w, "Head-to-head win rates", Charts.winMatrix({ models, rates }));
  }
  if (A.win_rates.length) w.append(card("Head-to-head win rates", null, autoTable(A.win_rates)));
  return w;
}

async function loadArena(runId) {
  S.aRun = runId; S.arena = null; render();
  if (!runId) return;
  try { S.arena = await api("/api/arena?run_id=" + encodeURIComponent(runId)); }
  catch (e) { S.arena = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Catalogue + Providers
   -------------------------------------------------------------------------- */
function viewCatalogue() {
  const w = el("div", {}, head("Reference", "Benchmark catalogue",
    "Every spec on disk. One without an adapter is a half-built benchmark that " +
    "fails only when someone selects it, so it is listed, not hidden."));
  w.append(table(["id", "family", "version", "mode", "chance_level", "licence",
                  "commercial_use", "adapter", "spec_hash"], S.benchmarks, b => [
    el("td", { class: "m" }, b.id),
    el("td", { class: "m" }, b.family || "n/a"),
    el("td", { class: "n" }, b.version ?? "n/a"),
    el("td", { class: "m" }, b.hashed ? b.hashed.mode : "n/a"),
    el("td", { class: "n" }, b.hashed ? b.hashed.chance_level : "n/a"),
    el("td", { class: "m" }, b.licence || "n/a"),
    el("td", {}, el("span", { class: "tag " + (b.invalid ? "dim" : b.commercial_use ? "ok" : "warn") },
      b.invalid ? "n/a" : b.commercial_use ? "yes" : "ack required")),
    el("td", {}, el("span", { class: "tag " + (b.has_adapter ? "ok" : "crit") },
      b.has_adapter ? "yes" : "MISSING")),
    el("td", { class: "m" }, b.spec_hash ? b.spec_hash.slice(0, 12) : "n/a"),
  ]));
  w.append(el("p", { class: "note" },
    "Adding one is a YAML spec, an adapter module, a registry line and a " +
    "fixture, never the runner, the store, the stats layer or the reporter."));
  return w;
}

function viewProviders() {
  const w = el("div", {}, head("Reference", "Providers",
    "Capability comes from the endpoint table and the adapter agreeing, so the " +
    "two cannot drift. A provider that genuinely cannot embed simply does not " +
    "define the method."));
  const withKey = S.providers.filter(p => p.key_present).length;
  w.append(kpis([
    [String(S.providers.length), "known providers"],
    [String(withKey), "reachable now", withKey ? "ok" : "warn"],
    [String(S.providers.filter(p => p.local).length), "local / offline"],
    [String(S.providers.filter(p => p.rerank).length), "with rerank"],
  ]));
  w.append(table(["provider", "key env", "key", "embeddings", "rerank", "seed", "local", "base url"],
    S.providers, p => [
      el("td", { class: "m" }, p.provider),
      el("td", { class: "m" }, p.api_key_env || "none"),
      el("td", {}, el("span", { class: "tag " + (p.key_present ? "ok" : "dim") },
        p.key_present ? "present" : "absent")),
      el("td", {}, el("span", { class: "tag " + (p.embeddings ? "ok" : "dim") },
        p.embeddings ? "yes" : "no")),
      el("td", {}, el("span", { class: "tag " + (p.rerank ? "ok" : "dim") },
        p.rerank ? "yes" : "no")),
      el("td", {}, el("span", { class: "tag " + (p.seed ? "ok" : "dim") }, p.seed ? "yes" : "no")),
      el("td", {}, el("span", { class: "tag " + (p.local ? "accent" : "dim") },
        p.local ? "yes" : "no")),
      el("td", { class: "m" }, p.base_url || "default"),
    ]));
  w.append(el("p", { class: "note" },
    "Embeddings, reranking and judging are fixed apparatus (the machinery " +
    "around the model under test), pinned in models.yaml rather than " +
    "following that model. If two models saw different retrieved passages, " +
    "the run would measure the embedders, not the models (I2, apparatus is " +
    "pinned)."));
  return w;
}

/* --------------------------------------------------------------------------
   RAG readiness
   -------------------------------------------------------------------------- */
function viewRag() {
  const w = el("div", {}, head("Before you spend", "Retrieval",
    "A RAG profile with an empty index and a RAG profile that is ready look " +
    "identical right up until the run produces uniformly terrible numbers. " +
    "This screen is the difference."));

  const R = S.ragStatus;
  if (!R) { w.append(el("p", { class: "note" }, "Loading…")); return w; }
  if (R.error) { w.append(errMsg(R.error)); return w; }

  const ready = R.profiles.filter(p => p.ready).length;
  w.append(kpis([
    [R.vector_store, "vector store", R.vector_store === "none" ? "crit" : "accent"],
    [R.reachable ? "yes" : "no", "reachable", R.reachable ? "ok" : "crit"],
    [String(R.profiles.length), "RAG profiles"],
    [String(ready), "ingested", ready ? "ok" : "warn"],
  ]));

  if (!R.reachable) w.append(el("div", { class: "msg crit" },
    R.vector_store === "none"
      ? "No vector store configured. Set `qdrant_path` in configs/run.yaml for " +
        "embedded mode (a folder: no Docker, no server, no ports), or point " +
        "`qdrant_url` at a running instance."
      : "Configured for a Qdrant server that is not answering. Start it, or " +
        "switch to embedded mode with `qdrant_path`."));

  w.append(card("Profiles",
    "A run against an empty index retrieves nothing and scores every item as a " +
    "model failure, so the dashboard refuses to start one.",
    table(["profile", "corpus", "chunks", "embedder", "collection", "status"],
      R.profiles, p => [
        el("td", { class: "m" }, p.profile),
        el("td", {}, el("span", { class: "tag " + (p.corpus_present ? "ok" : "warn") },
          p.corpus_present ? "present" : "missing")),
        numTd(p.chunks_indexed, 0),
        el("td", { class: "m" }, p.embedding_model || "none"),
        el("td", { class: "m" }, (p.collection || "none").slice(0, 46)),
        el("td", {}, el("span", { class: "tag " + (p.ready ? "ok" : "crit") },
          p.ready ? "ready" : "not ingested")),
      ])));

  w.append(el("p", { class: "note" }, R.note));
  w.append(card("Ingest",
    "Ingest is a CLI command, not a button: it rewrites the index every " +
    "profile shares, and on embedded Qdrant a second ingest currently appends " +
    "rather than replacing (open debt item R-18 in docs/DEBT.md), so it should " +
    "be a deliberate act.",
    el("pre", { class: "out" },
      "python main.py ingest --profile <name> --qdrant-path workspace/qdrant")));
  return w;
}

async function loadRagStatus() {
  try { S.ragStatus = await api("/api/rag-status"); }
  catch (e) { S.ragStatus = { error: e.message }; }
  render();
}


/* --------------------------------------------------------------------------
   Connections: Together, OpenRouter and the rest
   -------------------------------------------------------------------------- */
function viewConnections() {
  const w = el("div", {}, head("Reference", "Connections",
    "A key being set and a key working are different facts. Every row here " +
    "comes from a real /models request, so 'live' means the provider " +
    "answered, not that an environment variable exists."));

  const C = S.connections;
  if (!C) { w.append(el("p", { class: "note" }, "Checking…")); return w; }
  if (C.error) { w.append(errMsg(C.error)); return w; }

  const live = C.connections.filter(c => c.ok);
  w.append(kpis([
    [String(live.length), "live", live.length ? "ok" : "warn"],
    [String(C.connections.length), "checked"],
    [String(live.reduce((a, c) => a + c.n_models, 0)), "models reachable", "accent"],
    [live.length ? Math.round(Math.min(...live.map(c => c.latency_ms))) + "ms" : "n/a",
     "fastest /models"],
  ]));

  w.append(card("Providers", C.note,
    table(["provider", "status", "key", "models", "latency_ms", "detail"],
      C.connections, c => [
        el("td", { class: "m" }, c.provider),
        el("td", {}, el("span", { class: "tag " + (c.ok ? "ok" : c.key_source === "none" ? "dim" : "crit") },
          c.ok ? "live" : c.key_source === "none" ? "no key" : "failed")),
        el("td", { class: "m" }, c.key_masked || "none"),
        numTd(c.n_models, 0),
        el("td", { class: "n" }, c.latency_ms ? Math.round(c.latency_ms) + "ms" : "n/a"),
        (c.error || "").slice(0, 70)
          ? el("td", { class: "m" }, c.error.slice(0, 70))
          : el("td", {}, el("span", { class: "null" }, "none")),
      ])));

  w.append(el("hr", { class: "rule" }));

  w.append(card("Check one provider",
    "A key typed here is used for this single request and then dropped: it " +
    "is never written to a config, a cache key, the trace store or a log. The " +
    "durable place for it is your environment or a .env file (§6, the " +
    "security rules).",
    el("div", { class: "row" },
      el("div", { class: "f" }, el("label", {}, "Provider"),
        el("select", { onchange: e => { S.connProvider = e.target.value; } },
          C.listable.map(p => el("option", { value: p, selected: p === S.connProvider }, p)))),
      el("div", { class: "f", style: "flex:2" },
        el("label", {}, "API key (optional; env is used when blank)"),
        el("input", { type: "password", autocomplete: "off", spellcheck: "false",
          placeholder: "leave blank to use the environment",
          oninput: e => { S.connKey = e.target.value; } })),
      el("div", { class: "f", style: "flex:0 0 auto" }, el("label", {}, " "),
        el("button", { class: "btn", onclick: runConnectionCheck }, "Check")))));

  const D = S.connDetail;
  if (!D) return w;
  if (!D.ok) {
    w.append(errMsg(D.error || "failed"));
    if (D.hint) w.append(el("p", { class: "note" }, D.hint));
    return w;
  }

  w.append(el("div", { class: "msg ok" },
    `${D.provider} answered in ${Math.round(D.latency_ms)}ms: ` +
    `${D.n_models} model(s), key from ${D.key_source} (${D.key_masked}).`));

  const picked = S.connPicked;
  w.append(card("Catalogue",
    "Click a model to add it to a models.yaml fragment. The harness proposes; " +
    "you commit. Model choice changes results, so §7 (the working agreement) " +
    "reserves the edit for a human.",
    el("div", { class: "pills", style: "max-height:260px;overflow:auto" },
      D.models.map(m => el("button", {
        class: "pick", "aria-pressed": picked.has(m.id),
        title: m.context ? `context ${m.context}` : "",
        onclick: () => { picked.has(m.id) ? picked.delete(m.id) : picked.add(m.id); render(); },
      }, m.id)))));

  if (picked.size) {
    w.append(el("div", { class: "row" },
      el("button", { class: "btn", onclick: buildModelsYaml },
        `Build models.yaml for ${picked.size} model(s)`)));
  }
  if (S.modelsYaml) {
    w.append(card("configs/models.yaml", S.modelsYaml.note,
      el("pre", { class: "out" }, S.modelsYaml.yaml)));
  }
  return w;
}

async function loadConnections() {
  try { S.connections = await api("/api/connections"); }
  catch (e) { S.connections = { error: e.message }; }
  if (S.connections.listable && !S.connProvider) {
    const live = (S.connections.connections || []).find(c => c.ok);
    S.connProvider = live ? live.provider : S.connections.listable[0];
  }
  render();
}

async function runConnectionCheck() {
  S.connDetail = null; S.modelsYaml = null; S.connPicked = new Set(); render();
  try {
    S.connDetail = await post("/api/connection-check", {
      provider: S.connProvider, api_key: S.connKey || null,
    });
  } catch (e) { S.connDetail = { ok: false, error: e.message }; }
  // The typed key is dropped from client state too, not just from the server.
  S.connKey = "";
  render();
}

async function buildModelsYaml() {
  try {
    S.modelsYaml = await post("/api/models-yaml", {
      provider: S.connProvider, models: [...S.connPicked],
    });
  } catch (e) { S.modelsYaml = { yaml: "", note: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Guide: what this platform is, end to end.

   Written for someone who has not used it and does not yet know why it
   refuses things. The order is the order of the work: what problem it exists
   for, what happens to a single item, then the four ways an eval lies and
   what is done about each, then the tour of the screens.

   Every chart on this page is an ILLUSTRATION except the denominator one,
   which is the real live run, and that one is labelled as real, because a
   teaching chart mistaken for a result is exactly the failure this harness
   was built to prevent.
   -------------------------------------------------------------------------- */

/* A small numbered step, for the walkthrough. */
const step = (n, title, body, ...kids) => el("div", { class: "step" },
  el("div", { class: "stepn" }, String(n)),
  el("div", { class: "stepbody" },
    el("b", {}, title),
    el("p", {}, body),
    ...kids));

const FAILURES = [
  ["Measuring noise",
   "A three-point gap between two models on 200 items is usually nothing. " +
   "Report it as a win and you have made a procurement decision out of a " +
   "coin flip.",
   "Every comparison is paired at the item level and corrected across the " +
   "family. When the interval straddles zero the harness says so and " +
   "estimates how many more items would settle it. It will not print an " +
   "ordering to fill the space."],
  ["Under-counted cost",
   "The judge model, the embedder and the reranker are all billed, and they " +
   "are almost never in the number anyone quotes. A cheap model with an " +
   "expensive judge is not cheap.",
   "Every paid call is attributed to a bucket from the provider's own usage " +
   "block. Nothing is estimated unless the profile opts in, and then the row " +
   "records that it was estimated. A model with no price fails validation " +
   "rather than costing zero."],
  ["Untested failure modes",
   "Accuracy says nothing about what happens when the context contains an " +
   "instruction, when the answer is not in the corpus, or when the model is " +
   "asked to abstain.",
   "Injection, fabrication and leakage run as their own probe families with " +
   "their own rates. Refusal and over-refusal are reported together: a " +
   "model that refuses everything is not safe, it is useless."],
  ["The wrong question",
   "\"Which model scores highest\" is rarely the decision. The decision is " +
   "\"what is the cheapest model that clears our bar at our volume\".",
   "Decide takes requirements and a daily volume and answers that question " +
   "directly. Cost carries a negative weight by construction; a positive one " +
   "is rejected at config load, because that is how a leaderboard gets " +
   "fixed by flipping a sign."],
];

const SCREENS = [
  ["Prepare", "Before you spend anything.", [
    ["Preflight", "Validates a profile end to end (dataset present, every " +
     "model priced, routing resolvable), then estimates the run's cost " +
     "before it happens. A missing price stops you here rather than " +
     "surfacing as a zero in the report."],
    ["Retrieval", "The state of the RAG corpus: what is ingested, what the " +
     "embedder is, whether the index is reachable. The apparatus (the " +
     "embedder, reranker and judge around the model under test) is pinned, " +
     "so this is the thing that must not move between compared runs."],
    ["Probes", "The adversarial families: injection, fabrication, leakage. " +
     "Generated deterministically from a seed, so the same probe set can be " +
     "regenerated for any past run."],
    ["Extraction", "Paste a raw model response and watch the extraction " +
     "chain run on it, step by step. This is where you find out that your " +
     "regex misses the format your model actually uses."]]],
  ["Run", "The matrix walk.", [
    ["Profile run", "A full model × profile × pass matrix against your own " +
     "data and passes, with live cost metering and a budget ceiling."],
    ["Benchmark run", "The same machinery pointed at a public benchmark " +
     "spec. Same trace store, same cost meter, same statistics; there is no " +
     "second pipeline for benchmarks."]]],
  ["Analyse", "Reading rows, never re-running them.", [
    ["Profile report", "Leaderboard with intervals, failure rates beside " +
     "accuracy rather than inside it, and the paired test."],
    ["Benchmark results", "The same, plus chance-adjusted accuracy and " +
     "extraction-failure rate: a 25% score on four-way multiple choice is " +
     "not 25% good."],
    ["Decide", "Requirements in, a model out, with the cost at your volume. " +
     "Refuses to answer when the comparison is not separable."],
    ["Gate", "Baseline versus candidate for CI. Exit code, tolerance, and a " +
     "regression check that will not pass a partial run without being told to."],
    ["Arena", "Head-to-head item-level disagreement: where two models " +
     "actually differ, not just by how much."],
    ["Saved reports", "Frozen findings. A report captures its numbers and " +
     "stops moving, so fixing a price tomorrow does not silently rewrite " +
     "last month's conclusion."]]],
  ["Reference", "What is installed and what it costs.", [
    ["Catalogue", "Every benchmark spec: family, licence, chance level, " +
     "spec hash. Licence is a required field, and a non-commercial set needs " +
     "an explicit acknowledgement before it will run."],
    ["Providers", "Configured providers and their rate limits."],
    ["Connections", "Test an API key, list what the account can see, and " +
     "probe whether a listed model can actually be invoked; those are not " +
     "the same thing."]]],
];

function viewGuide() {
  const w = el("div", {}, head("Guide", "How this platform works",
    "An evaluation harness that walks a model × profile × pass matrix, " +
    "scores every item, meters the real spend, and produces a decision " +
    "rather than a leaderboard. This page is the whole thing, end to end."));

  // --- the thesis ----------------------------------------------------------
  const t = card("What it is for",
    "Most internal evals fail quietly. They produce a number that looks " +
    "reasonable, nobody can reproduce it, and a decision gets made on it.");
  t.append(el("p", { class: "lead" },
    "This harness is built against four specific ways that happens. Every " +
    "feature traces back to one of them, and a change that makes the tool " +
    "faster or prettier while weakening one is treated as a regression."));
  w.append(t);

  // --- the pipeline --------------------------------------------------------
  charted(w, "Pipeline", Charts.pipeline());

  const p = card("One item, eight stages",
    "The dashed line is the important part of that diagram.");
  p.append(el("div", { class: "grid2" },
    el("div", {},
      el("b", {}, "Left of the line: running"),
      el("p", { class: "note" },
        "Prompt assembly, the provider call, extraction and scoring. Only " +
        "the provider adapter may open a connection; everything else is a " +
        "pure function of its inputs. That is why a run can be replayed " +
        "from cache and why scorers can be property-tested.")),
    el("div", {},
      el("b", {}, "Right of the line: analysis"),
      el("p", { class: "note" },
        "Nothing here may touch the network. Every number in every report " +
        "is computed from rows already on disk, which is what makes a " +
        "result reproducible months later from the run directory alone."))));
  w.append(p);

  w.append(el("hr", { class: "rule" }));

  // --- the four failure modes ---------------------------------------------
  w.append(card("The four ways an eval lies",
    "Named, because a failure mode with a name is one you can check for."));
  const f = el("div", { class: "grid2" });
  for (const [name, problem, answer] of FAILURES) {
    f.append(el("div", { class: "failcard" },
      el("div", { class: "failhead" }, name),
      el("p", { class: "note" }, problem),
      el("p", { class: "fix" }, answer)));
  }
  w.append(el("div", { class: "card" }, f));

  // --- significance --------------------------------------------------------
  charted(w, "Significance", Charts.significanceDemo());
  w.append(card(null, null, el("p", { class: "note" },
    "Illustration, not a run. The test itself is exact McNemar for binary " +
    "metrics and a paired bootstrap for continuous ones, selected " +
    "automatically from the metric's type and recorded in the output, with " +
    "Holm–Bonferroni across the comparison family and the family size " +
    "printed beside the result.")));

  // --- the real finding ----------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  const real = card("What that refusal looks like in practice",
    "This one is not an illustration. It is the live Together AI run on " +
    "MMLU-Pro, and it is the clearest example of why the rules above exist.");
  real.append(Charts.pairingDemo([
    { model: "DeepSeek-V4-Flash", accuracy: 0.978, scored: 46 },
    { model: "GLM-5.3-Flash", accuracy: 0.962, scored: 52 },
    { model: "gpt-oss-120b", accuracy: 0.907, scored: 54 },
  ], { n: 60 }));
  real.append(el("div", { class: "msg warn", style: "margin-top:18px" },
    "The harness refused to rank these. Because the three models were not " +
    "scored on an identical item set, the paired test raised rather than " +
    "quietly comparing them on whatever items happened to overlap, which " +
    "would have reported a seven-point win that does not exist."));
  w.append(real);

  // --- failure is not wrongness -------------------------------------------
  charted(w, "Failure is not wrongness", Charts.scoredSplit([
    { model: "model-a", n_items: 60, n_scored: 54,
      extraction_failure_rate: 0.033, refusal_rate: 0.033, truncation_rate: 0.033 },
    { model: "model-b", n_items: 60, n_scored: 46,
      extraction_failure_rate: 0.017, refusal_rate: 0.0, truncation_rate: 0.217 },
  ]));
  w.append(card(null, null, el("p", { class: "note" },
    "Illustration. A parse failure, a truncation, a refusal and a sandbox " +
    "violation are four different events, and none of them is a wrong " +
    "answer. Folding them into the accuracy numerator is the single easiest " +
    "way to make a model look worse than it is. As the chart above shows, " +
    "excluding them without watching the denominator is the easiest way to " +
    "make one look better.")));

  // --- cost ----------------------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  charted(w, "Cost", Charts.costBuckets([
    { bucket: "generation", usd: 0.0412 },
    { bucket: "judge", usd: 0.0197 },
    { bucket: "embedding", usd: 0.0068 },
    { bucket: "rerank", usd: 0.0031 },
    { bucket: "cached", usd: 0.0009 },
  ]));
  w.append(card(null, null, el("p", { class: "note" },
    "Illustration of the shape, with realistic proportions. In that split " +
    "the judge is a third of the bill, and the judge is the line almost " +
    "every internal eval leaves out. Money is held as Decimal and rounded " +
    "only for display, and the four buckets plus cached must sum exactly to " +
    "the reported total, which is asserted by a test.")));

  // --- architecture --------------------------------------------------------
  charted(w, "Architecture", Charts.layers());

  // --- case studies --------------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  const cs = card("Worked examples: the case studies",
    "Five evaluations of language models on the work of Indian government " +
    "offices, each written up the same way and each runnable from this " +
    "platform: routing citizen grievances, translating public notices into " +
    "Hindi, drafting RTI applications, answering scheme questions from the " +
    "documents, and answering RTI-framework questions with citations. Every " +
    "dataset is fictional and illustrative; the point is the method, and the " +
    "method is the same one you would use on your own data. Beside them, an " +
    "eight-benchmark public sweep of four of the same models is frozen under " +
    "Saved reports, with its extraction-failure rates printed next to every score.");
  cs.append(el("div", { class: "row", style: "margin-top:12px" },
    el("button", { class: "btn", onclick: () => go("casestudies") }, "Open the case studies"),
    el("button", { class: "btn ghost", onclick: () => go("reports") }, "Saved reports")));
  w.append(cs);

  // --- a new attempt ------------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  const na = card("Start a new attempt",
    "The exact clicks, in order. Each step names the screen it happens on, " +
    "and every one of them is reachable from the task bar above.");
  const NEW_ATTEMPT = [
    ["Evaluate (start here)",
     "Pick the kind of task, see the data format with sample rows, and have the " +
     "profile written for you. Upload your own JSONL if you have one; it is checked " +
     "line by line before anything is written.", "evaluate"],
    ["Reference › Connections",
     "Test the provider key, then PROBE the models you intend to run: a " +
     "model the catalogue lists is not a model the account can invoke. Only " +
     "probed-invokable models are worth putting in a run.", "connections"],
    ["Prepare › Preflight",
     "Pick the profile. Validate confirms the dataset is on disk and every " +
     "model and the judge are priced. Read the estimate: it itemises judge " +
     "and embedding spend, which are the lines most evals forget.", "preflight"],
    ["Prepare › Retrieval  (RAG profiles only)",
     "Confirm the corpus is ingested and the embedder is the one the profile " +
     "pins. An empty index does not fail; it retrieves nothing and makes " +
     "every model look uniformly bad.", "rag"],
    ["Run › Profile run",
     "Choose the profile, toggle the latency lane, pick two or more models, " +
     "and Run. Progress and metered spend update live; a budget ceiling " +
     "aborts cleanly and keeps every row already written.", "runprofile"],
    ["Analyse › Profile report",
     "The interval chart first, then the metric profile, then the paired " +
     "test. If it says not separable, that is the finding; it will tell you " +
     "how many more items would settle it. Read the truncation table before " +
     "anything else: a reasoning model that ran out of tokens has an empty " +
     "answer, not a wrong one, and the table now shows how much of each " +
     "answer was hidden reasoning.", "preport"],
    ["Analyse › Saved reports",
     "Freeze the attempt with its caveats written down. The numbers stop " +
     "moving even if a price changes tomorrow, and it appears in Past " +
     "attempts on the Overview.", "reports"],
  ];
  NEW_ATTEMPT.forEach(([where, body, view], i) =>
    na.append(step(i + 1, where, body,
      el("button", { class: "btn ghost sm", onclick: () => go(view) }, "Go there"))));
  na.append(el("div", { class: "msg info", style: "margin-top:14px" },
    "Or from a terminal, the same path: python main.py validate --profile " +
    "<p> → estimate → ingest (RAG) → probes --append → run --models a b c " +
    "--budget <usd> → report --run-id <id>. The browser and the CLI write " +
    "identical TraceRows (one trace row per evaluated item) to the same store."));
  w.append(na);

  // --- the walkthrough -----------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  const walk = card("A first run, start to finish",
    "Six steps. Nothing here needs an API key except step four, and there " +
    "is a fake provider if you would rather not spend anything yet.");
  [
    [1, "Check the catalogue", "See which benchmarks are installed, what " +
     "licence each carries and what its chance level is.", "catalogue"],
    [2, "Preflight a profile", "Confirm the dataset is present and every " +
     "model has a price, then read the cost estimate before committing.",
     "preflight"],
    [3, "Test the extraction chain", "Paste a response in the shape your " +
     "model actually produces and check the chain finds the answer.",
     "extract"],
    [4, "Run the matrix", "Pick models and items and go. Cost meters live " +
     "and the budget ceiling aborts cleanly, keeping every row written.",
     "run"],
    [5, "Read the result", "Leaderboard with intervals, failure rates " +
     "beside accuracy, and the paired test, or a refusal to rank.",
     "results"],
    [6, "Freeze it", "Save a report so the finding survives a later price " +
     "change, then gate a candidate against it in CI.", "reports"],
  ].forEach(([n, title, body, view]) =>
    walk.append(step(n, title, body,
      el("button", { class: "btn ghost sm", onclick: () => go(view) },
        "Open " + title.toLowerCase().replace(/^(check|test|read) /, "")))));
  w.append(walk);

  // --- screen tour ---------------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  w.append(card("Every screen, and what it is for", null));
  for (const [section, blurb, items] of SCREENS) {
    const c = card(section, blurb);
    const grid = el("div", { class: "grid2" });
    for (const [title, body] of items) {
      grid.append(el("div", { class: "screencard" },
        el("b", {}, title), el("p", { class: "note" }, body)));
    }
    c.append(grid);
    w.append(c);
  }

  // --- the refusals --------------------------------------------------------
  w.append(el("hr", { class: "rule" }));
  const no = card("Things it will refuse to do",
    "Each of these is a deliberate stop, not a missing feature. They are " +
    "the reason to use this rather than a spreadsheet.");
  no.append(el("ul", { class: "plain refuse" },
    el("li", {}, "Rank models that were not scored on the same items; it " +
      "raises rather than silently intersecting."),
    el("li", {}, "Declare a winner when the interval straddles zero."),
    el("li", {}, "Report a cost for a model with no pricing entry."),
    el("li", {}, "Count a parse failure, a truncation or a refusal as a " +
      "wrong answer."),
    el("li", {}, "Compare two runs whose apparatus or spec hash differs."),
    el("li", {}, "Accept a composite weight that rewards being expensive."),
    el("li", {}, "Mutate a trace row after it is written."),
    el("li", {}, "Rank a run that aborted on budget, unless explicitly told " +
      "to allow partials.")));
  w.append(no);

  w.append(card("Where to look next", null,
    el("div", { class: "row" },
      el("button", { class: "btn", onclick: () => go("overview") },
        "Overview"),
      el("button", { class: "btn ghost", onclick: () => go("about") },
        "About, including what is not done"),
      el("button", { class: "btn ghost", onclick: () => go("reports") },
        "Saved reports"))));

  return w;
}

/* --------------------------------------------------------------------------
   About: what each screen does, and what it refuses to do
   -------------------------------------------------------------------------- */
const ABOUT = [
  ["The four failure modes", null, [
    ["Measuring noise",
     "Forty questions, 0.82 against 0.78, ship it. That gap is inside the " +
     "margin of error and half the time it flips on a re-run. Every screen " +
     "that shows a ranking also shows the interval, and the paired test " +
     "decides whether the order means anything."],
    ["Under-counted cost",
     "Judge, embedding and rerank calls are billed but usually not reported, " +
     "so cost reads as a fraction of the truth. Because cost carries a " +
     "NEGATIVE weight in the composite, that does not merely understate " +
     "dollars; it mis-ranks the models. Cost here is metered from real usage " +
     "blocks and split four ways."],
    ["Untested failure modes",
     "Accuracy says nothing about whether a model will obey an instruction " +
     "hidden in a retrieved document, invent an answer it should have " +
     "refused, or leak data. That is what Probes are for."],
    ["The wrong question",
     "A leaderboard says which model scored highest. Nobody gets to make that " +
     "decision. The real one is 'the cheapest model that clears our bar at " +
     "our volume', which is the Decide screen."],
  ]],

  ["Evaluate", "Start here with your own model and data.", [
    ["Evaluate",
     "Four kinds of task (label, generate, answer from documents, public benchmark), " +
     "each with its data format and sample rows, a profile written from a few choices " +
     "and validated as you type, an upload checked line by line, and the run path " +
     "in order with the matching terminal commands."],
  ]],

  ["Case studies", "Worked evaluations, written up so the method can be copied.", [
    ["Case studies",
     "Each one states the use case, who would run it, what a wrong answer " +
     "costs, the metrics and why they are weighted as they are, the caveats " +
     "that travel with the numbers, and the results of the newest run. The " +
     "figures on that screen are the Profile report's figures for the same " +
     "run; nothing is recomputed."],
    ["Indian-government use cases",
     "Five profiles built for the work of district and state offices: " +
     "grievance routing, notice translation into Hindi, RTI drafting, scheme " +
     "question answering and RTI-framework question answering. Datasets are " +
     "fictional and illustrative, in English, Hindi and Hinglish, and one " +
     "new metric (native script ratio) keeps 'fluent but in the wrong " +
     "script' apart from 'wrong'."],
  ]],

  ["Before you spend", "The cheap half of the workflow. Everything here is " +
   "free and most of it is instant.", [
    ["Preflight",
     "Validate loads the profile, counts the evalset, and checks every gold " +
     "passage id resolves to a real chunk; a gold id that matches nothing " +
     "makes that item's retrieval metrics meaningless. Estimate then " +
     "forecasts the bill INCLUDING judge calls, which on a judge-scored " +
     "profile are routinely the largest line."],
    ["Retrieval",
     "Which RAG profiles have an ingested corpus. A profile with an empty " +
     "index and one that is ready look identical right up until the run " +
     "returns uniformly terrible numbers, so a run against an empty index is " +
     "refused rather than started."],
    ["Probes",
     "Five adversarial families derived from your own evalset: injection, " +
     "unanswerable, noise, paraphrase, positional. Derived rather than " +
     "hand-written because a probe set that does not track your real corpus " +
     "tests nothing about your real system. Generation is pure and " +
     "deterministic given the seed, so it costs nothing."],
    ["Datasets",
     "Public benchmarks are fetched from HuggingFace's datasets server into a " +
     "local cache, with a sha256 recorded in the spec. The checksum is the " +
     "point: a dataset that changes under you produces a silently changed " +
     "score, and spec_hash (the format fingerprint every run records) covers " +
     "the checksum so it cannot happen quietly. A non-commercial licence is " +
     "a gate, not a warning."],
    ["Extraction",
     "Paste raw model output and watch the benchmark's extraction chain run, " +
     "link by link. This is here because extraction is quietly the whole " +
     "benchmark: the gap between a reported 0.61 and a reported 0.78 is " +
     "usually not the model, it is whether the harness could find the answer " +
     "in the prose."],
  ]],

  ["Run", "Both paths write the same TraceRows (one trace row per evaluated " +
   "item) into the same store, meter through the same cost meter, and are " +
   "analysed by the same statistics.", [
    ["Profile run",
     "The original apparatus (the retrieval, prompting and grading machinery " +
     "around the model under test): retrieve, prompt, generate, score, meter. " +
     "A profile describes one workload: its evalset, its retrieval settings, " +
     "which metrics are active and how they are weighted."],
    ["Benchmark run",
     "The §10 benchmark subsystem: public or private benchmarks through that " +
     "same apparatus. Every model sees the identical item set in the " +
     "identical order, sampled deterministically from the seed; that is what " +
     "makes the comparison paired."],
    ["The fake: models",
     "A deterministic local provider. No key, no network, real token counts " +
     "so the cost meter works end to end. It exists so the run path can be " +
     "exercised before any key does, but its scores are not results."],
  ]],

  ["Analyse", null, [
    ["Profile report",
     "Weighted composite from the profile's own weights, the Pareto frontier, " +
     "bootstrap confidence intervals, the paired significance test, power, " +
     "tuning gain, and the diagnostics: error attribution, truncation rate, " +
     "and how much of the cost was estimated rather than measured."],
    ["Benchmark results",
     "The leaderboard with two columns most benchmark reports omit: " +
     "chance-adjusted accuracy (25% on four-way multiple choice is not '25% " +
     "good') and the count of items EXCLUDED from the denominator."],
    ["Decide",
     "Constraint-based selection. Give it 'accuracy >= 0.9' and a daily " +
     "volume and it returns the cheapest qualifying model, the projected " +
     "monthly spend, and the price of each extra accuracy point. When nothing " +
     "qualifies it says which constraint each model failed and by how much. " +
     "'No model qualifies' is not actionable; 'everything failed latency, " +
     "relax it to 2.4s' is."],
    ["Gate",
     "The CI regression gate. Fails a build when quality drops or cost rises, " +
     "and only on differences that are statistically real, so it does not " +
     "flap on noise."],
    ["Arena",
     "Pairwise Elo and head-to-head win rates, each pair judged in both " +
     "orders because a judge shown the same two answers reversed does not " +
     "always agree with itself. It reports stored judgements rather than " +
     "offering a button, because judging bills per pair."],
  ]],

  ["What it refuses to do", "The refusals are the product.", [
    ["Refuses to compare unpaired runs",
     "If two models were not scored on the same items, comparing them selects " +
     "the item set by one model's failures. The harness raises and names the " +
     "missing items rather than quietly intersecting them."],
    ["Refuses to bill a missing usage block as free",
     "A provider that returns no token counts raises. Substituting zero would " +
     "make that model look free, and because cost is negatively weighted, " +
     "free promotes it up the leaderboard."],
    ["Refuses to call a parse failure a wrong answer",
     "An answer the extractor could not read scores null, not zero: absent " +
     "from both numerator and denominator, counted instead in " +
     "extraction_failure_rate."],
    ["Refuses to rank a non-significant difference",
     "It reports 'not separable at n=X' and says how many more items you " +
     "would need."],
    ["Refuses to average across formats",
     "spec_hash, the format fingerprint, covers the prompt template, few-shot " +
     "count, decoding params and extraction chain. Two runs of the 'same' " +
     "benchmark under different " +
     "formats are not the same benchmark, and the reporter will not mix them."],
  ]],
];

/* The enforcement register, current as of the last audit of the code, not
   the Phase 0 baseline. Several of these moved from "unenforced" to
   "enforced" during that work, and the two that have not are named as such
   rather than rounded up. */
const INVARIANTS = [
  ["I1", "Paired comparison", "code",
   "paired_values raises UnpairedItemsError rather than intersecting"],
  ["I2", "Apparatus pinned", "code",
   "apparatus_hash computed on both run paths, compared in comparable_with"],
  ["I3", "All spend metered", "code",
   "a missing usage block raises MissingUsageError; no silent zero"],
  ["I4", "Cost negatively weighted", "code",
   "a positive weight on cost or latency is rejected at config load"],
  ["I5", "Paired, corrected significance", "code",
   "exact McNemar or paired bootstrap, selected automatically, Holm-corrected"],
  ["I6", "Non-significant stays non-significant", "code",
   "refuses to rank; states how many more items are needed"],
  ["I7", "Failure is not wrongness", "convention",
   "extraction/refusal/truncation/format rates exist; over-refusal and " +
   "sandbox violation are documented but not computed"],
  ["I8", "Traces append-only", "code",
   "one Parquet part per checkpoint; rows are never rewritten"],
  ["I9", "Reproducibility manifest", "code",
   "RunManifest captured before the first call and written beside the traces"],
  ["I10", "Determinism where claimed", "code",
   "probes, sampling and ordering are pure functions of (seed, dataset_hash)"],
  ["I11", "Config is the spine", "convention",
   "profiles, models, pricing and specs are YAML; a provider still needs an " +
   "adapter, which the contract permits"],
  ["I12", "Secrets and canaries never leak", "unenforced",
   "no redaction processor exists, and canaries are stored as plaintext in " +
   "assembled_prompt; the largest open correctness gap"],
];

/* Counted in surfaces, and deliberately conservative: "verified live" means a
   real call to a real provider was made and the result inspected during this
   work, not that a test exercised the code path. */
const VERIFIED = [
  { area: "benchmark specs", live: 3, offline: 3, unexercised: 0 },
  { area: "bench adapters", live: 3, offline: 1, unexercised: 0 },
  { area: "provider adapters", live: 2, offline: 1, unexercised: 1 },
  { area: "§10.4 families", live: 2, offline: 1, unexercised: 9 },
];

function viewAbout() {
  const w = el("div", {}, head("Reference", "About this harness",
    "What each screen does, why it exists, and what it deliberately will not " +
    "do. Every refusal below is there because the alternative produces a " +
    "number that looks fine and is wrong."));

  w.append(card("New here?",
    "This page is the reference. The Guide is the explanation: what the " +
    "platform is for, what happens to one item end to end, and why it " +
    "refuses the things it refuses.",
    el("div", { class: "row" },
      el("button", { class: "btn", onclick: () => go("guide") },
        "Read the Guide"))));

  // The two charts that report on the harness rather than on a model. They
  // sit above the feature tour on purpose: what is enforced and what has
  // actually been run matter more than what exists.
  charted(w, "Invariants (the rules the harness must never break)", Charts.invariantStatus(
    INVARIANTS.map(([id, name, status, note]) => ({ id, name, status, note }))));
  charted(w, "Verification", Charts.verification(VERIFIED));

  w.append(el("hr", { class: "rule" }));

  for (const [section, blurb, items] of ABOUT) {
    const c = card(section, blurb);
    const grid = el("div", { class: "grid2" });
    for (const [title, body] of items) {
      grid.append(el("div", {
        style: "border-left:2px solid var(--hair-2);padding-left:14px",
      },
        el("div", { style: "font-weight:600;font-size:13px;margin-bottom:5px" }, title),
        el("div", { style: "font-size:12.5px;color:var(--ink-2)" }, body)));
    }
    c.append(grid);
    w.append(c);
  }

  w.append(el("hr", { class: "rule" }));
  w.append(card("Honestly, what is not done",
    "Kept here rather than in a changelog, because a platform that lists only " +
    "its features is the thing this harness was built to distrust.",
    el("ul", { class: "plain", style: "margin:0;padding-left:18px;color:var(--ink-2);font-size:13px" },
      el("li", {}, "No statistical simulation suite. The paired tests are " +
        "asserted by unit tests, not by a calibration run; the false-positive " +
        "rate under the null has never been measured. Largest open gap."),
      el("li", {}, "The RAG path is not offline: sparse retrieval downloads a " +
        "BM25 model from huggingface.co at ingest and at every query."),
      el("li", {}, "Re-ingesting a corpus on embedded Qdrant doubles it. The " +
        "delete reports success and the points come back."),
      el("li", {}, "Three benchmarks now run on the real public datasets " +
        "(GSM8K, MMLU-Pro, ARC-Challenge) fetched from HuggingFace with " +
        "checksums. The other three still use synthetic fixtures, and their " +
        "numbers are not comparable to published ones."),
      el("li", {}, "configs/pricing.yaml is stale for 2 of 5 Together models. " +
        "Cost is negatively weighted, so a stale price mis-ranks; left " +
        "unedited because a pricing change is a results change (debt item " +
        "R-19 in docs/DEBT.md)."),
      el("li", {}, "A provider listing a model does not mean the account can " +
        "call it: 8 of 10 probed Together models returned 'non-serverless'. " +
        "Probe before trusting a catalogue (debt item R-20)."),
      el("li", {}, "No sandbox, so no code-execution benchmarks. No " +
        "contamination probes. Log-likelihood scoring validates but has no " +
        "code path."))));
  w.append(el("p", { class: "note" },
    "Full detail, with reproductions, in docs/DEBT.md and docs/INVENTORY.md."));
  return w;
}


/* --------------------------------------------------------------------------
   Reports: findings, frozen and kept
   -------------------------------------------------------------------------- */
function viewReports() {
  const w = el("div", {}, head("Analyse", "Saved reports",
    "A run's rows live in the trace store; a report is the other thing: what " +
    "was found, on which runs, under which prices, on a given day. Saved " +
    "reports are frozen: fix a pricing entry tomorrow and last month's report " +
    "keeps saying what it said, because a record that re-derives itself is " +
    "not a record."));

  // --- build ---------------------------------------------------------------
  const picked = S.repPicked;
  w.append(card("New report",
    "Pick the runs it should cover. Each contributes its leaderboard, its " +
    "intervals, its paired test and its manifest.",
    el("div", { class: "pills" }, S.allRuns.map(r => el("button", {
      class: "pick", "aria-pressed": picked.has(r.run_id),
      onclick: () => { picked.has(r.run_id) ? picked.delete(r.run_id) : picked.add(r.run_id); render(); },
    }, `${r.subject} · ${r.run_id}`))),
    el("div", { class: "row", style: "margin-top:16px" },
      el("div", { class: "f", style: "flex:2" }, el("label", {}, "Title"),
        el("input", { type: "text", value: S.repTitle,
          oninput: e => { S.repTitle = e.target.value; } })),
      el("div", { class: "f" }, el("label", {}, "Provider"),
        el("input", { type: "text", value: S.repProvider,
          oninput: e => { S.repProvider = e.target.value; } }))),
    el("div", { class: "f", style: "margin-top:14px" },
      el("label", {}, "Notes and caveats, one per line"),
      el("textarea", { style: "min-height:84px",
        oninput: e => { S.repNotes = e.target.value; } }, S.repNotes)),
    el("div", { class: "row", style: "margin-top:14px" },
      el("button", { class: "btn", disabled: !picked.size, onclick: saveReport },
        `Save report from ${picked.size} run(s)`))));

  if (S.repError) w.append(errMsg(S.repError));

  // --- list ----------------------------------------------------------------
  if (!S.reports) { w.append(el("p", { class: "note" }, "Loading…")); return w; }
  if (!S.reports.length) {
    w.append(empty("No saved reports yet",
      "Run something, then freeze it here."));
    return w;
  }

  w.append(el("hr", { class: "rule" }));
  w.append(card("Saved", null,
    table(["title", "created", "runs", "models", "total_cost_usd", "headline", ""],
      S.reports, r => [
        el("td", { class: "m" }, r.title),
        el("td", { class: "m" }, new Date(r.created * 1000).toLocaleString()),
        numTd(r.n_runs, 0),
        el("td", { class: "m" }, (r.models || []).length),
        numTd(r.total_cost_usd, 5),
        el("td", { class: "m" }, (r.summary || "").slice(0, 70)),
        el("td", {}, el("button", { class: "btn ghost sm",
          onclick: () => openReport(r.id) }, "open")),
      ])));

  const R = S.reportOpen;
  if (!R) return w;

  w.append(el("hr", { class: "rule" }));
  const hdr = card(R.title,
    `${R.created_human}${R.provider ? " · " + R.provider : ""} · ` +
    `${R.run_ids.length} run(s) · $${(R.total_cost_usd || 0).toFixed(5)} metered`);
  hdr.append(el("p", { class: "note" }, R.summary));
  if (R.notes) hdr.append(el("pre", { class: "out", style: "margin-top:12px" }, R.notes));
  if (R.caveats && R.caveats.length) {
    // Caveats sit ABOVE the numbers on purpose. A limitation printed under a
    // table is a limitation nobody reads before quoting the table.
    const c = el("div", { class: "msg warn" }, "Read before quoting these numbers:");
    const ul = el("ul", { style: "margin:8px 0 0;padding-left:18px" });
    for (const cav of R.caveats) ul.append(el("li", {}, cav));
    c.append(ul);
    hdr.append(c);
  }
  hdr.append(el("div", { class: "row", style: "margin-top:14px" },
    el("button", { class: "btn ghost sm", onclick: () => downloadReport(R) },
      "Download JSON"),
    el("button", { class: "btn ghost sm", onclick: () => removeReport(R.id) },
      "Delete")));
  w.append(hdr);

  for (const s of R.sections) {
    const sec = card(`${s.subject} · ${s.run_id}`,
      `${s.kind} · ${s.n_rows} rows` +
      (s.chance_level ? ` · chance ${s.chance_level}` : "") +
      (s.excluded_total ? ` · ${s.excluded_total} excluded` : ""));

    const ciFig = Charts.accuracyCI(
      (s.ci && s.ci.length) ? s.ci
        : s.summary.map(r => ({ model: r.model, mean: r.accuracy, n: r.n_scored })),
      { metric: "accuracy", chance: s.chance_level });
    if (ciFig) sec.append(ciFig, el("div", { style: "height:18px" }));

    sec.append(table(["model", "n_items", "n_scored", "accuracy", "accuracy_chance_adjusted",
                      "extraction_failure_rate", "cost_usd", "cost_per_correct_answer"],
      s.summary, r => [
        el("td", { class: "m" }, r.model),
        numTd(r.n_items, 0), numTd(r.n_scored, 0),
        numTd(r.accuracy), numTd(r.accuracy_chance_adjusted),
        numTd(r.extraction_failure_rate), numTd(r.cost_usd, 5),
        numTd(r.cost_per_correct_answer, 5)]));

    if (s.summary && s.summary.length)
      charted(sec, "Why items were excluded", Charts.failureSplit(s.summary.map(r => ({
        model: r.model, extraction_failure_rate: r.extraction_failure_rate,
        truncation_rate: r.truncation_rate, refusal_rate: r.refusal_rate,
        error_rate: r.error_rate }))));
    if (s.significance && s.significance.length)
      charted(sec, "Which pairs are separable", Charts.pairwiseMatrix(s.significance));
    if (s.significance_error) {
      sec.append(errMsg(s.significance_error));
    } else if (s.significance.length) {
      sec.append(el("p", { class: "note", style: "margin-top:14px" },
        "Paired (same items for every model), Holm-corrected for multiple comparisons:"));
      for (const r of s.significance)
        sec.append(el("p", { class: "note" },
          (r.significant ? "✓ " : "· ") + r.model_a + " vs " + r.model_b +
          ": " + r.verdict));
    }
    if (s.power) sec.append(el("p", { class: "note" }, s.power));

    if (s.manifest) {
      const d = el("details", { style: "margin-top:14px" });
      d.append(el("summary", {
        style: "cursor:pointer;color:var(--ink-3);font-size:12.5px" },
        "Provenance (I9, the reproducibility record: what the run was made from)"));
      const keys = ["git_sha", "git_dirty", "dataset_hash", "apparatus_hash",
                    "spec_hash", "pricing_as_of", "n_rows", "n_errors"];
      d.append(el("div", { class: "kv", style: "margin-top:10px" },
        keys.filter(k => k in s.manifest).map(k =>
          el("div", {}, el("dt", {}, termLabel(k)), el("dd", {}, String(s.manifest[k]))))));
      sec.append(d);
    }
    w.append(sec);
  }
  return w;
}

async function loadReports() {
  try { S.reports = (await api("/api/reports")).reports; }
  catch (e) { S.repError = e.message; S.reports = []; }
  render();
}

async function saveReport() {
  S.repError = "";
  const lines = (S.repNotes || "").split("\n").map(s => s.trim()).filter(Boolean);
  try {
    const doc = await post("/api/save-report", {
      run_ids: [...S.repPicked], title: S.repTitle,
      provider: S.repProvider, caveats: lines,
    });
    S.repPicked = new Set();
    S.reportOpen = doc;
    await loadReports();
  } catch (e) { S.repError = e.message; render(); }
}

async function openReport(id) {
  try { S.reportOpen = await api("/api/report?id=" + encodeURIComponent(id)); }
  catch (e) { S.repError = e.message; }
  render();
}

async function removeReport(id) {
  try {
    await post("/api/delete-report", { id });
    S.reportOpen = null;
    await loadReports();
  } catch (e) { S.repError = e.message; render(); }
}

function downloadReport(doc) {
  const blob = new Blob([JSON.stringify(doc, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${doc.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* --------------------------------------------------------------------------
   Evaluate (start here)
   -------------------------------------------------------------------------- */
const EV_TASKS = [
  ["classify", "Label things",
   "Route tickets, tag grievances, sort documents. The model must reply with exactly " +
   "one label from your list. Scored exact; a reply that is not a label is a format " +
   "failure, not a wrong label."],
  ["direct", "Generate or transform",
   "Translate, draft, summarise, extract, calculate. An instruction in, an answer out, " +
   "scored against your reference by the judge (or exact, numeric, token overlap)."],
  ["rag", "Answer from your documents",
   "Questions over a corpus you supply. Retrieval is pinned as apparatus, and " +
   "faithfulness, abstention and injection resistance are scored beside accuracy."],
  ["benchmark", "Run a public benchmark",
   "Pick from the catalogue (MMLU-Pro, GSM8K, IFEval, HellaSwag, MGSM and more), run it " +
   "live, and read chance-adjusted accuracy with failures kept apart from wrongness."],
];

function copyText(text) {
  try { navigator.clipboard.writeText(text); } catch { /* no clipboard: the text is on screen */ }
}
function downloadText(name, text, type = "text/plain") {
  const blob = new Blob([text], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
const codeBox = (text, fname) => el("div", { class: "evcode" },
  el("pre", { class: "out" }, text),
  el("div", { class: "row", style: "margin-top:8px" },
    el("button", { class: "btn ghost sm", onclick: () => copyText(text) }, "Copy"),
    fname ? el("button", { class: "btn ghost sm", onclick: () => downloadText(fname, text) },
      "Download " + fname) : null));

function fieldsTable(fields) {
  return el("div", { class: "tablewrap" }, el("table", { class: "cs-table" },
    el("thead", {}, el("tr", {}, el("th", {}, "field"), el("th", {}, "required"), el("th", {}, "meaning"))),
    el("tbody", {}, fields.map(f => el("tr", {},
      el("td", { class: "m" }, f.name), el("td", {}, f.required ? "yes" : "no"), el("td", {}, f.meaning))))));
}

function viewEvaluate() {
  const E = S.ev;
  const w = el("div", {}, head("Start here", "Evaluate a model on your data",
    "The whole path for someone new: pick the kind of task, see the data format " +
    "with sample rows, get a profile written for you, upload your own file if you " +
    "have one, then run, read and freeze the result. Every step names the screen " +
    "it happens on and the command that does the same thing from a terminal."));

  // --- 1. what kind of task ------------------------------------------------
  const pick = card("1. What are you evaluating?",
    "The choice fixes the data format, the scorer and the metrics that make sense.");
  pick.append(el("div", { class: "evpicks" }, EV_TASKS.map(([id, title, why]) =>
    el("button", { class: "evpick", "aria-pressed": String(E.task === id),
      onclick: () => { E.task = id; E.spec = null; E.yaml = ""; E.created = null; E.uploads = {}; render(); loadEvalSpec(); } },
      el("b", {}, title), el("span", {}, why)))));
  w.append(pick);

  if (E.task === "benchmark") { w.append(...evaluateBenchmark()); return w; }

  const spec = E.spec;
  if (!spec) { w.append(el("p", { class: "note" }, "Loading the format…")); return w; }

  // --- 2. data format -------------------------------------------------------
  const data = card("2. Prepare your data",
    "One JSON object per line (JSONL), UTF-8. Two or three rows are enough to try " +
    "the whole path; sixty or more before a paired test can separate models a few points apart.");
  data.append(el("h3", { class: "cs-h" }, "evalset.jsonl"), fieldsTable(spec.fields),
    el("p", { class: "note" }, "Sample rows for this task:"),
    codeBox(spec.samples_jsonl, "evalset.jsonl"));
  if (E.task === "rag") {
    data.append(el("h3", { class: "cs-h" }, "corpus.jsonl (the documents)"),
      fieldsTable(spec.corpus_fields),
      el("p", { class: "note" }, "Sample documents; the questions above point at them by doc_id#0:"),
      codeBox(spec.corpus_jsonl, "corpus.jsonl"));
  }
  if (spec.languages) {
    data.append(el("h3", { class: "cs-h" }, "Languages"),
      el("p", { class: "note" }, "Tag each item with meta.language using one of these codes and every " +
        "write-up gets a per-language reading. The script column is what native_script_ratio " +
        "checks when a direct task asks for that language."),
      el("div", { class: "tablewrap" }, el("table", { class: "cs-table" },
        el("thead", {}, el("tr", {}, el("th", {}, "code"), el("th", {}, "language"), el("th", {}, "script"),
          el("th", {}, "multilingual embedder"), el("th", {}, "note"))),
        el("tbody", {}, spec.languages.map(l => el("tr", {},
          el("td", { class: "m" }, l.code), el("td", {}, l.name), el("td", {}, l.script),
          el("td", {}, l.embedder_listed ? "listed on the model card" : "not listed; unmeasured"),
          el("td", {}, l.note || "")))))));
  }
  data.append(el("h3", { class: "cs-h" }, "Before you write yours"),
    el("ul", { class: "cs-ul" }, spec.notes.map(n => el("li", {}, n))));
  w.append(data);

  // --- 3. the profile -------------------------------------------------------
  const form = card("3. Describe the evaluation",
    "These choices become a profile file. It is validated as you type; the YAML on the " +
    "right is what gets written, and every line is commented so you can edit it later.");
  const f = (label, control) => el("div", { class: "f" }, el("label", {}, label), control);
  const left = el("div", {},
    f("Name (letters, digits, underscores)", el("input", { type: "text", value: E.name, placeholder: "my_support_triage",
      oninput: e => { E.name = e.target.value; previewProfile(); } })),
    E.task === "classify" ? f("Labels, comma-separated", el("textarea", { style: "min-height:56px",
      oninput: e => { E.labels = e.target.value; previewProfile(); } }, E.labels)) : null,
    f("Scorer for accuracy", el("select", { onchange: e => { E.scorer = e.target.value; previewProfile(); } },
      spec.scorers.map(s => el("option", { value: s, selected: s === (E.scorer || spec.defaults.scorer) }, s)))),
    f("Task instruction (the system prompt every model gets)", el("textarea", { style: "min-height:96px",
      oninput: e => { E.sys = e.target.value; previewProfile(); } }, E.sys)),
    f("Max tokens per answer", el("input", { type: "number", min: 16, value: E.maxTokens,
      oninput: e => { E.maxTokens = e.target.value; previewProfile(); } })),
    f("Language of the answers (and questions)", el("select", { onchange: e => { E.language = e.target.value; previewProfile(); } },
      (spec.languages || []).map(l => el("option", { value: l.code, selected: l.code === (E.language || "en") },
        `${l.name} (${l.script})` + (l.embedder_listed ? "" : " · embedder coverage unmeasured"))))),
    (() => { const l = (spec.languages || []).find(x => x.code === (E.language || "en"));
             return l && l.note ? el("p", { class: "note" }, l.note) : null; })(),
    el("p", { class: "note" }, "Reasoning models spend the token budget thinking before the visible " +
      "answer; a small budget truncates before the first character and reads as accuracy zero."));
  const right = el("div", {},
    el("label", {}, "Profile preview"),
    E.yamlErr ? errMsg(E.yamlErr) : null,
    E.yaml ? codeBox(E.yaml, (E.name || "profile") + ".yaml") : el("p", { class: "note" }, "Type a name to see the profile."));
  form.append(el("div", { class: "grid2 evform" }, left, right));
  form.append(el("div", { class: "row", style: "margin-top:14px" },
    el("button", { class: "btn", disabled: !E.yaml || !!E.yamlErr, onclick: createScaffold },
      "Create the profile and sample data on disk"),
    el("label", { class: "chk" }, el("input", { type: "checkbox", checked: E.overwrite,
      onchange: e => { E.overwrite = e.target.checked; } }), " overwrite if the files exist")));
  if (E.createErr) form.append(errMsg(E.createErr));
  if (E.created) {
    form.append(el("div", { class: "msg ok", style: "margin-top:12px" },
      "Written: " + E.created.written.join(", ") + ". The profile validates and appears on every screen's profile picker."));
    form.append(el("div", { class: "row", style: "margin-top:10px" },
      el("button", { class: "btn", onclick: () => { S.vProfile = E.created.name; go("preflight"); runValidate(); } }, "Preflight it"),
      el("button", { class: "btn ghost", onclick: () => { S.rProfile = E.created.name; go("runprofile"); } }, "Go to Profile run")));
  }
  w.append(form);

  // --- 4. own data ----------------------------------------------------------
  const up = card("4. Use your own data (optional)",
    "Upload a JSONL in the format above. It is checked line by line and written to " +
    "data/<name>/ only when every line is clean; the problems name the line otherwise. " +
    "Files above 20 MB: copy them into data/<name>/ by hand instead.");
  const kinds = E.task === "rag" ? ["evalset", "corpus"] : ["evalset"];
  for (const kind of kinds) {
    const r = E.uploads[kind];
    up.append(el("div", { class: "f" }, el("label", {}, kind + ".jsonl"),
      el("input", { type: "file", accept: ".jsonl,.json,.txt", onchange: e => uploadDataset(kind, e.target.files[0]) })),
      r ? (r.problems ? el("div", { class: "msg crit" }, el("div", {}, "Not written. Problems:"),
              el("ul", { style: "margin:6px 0 0;padding-left:18px" }, r.problems.map(p => el("li", { class: "m" }, p))))
            : el("div", { class: "msg ok" }, `Written ${r.path}: ${r.rows} rows` +
              (r.item_types ? " (" + Object.entries(r.item_types).map(([k, v]) => `${v} ${k}`).join(", ") + ")" : ""))) : null);
  }
  if (!E.name) up.append(el("p", { class: "note" }, "Give the evaluation a name in step 3 first; uploads are filed under it."));
  if (E.uploadErr) up.append(errMsg(E.uploadErr));
  w.append(up);

  // --- 5. run it -------------------------------------------------------------
  const name = E.name || "<name>";
  const run = card("5. Run it, end to end",
    "In this order. Each step is a screen in the task bar; the commands below do the same from a terminal.");
  const STEPS = [
    ["Prepare › Preflight", "Validate loads the profile and counts the items; Estimate itemises the bill including the judge. Nothing is spent yet.", "preflight"],
    E.task === "rag" ? ["Prepare › Retrieval", "Index the corpus with the pinned embedder. An empty index does not fail, it makes every model look bad.", "rag"] : null,
    E.task === "rag" ? ["Prepare › Probes", "Derive unanswerable, noise, injection and paraphrase items from yours and append them.", "probes"] : null,
    ["Reference › Connections", "Test the provider key and probe that the models you want are invokable on the account.", "connections"],
    ["Run › Profile run", "Pick two or more models, set a budget, run. Progress and metered spend update live.", "runprofile"],
    ["Analyse › Profile report", "Intervals first, then the paired test. Read the truncation table before anything else.", "preport"],
    ["Analyse › Decide", "The cheapest model that clears your bar at your daily volume.", "decide"],
    ["Analyse › Saved reports", "Freeze the attempt with its caveats; it appears under Past attempts.", "reports"],
  ].filter(Boolean);
  STEPS.forEach(([where, body, view], i) => run.append(step(i + 1, where, body,
    el("button", { class: "btn ghost sm", onclick: () => go(view) }, "Go there"))));
  const cli = [
    `python main.py validate --profile configs/profiles/${name}.yaml`,
    `python main.py estimate --profile configs/profiles/${name}.yaml`,
    E.task === "rag" ? `python main.py ingest   --profile configs/profiles/${name}.yaml` : null,
    E.task === "rag" ? `python main.py probes   --profile configs/profiles/${name}.yaml --append` : null,
    `python main.py run      --profile configs/profiles/${name}.yaml --models <model-a> <model-b> --budget 5`,
    `python main.py report   --profile configs/profiles/${name}.yaml`,
    `python main.py decide   --profile configs/profiles/${name}.yaml --require "accuracy>=0.85" --optimise cost_usd --qpd 10000`,
  ].filter(Boolean).join("\n");
  run.append(el("h3", { class: "cs-h" }, "The same path from a terminal"), codeBox(cli));
  w.append(run);

  // --- 6. your own model -------------------------------------------------------
  const own = card("6. Bring your own model",
    "Any OpenAI-compatible endpoint works; the harness needs three things before it will spend.");
  own.append(el("ul", { class: "cs-ul" },
    el("li", {}, "The key in .env (never in YAML): TOGETHER_API_KEY, OPENROUTER_API_KEY, or the provider's own. Check it on Reference › Connections and probe the model there."),
    el("li", {}, "A line in configs/models.yaml under models: (for another provider, prefix it: openrouter:vendor/model)."),
    el("li", {}, "A price in configs/pricing.yaml (input and output per million tokens). An unpriced model fails Preflight on purpose: cost is a ranked metric, and a missing price would rank the model as free.")));
  own.append(el("div", { class: "row", style: "margin-top:10px" },
    el("button", { class: "btn ghost sm", onclick: () => go("connections") }, "Connections"),
    el("button", { class: "btn ghost sm", onclick: () => go("providers") }, "Providers")));
  w.append(own);
  return w;
}

function evaluateBenchmark() {
  const E = S.ev;
  const out = [];
  const b = card("2. Pick a benchmark",
    "Every benchmark in the catalogue has a spec (prompt format, extraction chain, chance level) " +
    "and an adapter tested offline on a committed fixture. The run shares the trace store, cost " +
    "meter and paired test with everything else.");
  const list = S.benchmarks || [];
  b.append(el("div", { class: "f" }, el("label", {}, "Benchmark"),
    el("select", { onchange: e => { E.bench = e.target.value; render(); } },
      [el("option", { value: "" }, "pick one"),
       ...list.map(x => el("option", { value: x.id, selected: x.id === E.bench },
         `${x.id} · ${x.family || ""}` + (x.commercial_use === false ? " · non-commercial licence" : "")))])));
  const chosen = list.find(x => x.id === E.bench);
  if (chosen) b.append(el("p", { class: "note" },
    `${chosen.id}: ${chosen.family || "?"} family, ${chosen.licence || "licence not stated"}` +
    (chosen.commercial_use === false ? ". Non-commercial: the run needs the licence acknowledged." : ".")));
  b.append(el("div", { class: "row", style: "margin-top:12px" },
    el("button", { class: "btn", disabled: !E.bench, onclick: () => { S.benchmark = E.bench; go("run"); } }, "Open Benchmark run"),
    el("button", { class: "btn ghost", onclick: () => go("catalogue") }, "See the catalogue")));
  out.push(b);

  const s = card("3. Run it, end to end", "Fetch caches the dataset with its checksum; run, then read.");
  [["Reference › Catalogue", "Licence, family, chance level and the extraction chain for each set.", "catalogue"],
   ["Run › Benchmark run", "Pick the models, an item limit and a seed; the first run fetches the dataset (licence acknowledged where needed).", "run"],
   ["Analyse › Benchmark results", "Accuracy with n_scored beside it, chance-adjusted accuracy, extraction failures and truncation kept apart, the paired test.", "results"],
   ["Analyse › Saved reports", "Freeze the sweep with its caveats.", "reports"],
  ].forEach(([where, body, view], i) => s.append(step(i + 1, where, body,
    el("button", { class: "btn ghost sm", onclick: () => go(view) }, "Go there"))));
  const id = E.bench || "<benchmark>";
  s.append(el("h3", { class: "cs-h" }, "The same path from a terminal"), codeBox([
    `python main.py bench fetch ${id}`,
    `python main.py bench run --benchmark ${id} --models <model-a> <model-b> --limit 50 --seed 1729 --budget 2`,
    `python main.py bench report --run-id <run id printed above>`,
  ].join("\n")));
  out.push(s);

  const add = card("4. Add a benchmark of your own",
    "A private set is the only one guaranteed uncontaminated. It takes one YAML spec, one adapter, one fixture, one registry line.");
  add.append(codeBox([
    "# configs/benchmarks/my_set.yaml",
    "id: my_set",
    "version: 1",
    "family: multiple_choice        # or short_answer, math, instruction, ...",
    "task: direct",
    "source: {kind: local, ref: data/my_set/items.jsonl, licence: private, commercial_use: true}",
    "sampling: {limit: null, seed: 1729, samples_per_item: 1}",
    "prompt: {decoding: {temperature: 0.0, max_tokens: 1024}}",
    "scoring:",
    "  mode: generative",
    "  extraction:",
    "    chain:",
    "      - {kind: regex, pattern: 'Answer:\\s*\\(?([A-D])\\)?'}",
    "      - {kind: last_capital_letter}",
    "  metric: accuracy",
    "  chance_level: 0.25",
  ].join("\n"), "my_set.yaml"));
  add.append(el("ul", { class: "cs-ul" },
    el("li", {}, "The adapter goes in harness/bench/adapters/ with pure load, prompt, extract and score methods; copy the closest family (arc.py for multiple choice, gsm8k.py for numeric)."),
    el("li", {}, "Register it in harness/bench/registry.py and commit a 20-item fixture under tests/bench/fixtures/my_set/ so the whole path is tested offline."),
    el("li", {}, "The shared adapter contract suite runs against it automatically; extraction failures stay separate from wrong answers.")));
  out.push(add);
  return out;
}

const _previewTimer = { id: null };
function previewProfile() {
  clearTimeout(_previewTimer.id);
  _previewTimer.id = setTimeout(async () => {
    const E = S.ev;
    if (!E.name) { E.yaml = ""; E.yamlErr = ""; render(); return; }
    try {
      const r = await post("/api/scaffold-preview", { name: E.name, task: E.task, options: evOptions() });
      E.yaml = r.yaml; E.yamlErr = "";
    } catch (e) { E.yaml = ""; E.yamlErr = e.message; }
    render();
  }, 250);
}
function evOptions() {
  const E = S.ev;
  return { labels: E.labels, scorer: E.scorer, system_prompt: E.sys, max_tokens: E.maxTokens,
           multilingual: E.multilingual, language: E.language || "en" };
}
async function loadEvalSpec() {
  const E = S.ev;
  if (E.task === "benchmark") { render(); return; }
  try {
    E.spec = await api("/api/scaffold-spec?task=" + encodeURIComponent(E.task));
    const d = E.spec.defaults;
    E.sys = d.system_prompt; E.maxTokens = String(d.max_tokens); E.scorer = d.scorer;
    if (E.task === "classify" && !E.labels) E.labels = (d.labels || []).join(", ");
  } catch (e) { E.spec = null; E.createErr = e.message; }
  render();
  previewProfile();
}
async function createScaffold() {
  const E = S.ev;
  E.createErr = ""; E.created = null;
  try {
    E.created = await post("/api/scaffold", { name: E.name, task: E.task, options: evOptions(),
                                              with_samples: true, overwrite: E.overwrite });
    S.profiles = (await api("/api/profiles")).profiles;
  } catch (e) { E.createErr = e.message; }
  render();
}
function uploadDataset(kind, file) {
  const E = S.ev;
  if (!file) return;
  if (!E.name) { E.uploadErr = "Give the evaluation a name in step 3 first."; render(); return; }
  const reader = new FileReader();
  reader.onload = async () => {
    E.uploadErr = "";
    try {
      E.uploads[kind] = await post("/api/upload-dataset", {
        name: E.name, kind, text: String(reader.result), overwrite: E.overwrite,
        task: E.task, labels: E.task === "classify" ? E.labels : "" });
    } catch (e) {
      const msg = e.message || "";
      E.uploads[kind] = msg.includes("Problems") || msg.includes("problems")
        ? { problems: msg.split("\n").slice(1).filter(Boolean) } : null;
      E.uploadErr = E.uploads[kind] ? "" : msg;
    }
    render();
  };
  reader.readAsText(file);
}

/* --------------------------------------------------------------------------
   Product pages (left sidebar). For a reader who wants the product before
   the numbers: what it is, what is built, how it is put together, what is
   trusted, what is next. Prose and status only; every figure quoted here is
   a count the product carries, never a metric computed in the browser.
   -------------------------------------------------------------------------- */
const STORY_GROUPS = [
  ["product", "Product", true, [
    ["capabilities", "Capabilities", "every module, with its honest build status", "◈"],
    ["architecture", "Architecture", "layers, the TraceRow boundary, twelve invariants", "⌗"],
    ["trust", "Trust and security", "secrets, sessions, append-only traces", "⛨"],
    ["roadmap", "Roadmap", "built, open, next", "↗"],
    ["deployment", "Deployment", "run it, integrate it, share it", "⇪"],
    ["about", "About", "the four failure modes and every screen", "?"],
    ["guide", "Guide", "the whole platform end to end", "◷"],
  ]],
  ["evaluations", "Evaluations", true, [
    ["casestudies", "Case studies", "five Indian-government use cases, live results", "☷"],
    ["evaluate", "Evaluate a model", "data format, sample rows, the run path", "✎"],
    ["reports", "Saved reports", "frozen with their caveats", "✦"],
  ]],
  ["workspace", "Workspace", false, [
    ["overview", "Overview", "runs, spend, past attempts", "◈"],
    ["preflight", "Preflight", "validate, estimate the bill", "✓"],
    ["rag", "Retrieval", "index a corpus", "⛁"],
    ["probes", "Probes", "adversarial items from your own", "◎"],
    ["extract", "Extraction", "how an answer is read out", "⌁"],
    ["runprofile", "Profile run", "pick models, set a budget, run", "▶"],
    ["run", "Benchmark run", "public benchmarks", "▸"],
    ["preport", "Profile report", "intervals, paired test", "▦"],
    ["results", "Benchmark results", "failures kept apart", "▤"],
    ["decide", "Decide", "cheapest model clearing your bar", "◆"],
    ["gate", "Gate", "regression gate between runs", "⊘"],
    ["arena", "Arena", "pairwise judge", "⚔"],
    ["catalogue", "Catalogue", "benchmarks with licence", "≡"],
    ["providers", "Providers", "what each provider offers", "◇"],
    ["connections", "Connections", "keys and model probing", "⇄"],
  ]],
];

const status = (s) => el("span", { class: "tag " + (s === "Built" ? "ok" : s === "Partial" ? "warn" : "dim") }, s);

function storyList(rows) {
  return el("div", { class: "story-list" }, rows.map(([title, st, body], i) =>
    el("div", { class: "story-row" },
      el("div", { class: "story-n" }, String(i + 1)),
      el("div", {}, el("b", {}, title, status(st)), el("p", {}, body)))));
}

function viewCapabilities() {
  const w = el("div", {}, head("Product", "Capabilities",
    "Every module, honest about its build status. Built means it runs today and is " +
    "covered by the offline test suite; Partial means it runs with a stated gap; " +
    "Roadmap means it is specified, not shipped."));
  w.append(kpis([["17", "screens", "accent"], ["12", "invariants enforced"], ["5", "case studies"], ["11", "benchmark specs"]]));
  w.append(card("Measurement", "The core: what makes a number on this platform worth quoting.", storyList([
    ["Paired model comparison", "Built", "Every model sees the identical items in the identical order with the identical retrieved context. Binary metrics use an exact McNemar test, continuous ones a paired bootstrap, and Holm-Bonferroni corrects across the family. A gap inside the interval is reported as not separable, with the items it would take to settle it."],
    ["Metered cost, four buckets", "Built", "Generation, judge, embedding and rerank spend are read from each provider's own usage counts and sum to the total. Cost carries a negative weight in every composite, a positive one is rejected at load, and an unpriced model fails validation rather than ranking as free."],
    ["Failure kept apart from wrongness", "Built", "Truncation, refusal, extraction failure, format violation, abstention and injection are their own rates beside accuracy. Hidden reasoning tokens are recorded per row, so an answer that ran out of budget reads as a budget finding."],
    ["Decision layer", "Built", "Decide names the cheapest model that clears your bar at your daily volume and prices every extra point; Gate compares two runs with tolerances; Arena runs a pairwise judge with position swap."],
    ["Reproducibility", "Built", "Every run writes a manifest: git commit and dirty flag, dataset hash, apparatus hash, seeds, pricing version. A run without a manifest is shown as such, never as a result."],
  ])));
  w.append(card("Workloads", "What can be evaluated, and how.", storyList([
    ["Your own data (classify, generate, answer from documents)", "Built", "A profile is YAML: task, scorer, metrics, weights, apparatus. The Evaluate screen writes it from a few choices, shows the data format with sample rows, and checks an uploaded file line by line."],
    ["Retrieval apparatus", "Built", "A pinned embedder served locally (English or multilingual), dense, sparse or hybrid retrieval, an embedded vector index, citation checks and abstention scoring. The apparatus is identical for every model and its hash travels with every row."],
    ["Adversarial probes", "Built", "Unanswerable, noise, injection, paraphrase and positional items derived from your own set, with canaries stored hashed so a leak is detectable and never re-injected."],
    ["Public benchmarks", "Built", "Eleven specs with pinned checksums and licence flags across multiple choice, maths, instruction following and multilingual maths (Bengali, Telugu), each with an offline fixture and the shared adapter contract suite. Chance-adjusted accuracy sits beside the raw one."],
    ["Code benchmarks in a sandbox", "Roadmap", "HumanEval-style sets need a process sandbox with no network and hard limits; specified, not shipped, so no code benchmark is offered."],
  ])));
  w.append(card("Languages and context", "Built for Indian-language work, honest about coverage.", storyList([
    ["Eight languages recognised", "Partial", "English, Hindi, Marathi, Bengali, Gujarati, Kannada, Telugu and Hinglish, with a native-script metric, Indic digit normalisation and per-language readings. The multilingual embedder lists Hindi, Marathi and Gujarati; Bengali, Kannada and Telugu retrieval quality is unmeasured and the screens say so."],
    ["Five case studies", "Built", "Grievance routing, notice translation into Hindi, RTI drafting, scheme question answering and RTI-framework question answering, each with a fictional dataset, a write-up and a live five-model run."],
    ["Reasoning-model awareness", "Partial", "Hidden reasoning is recorded and reported; there is no per-model switch to turn thinking off yet, so budgets are sized to cover it."],
    ["Judge calibration", "Partial", "Cohen's kappa against human labels is computed when a dataset carries them; none of the shipped case studies has been human-labelled yet."],
  ])));
  w.append(card("Platform", "Around the measurement.", storyList([
    ["Sign-in gate with server-side roles", "Built", "A landing page, a shared pilot password hashed before comparison, HttpOnly sessions, and an Assurance Lead role required to start a paid run. Pilot-grade: no accounts, SSO or MFA."],
    ["Saved reports and HTML export", "Built", "A report freezes runs with their caveats; a price fixed tomorrow does not change what a report said today."],
    ["Integration with Maha Evaluation Intelligence", "Built", "The harness is vendored unchanged into the Maha platform, which reads its API through a same-origin proxy and deep-links into every screen."],
  ])));
  return w;
}

const INVARIANTS_STORY = [
  ["I1 Paired comparison", "Every model sees the identical item set in the identical order with identical retrieved context; a missing score raises rather than dropping the item."],
  ["I2 Pinned apparatus", "Embedder, reranker and judge are fixed apparatus; changing one changes the apparatus hash and ends comparability with earlier runs."],
  ["I3 All spend metered", "Every paid call is attributed to a bucket from the provider's real usage block; an unpriced model is a hard failure, not a zero."],
  ["I4 Cost negatively weighted", "A positive weight on cost or latency is a config error rejected at load."],
  ["I5 No unpaired, uncorrected significance", "Exact McNemar or paired bootstrap, Holm-Bonferroni across the family, selected automatically and recorded."],
  ["I6 Non-significant means non-significant", "No winner is declared when the interval straddles zero; the harness says how many more items are needed."],
  ["I7 Failure is not wrongness", "Parse failures, truncations, refusals and violations are reported as their own rates."],
  ["I8 Traces are append-only", "Rows are written once; re-scoring produces a derived table; a budget abort keeps every row already written."],
  ["I9 Reproducibility", "Every run records git SHA, dirty flag, profile, dataset and apparatus hashes, seeds and pricing version."],
  ["I10 Determinism where claimed", "Probe generation, sampling and ordering are pure functions of the seed and the dataset hash."],
  ["I11 Config is the spine", "A profile, model, provider, benchmark or weight is YAML; if it needs code, the seam is wrong."],
  ["I12 Secrets and canaries never leak", "Keys are redacted from every log, trace, cache key, manifest and export; canaries are stored hashed."],
];

function viewArchitecture() {
  const w = el("div", {}, head("Product", "Architecture and foundations",
    "One boundary and twelve rules. The boundary is a row: everything above it " +
    "reads rows already on disk and may never open a network connection."));
  charted(w, "Pipeline", Charts.pipeline());
  const layers = card("Layers, imports point downward only",
    "An upward import is a build failure, enforced by a contract rather than by review habit.");
  [["cli / app / dashboard", "presentation: layout, state and calls, no business logic"],
   ["decide / report / arena / gate", "analysis over traces only; nothing here touches the network"],
   ["store (Parquet, DuckDB)", "TraceRow in, tables out; the running/analysis boundary"],
   ["run / passes / bench", "orchestration: the model × profile × pass matrix, checkpoints, resume, budget"],
   ["scorers / judges", "pure functions over an item and a response; no clock, no I/O, an injected random generator"],
   ["providers", "the only code allowed to open a connection, one adapter per provider, one shared contract suite"],
   ["config / contracts", "depended on by all, depends on nothing"],
  ].forEach(([c, s]) => layers.append(el("div", { class: "layer" }, el("code", {}, c), el("span", {}, s))));
  w.append(layers);
  w.append(card("The TraceRow", "One flat, fully denormalised row per evaluated item, written once.",
    el("p", { class: "note" }, "Model, profile, pass, item, the retrieved chunks, the answer, every metric, every " +
      "token count including hidden reasoning, cost by bucket, latency, the apparatus and dataset hashes, and the " +
      "manifest that ties it to a commit. The schema is versioned and evolves additively; a reader of an old " +
      "file is a test in the suite.")));
  const inv = card("The twelve invariants", "Violating any of these is a top-severity bug, even if every test passes.");
  inv.append(storyList(INVARIANTS_STORY.map(([t, b]) => [t, "Built", b])));
  w.append(inv);
  return w;
}

function viewTrust() {
  const w = el("div", {}, head("Product", "Trust and security",
    "What protects the keys, the data and the numbers, stated specifically, and what is " +
    "deliberately not claimed."));
  w.append(card("Secrets", null, storyList([
    ["Keys live in the environment", "Built", "Provider keys are read from .env or the environment, never from YAML, and a redaction step strips them from every log, trace, cache key, manifest and export before it is written."],
    ["Canaries stored hashed", "Built", "Injection probes plant a canary string; it is kept salted and hashed so a later prompt can never be contaminated with it, and the report counts hits without printing the text."],
  ])));
  w.append(card("Access", null, storyList([
    ["Loopback only", "Built", "The server binds to 127.0.0.1 and refuses to be framed by another site. A key that can spend money is never reachable from the network by default."],
    ["Sign-in gate", "Built", "A shared pilot password from the environment, hashed with PBKDF2-HMAC-SHA256 (200,000 iterations) before a constant-time comparison; HttpOnly, SameSite cookies that expire after twelve hours; every API route and the app page answer 401 without a session."],
    ["Roles enforced by the server", "Built", "The two endpoints that start paid runs require the Assurance Lead role and answer 403 otherwise. An Evaluator reads everything else."],
    ["Accounts, SSO, MFA", "Roadmap", "Not present. The gate suits a pilot behind a firewall; exposing the server beyond loopback needs real authentication in front of it."],
  ])));
  w.append(card("Data and numbers", null, storyList([
    ["Traces are append-only", "Built", "A row is written once with its manifest; re-scoring writes a derived table; an aborted run keeps every row already paid for."],
    ["Uploads checked line by line", "Built", "A dataset upload is validated before anything is written, capped in size, and confined to the project's data folder under a safe name; nothing is overwritten unless asked."],
    ["Adversarial content is inert", "Built", "Model output and retrieved passages are treated as data: rendered escaped, never concatenated into a later prompt without an explicit, tested step."],
    ["Untrusted code runs nowhere", "Roadmap", "Code benchmarks need a process sandbox with no network, no filesystem beyond a scratch directory and hard limits; until it exists, no candidate code is executed by the harness."],
  ])));
  return w;
}

function viewRoadmap() {
  const w = el("div", {}, head("Product", "Roadmap and honest status",
    "What is built, what is open with a number in the debt register, and what comes next."));
  w.append(card("Open now, recorded in docs/DEBT.md", "Each has a register entry with evidence; none is hidden.", storyList([
    ["R-30 Empty answers in the accuracy denominator", "Partial", "For profile runs an answer that ran out of tokens is scored as wrong rather than absent. The benchmark path already keeps them apart. Which definition to adopt changes reported numbers, so it is a human decision, not a silent fix."],
    ["R-31 No per-model switch for hidden reasoning", "Roadmap", "Some providers can turn thinking off per request; the seam must reach the cache key and the manifest before it is offered. Until then budgets are sized to cover the reasoning."],
    ["R-32 A provider reports zero reasoning tokens beside real reasoning", "Partial", "Recorded as unknown, never as zero; the reasoning-rate column is the robust signal."],
    ["R-33 Resume re-ran the latency lane", "Built", "Fixed the day it was found, with a regression test."],
    ["R-34 Listing runs loads the whole store", "Partial", "Every screen open waits a few seconds while the store grows; the DuckDB summary query exists and the listing should use it."],
  ])));
  w.append(card("Next", "In the order they would be picked up.", storyList([
    ["Thinking switch seam", "Roadmap", "Per-model request parameters in models.yaml, entering the cache key and manifest, so a classifier can be measured in the mode it would be deployed in."],
    ["Store listing on DuckDB", "Roadmap", "Sub-second run lists regardless of store size."],
    ["Code benchmark sandbox", "Roadmap", "Container or hardened subprocess with the hostile battery from the spec: fork bomb, infinite loop, socket, path escape."],
    ["Human-labelled subsets", "Roadmap", "Thirty labelled items per case study to calibrate the judge and report its agreement."],
    ["Measured multilingual retrieval", "Roadmap", "Bengali, Kannada and Telugu retrieval checked the way Hindi was, and the coverage note updated from evidence."],
    ["Accounts in front of the gate", "Roadmap", "SSO or individual accounts when the tool leaves the pilot's laptop."],
  ])));
  return w;
}

function viewDeployment() {
  const w = el("div", {}, head("Product", "Deployment and integration",
    "How it runs, how another platform reads it, and how to hand it to a new team."));
  w.append(card("Run it", "A Python process on one machine.", storyList([
    ["Local server", "Built", "python main.py serve starts the UI on 127.0.0.1:8010 behind the sign-in. The provider key sits in .env; the pilot password in HARNESS_PILOT_PASSWORD."],
    ["Embedded index", "Built", "Retrieval uses an embedded vector store under workspace/; one process at a time holds it, which the runner serialises."],
    ["Offline by default", "Built", "The whole test suite runs with no key and blocked sockets; live runs are opt-in and budget-capped."],
  ])));
  w.append(card("Integrate it", "The Maha Evaluation Intelligence platform carries the harness unchanged.", storyList([
    ["Vendored copy", "Built", "The product is copied into the Maha repository as evaluation-harness/ and started by its dev launcher when its environment exists."],
    ["Same-origin API path", "Built", "The Maha SPA reads runs, case studies, reports and the Evaluate format through /harness-api, a proxy to this server's /api; it computes no metric of its own."],
    ["Deep links", "Built", "Every harness screen has a hash the other platform links to; the sign-in preserves it, so a link lands on the intended screen."],
    ["Static hosting", "Partial", "The Maha frontend deploys as static files; the harness is a Python server that must be hosted beside it, with a rewrite for /harness-api and real authentication in front."],
  ])));
  w.append(card("Hand it to a new team", "Ten minutes from clone to first result on their own data.",
    el("div", { class: "row" },
      el("button", { class: "btn", onclick: () => go("evaluate") }, "Open Evaluate"),
      el("button", { class: "btn ghost", onclick: () => go("guide") }, "Open the Guide"),
      el("button", { class: "btn ghost", onclick: () => go("casestudies") }, "See the case studies"))));
  return w;
}

const STORIES = {
  capabilities: viewCapabilities, architecture: viewArchitecture, trust: viewTrust,
  roadmap: viewRoadmap, deployment: viewDeployment,
};

function buildSidebar() {
  const side = $("#sidebar");
  let open = {};
  try { open = JSON.parse(localStorage.getItem("harness-sidebar") || "{}") || {}; } catch { /* ignore */ }
  const isOpen = (id, dflt) => (id in open ? open[id] : dflt);
  const draw = () => side.replaceChildren(...STORY_GROUPS.map(([id, label, dflt, items]) => {
    const o = isOpen(id, dflt);
    const g = el("div", { class: "sidebar-group" },
      el("button", { class: "sidebar-group-head", "aria-expanded": String(o),
        onclick: () => { open[id] = !isOpen(id, dflt); try { localStorage.setItem("harness-sidebar", JSON.stringify(open)); } catch { /* ignore */ } draw(); } },
        el("span", {}, label), el("span", { class: "chev" }, "›")));
    if (o) g.append(el("div", { class: "sidebar-group-items" }, items.map(([view, l, d, ic]) =>
      el("button", { class: "sidebar-item", "data-view": view, "aria-current": String(view === S.view),
        onclick: () => { go(view); side.classList.remove("open"); } },
        el("span", { class: "ic" }, ic), el("span", {}, el("b", {}, l), el("span", {}, d))))));
    return g;
  }));
  draw();
  $("#sidebtn").addEventListener("click", e => {
    const on = side.classList.toggle("open");
    e.currentTarget.setAttribute("aria-expanded", String(on));
  });
  // The sidebar sits under the sticky task bar; measure it so both stick.
  const setTop = () => document.documentElement.style.setProperty("--topbar-h", `${$(".topbar").offsetHeight}px`);
  setTop();
  addEventListener("resize", setTop);
}

function renderSidebar() {
  for (const b of document.querySelectorAll(".sidebar-item"))
    b.setAttribute("aria-current", String(b.dataset.view === S.view));
}

/* --------------------------------------------------------------------------
   Case studies
   -------------------------------------------------------------------------- */
const CS_LANG = { en: "English", hi: "Hindi", mr: "Marathi", bn: "Bengali", gu: "Gujarati",
                  kn: "Kannada", te: "Telugu", hinglish: "Hinglish" };

function csBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    if (b.type === "h2") out.push(el("h3", { class: "cs-h" }, b.text));
    else if (b.type === "p") out.push(el("p", { class: "note cs-p" }, b.text));
    else if (b.type === "ul") out.push(el("ul", { class: "cs-ul" }, b.items.map(t => el("li", {}, t))));
    else if (b.type === "table") {
      const t = el("table", { class: "cs-table" });
      t.append(el("thead", {}, el("tr", {}, b.header.map(h => el("th", {}, h)))));
      t.append(el("tbody", {}, b.rows.map(r => el("tr", {}, r.map(c => el("td", {}, c))))));
      out.push(el("div", { class: "tablewrap" }, t));
    }
  }
  return out;
}

function csResultsFigures(R) {
  // Drawn from the same profile_results payload the Profile report reads;
  // nothing is computed here (the figures only lay out numbers).
  const box = el("div", { class: "cs-results" });
  if (!R) { box.append(el("p", { class: "note" }, "Loading the run…")); return box; }
  if (R.error) { box.append(errMsg(R.error)); return box; }
  box.append(kpis([
    [R.run_id || "n/a", "run", "accent"],
    [String((R.models || []).length), "models"],
    [(R.n_rows || 0).toLocaleString(), "rows"],
    [term(R.metric), el("span", {}, "metric", tid(R.metric, true))],
  ]));
  if (R.ci) charted(box, R.metric, Charts.accuracyCI(R.ci, { metric: R.metric }));
  if (R.pareto) charted(box, "Quality against cost", Charts.qualityCost(
    R.pareto.map(p => ({ model: p.model, accuracy: p.accuracy,
                         cost: p.cost, on_frontier: !!p.on_frontier }))));
  if (R.composite && R.composite.length)
    charted(box, "Metric profile", Charts.metricProfile(R.composite,
      (R.active_metrics || []).filter(m => m in (R.composite[0] || {})),
      { lowerIsBetter: Object.entries(R.weights || {})
          .filter(([, v]) => v < 0).map(([k]) => k) }));
  if (R.significance && R.significance.length)
    charted(box, "Which pairs are separable", Charts.pairwiseMatrix(R.significance));
  if (R.cost_per_correct && R.cost_per_correct.length)
    charted(box, "cost_per_correct_answer", Charts.costPerCorrect(R.cost_per_correct));
  if (R.composite) box.append(card(termLabel("composite"),
    "Deterministic arithmetic from the profile's own weights; a negative weight means lower is better.",
    autoTable(R.composite)));
  box.append(significanceCard({
    significance: R.significance, significance_error: R.significance_error,
    power: R.power,
  }));
  return box;
}

function viewCaseStudies() {
  const w = el("div", {}, head("Worked examples", "Case studies",
    "Worked evaluations, each written up the same way: the use case, who " +
    "would run it, what a wrong answer costs, the metrics and why, the " +
    "caveats, and the results of the newest run. The numbers on this screen " +
    "are the Profile report's numbers for that run, not a second computation."));

  if (S.csError) w.append(errMsg(S.csError));
  const list = S.caseStudies;
  if (!list) { w.append(el("p", { class: "note" }, "Loading…")); return w; }
  if (!list.length) {
    w.append(empty("No case studies yet",
      "Add a Markdown file under docs/case-studies named after a profile."));
    return w;
  }

  const withRun = list.filter(c => c.run).length;
  // Languages across the case studies: how many, and behind the info button,
  // which ones with their item counts (from each profile's meta.language tags).
  const langCount = {};
  for (const c of list) {
    for (const [k, n] of Object.entries((c.profile && c.profile.languages) || {})) {
      const e = langCount[k] || (langCount[k] = { items: 0, studies: 0 });
      e.items += n; e.studies += 1;
    }
  }
  const langCodes = Object.keys(langCount);
  const langTip = infoTip("Which languages the case studies cover",
    el("div", {}, el("b", {}, "Languages in the case studies"),
      el("ul", { class: "tiplist" }, langCodes.map(k => el("li", {},
        `${CS_LANG[k] || k} (${k}): ${langCount[k].items} items in ${langCount[k].studies} ` +
        (langCount[k].studies === 1 ? "case study" : "case studies")))),
      el("span", { class: "note" }, "Counted from each item's meta.language tag; items without a tag are not counted.")));
  w.append(kpis([
    [String(list.length), "case studies", "accent"],
    [String(withRun), "with a live run"],
    [String(list.length - withRun), "not run yet"],
    [String(langCodes.length), el("span", {}, "languages", langTip)],
  ]));

  for (const c of list) {
    const p = c.profile || {};
    const open = !!S.csOpen[c.id];
    const sub = p.description || (p.invalid ? "Profile invalid: " + p.invalid : "No profile for this case study.");
    const cardEl = card(c.title, sub);

    const tags = el("div", { class: "pills cs-tags" });
    if (p.task) tags.append(el("span", { class: "pill" }, term(p.task)));
    if (p.n_items) tags.append(el("span", { class: "pill" }, `${p.n_items} items`));
    for (const [k, n] of Object.entries(p.languages || {}))
      tags.append(el("span", { class: "pill" }, `${CS_LANG[k] || k} ${n}`));
    if (p.accuracy_scorer) tags.append(el("span", { class: "pill" }, `${term("accuracy")} scored by ${p.accuracy_scorer}`));
    if (p.target_script) tags.append(el("span", { class: "pill" }, `answers in ${p.target_script}`));
    if (p.embedding_model) tags.append(el("span", { class: "pill" }, `embedder ${p.embedding_model.split("/").pop()}`));
    cardEl.append(tags);

    if (p.weights) cardEl.append(el("p", { class: "note", style: "margin-top:10px" },
      "Composite weights: " + Object.entries(p.weights).map(([k, v]) => `${term(k)} ${v}`).join(" · ") +
      ". Negative means lower is better."));

    const r = c.run;
    if (r) {
      cardEl.append(el("div", { class: "msg info", style: "margin-top:12px" },
        `Newest run ${r.run_id}: ${r.models.length} models, ${r.rows.toLocaleString()} rows, ` +
        `$${(r.cost_usd || 0).toFixed(4)} metered` + (r.errors ? `, ${r.errors} errors` : "") +
        (r.aborted ? ", stopped early (partial)" : "") +
        (c.other_runs.length ? `. Earlier attempts: ${c.other_runs.join(", ")}.` : ".")));
    } else {
      cardEl.append(el("div", { class: "msg warn", style: "margin-top:12px" },
        "Not run yet. The write-up describes the evaluation; the Results section stays empty until a run exists."));
    }
    if (!c.results_filled && r) cardEl.append(el("p", { class: "note" },
      "The write-up's Results section has not been filled in yet; the figures below are read live from the run."));

    const actions = el("div", { class: "row", style: "margin-top:14px" },
      el("button", { class: "btn" + (open ? "" : " ghost"), onclick: () => toggleCaseStudy(c.id) },
        open ? "Hide the write-up" : "Read the write-up"));
    if (r) {
      actions.append(el("button", { class: "btn ghost", onclick: () => loadProfileReport({ profile: c.id }) },
        "Open in Profile report"));
      actions.append(el("button", { class: "btn ghost", onclick: () => { S.dProfile = c.id; go("decide"); } },
        "Decide on this profile"));
    }
    cardEl.append(actions);

    if (open) {
      cardEl.append(el("hr", { class: "rule" }));
      cardEl.append(...csBlocks(c.blocks));
      if (r) {
        cardEl.append(el("h3", { class: "cs-h" }, "Results, read live from the newest run"));
        cardEl.append(csResultsFigures(S.csResults[c.id]));
      }
    }
    w.append(cardEl);
  }
  return w;
}

async function loadCaseStudies() {
  S.csError = "";
  try { S.caseStudies = (await api("/api/case-studies")).case_studies; }
  catch (e) { S.csError = e.message; S.caseStudies = []; }
  render();
}

async function toggleCaseStudy(id) {
  S.csOpen[id] = !S.csOpen[id];
  render();
  const c = (S.caseStudies || []).find(x => x.id === id);
  if (!S.csOpen[id] || !c || !c.run || S.csResults[id]) return;
  try {
    S.csResults[id] = await api("/api/profile-results?" +
      new URLSearchParams({ profile: id, metric: "accuracy" }));
  } catch (e) { S.csResults[id] = { error: e.message }; }
  render();
}

/* --------------------------------------------------------------------------
   Shell
   -------------------------------------------------------------------------- */
const VIEWS = {
  overview: viewOverview, preflight: viewPreflight, probes: viewProbes,
  extract: viewExtract, runprofile: viewRunProfile, run: viewRun,
  preport: viewProfileReport, results: viewResults, decide: viewDecide,
  gate: viewGate, arena: viewArena, catalogue: viewCatalogue,
  providers: viewProviders, rag: viewRag,
  connections: viewConnections, about: viewAbout, reports: viewReports,
  guide: viewGuide, casestudies: viewCaseStudies, evaluate: viewEvaluate,
};

/* The task bar's two tiers, in the order the work actually happens: check
   before you spend, run, then analyse. Reference and the explainers sit at the
   end because you visit them once, not every session.

   A section with one screen renders no sub-bar: a second tier containing a
   single item is a row of chrome that tells the reader nothing. */
const SECTIONS = [
  ["overview", "Overview", [["overview", "Overview", "◈"]]],
  // Start here: a new user with a model and some data is walked to a result.
  ["evaluate", "Evaluate", [["evaluate", "Evaluate", "✎"]]],
  // Case studies sit in the top bar as their own button: they are the front
  // door for a reader who wants the worked examples, not a sub-tab of Analyse.
  ["casestudies", "Case studies", [["casestudies", "Case studies", "☷"]]],
  ["prepare", "Prepare", [
    ["preflight", "Preflight", "✓"], ["rag", "Retrieval", "⛁"],
    ["probes", "Probes", "◎"], ["extract", "Extraction", "⌁"]]],
  ["run", "Run", [
    ["runprofile", "Profile run", "▶"], ["run", "Benchmark run", "▸"]]],
  ["analyse", "Analyse", [
    ["preport", "Profile report", "▦"], ["results", "Benchmark results", "▤"],
    ["decide", "Decide", "◆"], ["gate", "Gate", "⊘"],
    ["arena", "Arena", "⚔"], ["reports", "Saved reports", "✦"]]],
  ["reference", "Reference", [
    ["catalogue", "Catalogue", "≡"], ["providers", "Providers", "◇"],
    ["connections", "Connections", "⇄"]]],
  ["guide", "Guide", [["guide", "Guide", "◷"]]],
  ["about", "About", [["about", "About", "?"]]],
];

const SECTION_OF = Object.fromEntries(
  SECTIONS.flatMap(([id, , views]) => views.map(([v]) => [v, id])));

/* Each screen's first-open fetch, in one place. Keeping it here rather than in
   the click handler means the tab bar, the sub-bar, a deep link and any
   in-page "go to X" button all warm the same data; a screen that loads only
   when reached one particular way is the kind of bug that looks like an empty
   page. */
const ON_ENTER = {
  results: () => { if (S.benchRuns.length && !S.results) loadResults(S.benchRuns[0].run_id); },
  extract: () => { if (!S.extract) runExtract(); },
  preflight: () => { if (!S.validation) runValidate(); },
  probes: () => { if (!S.probes) runProbes(); },
  rag: () => { if (!S.ragStatus) loadRagStatus(); },
  connections: () => { if (!S.connections) loadConnections(); },
  reports: () => { if (!S.reports) loadReports(); },
  casestudies: () => { if (!S.caseStudies) loadCaseStudies(); },
  evaluate: () => { if (!S.ev.spec && S.ev.task !== "benchmark") loadEvalSpec(); },
};

function go(view) {
  if (!VIEWS[view] && !STORIES[view]) return;
  S.view = view;
  document.querySelector(".tabs")?.classList.remove("open");
  document.querySelector("#menubtn")?.setAttribute("aria-expanded", "false");
  try { location.hash = view; } catch { /* file:// */ }
  render();
  ON_ENTER[view]?.();
}

function buildNav() {
  const sections = $("#sections");
  sections.replaceChildren(...SECTIONS.map(([id, label, views]) =>
    el("button", {
      class: "tab", "data-section": id,
      onclick: () => go(S.sectionLast[id] || views[0][0]),
    }, label)));

  $("#menubtn").addEventListener("click", e => {
    const open = sections.classList.toggle("open");
    e.currentTarget.setAttribute("aria-expanded", String(open));
  });

  addEventListener("hashchange", () => {
    const v = location.hash.replace(/^#/, "");
    if (v && v !== S.view) go(v);
  });
}

function renderNav() {
  renderSidebar();
  if (STORIES[S.view]) {
    // A product page: no task-bar section is current and there is no sub-bar.
    for (const b of document.querySelectorAll(".tab")) b.setAttribute("aria-current", "false");
    $("#subbar").replaceChildren();
    return;
  }
  const sec = SECTION_OF[S.view] || "overview";
  S.sectionLast[sec] = S.view;

  for (const b of document.querySelectorAll(".tab"))
    b.setAttribute("aria-current", String(b.dataset.section === sec));

  const views = (SECTIONS.find(s => s[0] === sec) || [])[2] || [];
  const bar = $("#subbar");
  if (views.length < 2) { bar.replaceChildren(); return; }
  bar.replaceChildren(...views.map(([v, label, ic]) =>
    el("button", {
      class: "subtab", "aria-current": String(v === S.view),
      onclick: () => go(v),
    }, el("span", { class: "ic" }, ic), label)));
}

function paint() {
  renderNav();
  $("#main").replaceChildren((VIEWS[S.view] || STORIES[S.view] || viewOverview)());
  if (S.view === "run") renderKpis();
  window.scrollTo({ top: 0, behavior: "instant" });
}

let LAST_VIEW = null;
let TRANSITION = null;
function render() {
  // The first paint has nothing to transition FROM, and a render issued while
  // a transition is still in flight aborts it with an InvalidStateError that
  // surfaces as an uncaught rejection. Both cases paint directly.
  const first = LAST_VIEW === null;
  const changed = LAST_VIEW !== S.view;
  LAST_VIEW = S.view;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  // Cross-fade only on a screen change. Re-renders inside one screen (a poll
  // tick, a toggled pill) must not flash, so they paint directly.
  // A hidden document (background tab, minimised window, a preview pane
  // that is not drawing) cannot run a transition: the API rejects with
  // InvalidStateError. Paint directly there, and observe every promise the
  // transition exposes: `ready` rejects independently of `finished`, and an
  // unobserved rejection is an "uncaught" error in the console.
  const visible = document.visibilityState === "visible";
  if (!first && changed && !still && !TRANSITION && visible &&
      typeof document.startViewTransition === "function") {
    try {
      TRANSITION = document.startViewTransition(paint);
      const done = () => { TRANSITION = null; };
      const quiet = () => {};
      TRANSITION.ready.then(quiet, quiet);
      TRANSITION.updateCallbackDone.then(quiet, quiet);
      TRANSITION.finished.then(done, done);
    } catch {
      TRANSITION = null;
      paint();
    }
  } else {
    paint();
  }
}

async function refreshRuns() {
  const [all, bench] = await Promise.all([api("/api/runs"), api("/api/bench-runs")]);
  S.allRuns = all.runs;
  S.benchRuns = bench.runs;
  const p = S.allRuns.filter(r => r.kind === "profile").length;
  const b = S.allRuns.filter(r => r.kind === "benchmark").length;
  $("#storeline").textContent = `${p} profile · ${b} benchmark run(s)`;
  if (!S.gBase && S.allRuns.length > 1) {
    S.gBase = S.allRuns[1].run_id;
    S.gCand = S.allRuns[0].run_id;
  }
  // Default the pickers to a profile that actually HAS runs. Defaulting to the
  // alphabetically-first profile produced "No rows for that selection" on
  // first open, which reads as a broken page rather than an empty store.
  const withRuns = S.allRuns.find(r => r.kind === "profile");
  if (withRuns) {
    if (!S.dProfile || !S.allRuns.some(r => r.subject === S.dProfile))
      S.dProfile = withRuns.subject;
    if (!S.pProfile && !S.pRun) S.pProfile = withRuns.subject;
  }
}

function initTheme() {
  let t = null;
  try { t = localStorage.getItem("harness-theme"); } catch { /* private mode */ }
  if (t) document.documentElement.dataset.theme = t;
  $("#themebtn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("harness-theme", next); } catch { /* ignore */ }
    // Charts follow the theme through CSS variables; anything that read a
    // token at build time is rebuilt here so nothing keeps the old look.
    render();
  });
}

async function boot() {
  initTheme();
  initUser();
  buildNav();
  buildSidebar();
  document.querySelector(".brand").addEventListener("click", e => {
    e.preventDefault();
    go("overview");
  });

  try {
    const [b, m, prof, prov] = await Promise.all([
      api("/api/benchmarks"), api("/api/models"),
      api("/api/profiles"), api("/api/providers"),
    ]);
    S.benchmarks = b.benchmarks;
    S.models = m;
    S.profiles = prof.profiles;
    S.providers = prov.providers;
    S.benchmark = (S.benchmarks.find(x => x.has_adapter) || S.benchmarks[0] || {}).id || "";
    S.models.local.slice(0, 2).forEach(x => { S.picked.add(x); S.rPicked.add(x); });
    S.raw = SAMPLES[S.benchmark] || "Answer: C";
    // A profile whose dataset is missing opens every screen on "file not
    // found", so default to one that can actually run.
    const usable = S.profiles.find(p => !p.invalid && p.runnable)
      || S.profiles.find(p => !p.invalid) || {};
    S.vProfile = S.probeProfile = S.rProfile = usable.name || "";
    await refreshRuns();
  } catch (e) {
    $("#main").replaceChildren(errMsg("Could not reach the API: " + e.message));
    return;
  }
  const deep = location.hash.replace(/^#/, "");
  if (deep && VIEWS[deep]) { go(deep); return; }
  render();
}
boot();
