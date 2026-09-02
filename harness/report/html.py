"""
Self-contained HTML report.

A benchmark result that lives in a terminal buffer or a Parquet file reaches
exactly one person. The decision it informs is usually made by several — an
engineer, whoever owns the budget, sometimes whoever owns the risk. This writes
a single file with no external assets, so it can be attached to a ticket, mailed,
or committed as a build artifact.

Deliberately dependency-free: no plotting library, no CDN, no template engine.
Bars are styled divs. A report that needs a network fetch to render is not a
report you can attach to anything.
"""

from __future__ import annotations

import html as _html
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import aggregate as A
from . import stats as S
from .decide import model_aggregates, project_cost

CSS = """
:root{--bg:#fff;--fg:#16181d;--muted:#5b6472;--line:#e3e6ea;--accent:#2f6fed;
--good:#1a7f4b;--bad:#b3261e;--warn:#8a6100;--card:#f7f8fa;}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--fg:#e8eaed;
--muted:#9aa4b2;--line:#2a2f37;--accent:#6f9bff;--good:#4ade80;--bad:#ff6b6b;
--warn:#fbbf24;--card:#1c1f25;}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.6rem;margin:0 0 .25rem}
h2{font-size:1.15rem;margin:2.5rem 0 .5rem;padding-bottom:.35rem;
border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 2rem;font-size:.9rem}
.note{color:var(--muted);font-size:.87rem;margin:.35rem 0 .9rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.5rem 0}
th,td{text-align:left;padding:.45rem .6rem;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:.8rem;text-transform:uppercase;
letter-spacing:.03em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr.top td{font-weight:600}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.bar{height:8px;background:var(--accent);border-radius:4px;display:inline-block;
vertical-align:middle}
.bartrack{background:var(--line);border-radius:4px;width:120px;display:inline-block;
vertical-align:middle;margin-right:.5rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
gap:.75rem;margin:1rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:.8rem .9rem}
.card .k{color:var(--muted);font-size:.75rem;text-transform:uppercase;
letter-spacing:.04em}
.card .v{font-size:1.35rem;font-weight:650;margin-top:.15rem;
font-variant-numeric:tabular-nums}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.pill{display:inline-block;padding:.08rem .45rem;border-radius:99px;
font-size:.75rem;border:1px solid var(--line)}
.verdict{background:var(--card);border-left:3px solid var(--accent);
padding:.6rem .8rem;border-radius:0 8px 8px 0;margin:.5rem 0}
footer{margin-top:3rem;color:var(--muted);font-size:.8rem;
border-top:1px solid var(--line);padding-top:1rem}
"""


def _esc(x) -> str:
    return _html.escape("" if x is None else str(x))


def _fmt(v, places: int = 4) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{places}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 \
            else f"{v:,.0f}"
    return str(v)


def _table(df: pd.DataFrame, highlight_first: bool = False,
           places: int = 4) -> str:
    if df is None or df.empty:
        return '<p class="note">No data.</p>'
    head = "".join(
        f'<th class="{"num" if pd.api.types.is_numeric_dtype(df[c]) else ""}">'
        f"{_esc(c)}</th>" for c in df.columns)
    body = []
    for i, (_, row) in enumerate(df.iterrows()):
        cells = "".join(
            f'<td class="{"num" if isinstance(v, (int, float, np.number)) else ""}">'
            f"{_esc(_fmt(v, places))}</td>" for v in row)
        cls = ' class="top"' if highlight_first and i == 0 else ""
        body.append(f"<tr{cls}>{cells}</tr>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _bar_table(df: pd.DataFrame, label_col: str, value_col: str,
               places: int = 4) -> str:
    """A table with an inline bar, so relative magnitude reads at a glance."""
    if df is None or df.empty or value_col not in df.columns:
        return '<p class="note">No data.</p>'
    vals = df[value_col].astype(float)
    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        v = float(r[value_col])
        pct = max(2.0, 100.0 * (v - lo) / span) if hi > lo else 100.0
        cls = ' class="top"' if i == 0 else ""
        rows.append(
            f"<tr{cls}><td>{_esc(r[label_col])}</td>"
            f'<td class="num">{_esc(_fmt(v, places))}</td>'
            f'<td><span class="bartrack"><span class="bar" '
            f'style="width:{pct:.0f}%"></span></span></td></tr>')
    return (f'<div class="scroll"><table><thead><tr><th>model</th>'
            f'<th class="num">{_esc(value_col)}</th><th></th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def _cards(items: list[tuple[str, str, str]]) -> str:
    cards = "".join(
        f'<div class="card"><div class="k">{_esc(k)}</div>'
        f'<div class="v {cls}">{_esc(v)}</div></div>' for k, v, cls in items)
    return f'<div class="cards">{cards}</div>'


def build_html_report(df: pd.DataFrame, profile, metric: str = "accuracy",
                      title: str | None = None) -> str:
    """Render the full report as one HTML string."""
    quality = (df[df["pass_"].isin(["baseline", "adapted"])]
               if "pass_" in df.columns else df)
    parts: list[str] = []
    name = title or f"{profile.name} evaluation"

    # -- headline ------------------------------------------------------- #
    n_models = df["model"].nunique() if "model" in df.columns else 0
    n_items = df["item_id"].nunique() if "item_id" in df.columns else 0
    total_cost = (float(df["cost_usd"].sum(skipna=True))
                  if "cost_usd" in df.columns else 0.0)
    err_rate = (float(df["error"].notna().mean())
                if "error" in df.columns and len(df) else 0.0)

    parts.append(f"<h1>{_esc(name)}</h1>")
    parts.append(f'<p class="sub">{_esc(profile.description or "")} '
                 f'Generated {time.strftime("%Y-%m-%d %H:%M")} · '
                 f'task={_esc(profile.task.value)} · '
                 f'scorer={_esc(profile.accuracy_scorer)}</p>')
    parts.append(_cards([
        ("models", str(n_models), ""),
        ("eval items", f"{n_items:,}", ""),
        ("rows", f"{len(df):,}", ""),
        ("run cost", f"${total_cost:,.4f}", ""),
        ("error rate", f"{err_rate * 100:.1f}%",
         "bad" if err_rate > 0.05 else "good"),
    ]))

    # -- leaderboard ----------------------------------------------------- #
    comp = A.weighted_composite(quality, profile.metric_weights)
    parts.append("<h2>Leaderboard (weighted composite)</h2>")
    parts.append('<p class="note">Deterministic arithmetic over the profile\'s '
                 'metric weights. Lower-is-better metrics carry negative '
                 'weights.</p>')
    parts.append(_bar_table(comp, "model", "composite"))
    if not comp.empty:
        parts.append(_table(comp))

    # -- significance ---------------------------------------------------- #
    sig = S.significance_matrix(quality, metric=metric)
    parts.append(f"<h2>Is the difference real? ({_esc(metric)})</h2>")
    parts.append('<p class="note">Paired tests on the same items, '
                 'Holm-Bonferroni corrected. Two models with overlapping '
                 'confidence intervals can still differ significantly - and a '
                 'visible gap on the leaderboard can still be noise.</p>')
    if not sig.empty:
        for _, r in sig.iterrows():
            mark = ("<span class='pill good'>significant</span>"
                    if r["significant"] else "<span class='pill'>n.s.</span>")
            parts.append(f'<div class="verdict">{mark} '
                         f'{_esc(r["model_a"])} vs {_esc(r["model_b"])}: '
                         f'{_esc(r["verdict"])}</div>')
        parts.append(_table(sig[["model_a", "model_b", "n_pairs", "diff",
                                 "p_value", "p_adjusted", "test"]]))

    pr = S.power_report(quality, metric)
    parts.append("<h2>Statistical power</h2>")
    parts.append(f'<p class="note">{_esc(pr.summary())}. Differences smaller '
                 f'than that are not measurable with this evalset.</p>')
    if pr.suggestions:
        rows = pd.DataFrame([{"to detect a gap of": k.split("_")[1],
                              "paired items needed": v}
                             for k, v in sorted(pr.suggestions.items())])
        parts.append(_table(rows, places=0))

    # -- efficiency / Pareto --------------------------------------------- #
    parts.append("<h2>Cost, latency and the frontier</h2>")
    parts.append('<p class="note">Latency comes from the low-concurrency lane '
                 'where available; p95 rather than the mean, because SLAs are '
                 'written on tails.</p>')
    parts.append(_table(model_aggregates(df)))
    pf = A.pareto_frontier(df)
    if not pf.empty:
        parts.append("<h2>Pareto frontier</h2>")
        parts.append('<p class="note">Models not beaten on all of accuracy, '
                     'cost and latency at once. When leaders tie on quality, '
                     'this is where the decision actually lives.</p>')
        parts.append(_table(pf))

    proj = project_cost(df, 10_000)
    if not proj.empty:
        parts.append("<h2>Projected production spend (10,000 queries/day)</h2>")
        parts.append('<p class="note">Generation cost only - the judge is '
                     'evaluation infrastructure, not something you pay for in '
                     'production.</p>')
        parts.append(_table(proj, places=2))

    # -- quality detail --------------------------------------------------- #
    qual_cols = [c for c in ("accuracy", "faithfulness", "answer_relevance",
                             "completeness", "token_f1",
                             "citation_valid_pointer", "citation_supporting",
                             "abstention_correct")
                 if c in quality.columns and quality[c].notna().any()]
    if qual_cols:
        parts.append("<h2>Answer &amp; citation quality</h2>")
        parts.append(_table(quality.groupby("model")[qual_cols]
                            .mean(numeric_only=True).reset_index()))

    retr_cols = [c for c in ("hit_rate_at_k", "mrr", "ndcg_at_k",
                             "context_recall", "context_precision",
                             "average_precision", "rerank_mrr_delta")
                 if c in quality.columns and quality[c].notna().any()]
    if retr_cols:
        parts.append("<h2>Retrieval quality</h2>")
        parts.append(_table(quality.groupby("model")[retr_cols]
                            .mean(numeric_only=True).reset_index()))

    # -- safety ----------------------------------------------------------- #
    safety_cols = [c for c in ("injection_resisted", "abstention_correct",
                               "pii_leaked")
                   if c in df.columns and df[c].notna().any()]
    if safety_cols:
        parts.append("<h2>Robustness &amp; security probes</h2>")
        parts.append('<p class="note">Injection resistance is the fraction of '
                     'planted hostile instructions the model ignored. Anything '
                     'below 1.0 means the model took orders from a retrieved '
                     'passage.</p>')
        parts.append(_table(df.groupby("model")[safety_cols]
                            .mean(numeric_only=True).reset_index()))

    cons = A.consistency_report(df)
    if not cons.empty:
        parts.append("<h2>Paraphrase consistency</h2>")
        parts.append('<p class="note">How stable each model\'s answer is when '
                     'the same question is reworded.</p>')
        parts.append(_table(cons))

    # -- tuning gain ------------------------------------------------------ #
    if "pass_" in df.columns and {"baseline", "adapted"} <= set(df["pass_"].unique()):
        metrics = [m for m in profile.active_metrics if m in df.columns]
        parts.append("<h2>Tuning gain (adapted &minus; baseline)</h2>")
        parts.append('<p class="note">Every model received an identical tuning '
                     'budget, so this is how much each one benefits from '
                     'tuning - not how hard someone tried.</p>')
        parts.append(_table(A.tuning_gain(df, metrics)))

    # -- diagnostics ------------------------------------------------------ #
    err = A.error_attribution(df)
    if not err.empty and err.get("errors", pd.Series([0])).sum():
        parts.append("<h2>Error attribution</h2>")
        parts.append('<p class="note">What failed and why. rate_limit means '
                     'lower your concurrency; context_length means lower k; '
                     'content_filter is a finding about the model.</p>')
        parts.append(_table(err))

    trunc = A.truncation_report(df)
    if not trunc.empty and trunc["truncated"].sum():
        parts.append("<h2>Truncated answers</h2>")
        parts.append('<p class="note">Answers cut off by max_tokens. Their '
                     'completeness and citation scores are not valid - raise '
                     'max_tokens and re-run.</p>')
        parts.append(_table(trunc))

    if "human_label" in df.columns and df["human_label"].notna().any():
        cal = A.judge_calibration(quality)
        parts.append("<h2>Judge calibration</h2>")
        parts.append(_table(pd.DataFrame([cal])))

    parts.append(
        '<footer>Generated by the multi-model LLM evaluation harness. '
        'Every number here is reproducible from the trace store; cost figures '
        'depend on the dated prices in configs/pricing.yaml.</footer>')

    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_esc(name)}</title><style>{CSS}</style></head><body>"
            f"<div class='wrap'>{''.join(parts)}</div></body></html>")


def write_html_report(df: pd.DataFrame, profile, out_path: str,
                      metric: str = "accuracy") -> str:
    html_text = build_html_report(df, profile, metric=metric)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_text, encoding="utf-8")
    return str(p)
