"""
The per-item runner: executes ONE eval item end-to-end and returns a TraceRow.

The pipeline, wired together:

  retrieve -> (rerank) -> (probe attack) -> assemble prompt -> generate
  -> score (via the scorer registry) -> attribute cost

Kept separate from the matrix walk (orchestrator.py) so it can be called from
either concurrency lane, from the tuning search, and from tests.

**What changed.** This function used to *be* the scoring logic: ~200 lines of
`if "metric" in active` blocks for retrieval, citations, abstention, probes,
accuracy, classification and the judge, inline with retrieval, prompting,
generation, caching and cost. Adding a metric meant editing the busiest
function in the codebase.

Scoring now lives in `harness/eval/scoring.py` as a registry of independent
scorers, and this module keeps only the responsibilities that are genuinely
its own: run the pipeline stages in order, handle the cache, attribute cost,
and turn failures into a recorded row rather than a crashed matrix. Each stage
below is a small named function, so the shape of the pipeline is visible at a
glance instead of being buried in a 200-line try block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cache.cache import KeyValueCache, stable_hash
from ..clients.pricing import PricingProvider
from ..eval import probes as P
from ..eval.judge import Judge
from ..eval.scoring import (
    DEFAULT_SCORERS,
    Scorer,
    ScoringContext,
    parse_label,
    run_scorers,
)
from ..rag.prompt import PromptConfig, assemble_messages
from ..rag.rerank import Reranker
from ..rag.retrieve import RetrievalConfig, Retriever
from ..store.schema import EvalItem, Pass, TaskType, TraceRow, classify_error

__all__ = ["RunContext", "run_item", "parse_label"]


@dataclass
class RunContext:
    """Everything a run needs that isn't the eval item itself.

    Depends on protocols (`KeyValueCache`) rather than concrete classes where a
    substitution is genuinely useful — the latency lane swaps in a null cache,
    and tests swap in fakes.
    """

    client: object
    retriever: Retriever | None
    pricing: PricingProvider
    judge: Judge | None
    cache: KeyValueCache
    embedding_model: str
    reranker: Reranker | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0
    meter: object = None          # CostMeter, for run-level spend and the budget
    task: TaskType = TaskType.RAG
    stream: bool = False          # set by the latency lane, to capture TTFT
    provider: str = ""
    abstention_judge: bool = False
    label_set: list[str] = field(default_factory=list)
    # Injectable so a caller can add a metric family without touching this
    # module, and so tests can run one scorer in isolation.
    scorers: tuple[Scorer, ...] = DEFAULT_SCORERS


def _meter_delta(before: dict, after: dict, key: str) -> float:
    """Spend on one subsystem attributable to this item, from the shared meter.

    Deriving per-item cost from a meter delta rather than re-pricing tokens
    locally means it always reconciles with the run total — including cache
    hits, which correctly cost nothing.

    Caveat worth naming: under concurrency, other threads bill against the same
    meter between the two reads, so a single row's non-generation costs are an
    apportionment rather than an exact attribution. The run-level totals are
    exact, which is what the budget and the cost report are built on.
    """
    return max(0.0, after.get(key, 0.0) - before.get(key, 0.0))


# --------------------------------------------------------------------------- #
#  Pipeline stages
# --------------------------------------------------------------------------- #
def _check_budget(ctx: RunContext) -> None:
    """Stop before spending, not after.

    Once the ceiling is crossed, every queued item would otherwise still issue
    its API call and only then discover the budget is gone — you would pay for
    the entire remaining matrix and throw all of it away.
    """
    if ctx.meter is not None and getattr(ctx.meter, "tripped", False):
        from ..clients.cost import BudgetExceeded

        raise BudgetExceeded(ctx.meter.spent, ctx.meter.budget_usd)


def _retrieve(item: EvalItem, ctx: RunContext, cfg: RetrievalConfig,
              do_rerank: bool, rerank_top_n: int) -> tuple[list, list[str], list[str]]:
    """Returns (chunks, pre_rerank_ids, reranked_ids). Empty for non-RAG tasks."""
    if ctx.task != TaskType.RAG or ctx.retriever is None:
        return [], [], []
    chunks = ctx.retriever.retrieve(item.query, cfg)
    pre_rerank_ids = [c.chunk_id for c in chunks]
    reranked_ids: list[str] = []
    if do_rerank and ctx.reranker is not None:
        chunks = ctx.reranker.rerank(item.query, chunks, top_n=rerank_top_n)
        reranked_ids = [c.chunk_id for c in chunks]
    return chunks, pre_rerank_ids, reranked_ids


def _generate(model: str, messages: list[dict], ctx: RunContext,
              row: TraceRow) -> None:
    """Generate, or serve from cache, writing the outcome onto `row`."""
    gen_params = {"temperature": ctx.temperature,
                  "max_tokens": ctx.max_tokens, "seed": ctx.seed}
    cache_key = ctx.cache.key_for_generation(model, messages, gen_params)
    cached = ctx.cache.get(cache_key)

    if cached is not None:
        row.cache_hit = True
        row.raw_output = cached["text"]
        row.prompt_tokens = cached.get("prompt_tokens")
        row.completion_tokens = cached.get("completion_tokens")
        row.finish_reason = cached.get("finish_reason")
        row.truncated = cached.get("finish_reason") == "length"
        if ctx.meter is not None:
            ctx.meter.record_cache_hit()
        # Latency is deliberately NOT restored from cache — a cached timing is
        # a fabrication. The latency lane bypasses the cache entirely.
        return

    gen = ctx.client.generate(model, messages, stream=ctx.stream, **gen_params)
    row.raw_output = gen.text
    row.prompt_tokens = gen.prompt_tokens
    row.completion_tokens = gen.completion_tokens
    row.latency_ms = gen.latency_ms
    row.ttft_ms = gen.ttft_ms
    row.finish_reason = gen.finish_reason
    row.truncated = gen.truncated
    ctx.cache.set(cache_key, {
        "text": gen.text, "prompt_tokens": gen.prompt_tokens,
        "completion_tokens": gen.completion_tokens,
        "finish_reason": gen.finish_reason,
    })


def _attribute_cost(row: TraceRow, ctx: RunContext, model: str,
                    meter_before: dict) -> None:
    """Split this item's spend by the subsystem that caused it.

    Generation is priced directly; judge, embedding and rerank come from meter
    deltas. An unpriced model leaves the field None rather than 0.0 — a
    zero-cost model would win any cost-weighted leaderboard.
    """
    if row.prompt_tokens is not None:
        try:
            row.gen_cost_usd = ctx.pricing.generation_cost(
                model, row.prompt_tokens, row.completion_tokens or 0)
        except KeyError:
            row.gen_cost_usd = None

    if ctx.meter is not None:
        after = ctx.meter.summary()
        row.judge_cost_usd = _meter_delta(meter_before, after, "judge_usd")
        row.embed_cost_usd = _meter_delta(meter_before, after, "embedding_usd")
        row.rerank_cost_usd = _meter_delta(meter_before, after, "rerank_usd")

    row.cost_usd = sum(
        v for v in (row.gen_cost_usd, row.judge_cost_usd,
                    row.embed_cost_usd, row.rerank_cost_usd)
        if v is not None
    ) or None


# --------------------------------------------------------------------------- #
#  The runner
# --------------------------------------------------------------------------- #
def run_item(
    item: EvalItem,
    model: str,
    profile_name: str,
    pass_: Pass,
    retrieval_cfg: RetrievalConfig,
    prompt_cfg: PromptConfig,
    ctx: RunContext,
    run_id: str,
    do_rerank: bool = False,
    rerank_top_n: int = 5,
    active_metrics: list[str] | None = None,
    accuracy_scorer: str = "judge",
    profile_cfg_hash: str = "",
) -> TraceRow:
    active = set(active_metrics or [])
    row = TraceRow(
        run_id=run_id, item_id=item.item_id, model=model, profile=profile_name,
        pass_=pass_, item_type=item.item_type,
        retrieval_mode=retrieval_cfg.mode if ctx.task == TaskType.RAG else None,
        profile_cfg_hash=profile_cfg_hash,
        provider=ctx.provider,
        retrieval_cfg_hash=stable_hash(retrieval_cfg.__dict__),
        prompt_cfg_hash=stable_hash(prompt_cfg.__dict__),
        gen_params_hash=stable_hash({"t": ctx.temperature, "mt": ctx.max_tokens,
                                     "seed": ctx.seed}),
        consistency_group=(item.meta or {}).get("consistency_group"),
        human_label=item.human_label,
    )

    meter_before = ctx.meter.summary() if ctx.meter is not None else {}

    try:
        _check_budget(ctx)

        chunks, pre_rerank_ids, reranked_ids = _retrieve(
            item, ctx, retrieval_cfg, do_rerank, rerank_top_n)
        row.retrieved_ids = pre_rerank_ids
        row.reranked_ids = reranked_ids

        # The probe mutates the context the MODEL sees. Retrieval metrics are
        # scored against the ids above, so the retriever is judged on its own
        # work and the model on the adversarial context it was handed.
        attacked = P.apply_context_attack(chunks, item) if chunks else chunks

        messages = assemble_messages(item.query, attacked, prompt_cfg,
                                     history=item.history or None,
                                     task=ctx.task, label_set=ctx.label_set)
        row.assembled_prompt = messages[-1]["content"]

        _generate(model, messages, ctx, row)

        scoring = run_scorers(
            ScoringContext(
                item=item, model=model, answer=row.raw_output, active=active,
                accuracy_scorer=accuracy_scorer, task=ctx.task,
                retrieved_ids=pre_rerank_ids, reranked_ids=reranked_ids,
                k=retrieval_cfg.k, chunks_shown=attacked,
                context_text="\n\n".join(
                    f"[{c.chunk_id}] {c.text}" for c in attacked),
                judge=ctx.judge, abstention_judge=ctx.abstention_judge,
                label_set=ctx.label_set,
            ),
            scorers=ctx.scorers,
        )
        for field_name, value in scoring.values.items():
            setattr(row, field_name, value)
        if scoring.failures:
            # Recorded, not raised: the generation was paid for and every other
            # metric from it is still valid.
            row.error = "; ".join(f"{k}: {v}" for k, v in scoring.failures.items())
            row.error_kind = "scoring"

        _attribute_cost(row, ctx, model, meter_before)

    except Exception as e:  # noqa: BLE001 — record failure, don't kill the matrix
        row.error = f"{type(e).__name__}: {e}"
        row.error_kind = classify_error(e)
        # A budget trip must stop the whole run, not just mark one item. Every
        # remaining item would fail identically, burning wall-clock to write
        # thousands of identical error rows.
        if row.error_kind == "budget":
            raise

    return row
