"""
The per-item runner: executes ONE eval item end-to-end and returns a TraceRow.

This is the heart of the pipeline (steps 2-8) wired together:
  embed query -> retrieve -> (rerank) -> assemble prompt -> generate (cache-checked)
  -> parse citations -> score retrieval/citation/accuracy/abstention -> judge.

Kept separate from the matrix walk (orchestrator.py) so it can be called from
either concurrency lane, from the tuning search, and from tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..cache.cache import Cache, stable_hash
from ..clients.pricing import PricingRegistry
from ..clients.together_client import TogetherClient
from ..eval import metrics as M
from ..eval.judge import Judge
from ..rag.prompt import PromptConfig, assemble_messages
from ..rag.rerank import Reranker
from ..rag.retrieve import RetrievalConfig, Retriever
from ..store.schema import EvalItem, ItemType, Pass, TraceRow


@dataclass
class RunContext:
    """Everything a run needs that isn't the eval item itself."""
    client: TogetherClient
    retriever: Retriever
    pricing: PricingRegistry
    judge: Optional[Judge]
    cache: Cache
    embedding_model: str
    reranker: Optional[Reranker] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0


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
    active_metrics: Optional[list[str]] = None,
    accuracy_scorer: str = "judge",
    profile_cfg_hash: str = "",
) -> TraceRow:
    active = set(active_metrics or [])
    row = TraceRow(
        run_id=run_id, item_id=item.item_id, model=model, profile=profile_name,
        pass_=pass_, item_type=item.item_type,
        retrieval_mode=retrieval_cfg.mode,
        profile_cfg_hash=profile_cfg_hash,
        retrieval_cfg_hash=stable_hash(retrieval_cfg.__dict__),
        prompt_cfg_hash=stable_hash(prompt_cfg.__dict__),
        gen_params_hash=stable_hash({"t": ctx.temperature, "mt": ctx.max_tokens,
                                     "seed": ctx.seed}),
    )

    try:
        # --- retrieve ------------------------------------------------------
        chunks = ctx.retriever.retrieve(item.query, retrieval_cfg)
        pre_rerank_ids = [c.chunk_id for c in chunks]
        row.retrieved_ids = pre_rerank_ids

        if do_rerank and ctx.reranker is not None:
            chunks = ctx.reranker.rerank(item.query, chunks, top_n=rerank_top_n)
            row.reranked_ids = [c.chunk_id for c in chunks]

        final_ids = row.reranked_ids or row.retrieved_ids
        k = retrieval_cfg.k

        # --- retrieval metrics (only for items that have gold passages) ----
        if item.gold_passage_ids:
            if "hit_rate_at_k" in active:
                row.hit_rate_at_k = M.hit_rate_at_k(final_ids, item.gold_passage_ids, k)
            if "mrr" in active:
                row.mrr = M.mrr(final_ids, item.gold_passage_ids)
            if "ndcg_at_k" in active:
                row.ndcg_at_k = M.ndcg_at_k(final_ids, item.gold_passage_ids, k)
            if "context_recall" in active:
                row.context_recall = M.context_recall(final_ids, item.gold_passage_ids, k)
            if "rerank_hit_delta" in active and row.reranked_ids:
                before = M.hit_rate_at_k(pre_rerank_ids, item.gold_passage_ids, k)
                after = M.hit_rate_at_k(row.reranked_ids, item.gold_passage_ids, k)
                row.rerank_hit_delta = after - before

        # --- assemble prompt ----------------------------------------------
        messages = assemble_messages(item.query, chunks, prompt_cfg,
                                     history=item.history or None)
        row.assembled_prompt = messages[-1]["content"]

        # --- generate (cache-checked) -------------------------------------
        gen_params = {"temperature": ctx.temperature, "max_tokens": ctx.max_tokens,
                      "seed": ctx.seed}
        cache_key = ctx.cache.key_for_generation(model, messages, gen_params)
        cached = ctx.cache.get(cache_key)
        if cached is not None:
            row.cache_hit = True
            row.raw_output = cached["text"]
            row.prompt_tokens = cached["prompt_tokens"]
            row.completion_tokens = cached["completion_tokens"]
            # NOTE: latency is NOT restored from cache — cached latency is
            # meaningless. Latency-lane runs should bypass the cache.
        else:
            gen = ctx.client.generate(model, messages, **gen_params)
            row.raw_output = gen.text
            row.prompt_tokens = gen.prompt_tokens
            row.completion_tokens = gen.completion_tokens
            row.latency_ms = gen.latency_ms
            row.ttft_ms = gen.ttft_ms
            ctx.cache.set(cache_key, {
                "text": gen.text, "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
            })

        # --- cost ----------------------------------------------------------
        if "cost_usd" in active and row.prompt_tokens is not None:
            row.cost_usd = ctx.pricing.generation_cost(
                model, row.prompt_tokens, row.completion_tokens)

        # --- citations -----------------------------------------------------
        cited = M.extract_citations(row.raw_output)
        row.cited_ids = cited
        if "citation_valid_pointer" in active:
            row.citation_valid_pointer = M.citation_valid_pointer(cited, final_ids)
        if "citation_supporting" in active and item.gold_passage_ids:
            row.citation_supporting = M.citation_supporting(cited, item.gold_passage_ids)

        # --- abstention (behavioural) -------------------------------------
        if "abstention" in active or item.item_type == ItemType.UNANSWERABLE:
            row.abstained = M.detect_abstention(row.raw_output)
            row.abstention_correct = M.abstention_correct(
                row.abstained, item.item_type == ItemType.ANSWERABLE)

        # --- accuracy (non-judge scorers) ---------------------------------
        if item.gold_answer and accuracy_scorer != "judge":
            scorer = {
                "exact": M.exact_match, "contains": M.contains_match,
                "numeric": M.numeric_match,
            }.get(accuracy_scorer, M.contains_match)
            row.accuracy = scorer(row.raw_output, item.gold_answer)

        # --- judge-based metrics ------------------------------------------
        needs_judge = ctx.judge is not None and (
            "faithfulness" in active or "answer_relevance" in active or
            "completeness" in active or accuracy_scorer == "judge")
        if needs_judge:
            context_text = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in chunks)
            js = ctx.judge.score(model, item.query, context_text,
                                 row.raw_output, item.gold_answer)
            if "faithfulness" in active:
                row.faithfulness = js.faithfulness
            if "answer_relevance" in active:
                row.answer_relevance = js.answer_relevance
            if "completeness" in active:
                row.completeness = js.completeness
            if accuracy_scorer == "judge" and js.accuracy is not None:
                row.accuracy = js.accuracy

    except Exception as e:  # noqa: BLE001 — record failure, don't crash the matrix
        row.error = f"{type(e).__name__}: {e}"

    return row
