"""
Equal-budget tuning search (the adapted pass).

The core mechanism: for each model, generate up to N candidate configs from the
profile's declared knobs, evaluate each on a HELD-OUT DEV split, keep the best.
N is identical for every model, which is what makes "equal tuning budget"
literally true rather than aspirational.

Model weights are frozen — tuning only searches prompt, few-shot, context order
and retrieval params. The winner is then run once on the TEST split; that run is
the adapted result, and the baseline-vs-adapted delta on test is the tuning-gain
signal.

Two things beyond the original:

**Successive halving.** Pure random search spends the same dev budget on an
obviously bad candidate as on the winner. Halving evaluates all candidates on a
small dev slice, keeps the top half, doubles the slice, repeats. Same total
evaluations, far more of them spent on candidates that might actually win — and
because the schedule depends only on the budget, it stays identical across
models, so equal-budget fairness survives.

**Deduplication.** The grid can contain configs that differ only in a parameter
the rest of the config makes irrelevant — `rerank_top_n` when `rerank` is False
is the obvious one. Sampling those wastes a slot on a literal duplicate of
another candidate, and it does so unequally across profiles.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable
from dataclasses import dataclass

from ..profiles.profile import Profile
from ..rag.prompt import DEFAULT_SYSTEM_PROMPT, PromptConfig
from ..rag.retrieve import RetrievalConfig
from ..store.schema import RetrievalMode


@dataclass
class Candidate:
    """One point in the search space: a retrieval config + a prompt config."""
    retrieval: RetrievalConfig
    prompt: PromptConfig
    rerank: bool
    rerank_top_n: int

    def identity(self) -> tuple:
        """What makes this candidate meaningfully distinct from another.

        `rerank_top_n` is included only when reranking is on: otherwise two
        candidates identical in every effective respect look different and both
        consume a slot from the budget.
        """
        return (
            self.retrieval.mode.value, self.retrieval.k,
            self.prompt.system_prompt, self.prompt.context_order,
            len(self.prompt.few_shot),
            tuple(sorted(m.get("content", "") for m in self.prompt.few_shot)),
            self.rerank, self.rerank_top_n if self.rerank else None,
        )

    def describe(self) -> str:
        """Human-readable summary, for reporting which config actually won."""
        bits = [f"mode={self.retrieval.mode.value}", f"k={self.retrieval.k}",
                f"order={self.prompt.context_order}"]
        if self.rerank:
            bits.append(f"rerank@{self.rerank_top_n}")
        if self.prompt.few_shot:
            bits.append(f"fewshot={len(self.prompt.few_shot)}")
        bits.append(f"prompt#{abs(hash(self.prompt.system_prompt)) % 1000:03d}")
        return " ".join(bits)


def enumerate_candidates(profile: Profile, budget: int,
                         seed: int = 0) -> list[Candidate]:
    """Build the candidate grid from the profile's knobs, then sample to `budget`.

    If the deduplicated grid is smaller than the budget we use all of it; if
    larger, we sample without replacement using a fixed seed, so every model
    sees the identical candidate set. That identity is the fairness guarantee.
    """
    knobs = profile.knobs
    system_prompts = knobs.system_prompts or [DEFAULT_SYSTEM_PROMPT]
    few_shot_sets = knobs.few_shot_sets or [[]]
    context_orders = knobs.context_orders or ["as_is"]

    grid = list(itertools.product(
        knobs.retrieval_modes or [profile.retrieval_mode.value],
        knobs.k_values or [profile.k],
        knobs.rerank_options or [profile.rerank],
        knobs.rerank_top_n or [profile.rerank_top_n],
        system_prompts,
        few_shot_sets,
        context_orders,
    ))

    candidates: list[Candidate] = []
    seen: set[tuple] = set()
    for mode, k, do_rerank, top_n, sys_prompt, fewshot, order in grid:
        cand = Candidate(
            retrieval=RetrievalConfig(
                mode=RetrievalMode(mode), k=k,
                embedding_model=profile.embedding_model,
                dense_weight=profile.dense_weight,
                sparse_weight=profile.sparse_weight,
                # Reranking can only reorder what retrieval handed it, so a
                # candidate that reranks must over-fetch or there is nothing to
                # promote and rerank gain measures as zero by construction.
                candidate_multiplier=3 if do_rerank else profile.candidate_multiplier,
            ),
            prompt=PromptConfig(system_prompt=sys_prompt, few_shot=fewshot,
                                max_context_chunks=k, context_order=order,
                                order_seed=seed),
            rerank=do_rerank,
            rerank_top_n=top_n,
        )
        ident = cand.identity()
        if ident in seen:
            continue
        seen.add(ident)
        candidates.append(cand)

    if len(candidates) > budget:
        candidates = random.Random(seed).sample(candidates, budget)
    return candidates


# ---------------------------------------------------------------------------
#  Search strategies
# ---------------------------------------------------------------------------
def search(profile: Profile, model: str,
           evaluate_candidate: Callable[[Candidate], float],
           seed: int = 0) -> tuple[Candidate, float, list[float]]:
    """Exhaustive search: evaluate every candidate on the full dev split.

    `evaluate_candidate` is injected by the orchestrator, so this module has no
    dependency on the RAG or eval machinery and stays trivially testable.

    Returns (best_candidate, best_score, all_scores).
    """
    candidates = enumerate_candidates(profile, profile.tuning_budget, seed=seed)
    if not candidates:
        raise ValueError(
            f"Profile '{profile.name}' produced no tuning candidates. Check its "
            f"`knobs:` block — every list must have at least one entry.")
    scores: list[float] = []
    best_idx, best_score = 0, float("-inf")
    for i, cand in enumerate(candidates):
        s = evaluate_candidate(cand)
        scores.append(s)
        if s > best_score:
            best_idx, best_score = i, s
    return candidates[best_idx], best_score, scores


def halving_schedule(n_candidates: int, n_dev: int,
                     min_slice: int = 4) -> list[tuple[int, int]]:
    """Plan a successive-halving search: [(n_survivors, dev_items_each), ...].

    Starts wide and shallow, ends narrow and deep. The schedule is a pure
    function of (n_candidates, n_dev), so every model runs the identical
    schedule and the equal-budget property is preserved exactly.
    """
    schedule: list[tuple[int, int]] = []
    survivors = max(1, n_candidates)
    slice_size = max(min_slice, n_dev // max(1, n_candidates))
    while survivors > 1 and slice_size < n_dev:
        schedule.append((survivors, min(slice_size, n_dev)))
        survivors = max(1, survivors // 2)
        slice_size *= 2
    schedule.append((survivors, n_dev))  # final round always uses full dev
    return schedule


def search_halving(
    profile: Profile,
    evaluate: Callable[[Candidate, int], float],
    n_dev: int,
    seed: int = 0,
) -> tuple[Candidate, float, list[tuple[Candidate, float]]]:
    """Successive-halving search.

    `evaluate(candidate, n_items)` scores a candidate on the first `n_items` of
    the dev split. Using a *prefix* rather than a random subsample keeps early
    rounds comparable across candidates — otherwise a candidate could win by
    drawing an easier slice, which is the failure mode this is meant to avoid.
    """
    candidates = enumerate_candidates(profile, profile.tuning_budget, seed=seed)
    if not candidates:
        raise ValueError(
            f"Profile '{profile.name}' produced no tuning candidates.")

    alive = list(candidates)
    history: list[tuple[Candidate, float]] = []
    last_scores: dict[int, float] = {}

    for survivors, n_items in halving_schedule(len(candidates), n_dev):
        scored = []
        for cand in alive:
            score = evaluate(cand, n_items)
            scored.append((cand, score))
            last_scores[id(cand)] = score
        history.extend(scored)
        scored.sort(key=lambda t: t[1], reverse=True)
        alive = [c for c, _ in scored[:max(1, survivors)]]
        if len(alive) <= 1:
            break

    best = alive[0]
    return best, last_scores.get(id(best), float("-inf")), history
