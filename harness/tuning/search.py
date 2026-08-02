"""
Equal-budget tuning search (the adapted pass).

This is the core novel mechanism: for each model, generate up to N candidate
configs from the profile's declared knobs, evaluate each on a HELD-OUT DEV split,
and keep the best. N (the tuning budget) is identical for every model, which is
what makes "equal tuning budget" literally true rather than aspirational.

We freeze model weights — tuning only searches over prompt + few-shot +
retrieval params. The winner is then run once on the TEST split by the
orchestrator; that run is the adapted-pass result. The baseline-vs-adapted delta
on the test split is the tuning-gain signal.

Candidate generation is deterministic (seeded) so a re-run picks the same
candidates — reproducibility again.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Callable

from ..profiles.profile import Profile
from ..rag.prompt import PromptConfig, DEFAULT_SYSTEM_PROMPT
from ..rag.retrieve import RetrievalConfig
from ..store.schema import RetrievalMode


@dataclass
class Candidate:
    """One point in the tuning search space: a retrieval config + a prompt config."""
    retrieval: RetrievalConfig
    prompt: PromptConfig
    rerank: bool
    rerank_top_n: int


def enumerate_candidates(profile: Profile, budget: int,
                         seed: int = 0) -> list[Candidate]:
    """
    Build the candidate grid from the profile's knobs, then take a reproducible
    sample of size `budget`. If the full grid is smaller than the budget we use
    all of it; if larger, we sample WITHOUT replacement using a fixed seed so
    every model sees the same candidate set.
    """
    knobs = profile.knobs
    system_prompts = knobs.system_prompts or [DEFAULT_SYSTEM_PROMPT]
    few_shot_sets = knobs.few_shot_sets or [[]]

    grid = list(itertools.product(
        knobs.retrieval_modes or ["dense"],
        knobs.k_values or [profile.k],
        knobs.rerank_options or [profile.rerank],
        knobs.rerank_top_n or [profile.rerank_top_n],
        system_prompts,
        few_shot_sets,
    ))

    rng = random.Random(seed)
    if len(grid) > budget:
        grid = rng.sample(grid, budget)

    candidates: list[Candidate] = []
    for mode, k, do_rerank, top_n, sys_prompt, fewshot in grid:
        candidates.append(Candidate(
            retrieval=RetrievalConfig(
                mode=RetrievalMode(mode), k=k,
                embedding_model=profile.embedding_model,
            ),
            prompt=PromptConfig(system_prompt=sys_prompt, few_shot=fewshot,
                                max_context_chunks=k),
            rerank=do_rerank,
            rerank_top_n=top_n,
        ))
    return candidates


def search(profile: Profile, model: str,
           evaluate_candidate: Callable[[Candidate], float],
           seed: int = 0) -> tuple[Candidate, float, list[float]]:
    """
    Run the budget-limited search for one model.

    `evaluate_candidate` is injected by the orchestrator: it runs the candidate
    over the dev split and returns a single scalar score (the profile's weighted
    composite on dev). Keeping it injected means this module has no dependency on
    the RAG/eval machinery and is trivially testable.

    Returns (best_candidate, best_score, all_scores).
    """
    candidates = enumerate_candidates(profile, profile.tuning_budget, seed=seed)
    scores = []
    best_idx, best_score = 0, float("-inf")
    for i, cand in enumerate(candidates):
        s = evaluate_candidate(cand)
        scores.append(s)
        if s > best_score:
            best_idx, best_score = i, s
    return candidates[best_idx], best_score, scores
