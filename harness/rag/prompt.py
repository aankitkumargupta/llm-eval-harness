"""
Prompt assembly (pipeline step 5), the single most important tuning knob.

In the BASELINE pass this template is fixed and identical for every model. In the
ADAPTED pass, tuning may change the system prompt, the few-shot exemplars, and
the ordering of context. Everything the model sees is built here, so this is
where "how much does a model gain from tuning" is actually exercised.

The assembled prompt is logged verbatim on the TraceRow.

Two additions:
  * `context_order` gained `middle_gold` and `shuffle`, so the lost-in-the-middle
    effect is something the harness can *test* rather than a comment. It was
    declared as a knob but only ever `as_is` or `reverse`, and neither was
    reachable from any profile's tuning space.
  * Non-RAG task shapes (DIRECT, CLASSIFY) get their own assembly, which is what
    lets the same runner evaluate classification and free-form prompting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..store.schema import RetrievedChunk, TaskType

# A neutral default for the baseline pass. Instructs citation by chunk_id so
# citation validity can be scored. Deliberately plain, the point of a baseline
# is that no model gets a hand-tailored advantage.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context passages. Cite the passages you use by their id in square "
    "brackets, e.g. [doc1#3]. If the answer is not contained in the context, "
    "say you don't know rather than guessing."
)

DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question accurately and "
    "concisely."
)

CLASSIFY_SYSTEM_PROMPT = (
    "You are a precise classifier. Read the input and respond with exactly one "
    "label from the allowed set. Output only the label, with no explanation."
)


@dataclass
class PromptConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    few_shot: list[dict] = field(default_factory=list)  # [{role, content}, ...]
    # as_is | reverse | middle_gold | shuffle
    context_order: str = "as_is"
    max_context_chunks: int = 10
    order_seed: int = 0  # keeps `shuffle` reproducible


def _format_context(chunks: list[RetrievedChunk], order: str, limit: int,
                    gold_ids: set[str] | None = None,
                    seed: int = 0) -> str:
    selected = list(chunks[:limit])

    if order == "reverse":
        selected.reverse()
    elif order == "shuffle":
        random.Random(seed).shuffle(selected)
    elif order == "middle_gold" and gold_ids:
        # Force the gold passage into the middle of the window, the position
        # where long-context models most often lose it. A model that reads its
        # whole context is unaffected; a model that skims the ends falls over,
        # and that difference is invisible to any accuracy average.
        gold = [c for c in selected if c.chunk_id in gold_ids]
        rest = [c for c in selected if c.chunk_id not in gold_ids]
        mid = len(rest) // 2
        selected = rest[:mid] + gold + rest[mid:]

    # The bracketed id is what the model is asked to cite, and what citation
    # validity resolves against.
    return "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in selected)


def assemble_messages(query: str, chunks: list[RetrievedChunk],
                      cfg: PromptConfig,
                      history: list[tuple[str, str]] | None = None,
                      task: TaskType = TaskType.RAG,
                      label_set: list[str] | None = None,
                      gold_ids: set[str] | None = None) -> list[dict]:
    """Build the OpenAI-style messages list.

    Order: system prompt -> few-shot exemplars -> prior turns -> the user turn.
    """
    system = cfg.system_prompt
    if task == TaskType.CLASSIFY and system == DEFAULT_SYSTEM_PROMPT:
        system = CLASSIFY_SYSTEM_PROMPT
    elif task == TaskType.DIRECT and system == DEFAULT_SYSTEM_PROMPT:
        system = DIRECT_SYSTEM_PROMPT

    if task == TaskType.CLASSIFY and label_set:
        system = f"{system}\n\nAllowed labels: {', '.join(label_set)}"

    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend(cfg.few_shot)

    if history:
        for role, content in history:
            messages.append({"role": role, "content": content})

    if task == TaskType.RAG and chunks:
        context_block = _format_context(chunks, cfg.context_order,
                                        cfg.max_context_chunks, gold_ids,
                                        cfg.order_seed)
        user_content = f"Context passages:\n{context_block}\n\nQuestion: {query}"
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})
    return messages
