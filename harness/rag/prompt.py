"""
Prompt assembly (pipeline step 5) — the single most important tuning knob.

In the BASELINE pass this template is fixed and identical for every model. In the
ADAPTED pass, tuning is allowed to change the system prompt, the few-shot
exemplars, and the ordering of context. Everything the model sees is built here,
so this is where "how much does a model gain from tuning" is actually exercised.

The assembled prompt (and its token count) is logged verbatim on the TraceRow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..store.schema import RetrievedChunk


# A neutral default used by the baseline pass. Instructs citation by chunk_id so
# citation-validity can be scored. Deliberately plain — the point of baseline is
# that no model gets a hand-tailored advantage.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context passages. Cite the passages you use by their id in square "
    "brackets, e.g. [doc1#3]. If the answer is not contained in the context, "
    "say you don't know rather than guessing."
)


@dataclass
class PromptConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    few_shot: list[dict] = field(default_factory=list)  # [{role, content}, ...]
    context_order: str = "as_is"   # "as_is" | "reverse" (for lost-in-the-middle studies)
    max_context_chunks: int = 10


def _format_context(chunks: list[RetrievedChunk], order: str,
                    limit: int) -> str:
    selected = chunks[:limit]
    if order == "reverse":
        selected = list(reversed(selected))
    lines = []
    for c in selected:
        # The bracketed id is what the model is asked to cite, and what citation
        # validity resolves against.
        lines.append(f"[{c.chunk_id}] {c.text}")
    return "\n\n".join(lines)


def assemble_messages(query: str, chunks: list[RetrievedChunk],
                      cfg: PromptConfig,
                      history: list[tuple[str, str]] | None = None) -> list[dict]:
    """
    Build the OpenAI-style messages list. Order:
      system prompt -> few-shot exemplars -> prior turns (multi-turn) ->
      the user turn (context passages + the question).
    """
    messages: list[dict] = [{"role": "system", "content": cfg.system_prompt}]
    messages.extend(cfg.few_shot)

    if history:
        for role, content in history:
            messages.append({"role": role, "content": content})

    context_block = _format_context(chunks, cfg.context_order, cfg.max_context_chunks)
    user_content = (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {query}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages
