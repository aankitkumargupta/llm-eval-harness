"""
The provider contract.

Everything above this file is provider-agnostic. Adding a vendor means writing
one adapter, never touching the runner, the metrics, or the report layer.

**Interface segregation.** A RAG evaluation needs three capabilities, and almost
no provider has all three. Anthropic serves no embeddings; Groq serves neither
embeddings nor reranking; OpenRouter has embeddings but no rerank endpoint. A
single fat `LLMClient` forced every adapter to implement all four methods and
made the missing ones raise, so "can this provider embed?" was answered by a
*boolean flag* that could, and did, disagree with the code. (OpenRouter shipped
declared as embedding-incapable while its adapter would have handled it fine.)

So the contract is split by capability:

    TextGenerator      generate() / judge(), every provider
    Embedder           embed(), most, not all
    DocumentReranker   rerank(), few

Consumers depend on the narrowest protocol they actually use: `Retriever` needs
an `Embedder`, `Reranker` needs a `DocumentReranker`. `LLMClient` remains as the
union for callers that legitimately need everything.

**Capability checks have one source of truth.** `supports()` requires the method
to be *present* AND the provider to declare it. An adapter that genuinely cannot
do something (Anthropic embedding) simply does not define the method, so the two
signals cannot drift apart.

Token counts always come from the provider's usage block. Every model has a
different tokenizer, so a local count would disagree with the bill, and cost is
a headline metric here, not a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class MissingUsageError(RuntimeError):
    """A provider returned no usage block, and estimation was not opted into.

    This is deliberately fatal. The tempting alternative, substituting zero,
    is the worst available option, because cost carries a *negative* weight in
    the leaderboard composite: a model whose usage block went missing would be
    billed $0.00 and would therefore *rise* in the ranking. A run that silently
    under-bills is not a cheaper run, it is a wrong one.

    Estimation is available, but only on explicit opt-in, and it marks every
    row it touches with `usage_estimated=True` so the report can surface the
    rate rather than let it pass as measured.
    """


@dataclass
class GenResult:
    """Everything one generation call yields, for the TraceRow."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float | None = None      # populated only when streaming
    finish_reason: str | None = None  # "stop" | "length" | "content_filter" | ...
    model: str = ""                   # what the provider says it served
    # False means the counts came from the provider's own usage block. True
    # means they were estimated under an explicit opt-in and must be reported
    # as such, never mixed into a cost figure presented as measured (I3).
    usage_estimated: bool = False
    # Hidden reasoning, for models that think before they answer: tokens as
    # the provider reported them (None = not reported), and chars of the
    # reasoning text it returned beside the answer (0 = none). Billed inside
    # completion_tokens, so cost needs no correction; recorded so a report can
    # separate the visible answer from the hidden work (I3).
    reasoning_tokens: int | None = None
    reasoning_chars: int = 0

    @property
    def truncated(self) -> bool:
        """True when the model ran out of `max_tokens` mid-answer.

        Worth surfacing: a truncated answer scores badly on completeness and
        citations for a reason that has nothing to do with model quality, and a
        run where one model truncates constantly is measuring your `max_tokens`,
        not the model.
        """
        return self.finish_reason == "length"


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int  # embeddings bill on input tokens only
    usage_estimated: bool = False


class Capability(str, Enum):
    """What a provider can be asked to do."""
    GENERATE = "generate"
    EMBED = "embed"
    RERANK = "rerank"


@dataclass
class ProviderInfo:
    """Static facts about a provider, used for reporting and sanity checks."""
    name: str                       # "together" | "openai" | "anthropic" | ...
    base_url: str = ""
    supports_rerank: bool = False
    supports_embeddings: bool = True
    # Some self-hosted servers (llama.cpp, older vLLM) reject `seed`. Adapters
    # declare it so the runner can drop the arg instead of failing every call.
    supports_seed: bool = True
    extra: dict = field(default_factory=dict)

    def declares(self, capability: Capability) -> bool:
        return {
            Capability.GENERATE: True,
            Capability.EMBED: self.supports_embeddings,
            Capability.RERANK: self.supports_rerank,
        }[capability]


# --------------------------------------------------------------------------- #
#  Segregated capability protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class TextGenerator(Protocol):
    """Generation. The one capability every provider has."""

    info: ProviderInfo

    def generate(self, model: str, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = 0,
                 stream: bool = False, **extra) -> GenResult: ...

    def judge(self, model: str, messages: list[dict], **kw) -> GenResult: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Anthropic and Groq have no such endpoint."""

    def embed(self, model: str, texts: list[str]) -> EmbedResult: ...


@runtime_checkable
class DocumentReranker(Protocol):
    """Cross-encoder reordering. Rare: Together has it, most others don't."""

    def rerank(self, model: str, query: str, documents: list[str],
               top_n: int | None = None) -> list[tuple[int, float]]: ...


@runtime_checkable
class LLMClient(TextGenerator, Embedder, DocumentReranker, Protocol):
    """The union, for callers that genuinely need all three.

    Prefer the narrow protocols in new code: a function that only embeds should
    say so in its signature, so a provider that cannot embed is a type error
    rather than a runtime surprise.
    """


# --------------------------------------------------------------------------- #
#  Capability checks
# --------------------------------------------------------------------------- #
def supports(client: object, capability: Capability) -> bool:
    """Can `client` actually do this?

    Both signals must agree: the method has to exist AND the provider has to
    declare the capability. Structural presence alone is not enough, because one
    adapter class serves several vendors, the same `OpenAICompatibleClient`
    backs Together (embeddings) and Groq (none), so capability is a property of
    the *instance*, not the class.
    """
    if not callable(getattr(client, capability.value, None)):
        return False
    info = getattr(client, "info", None)
    return info.declares(capability) if isinstance(info, ProviderInfo) else True


def require(client: object, capability: Capability, hint: str = "") -> None:
    """Raise a useful `CapabilityError` unless `client` supports `capability`.

    Called before the work rather than after, so a misconfigured run fails at
    startup with an actionable message instead of thousands of items in.
    """
    if supports(client, capability):
        return
    name = getattr(getattr(client, "info", None), "name", type(client).__name__)
    raise CapabilityError(
        f"Provider '{name}' does not support {capability.value}."
        + (f" {hint}" if hint else "")
    )


class ProviderError(RuntimeError):
    """A provider call failed in a way the harness should record, not swallow."""


class CapabilityError(ProviderError):
    """The provider cannot do what was asked (e.g. rerank against Ollama).

    Distinct from a transient failure so the resilience layer never retries it:
    no amount of backoff will teach a local server how to rerank.
    """
