"""
Client construction from config, plus cross-provider routing.

Two jobs.

**Build a client from a name.** `build_client("groq")` returns something that
satisfies `LLMClient`. Adding a vendor is a table entry, not a code change
anywhere above this file.

**Route capabilities to different providers.** This is the part that makes a
genuinely multi-provider run possible. Anthropic serves no embeddings; Groq
serves no embeddings and no rerank. Without routing, "benchmark Claude against
Llama on our RAG corpus" is simply impossible — you'd have nothing to build the
index with. `RoutedClient` sends generation wherever the model lives while
keeping embeddings and reranking on a provider that supports them, so the
*retrieval apparatus stays fixed* across every model under test. That fixed
apparatus is the whole basis of the comparison: if two models saw different
retrieved passages, the eval measures the embedders, not the models.

Model strings carry an optional `provider/` prefix so one models.yaml can mix
vendors:

    models:
      - "together:openai/gpt-oss-120b"
      - "anthropic:claude-opus-5"
      - "ollama:llama3.1:8b"
      - "Qwen/Qwen2.5-7B-Instruct-Turbo"   # no prefix -> default provider
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import (
    Capability,
    EmbedResult,
    GenResult,
    ProviderInfo,
    require,
    supports,
)
from .cost import CostMeter
from .resilience import RetryPolicy

# `provider:model` — chosen over "/" because model ids contain slashes
# ("openai/gpt-oss-120b") and a slash split would be ambiguous.
PROVIDER_SEP = ":"


def split_model_ref(ref: str, default_provider: str = "together") -> tuple[str, str]:
    """Split "provider:model" into (provider, model).

    Only splits on a *known* provider prefix. Ollama tags like "llama3.1:8b"
    contain a colon too, so a naive split would route them to a provider called
    "llama3.1" and fail with a baffling error.
    """
    from .endpoints import PROVIDER_ENDPOINTS

    if PROVIDER_SEP in ref:
        head, rest = ref.split(PROVIDER_SEP, 1)
        known = set(PROVIDER_ENDPOINTS) | {"anthropic"}
        if head in known:
            return head, rest
    return default_provider, ref


def build_client(
    provider: str = "together",
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 120.0,
    retry: RetryPolicy | None = None,
    rate_limit: float = 0.0,
    meter: CostMeter | None = None,
    pricing=None,
    stream_for_ttft: bool = False,
    **kwargs,
):
    """Construct the adapter for `provider`."""
    common = dict(api_key=api_key, timeout=timeout, retry=retry,
                  rate_limit=rate_limit, meter=meter, pricing=pricing,
                  stream_for_ttft=stream_for_ttft)
    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(base_url=base_url, **common, **kwargs)

    from .openai_compatible import OpenAICompatibleClient
    return OpenAICompatibleClient(provider=provider, base_url=base_url,
                                  **common, **kwargs)


@dataclass
class RouteSpec:
    """One provider entry from configs/models.yaml."""
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    rate_limit: float = 0.0
    timeout: float = 120.0


class RoutedClient:
    """An `LLMClient` that dispatches per model string, with fixed side-channels.

    Generation follows the model's own `provider:` prefix. Embeddings, reranking
    and judging are pinned to explicitly chosen providers — they are part of the
    apparatus, not the variable under test, so they must not drift when the
    model under test changes.
    """

    def __init__(
        self,
        default_provider: str = "together",
        specs: dict[str, RouteSpec] | None = None,
        *,
        embedding_provider: str | None = None,
        rerank_provider: str | None = None,
        judge_provider: str | None = None,
        meter: CostMeter | None = None,
        pricing=None,
        retry: RetryPolicy | None = None,
        stream_for_ttft: bool = False,
        timeout: float = 120.0,
    ):
        self.default_provider = default_provider
        self.specs = specs or {}
        self.meter = meter
        self.pricing = pricing
        self.retry = retry
        self.stream_for_ttft = stream_for_ttft
        self.timeout = timeout

        self.embedding_provider = embedding_provider or default_provider
        self.rerank_provider = rerank_provider or default_provider
        self.judge_provider = judge_provider or default_provider

        self._clients: dict[str, object] = {}
        self.info = ProviderInfo(name=f"routed({default_provider})",
                                 supports_rerank=True, supports_embeddings=True)

    def client_for(self, provider: str):
        """Lazily build and memoise one client per provider.

        Lazy on purpose: constructing every configured provider up front would
        demand API keys for vendors this particular run never touches.
        """
        if provider not in self._clients:
            spec = self.specs.get(provider, RouteSpec(provider=provider))
            self._clients[provider] = build_client(
                provider, api_key=spec.api_key, base_url=spec.base_url,
                timeout=spec.timeout or self.timeout, retry=self.retry,
                rate_limit=spec.rate_limit, meter=self.meter,
                pricing=self.pricing, stream_for_ttft=self.stream_for_ttft,
            )
        return self._clients[provider]

    # -- LLMClient surface ------------------------------------------------ #
    def generate(self, model: str, messages: list[dict], **kw) -> GenResult:
        provider, real_model = split_model_ref(model, self.default_provider)
        return self.client_for(provider).generate(real_model, messages, **kw)

    def judge(self, model: str, messages: list[dict], **kw) -> GenResult:
        provider, real_model = split_model_ref(model, self.judge_provider)
        return self.client_for(provider).judge(real_model, messages, **kw)

    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        provider, real_model = split_model_ref(model, self.embedding_provider)
        client = self.client_for(provider)
        # `require` rather than letting the call fail: a provider that cannot
        # embed no longer *has* an embed method, so the raw call would raise
        # AttributeError instead of saying which config key to change.
        require(client, Capability.EMBED,
                "Set `embedding_provider` in configs/models.yaml to a provider "
                "that serves embeddings (together, openai, openrouter).")
        return client.embed(real_model, texts)

    def rerank(self, model: str, query: str, documents: list[str],
               top_n: int | None = None) -> list[tuple[int, float]]:
        provider, real_model = split_model_ref(model, self.rerank_provider)
        client = self.client_for(provider)
        require(client, Capability.RERANK,
                "Leave `rerank_model` blank, or set `rerank_provider` to one "
                "that has a rerank endpoint (together).")
        return client.rerank(real_model, query, documents, top_n)

    # -- pre-flight -------------------------------------------------------- #
    def preflight(self, models: list[str], embedding_model: str = "",
                  rerank_model: str = "", judge_model: str = "") -> list[str]:
        """Check the routing table *before* a run and return human-readable problems.

        A capability mismatch discovered at item 4,000 has already cost real
        money and hours. Catching "Groq cannot embed" at startup costs nothing,
        so this runs before the first billable call rather than after.
        """
        problems: list[str] = []
        seen: set[str] = set()

        for m in models:
            provider, _ = split_model_ref(m, self.default_provider)
            seen.add(provider)

        if embedding_model:
            provider, _ = split_model_ref(embedding_model, self.embedding_provider)
            try:
                if not supports(self.client_for(provider), Capability.EMBED):
                    problems.append(
                        f"Embedding model '{embedding_model}' routes to provider "
                        f"'{provider}', which serves no embeddings. Set "
                        f"`embedding_provider` in configs/models.yaml.")
            except Exception as e:  # noqa: BLE001 — surface, don't crash preflight
                problems.append(f"Cannot reach embedding provider '{provider}': {e}")

        if rerank_model:
            provider, _ = split_model_ref(rerank_model, self.rerank_provider)
            try:
                if not supports(self.client_for(provider), Capability.RERANK):
                    problems.append(
                        f"Rerank model '{rerank_model}' routes to provider "
                        f"'{provider}', which has no rerank endpoint. Leave "
                        f"`rerank_model` blank or set `rerank_provider`.")
            except Exception as e:  # noqa: BLE001
                problems.append(f"Cannot reach rerank provider '{provider}': {e}")

        for provider in sorted(seen):
            try:
                self.client_for(provider)
            except Exception as e:  # noqa: BLE001
                problems.append(f"Cannot construct provider '{provider}': {e}")

        return problems


def build_from_config(models_cfg: dict, *, meter: CostMeter | None = None,
                      pricing=None, api_key: str | None = None,
                      retry: RetryPolicy | None = None,
                      stream_for_ttft: bool = False) -> RoutedClient:
    """Build a `RoutedClient` from the `providers:` block of configs/models.yaml.

    Absent that block, everything defaults to Together — so an existing config
    keeps working untouched.
    """
    default_provider = models_cfg.get("default_provider", "together")
    specs: dict[str, RouteSpec] = {}
    for name, raw in (models_cfg.get("providers") or {}).items():
        raw = raw or {}
        specs[name] = RouteSpec(
            provider=name,
            api_key=raw.get("api_key") or (api_key if name == default_provider else None),
            base_url=raw.get("base_url"),
            rate_limit=float(raw.get("rate_limit", 0.0)),
            timeout=float(raw.get("timeout", 120.0)),
        )
    if default_provider not in specs:
        specs[default_provider] = RouteSpec(provider=default_provider, api_key=api_key)
    elif api_key and not specs[default_provider].api_key:
        specs[default_provider].api_key = api_key

    return RoutedClient(
        default_provider=default_provider, specs=specs,
        embedding_provider=models_cfg.get("embedding_provider"),
        rerank_provider=models_cfg.get("rerank_provider"),
        judge_provider=models_cfg.get("judge_provider"),
        meter=meter, pricing=pricing, retry=retry,
        stream_for_ttft=stream_for_ttft,
    )
