"""
TogetherClient, kept as the zero-config default.

Together is still the default provider and the one the bundled configs target,
so this name stays valid and behaves exactly as before. It is now a thin
specialisation of `OpenAICompatibleClient` rather than a standalone
implementation, which is what lets the same code path serve OpenAI, Groq,
Fireworks, vLLM and Ollama.

Everything the original class did, it still does. What it gains for free:
retries with jittered backoff, optional rate limiting, real streaming
time-to-first-token, cost metering, and a hard budget ceiling.

New code should prefer `harness.clients.registry.build_client(...)`, which can
route different capabilities to different providers.
"""

from __future__ import annotations

from .base import EmbedResult, GenResult  # re-exported for backwards compatibility
from .cost import CostMeter
from .openai_compatible import PROVIDER_ENDPOINTS, OpenAICompatibleClient
from .resilience import RetryPolicy

TOGETHER_BASE_URL = PROVIDER_ENDPOINTS["together"]["base_url"]
TOGETHER_RERANK_URL = PROVIDER_ENDPOINTS["together"]["rerank_url"]

__all__ = ["TogetherClient", "GenResult", "EmbedResult",
           "TOGETHER_BASE_URL", "TOGETHER_RERANK_URL"]


class TogetherClient(OpenAICompatibleClient):
    """Together AI via its OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None, timeout: float = 120.0,
                 retry: RetryPolicy | None = None, rate_limit: float = 0.0,
                 meter: CostMeter | None = None, pricing=None,
                 stream_for_ttft: bool = False):
        super().__init__(
            provider="together", api_key=api_key, timeout=timeout, retry=retry,
            rate_limit=rate_limit, meter=meter, pricing=pricing,
            stream_for_ttft=stream_for_ttft,
        )
