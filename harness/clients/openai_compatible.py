"""
Adapter for every OpenAI-shaped endpoint.

One class covers Together, OpenAI itself, Groq, Fireworks, DeepInfra, OpenRouter,
and — the interesting one — anything you host yourself: vLLM, Ollama, LM Studio,
llama.cpp's server. They all speak the same wire format, so the only thing that
varies is `base_url` and which optional features actually work.

That last part matters. A self-hosted llama.cpp server will 400 on `seed`, and
Ollama has no rerank endpoint at all. Rather than let those surface as mysterious
per-item errors halfway through a run, `ProviderInfo` declares what an endpoint
supports and this adapter degrades honestly: it drops unsupported parameters and
raises `CapabilityError` for genuinely missing capabilities.

Everything network-facing goes through the resilience layer, and every billed
call is recorded on the cost meter, so a run's reported spend is measured rather
than assumed.
"""

from __future__ import annotations

import os
import time

import requests
from openai import OpenAI

from .base import CapabilityError, EmbedResult, GenResult, ProviderInfo
from .cost import CostMeter
from .endpoints import PROVIDER_ENDPOINTS
from .resilience import RateLimiter, RetryPolicy, retry_call

# The endpoint table lives in `endpoints.py` — plain data, importable
# without pulling in the openai SDK. Re-exported here so existing imports
# of `PROVIDER_ENDPOINTS` from this module keep working.


class OpenAICompatibleClient:
    """One adapter, many vendors. See module docstring for the supported set."""

    def __init__(
        self,
        provider: str = "together",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        retry: RetryPolicy | None = None,
        rate_limit: float = 0.0,
        meter: CostMeter | None = None,
        pricing=None,
        stream_for_ttft: bool = False,
        extra_headers: dict | None = None,
    ):
        spec = PROVIDER_ENDPOINTS.get(provider, {})
        self.provider = provider
        self.timeout = timeout
        self.retry_policy = retry or RetryPolicy()
        self.limiter = RateLimiter(rate_limit)
        self.meter = meter
        self.pricing = pricing
        self.stream_for_ttft = stream_for_ttft
        self._rerank_url = spec.get("rerank_url", "")

        resolved_base = base_url or spec.get("base_url")
        if not resolved_base:
            raise ValueError(
                f"Unknown provider '{provider}' and no base_url given. Either use "
                f"one of {sorted(PROVIDER_ENDPOINTS)} or pass base_url explicitly."
            )

        key_env = spec.get("api_key_env", "")
        self.api_key = (api_key or (os.environ.get(key_env) if key_env else None)
                        or spec.get("default_api_key"))
        if not self.api_key:
            raise RuntimeError(
                f"No API key for provider '{provider}'. Set {key_env} in your "
                f"environment or pass api_key=... explicitly."
            )

        self.info = ProviderInfo(
            name=provider,
            base_url=resolved_base,
            supports_rerank=bool(spec.get("supports_rerank", False)),
            supports_embeddings=bool(spec.get("supports_embeddings", True)),
            supports_seed=bool(spec.get("supports_seed", True)),
            extra={"local": bool(spec.get("local", False))},
        )

        # `max_retries=0` because our own retry_call owns backoff — leaving the
        # SDK's retries on too would multiply the attempts and the wait.
        self._client = OpenAI(
            api_key=self.api_key, base_url=resolved_base, timeout=timeout,
            max_retries=0, default_headers=extra_headers or None,
        )

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #
    def _guarded(self, fn):
        """Pace, then call with retries, counting retries on the meter."""
        self.limiter.acquire()

        def _on_retry(attempt: int, delay: float, exc: BaseException) -> None:
            if self.meter is not None:
                self.meter.record_retry()

        return retry_call(fn, self.retry_policy, on_retry=_on_retry)

    def _bill_generation(self, model: str, r: GenResult, is_judge: bool) -> None:
        if self.meter is None:
            return
        usd = 0.0
        if self.pricing is not None:
            try:
                usd = self.pricing.generation_cost(
                    model, r.prompt_tokens, r.completion_tokens)
            except KeyError:
                usd = 0.0  # unpriced model: counted, not costed (report flags it)
        self.meter.record_generation(usd, r.prompt_tokens, r.completion_tokens,
                                     is_judge=is_judge)

    # ------------------------------------------------------------------ #
    #  Generation                                                         #
    # ------------------------------------------------------------------ #
    def generate(self, model: str, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = 0,
                 stream: bool = False, _is_judge: bool = False,
                 **extra) -> GenResult:
        """One chat completion, timed around the network call.

        Latency here is real wall-clock, so callers that report it must invoke
        this from the low-concurrency lane — under load you would be measuring
        your own queue depth, not the model.
        """
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens, **extra)
        if seed is not None and self.info.supports_seed:
            kwargs["seed"] = seed

        # Streaming is the only way to observe time-to-first-token, which is the
        # number that actually governs perceived responsiveness in a chat UI.
        # It was declared on TraceRow but never populated, because nothing ever
        # asked for a stream.
        if stream or self.stream_for_ttft:
            result = self._guarded(lambda: self._generate_streaming(kwargs))
        else:
            result = self._guarded(lambda: self._generate_blocking(kwargs))

        self._bill_generation(model, result, _is_judge)
        return result

    def _generate_blocking(self, kwargs: dict) -> GenResult:
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0]
        usage = resp.usage
        return GenResult(
            text=choice.message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None),
            model=getattr(resp, "model", kwargs["model"]) or kwargs["model"],
        )

    def _generate_streaming(self, kwargs: dict) -> GenResult:
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        prompt_tokens = completion_tokens = 0
        finish_reason = None
        served_model = kwargs["model"]

        resp = self._client.chat.completions.create(
            **kwargs, stream=True, stream_options={"include_usage": True})
        for event in resp:
            if getattr(event, "model", None):
                served_model = event.model
            if event.choices:
                delta = event.choices[0].delta
                if getattr(delta, "content", None):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    chunks.append(delta.content)
                if getattr(event.choices[0], "finish_reason", None):
                    finish_reason = event.choices[0].finish_reason
            usage = getattr(event, "usage", None)
            if usage:
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        return GenResult(
            text="".join(chunks), prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            ttft_ms=ttft_ms, finish_reason=finish_reason, model=served_model,
        )

    def judge(self, model: str, messages: list[dict], **kw) -> GenResult:
        """Generation with a fixed judge model at temperature 0.

        Named separately so judge spend lands in its own bucket on the meter —
        "the judge cost more than the models under test" is a finding you want
        to be able to read straight off the report.
        """
        kw.setdefault("temperature", 0.0)
        kw.setdefault("stream", False)  # judge latency is not a reported metric
        return self.generate(model, messages, _is_judge=True, **kw)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                         #
    # ------------------------------------------------------------------ #
    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        if not self.info.supports_embeddings:
            raise CapabilityError(
                f"Provider '{self.provider}' serves no embedding models. Set the "
                f"profile's `embedding_provider` to one that does (e.g. together, "
                f"openai) while keeping the models under test on '{self.provider}'."
            )
        if not texts:
            return EmbedResult(vectors=[], prompt_tokens=0)

        resp = self._guarded(
            lambda: self._client.embeddings.create(model=model, input=texts))
        vectors = [d.embedding for d in resp.data]
        tokens = getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0

        if self.meter is not None:
            usd = 0.0
            if self.pricing is not None:
                try:
                    usd = self.pricing.embedding_cost(model, tokens)
                except KeyError:
                    usd = 0.0
            self.meter.record_embedding(usd, tokens)
        return EmbedResult(vectors=vectors, prompt_tokens=tokens)

    # ------------------------------------------------------------------ #
    #  Rerank                                                             #
    # ------------------------------------------------------------------ #
    def rerank(self, model: str, query: str, documents: list[str],
               top_n: int | None = None) -> list[tuple[int, float]]:
        """Returns [(original_index, score), ...] best-first.

        Rerank is not part of the OpenAI-compatible surface, so this is raw HTTP
        against the vendor's own endpoint and only exists where declared.
        """
        if not self.info.supports_rerank or not self._rerank_url:
            raise CapabilityError(
                f"Provider '{self.provider}' has no rerank endpoint. Leave "
                f"`rerank_model` blank, or route reranking to a provider that "
                f"supports it."
            )
        if not documents:
            return []

        payload = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n

        def _call():
            r = requests.post(
                self._rerank_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload, timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()

        data = self._guarded(_call)
        results = [(item["index"], item["relevance_score"])
                   for item in data.get("results", [])]
        results.sort(key=lambda x: x[1], reverse=True)

        if self.meter is not None:
            usd = 0.0
            if self.pricing is not None:
                try:
                    usd = self.pricing.rerank_cost(model, tokens=0, n_docs=len(documents))
                except KeyError:
                    usd = 0.0
            self.meter.record_rerank(usd)
        return results
