"""
TogetherClient — the flattened provider layer.

This is the ONLY module that makes network calls to Together and the ONLY module
that reads the `usage` block for token counts. Everything above it is
provider-agnostic: if you ever leave Together, only this file changes.

Together exposes an OpenAI-compatible API, so we use the official `openai` SDK
pointed at Together's base URL. The model is just a string.

Four capabilities, all through one account/endpoint:
  generate() — the model under test (synchronous, real latency measured here)
  embed()    — corpus + query embeddings
  rerank()   — cross-encoder reranking (Together's native rerank endpoint)
  judge()    — the fixed LLM-as-judge (same generate path, different model)

Token counts come from the API's usage block (authoritative — every model has a
different tokenizer, so counting locally would disagree with billing).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from openai import OpenAI

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_RERANK_URL = "https://api.together.xyz/v1/rerank"


@dataclass
class GenResult:
    """Everything a single generation call yields, for the TraceRow."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: Optional[float] = None  # only populated when streaming


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int  # embedding calls are billed on input tokens


class TogetherClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 120.0):
        # Key resolution: explicit arg > env var. Never hardcoded.
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "No Together API key. Set TOGETHER_API_KEY in your environment "
                "(or pass api_key=...)."
            )
        self.timeout = timeout
        self._client = OpenAI(api_key=self.api_key, base_url=TOGETHER_BASE_URL,
                              timeout=timeout)

    # ------------------------------------------------------------------ #
    #  Generation (models under test + judge share this path)            #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: Optional[int] = 0,
        stream: bool = False,
        **extra,
    ) -> GenResult:
        """
        One synchronous chat completion. Latency is measured around THIS call —
        so callers that care about clean latency must invoke generate() in the
        low-concurrency lane (see orchestration). Token counts are read from the
        usage block, never estimated.
        """
        t0 = time.perf_counter()

        if stream:
            # Streaming lets us separate time-to-first-token from total time.
            ttft_ms = None
            chunks: list[str] = []
            prompt_tokens = completion_tokens = 0
            resp = self._client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, seed=seed, stream=True,
                stream_options={"include_usage": True}, **extra,
            )
            for event in resp:
                if event.choices and event.choices[0].delta.content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000.0
                    chunks.append(event.choices[0].delta.content)
                # usage arrives on the final chunk when include_usage=True
                if getattr(event, "usage", None):
                    prompt_tokens = event.usage.prompt_tokens
                    completion_tokens = event.usage.completion_tokens
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return GenResult("".join(chunks), prompt_tokens, completion_tokens,
                             latency_ms, ttft_ms)

        # Non-streaming path
        resp = self._client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=max_tokens, seed=seed, **extra,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GenResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
        )

    def judge(self, model: str, messages: list[dict], **kw) -> GenResult:
        """
        The judge is just generation with a fixed model at temperature 0.
        Kept as a named method so callers can't accidentally judge with a
        varying model. Judge latency is NOT a reported metric, so judge calls
        are the natural candidate for the high-throughput / batch lane.
        """
        kw.setdefault("temperature", 0.0)
        return self.generate(model, messages, **kw)

    # ------------------------------------------------------------------ #
    #  Embeddings                                                         #
    # ------------------------------------------------------------------ #
    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        """Embed a batch of texts. Callers MUST batch ingestion themselves
        (a few hundred at a time) to stay within request-size limits and to
        keep memory flat during corpus ingest."""
        resp = self._client.embeddings.create(model=model, input=texts)
        vectors = [d.embedding for d in resp.data]
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        return EmbedResult(vectors=vectors, prompt_tokens=prompt_tokens)

    # ------------------------------------------------------------------ #
    #  Rerank (native Together endpoint, not OpenAI-shaped)              #
    # ------------------------------------------------------------------ #
    def rerank(self, model: str, query: str, documents: list[str],
               top_n: Optional[int] = None) -> list[tuple[int, float]]:
        """
        Returns [(original_index, relevance_score), ...] sorted best-first.
        Uses Together's dedicated /rerank endpoint via raw HTTP since it isn't
        part of the OpenAI-compatible surface.
        """
        payload = {"model": model, "query": query, "documents": documents}
        if top_n is not None:
            payload["top_n"] = top_n
        r = requests.post(
            TOGETHER_RERANK_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload, timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        # response shape: {"results": [{"index": i, "relevance_score": s}, ...]}
        results = [(item["index"], item["relevance_score"]) for item in data["results"]]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
