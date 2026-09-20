"""
Anthropic adapter, the one major vendor that is not OpenAI-shaped.

Everything else the harness talks to speaks the OpenAI wire format, so
`OpenAICompatibleClient` covers it with a `base_url` swap. Anthropic does not,
and the differences are not cosmetic, each one below is a silent wrong answer
or a hard 400 if you paper over it:

  * `system` is a **top-level request parameter**, not a message with
    `role: "system"`. The harness's prompt assembler emits an OpenAI-style
    system message; we lift it out here. Left in the array it would be rejected.
  * Token counts are `usage.input_tokens` / `usage.output_tokens`, not
    `prompt_tokens` / `completion_tokens`. Reading the wrong names yields zero
    tokens, which quietly reports every Claude model as costing $0.00, and cost
    carries a negative weight in the leaderboard composite.
  * **`temperature` is removed on the current top models** (Fable 5, Opus 5,
    Opus 4.8/4.7, Sonnet 5) and returns a 400. The harness sends
    `temperature=0.0` on every call, so an adapter that forwards it blindly
    fails 100% of items on exactly the models people most want to benchmark.
  * There is no `seed`, so bit-level reproducibility comes from the harness
    cache rather than the API, which was already the design here.
  * Anthropic serves **no embedding or rerank endpoint**. Declared up front so
    a profile misconfiguration fails with a clear message at startup instead of
    a confusing error thousands of items into an ingest.

The `anthropic` SDK is imported lazily: the harness must keep working for users
who never touch Claude and haven't installed it.
"""

from __future__ import annotations

import time

from .base import GenResult, ProviderInfo
from .cost import CostMeter
from .resilience import RateLimiter, RetryPolicy, retry_call

# Model families that reject sampling parameters. Sending `temperature` to any
# of these is a 400, so the adapter drops it rather than forwarding what the
# provider-agnostic layer above asked for.
_NO_SAMPLING_PREFIXES = (
    "claude-fable-5", "claude-mythos-5", "claude-opus-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5",
)

# Published list prices, USD per 1M tokens, as a fallback when a Claude model is
# missing from configs/pricing.yaml. The YAML always wins, this only exists so
# that adding a Claude model to models.yaml doesn't silently report zero cost.
DEFAULT_PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "claude-fable-5":   {"input": 10.00, "output": 50.00},
    "claude-mythos-5":  {"input": 10.00, "output": 50.00},
    "claude-opus-5":    {"input": 5.00,  "output": 25.00},
    "claude-opus-4-8":  {"input": 5.00,  "output": 25.00},
    "claude-opus-4-7":  {"input": 5.00,  "output": 25.00},
    "claude-opus-4-6":  {"input": 5.00,  "output": 25.00},
    "claude-sonnet-5":  {"input": 2.00,  "output": 10.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5": {"input": 1.00,  "output": 5.00},
}


def supports_sampling(model: str) -> bool:
    """False when the model rejects `temperature` / `top_p` / `top_k`."""
    return not any(model.startswith(p) for p in _NO_SAMPLING_PREFIXES)


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split OpenAI-style messages into (system_text, conversation).

    Every `role: "system"` entry is concatenated into the top-level system
    parameter. The remaining turns must begin with `user`; a few-shot set that
    opens on an assistant turn is legal in the OpenAI format but rejected here,
    so we prepend a minimal user turn rather than dropping the exemplar and
    quietly changing the prompt the model is evaluated on.
    """
    system_parts: list[str] = []
    convo: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            convo.append({"role": role, "content": content})

    if convo and convo[0]["role"] != "user":
        convo.insert(0, {"role": "user", "content": "(begin)"})
    return "\n\n".join(system_parts), convo


class AnthropicClient:
    """Adapter for the Claude Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 120.0,
        retry: RetryPolicy | None = None,
        rate_limit: float = 0.0,
        meter: CostMeter | None = None,
        pricing=None,
        stream_for_ttft: bool = False,
        thinking: dict | None = None,
        effort: str | None = None,
        base_url: str | None = None,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "The Anthropic provider needs the official SDK: pip install anthropic"
            ) from e
        import anthropic

        self.timeout = timeout
        # `max_retries=0`: our resilience layer owns backoff. Leaving the SDK's
        # own retries on as well would multiply both the attempts and the wait.
        kwargs = {"timeout": timeout, "max_retries": 0}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        # With no explicit key the SDK resolves ANTHROPIC_API_KEY, then
        # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile, so an unset
        # env var does not mean "no credentials".
        self._client = anthropic.Anthropic(**kwargs)

        self.retry_policy = retry or RetryPolicy()
        self.limiter = RateLimiter(rate_limit)
        self.meter = meter
        self.pricing = pricing
        self.stream_for_ttft = stream_for_ttft
        self.thinking = thinking
        self.effort = effort

        self.info = ProviderInfo(
            name="anthropic",
            base_url=base_url or "https://api.anthropic.com",
            supports_rerank=False,
            supports_embeddings=False,  # Anthropic serves no embedding models
            supports_seed=False,        # no `seed` parameter exists
        )

    # ------------------------------------------------------------------ #
    def _guarded(self, fn):
        self.limiter.acquire()

        def _on_retry(attempt, delay, exc):
            if self.meter is not None:
                self.meter.record_retry()

        return retry_call(fn, self.retry_policy, on_retry=_on_retry)

    def _price(self, model: str, in_tok: int, out_tok: int) -> float:
        """Cost in USD, preferring configs/pricing.yaml over the built-in table."""
        if self.pricing is not None:
            try:
                return self.pricing.generation_cost(model, in_tok, out_tok)
            except KeyError:
                pass  # fall through to published list prices
        for prefix, p in DEFAULT_PRICING_USD_PER_MTOK.items():
            if model.startswith(prefix):
                return (in_tok / 1e6) * p["input"] + (out_tok / 1e6) * p["output"]
        return 0.0

    def _build_kwargs(self, model: str, messages: list[dict], temperature: float,
                      max_tokens: int, extra: dict) -> dict:
        system, convo = split_system(messages)
        kwargs: dict = {"model": model, "max_tokens": max_tokens,
                        "messages": convo, **extra}
        if system:
            kwargs["system"] = system
        # The single most important guard in this file, see module docstring.
        if supports_sampling(model):
            kwargs["temperature"] = temperature
        if self.thinking is not None:
            kwargs["thinking"] = self.thinking
        if self.effort is not None:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs

    # ------------------------------------------------------------------ #
    #  Generation                                                         #
    # ------------------------------------------------------------------ #
    def generate(self, model: str, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = 0,
                 stream: bool = False, _is_judge: bool = False,
                 **extra) -> GenResult:
        # `seed` is accepted and ignored: the signature is the provider-agnostic
        # contract, and Anthropic has no equivalent. Reproducibility here comes
        # from the harness's content-addressed cache.
        kwargs = self._build_kwargs(model, messages, temperature, max_tokens, extra)
        want_stream = stream or self.stream_for_ttft
        result = self._guarded(
            lambda: (self._generate_streaming(kwargs) if want_stream
                     else self._generate_blocking(kwargs)))

        if self.meter is not None:
            self.meter.record_generation(
                self._price(model, result.prompt_tokens, result.completion_tokens),
                result.prompt_tokens, result.completion_tokens, is_judge=_is_judge)
        return result

    @staticmethod
    def _text_of(resp) -> str:
        """Concatenate the text blocks, skipping thinking/tool blocks.

        `content` is a list of typed blocks, so indexing `[0].text` breaks the
        moment thinking is on, the first block is then a thinking block.
        """
        return "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text")

    @staticmethod
    def _usage_of(resp) -> tuple[int, int]:
        u = getattr(resp, "usage", None)
        if u is None:
            return 0, 0
        # Cache reads/writes are billed at different rates but are still input
        # tokens; counting them keeps the token total honest even when the
        # dollar figure uses the uncached rate.
        input_tokens = (getattr(u, "input_tokens", 0) or 0)
        input_tokens += (getattr(u, "cache_read_input_tokens", 0) or 0)
        input_tokens += (getattr(u, "cache_creation_input_tokens", 0) or 0)
        return input_tokens, (getattr(u, "output_tokens", 0) or 0)

    def _generate_blocking(self, kwargs: dict) -> GenResult:
        t0 = time.perf_counter()
        resp = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        in_tok, out_tok = self._usage_of(resp)
        return GenResult(
            text=self._text_of(resp), prompt_tokens=in_tok,
            completion_tokens=out_tok, latency_ms=latency_ms,
            finish_reason=self._normalise_stop(getattr(resp, "stop_reason", None)),
            model=getattr(resp, "model", kwargs["model"]),
        )

    def _generate_streaming(self, kwargs: dict) -> GenResult:
        t0 = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                chunks.append(text)
            final = stream.get_final_message()
        in_tok, out_tok = self._usage_of(final)
        return GenResult(
            text="".join(chunks), prompt_tokens=in_tok, completion_tokens=out_tok,
            latency_ms=(time.perf_counter() - t0) * 1000.0, ttft_ms=ttft_ms,
            finish_reason=self._normalise_stop(getattr(final, "stop_reason", None)),
            model=getattr(final, "model", kwargs["model"]),
        )

    @staticmethod
    def _normalise_stop(stop_reason: str | None) -> str | None:
        """Map Anthropic stop reasons onto the harness's OpenAI-shaped vocabulary.

        `GenResult.truncated` keys off `"length"`, so `max_tokens` must be
        translated or every truncated Claude answer would look like a clean stop
        and its low completeness score would be blamed on the model.
        """
        return {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "refusal": "content_filter",
            "pause_turn": "pause_turn",
        }.get(stop_reason or "", stop_reason)

    def judge(self, model: str, messages: list[dict], **kw) -> GenResult:
        kw.setdefault("temperature", 0.0)
        kw.setdefault("stream", False)
        return self.generate(model, messages, _is_judge=True, **kw)

    # ------------------------------------------------------------------ #
    #  Capabilities Anthropic does not have                               #
    # ------------------------------------------------------------------ #
    # `embed` and `rerank` are deliberately NOT defined.
    #
    # Defining them just to raise made this class structurally satisfy the
    # Embedder and DocumentReranker protocols while being unable to honour
    # either, so `isinstance(client, Embedder)` answered True for a client that
    # can never embed, and the only real signal was a separate boolean flag that
    # could drift out of sync with the code. Omitting the methods makes the type
    # tell the truth; `base.require()` turns the resulting absence into the same
    # actionable message these stubs used to carry.
