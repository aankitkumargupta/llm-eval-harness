"""
Pricing registry — turns token counts into dollars.

Prices change, so this reads a DATED config (configs/pricing.yaml). The cost
metric is only as trustworthy as that file, so it must be refreshed from live
vendor pricing before a real benchmark. Prices are per 1,000,000 tokens, the
unit every vendor's pricing page uses.

    cost = prompt_tokens/1e6 * input_price + completion_tokens/1e6 * output_price

Design rule kept from the original: an unpriced model raises rather than
returning zero. Silently reporting $0.00 would not merely mis-state the dollar
figure — cost carries a *negative weight* in the leaderboard composite, so a
free-looking model would leap to the top of the ranking. A loud failure is the
only safe behaviour.

Two additions:
  * `staleness_days` — prices drift, and a benchmark quoting year-old rates as
    fact is misleading in a way nobody notices. The report surfaces the age.
  * per-query rerank pricing — rerank endpoints commonly bill per query or per
    document rather than per token.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml


@runtime_checkable
class PricingProvider(Protocol):
    """What the runner needs to turn tokens into dollars.

    An abstraction rather than the concrete registry because the *policy* of
    pricing is genuinely pluggable: a team with negotiated rates, a self-hosted
    fleet amortising GPU-hours, or a test that wants deterministic numbers all
    want different arithmetic behind the same three questions. The runner
    should not care which.
    """

    def generation_cost(self, model: str, prompt_tokens: int,
                        completion_tokens: int) -> float: ...

    def embedding_cost(self, model: str, tokens: int) -> float: ...

    def rerank_cost(self, model: str, tokens: int = 0, n_docs: int = 0) -> float: ...


class PricingRegistry:
    def __init__(self, config_path: str = "configs/pricing.yaml",
                 strict: bool = True):
        self.config_path = Path(config_path)
        data = yaml.safe_load(self.config_path.read_text()) or {}
        self.as_of = data.get("as_of", "unknown")
        self.models: dict = data.get("models", {}) or {}
        self.embeddings: dict = data.get("embeddings", {}) or {}
        self.rerank: dict = data.get("rerank", {}) or {}
        # strict=False is for pre-flight estimates and dashboards, where a
        # missing price should degrade to zero rather than abort the render.
        self.strict = strict

    # ------------------------------------------------------------------ #
    @staticmethod
    def _strip_provider(model: str) -> str | None:
        """Drop a leading `provider:` routing prefix, or None if there isn't one.

        Only strips *known* provider names, because Ollama tags contain a colon
        too — blindly splitting would turn `llama3.1:8b` into `8b`.
        """
        from .registry import split_model_ref

        provider, stripped = split_model_ref(model, default_provider="")
        return stripped if provider else None

    def _match(self, table: dict, model: str) -> dict | None:
        if model in table:
            return table[model]
        candidates = [k for k in table if model.startswith(k)]
        return table[max(candidates, key=len)] if candidates else None

    def _lookup(self, table: dict, model: str) -> dict | None:
        """Exact match, then longest-prefix, then the same again without the
        `provider:` routing prefix.

        Three behaviours matter here:

        * The prefix fallback lets one entry cover a family (`claude-opus-5`
          also matching a dated snapshot of itself), longest match winning so a
          specific price always beats a general one.
        * A **provider-qualified key wins over a bare one**, because the same
          model genuinely costs different amounts on different routers —
          `openrouter:openai/gpt-oss-120b` and a direct Together
          `openai/gpt-oss-120b` are the same weights at different prices.
        * Falling back to the bare name means adding a `provider:` prefix to a
          model string doesn't silently blank its cost. That mismatch was real:
          the meter billed with the stripped name while the trace row priced
          with the prefixed one, so per-row `cost_usd` came out empty and cost
          quietly dropped out of the composite, the Pareto frontier and
          `decide`.
        """
        hit = self._match(table, model)
        if hit is not None:
            return hit
        bare = self._strip_provider(model)
        return self._match(table, bare) if bare else None

    def generation_cost(self, model: str, prompt_tokens: int,
                        completion_tokens: int) -> float:
        p = self._lookup(self.models, model)
        if p is None:
            if not self.strict:
                return 0.0
            raise KeyError(
                f"No pricing for model '{model}' in {self.config_path} "
                f"(as_of {self.as_of}). Add it before running cost metrics — "
                f"an unpriced model would score as free and win the leaderboard."
            )
        return (prompt_tokens / 1e6) * p["input"] + \
               (completion_tokens / 1e6) * p["output"]

    def embedding_cost(self, model: str, tokens: int) -> float:
        p = self._lookup(self.embeddings, model)
        if p is None:
            if not self.strict:
                return 0.0
            raise KeyError(
                f"No embedding pricing for '{model}' in {self.config_path}")
        return (tokens / 1e6) * p["input"]

    def rerank_cost(self, model: str, tokens: int = 0, n_docs: int = 0) -> float:
        """Rerank endpoints bill per query, per document, or per token — support all three."""
        p = self._lookup(self.rerank, model)
        if p is None:
            if self.strict and self.rerank:
                # Only strict when a rerank table exists: a blank table means
                # the user isn't reranking, not that they forgot a price.
                raise KeyError(
                    f"No rerank pricing for '{model}' in {self.config_path}")
            return 0.0
        return (p.get("per_query", 0.0)
                + n_docs * p.get("per_doc", 0.0)
                + (tokens / 1e6) * p.get("input", 0.0))

    # ------------------------------------------------------------------ #
    def known_models(self) -> set[str]:
        return set(self.models)

    def missing(self, models: list[str]) -> list[str]:
        """Which of `models` have no price. Call this before a run, not after."""
        return [m for m in models if self._lookup(self.models, m) is None]

    def staleness_days(self, today: _dt.date | None = None) -> int | None:
        """Age of the price sheet in days, or None if `as_of` isn't a valid date."""
        try:
            as_of = _dt.date.fromisoformat(str(self.as_of))
        except (TypeError, ValueError):
            return None
        return ((today or _dt.date.today()) - as_of).days

    def staleness_warning(self, threshold_days: int = 90,
                          today: _dt.date | None = None) -> str | None:
        """A warning string when prices are old enough to be misleading, else None."""
        age = self.staleness_days(today)
        if age is None:
            return (f"Pricing file {self.config_path} has no valid `as_of` date; "
                    f"cost figures cannot be dated.")
        if age > threshold_days:
            return (f"Pricing is {age} days old (as_of {self.as_of}). Refresh "
                    f"{self.config_path} from live vendor pricing before "
                    f"quoting these cost figures.")
        return None
