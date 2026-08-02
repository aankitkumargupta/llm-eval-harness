"""
Pricing registry — turns token counts into dollars.

Prices change over time, so this reads a DATED config (configs/pricing.yaml).
The cost metric is only as trustworthy as this file, so it must be refreshed
from Together's live pricing when you run a real benchmark. Prices are per
1,000,000 tokens (the common unit on model pricing pages).

Cost per query = (prompt_tokens/1e6 * input_price)
               + (completion_tokens/1e6 * output_price)
and embedding/rerank costs are added on top by the caller for true end-to-end
cost-per-query.
"""

from __future__ import annotations

from pathlib import Path

import yaml


class PricingRegistry:
    def __init__(self, config_path: str = "configs/pricing.yaml"):
        self.config_path = Path(config_path)
        data = yaml.safe_load(self.config_path.read_text())
        self.as_of = data.get("as_of", "unknown")          # the date these prices are valid for
        self.models: dict = data.get("models", {})          # per-model input/output prices
        self.embeddings: dict = data.get("embeddings", {})  # per-model input price
        self.rerank: dict = data.get("rerank", {})          # per-model price (per 1M tokens or per query)

    def generation_cost(self, model: str, prompt_tokens: int,
                        completion_tokens: int) -> float:
        p = self.models.get(model)
        if p is None:
            # Unknown model price is a hard problem: silently returning 0 would
            # corrupt the cost metric. We surface it so it can't pass unnoticed.
            raise KeyError(
                f"No pricing for model '{model}' in {self.config_path} "
                f"(as_of {self.as_of}). Add it before running cost metrics."
            )
        return (prompt_tokens / 1e6) * p["input"] + \
               (completion_tokens / 1e6) * p["output"]

    def embedding_cost(self, model: str, tokens: int) -> float:
        p = self.embeddings.get(model)
        if p is None:
            raise KeyError(f"No embedding pricing for '{model}' in {self.config_path}")
        return (tokens / 1e6) * p["input"]

    def rerank_cost(self, model: str, tokens: int = 0) -> float:
        # Some rerank endpoints bill per query, some per token. We support both:
        p = self.rerank.get(model, {})
        return p.get("per_query", 0.0) + (tokens / 1e6) * p.get("input", 0.0)
