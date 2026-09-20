"""
Local embeddings: FastEmbed (ONNX, CPU), one capability, no network at use.

Why this exists. The RAG apparatus pins an embedder (I2), and on the account
this harness is run against Together lists exactly one embedding model and
serves none of them without a dedicated endpoint, the probe that found it is
recorded in docs/DEBT.md R-20. An apparatus that cannot be served is not an
apparatus. The alternatives were a hashed "fake" embedder, which makes dense
retrieval noise and would have been reported as a RAG result, or a second
paid vendor for which no key exists. This adapter is the honest third option:
the same BGE family the profile names, served locally, producing real
semantic vectors.

What it deliberately does NOT do:

  * **Generate or judge.** An embed-only adapter has no `generate` method at
    all, so `supports(client, GENERATE)` is false structurally rather than by
    a flag someone could flip. A local model under test would be a different
    adapter and a different decision.
  * **Estimate tokens.** FastEmbed exposes its tokenizer, so the count is
    exact and `usage_estimated` is False. If a model ever hides its
    tokenizer, this raises the same `MissingUsageError` the API adapters
    raise (I3), it does not guess.
  * **Swallow a pricing gap.** The API adapter turns a missing embedding price
    into `usd = 0.0` (docs/DEBT.md R-22). This one lets the KeyError
    propagate. The model is priced, at zero, by an explicit entry keyed
    `local:<model>` in pricing.yaml, following that file's own rule for
    self-hosted models, and "priced at zero" is a different fact from
    "unpriced".
  * **Reach the network once the weights are cached.** Loading tries
    `local_files_only=True` first. Only when the weights are absent does it
    fall back to a one-time fetch, and it records that it did so on
    `fetched_from_network`, because a run that downloaded 200MB of model is
    not the same run as one that did not.

Cost is metered into the `embedding` bucket at the priced rate, so the four
buckets still sum to the total and a local embedder shows as $0.00 metered,
not as an absent column.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .base import EmbedResult, MissingUsageError, ProviderInfo
from .cost import CostMeter

log = logging.getLogger(__name__)

#: How pricing.yaml keys this provider's models. Provider-qualified on
#: purpose: a bare "BAAI/bge-base-en-v1.5" entry would also claim Together's
#: hosted copy of the same weights is free, and it is not.
PRICING_PREFIX = "local:"


class LocalEmbedClient:
    """FastEmbed-backed `Embedder`. Embed only, see the module docstring."""

    provider = "local"

    def __init__(self, *, meter: CostMeter | None = None, pricing=None,
                 cache_dir: str | None = None,
                 backend: Callable[[str], object] | None = None,
                 allow_estimated_usage: bool = False, **_ignored) -> None:
        self.meter = meter
        self.pricing = pricing
        self.cache_dir = cache_dir
        self.allow_estimated_usage = allow_estimated_usage
        # Test seam: a factory returning an object with `.embed(texts)` and
        # `.model.tokenizer.encode(text).ids`. Production uses FastEmbed.
        self._backend_factory = backend
        self._backends: dict[str, object] = {}
        # Construction is serialised: the harness embeds from eight worker
        # threads, and an ONNX session being built in one thread while
        # another calls into it is the same race as the Qdrant one, one layer
        # down. Inference after construction is lock-free.
        self._build_lock = threading.Lock()
        self.fetched_from_network: set[str] = set()
        self.info = ProviderInfo(
            name="local", base_url="local://fastembed",
            supports_rerank=False, supports_embeddings=True,
            supports_seed=False)

    # ------------------------------------------------------------------ #
    def _backend(self, model: str):
        be = self._backends.get(model)
        if be is not None:
            return be
        with self._build_lock:
            be = self._backends.get(model)          # another thread may have won
            if be is None:
                be = (self._backend_factory(model) if self._backend_factory is not None
                      else self._load_fastembed(model))
                self._backends[model] = be
        return be

    def _load_fastembed(self, model: str):
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise RuntimeError(
                "Local embeddings need the `fastembed` package (installed with "
                "qdrant-client[fastembed]). Install it, or set "
                "`embedding_provider` in configs/models.yaml to a hosted "
                "provider that serves the profile's embedding model.") from e

        # Eager, not lazy: the session is built here, under the build lock,
        # so no thread ever calls into a half-built one.
        common = dict(cache_dir=self.cache_dir, lazy_load=False)
        try:
            return TextEmbedding(model, local_files_only=True, **common)
        except Exception as first:                       # noqa: BLE001
            # Weights not cached. One fetch, recorded, then local forever.
            log.warning("local embedder %s not cached (%s); fetching once",
                        model, type(first).__name__)
            be = TextEmbedding(model, local_files_only=False, **common)
            self.fetched_from_network.add(model)
            return be

    # ------------------------------------------------------------------ #
    def embed(self, model: str, texts: list[str]) -> EmbedResult:
        if not texts:
            return EmbedResult(vectors=[], prompt_tokens=0)

        be = self._backend(model)
        vectors = [[float(x) for x in v] for v in be.embed(texts)]
        tokens, estimated = self._count_tokens(be, model, texts)

        usd = 0.0
        if self.pricing is not None:
            # KeyError propagates: an unpriced model is a validation failure,
            # not a free one (I3). The registry ships `local:` entries at 0.
            usd = self.pricing.embedding_cost(PRICING_PREFIX + model, tokens)
        if self.meter is not None:
            self.meter.record_embedding(usd, tokens)

        return EmbedResult(vectors=vectors, prompt_tokens=tokens,
                           usage_estimated=estimated)

    def _count_tokens(self, be, model: str, texts: list[str]) -> tuple[int, bool]:
        """Exact token count from the model's own tokenizer.

        Returns (tokens, estimated). `estimated` is only ever True under the
        explicit opt-in, mirroring the API adapters' `usage_estimated` path.
        """
        tok = getattr(getattr(be, "model", None), "tokenizer", None)
        if tok is not None:
            try:
                return sum(len(tok.encode(t).ids) for t in texts), False
            except Exception as e:                       # noqa: BLE001
                log.warning("tokenizer failed for %s: %s", model, e)
        if not self.allow_estimated_usage:
            raise MissingUsageError(
                f"Local embedder '{model}' exposed no tokenizer, so the token "
                f"count cannot be measured. Refusing to estimate: set "
                f"`allow_estimated_usage: true` on the provider to opt in, "
                f"and the rows will carry usage_estimated=True.")
        return max(1, sum(len(t) for t in texts) // 4), True
