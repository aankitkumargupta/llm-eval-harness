"""
The local FastEmbed adapter, held to the same rules as a paid provider.

It exists because the pinned hosted embedder was not invokable (R-20). That
makes it apparatus (I2), so what matters is not "does it embed" but: does it
meter honestly (I3), does it refuse to be a model under test, and does it
stay off the network once its weights are on disk. Everything here runs
under the suite-wide socket block; only the last test loads the real model,
and it skips — with the reason — when the weights are not cached.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness.clients.base import (
    Capability,
    CapabilityError,
    MissingUsageError,
    require,
    supports,
)
from harness.clients.cost import CostMeter
from harness.clients.local_embed import PRICING_PREFIX, LocalEmbedClient
from harness.clients.pricing import PricingRegistry


class _StubBackend:
    """One token per whitespace word, a fixed 3-dim vector. Deterministic."""

    class _Tok:
        def encode(self, text: str):
            return SimpleNamespace(ids=text.split())

    def __init__(self, model: str) -> None:
        self.model = SimpleNamespace(tokenizer=self._Tok())
        self.name = model
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        for _ in texts:
            yield [0.1, 0.2, 0.3]


class _NoTokenizerBackend(_StubBackend):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.model = SimpleNamespace()   # no .tokenizer


@pytest.fixture
def client():
    return LocalEmbedClient(meter=CostMeter(), pricing=PricingRegistry(),
                            backend=_StubBackend)


# --------------------------------------------------------------------------- #
def test_it_is_an_embedder_and_nothing_else(client):
    """Apparatus must not be able to become a model under test. The refusal
    is structural — there is no `generate` method — not a flag."""
    assert supports(client, Capability.EMBED)
    assert not supports(client, Capability.GENERATE)
    assert not supports(client, Capability.RERANK)
    assert not hasattr(client, "generate")
    with pytest.raises(CapabilityError):
        require(client, Capability.GENERATE)


def test_token_counts_are_measured_not_estimated(client):
    r = client.embed("BAAI/bge-base-en-v1.5", ["one two three", "four five"])
    assert r.prompt_tokens == 5
    assert r.usage_estimated is False
    assert len(r.vectors) == 2 and len(r.vectors[0]) == 3


def test_spend_is_metered_into_the_embedding_bucket_at_the_priced_rate(client):
    """I3: every paid call lands in a bucket. A local model is priced at zero
    by an explicit entry, so the bucket must record the call and the tokens
    even though the dollars are 0 — "metered at $0" is a different fact from
    "not metered"."""
    client.embed("BAAI/bge-base-en-v1.5", ["a b c d"])
    assert client.meter.counts.embedding == 1
    assert client.meter.counts.embedding_tokens == 4
    assert client.meter.costs.embedding == pytest.approx(0.0, abs=1e-12)


def test_an_unpriced_local_model_raises_rather_than_costing_zero():
    """The hosted adapter swallows a pricing KeyError into $0 (R-22). This one
    must not: the pricing table is the only thing standing between "free"
    and "unknown", and an unknown price is a validation failure."""
    c = LocalEmbedClient(meter=CostMeter(), pricing=PricingRegistry(),
                         backend=_StubBackend)
    with pytest.raises(KeyError, match="no-such-model"):
        c.embed("no-such-model", ["x"])


def test_the_price_is_looked_up_under_the_local_prefix():
    """A bare "BAAI/bge-base-en-v1.5" entry at $0 would also declare Together's
    hosted copy free. The adapter must look up the provider-qualified key."""
    seen = {}

    class _Pricing:
        def embedding_cost(self, model, tokens):
            seen["model"] = model
            return 0.0

    c = LocalEmbedClient(meter=CostMeter(), pricing=_Pricing(), backend=_StubBackend)
    c.embed("BAAI/bge-base-en-v1.5", ["x"])
    assert seen["model"] == PRICING_PREFIX + "BAAI/bge-base-en-v1.5"


def test_a_model_without_a_tokenizer_refuses_to_guess():
    c = LocalEmbedClient(backend=_NoTokenizerBackend)
    with pytest.raises(MissingUsageError):
        c.embed("m", ["x"])


def test_estimation_is_opt_in_and_flagged():
    c = LocalEmbedClient(backend=_NoTokenizerBackend, allow_estimated_usage=True)
    r = c.embed("m", ["twelve characters!"])
    assert r.usage_estimated is True
    assert r.prompt_tokens >= 1


def test_empty_input_costs_nothing_and_touches_no_backend():
    loaded = []
    c = LocalEmbedClient(meter=CostMeter(), backend=lambda m: loaded.append(m) or _StubBackend(m))
    r = c.embed("m", [])
    assert r.vectors == [] and r.prompt_tokens == 0
    assert loaded == [], "no model should be loaded to embed nothing"
    assert c.meter.counts.embedding == 0


def test_the_backend_is_loaded_once_per_model():
    made = []
    c = LocalEmbedClient(backend=lambda m: made.append(m) or _StubBackend(m))
    c.embed("m", ["a"])
    c.embed("m", ["b"])
    c.embed("n", ["c"])
    assert made == ["m", "n"]


def test_sixteen_threads_cold_construct_the_backend_exactly_once():
    """The race this pins: eight harness workers embedding at once, before
    any of them has loaded the model. A slow factory makes the window wide;
    the assertion is that exactly one construction happens and every
    thread's vectors come from it."""
    import threading
    import time

    made = []
    lock = threading.Lock()

    def slow_factory(model: str):
        with lock:
            made.append(model)
        time.sleep(0.05)                 # the window a real ONNX load opens
        return _StubBackend(model)

    c = LocalEmbedClient(backend=slow_factory)
    out: list[list[float]] = []
    errors: list[str] = []

    def worker() -> None:
        try:
            r = c.embed("m", ["one two"])
            with lock:
                out.append(r.vectors[0])
        except Exception as e:                       # noqa: BLE001
            with lock:
                errors.append(repr(e))

    ts = [threading.Thread(target=worker) for _ in range(16)]
    for th in ts:
        th.start()
    for th in ts:
        th.join()

    assert not errors
    assert made == ["m"], f"backend constructed {len(made)} times"
    assert len(out) == 16 and all(v == [0.1, 0.2, 0.3] for v in out)


# --------------------------------------------------------------------------- #
#  Routing: the adapter participates through the ordinary seams.
# --------------------------------------------------------------------------- #
def test_the_local_prefix_routes_to_the_adapter():
    from harness.clients.registry import build_client, split_model_ref

    assert split_model_ref("local:BAAI/bge-base-en-v1.5") == ("local", "BAAI/bge-base-en-v1.5")
    assert isinstance(build_client("local"), LocalEmbedClient)


def test_preflight_accepts_a_locally_embedded_rag_profile():
    """The run path's own check, with embeddings routed locally and the models
    under test on the fake provider. Constructing the adapter must not load
    a model — preflight has to stay instant and offline."""
    from harness.clients.registry import build_from_config

    rc = build_from_config({"default_provider": "fake",
                            "embedding_provider": "local",
                            "models": ["fake:a"]})
    assert rc.preflight(["fake:a"], embedding_model="BAAI/bge-base-en-v1.5") == []
    assert rc.client_for("local")._backends == {}


def test_the_local_provider_is_in_every_table_the_ui_reads():
    from harness.clients.endpoints import NON_OPENAI_PROVIDERS, all_providers

    spec = NON_OPENAI_PROVIDERS["local"]
    assert spec["supports_embeddings"] and not spec["supports_rerank"]
    assert spec["local"] is True
    assert "local" in all_providers()


def test_the_pricing_table_prices_the_shipped_local_models():
    reg = PricingRegistry()
    for m in ("BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5"):
        assert reg.embedding_cost(PRICING_PREFIX + m, 1_000_000) == 0.0


# --------------------------------------------------------------------------- #
#  The real model, only when its weights are already on disk.
# --------------------------------------------------------------------------- #
def _cached(model: str) -> bool:
    root = os.environ.get("FASTEMBED_CACHE_PATH") or str(
        Path(os.environ.get("TEMP", "/tmp")) / "fastembed_cache")
    needle = model.split("/")[-1].lower()
    return Path(root).is_dir() and any(
        needle in p.name.lower() for p in Path(root).iterdir())


@pytest.mark.slow
def test_the_real_model_embeds_with_no_network_once_cached():
    """Under the suite-wide socket block. If this reaches the network the
    block fails it, which is the point: after the one-time fetch the adapter
    must be fully offline, or the RAG path is not."""
    model = "BAAI/bge-base-en-v1.5"
    if not _cached(model):
        pytest.skip(f"{model} weights not in the FastEmbed cache; run "
                    f"`python main.py ingest --profile regulated_qa` once")
    c = LocalEmbedClient(meter=CostMeter(), pricing=PricingRegistry())
    r = c.embed(model, ["Section 12 establishes penalties.", "Within 72 hours."])
    assert len(r.vectors) == 2 and len(r.vectors[0]) == 768
    assert r.prompt_tokens > 0 and r.usage_estimated is False
    assert c.fetched_from_network == set(), "must load from cache, not fetch"
    assert c.meter.costs.embedding == 0.0 and c.meter.counts.embedding == 1
