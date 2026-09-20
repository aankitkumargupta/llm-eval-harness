"""
Provider endpoint metadata, plain data, no SDK.

This table lives apart from the adapter that uses it because it is *just
strings*, and importing it should not cost anything. `openai_compatible.py`
imports the `openai` SDK, which on some machines takes many seconds; anything
that only wants to know "which providers exist and what can they do" was paying
that price for nothing.

That was not hypothetical. The Capabilities screen reads this table, Streamlit
executes every tab body on every rerun, and pulling the adapter in for a dict of
strings added ~19s to the first page load, the app looked hung.

Adding a provider is still one entry here. The adapter re-exports the table, so
existing imports keep working.
"""

from __future__ import annotations

# Known endpoints, so `provider: together` in a config is all a user has to
# write. An unlisted vendor still works, pass `base_url` explicitly.
PROVIDER_ENDPOINTS: dict[str, dict] = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "rerank_url": "https://api.together.xyz/v1/rerank",
        "supports_rerank": True,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "supports_rerank": False,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "supports_rerank": False,
        # Groq serves no embedding models; pair it with another provider for the
        # corpus index rather than discovering this mid-ingest.
        "supports_embeddings": False,
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "supports_rerank": False,
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
        "supports_rerank": False,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        # OpenRouter serves a dedicated OpenAI-shaped /v1/embeddings endpoint
        # (baai/bge-m3, openai/text-embedding-3-small, qwen/qwen3-embedding-8b,
        # ...), so it can build the index as well as run the models under test.
        # That makes an OpenRouter-only RAG evaluation possible.
        "supports_embeddings": True,
        # No rerank endpoint: chat, embeddings, images, video, audio only.
        "supports_rerank": False,
    },
    # --- offline ----------------------------------------------------------
    # Not a vendor: a deterministic local provider so CI, the §12 verification
    # gate and `bench run --models fake:a` work with no key and no network.
    # Listed here (rather than in tests/) so the code path CI exercises is the
    # same one a user runs.
    "fake": {
        "base_url": "local://fake",
        "api_key_env": "",
        "default_api_key": "none",
        "supports_rerank": False,
        "supports_embeddings": True,
        "local": True,
    },

    # --- self-hosted -------------------------------------------------------
    # These are the reason the abstraction exists: "is the hosted API worth it
    # versus what we can run ourselves?" is a cost question you cannot answer
    # from inside a single-vendor harness.
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY",
        "default_api_key": "ollama",  # server ignores it, SDK requires one
        "supports_rerank": False,
        "supports_seed": False,
        "local": True,
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "VLLM_API_KEY",
        "default_api_key": "EMPTY",
        "supports_rerank": False,
        "local": True,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "api_key_env": "LMSTUDIO_API_KEY",
        "default_api_key": "lm-studio",
        "supports_rerank": False,
        "supports_seed": False,
        "local": True,
    },
}

# Providers that are not OpenAI-shaped and so have their own adapter. Listed
# here so capability views can enumerate every provider without importing any
# SDK at all.
NON_OPENAI_PROVIDERS: dict[str, dict] = {
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "supports_embeddings": False,   # Anthropic serves no embedding models
        "supports_rerank": False,
        "supports_seed": False,
    },
    # FastEmbed on the local CPU. Embeddings ONLY: the adapter has no generate
    # method, so it can never be a model under test. Exists because the pinned
    # hosted embedder was not invokable on the account (docs/DEBT.md R-20).
    "local": {
        "base_url": "local://fastembed",
        "api_key_env": "",
        "default_api_key": "none",
        "supports_embeddings": True,
        "supports_rerank": False,
        "supports_seed": False,
        "local": True,
    },
}


def all_providers() -> dict[str, dict]:
    """Every provider the harness knows about, OpenAI-shaped or not."""
    return {**PROVIDER_ENDPOINTS, **NON_OPENAI_PROVIDERS}
