"""
Provider connection checks and remote model listing.

This lives in `harness/clients/` and nowhere else, because §3 is explicit that
the capability protocols and their adapters are "the **only** code permitted to
touch the network". A "check my Together key" button in the web UI is still a
network call; putting it in the web layer would move the network boundary for
the convenience of a screen.

Two things a user needs before spending anything:

  * **Is this key live?** Not "is a key set", an expired or wrong-project key
    is set too. The only honest test is a request that the provider answers.
  * **What can I actually run?** A models.yaml that names a model the account
    cannot reach fails per-item, mid-run, after money has been spent on the
    models that did work.

Both answers come from the OpenAI-shaped `GET /v1/models` that Together,
OpenRouter, Groq, OpenAI, Fireworks and DeepInfra all serve.

**No key is ever returned, logged or echoed.** The functions here take a key,
put it in an Authorization header, and return facts about the response. §6 says
secrets never reach YAML, traces, cache keys or exports, and a diagnostic
endpoint is the classic place that rule gets broken, so `mask()` exists and
every path that could surface a key goes through it.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import requests

from .endpoints import PROVIDER_ENDPOINTS, all_providers

#: Providers that serve an OpenAI-shaped model list. Anthropic is deliberately
#: absent: it has its own SDK shape, and guessing at it here would be worse
#: than saying so.
LISTABLE = ("together", "openrouter", "groq", "openai", "fireworks", "deepinfra")

DEFAULT_TIMEOUT = 15.0


def mask(key: str | None) -> str:
    """A key rendered so it can be shown without being leaked.

    Enough to tell two keys apart, never enough to use one. Short strings
    collapse entirely rather than revealing most of themselves.
    """
    if not key:
        return ""
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:4]}…{key[-4:]}"


@dataclass
class ConnectionResult:
    provider: str
    ok: bool
    status: int | None = None
    latency_ms: float | None = None
    n_models: int = 0
    error: str = ""
    key_source: str = ""        # "environment" | "supplied" | "none"
    key_masked: str = ""
    base_url: str = ""
    models: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        # Belt and braces: the dataclass holds no raw key, but this is the
        # boundary a key would cross if one were ever added.
        d.pop("_key", None)
        return d


def resolve_key(provider: str, supplied: str | None = None) -> tuple[str, str]:
    """Return (key, source). A supplied key wins for this call only.

    A key passed in is used for the request and then dropped, it is never
    written to a config, a cache key or the trace store. The environment is the
    durable place for it, exactly as §6 says.
    """
    if supplied:
        return supplied, "supplied"
    spec = all_providers().get(provider, {})
    env = spec.get("api_key_env", "")
    if env and os.environ.get(env):
        return os.environ[env], "environment"
    default = spec.get("default_api_key")
    if default:
        return default, "default"
    return "", "none"


def _classify(status: int, body: str) -> str:
    """Turn an HTTP status into something the user can act on.

    Status codes, not string matching, the same rule the retry policy follows.
    "401" alone tells a user nothing; "the key was rejected" tells them where
    to look.
    """
    if status == 401:
        return "Key rejected (401). Check it is current and for the right account."
    if status == 403:
        return ("Key accepted but not authorised (403). Common when a key is "
                "scoped to a project that lacks model access.")
    if status == 404:
        return "Endpoint not found (404). The base_url may be wrong."
    if status == 429:
        return "Rate limited (429). The key works; the account is throttled."
    if 500 <= status < 600:
        return f"Provider error ({status}). Their side, not yours."
    return f"HTTP {status}: {body[:160]}"


def check_provider(provider: str, *, api_key: str | None = None,
                   timeout: float = DEFAULT_TIMEOUT,
                   include_models: bool = True) -> ConnectionResult:
    """Ask a provider to list its models. Returns facts, never raises.

    Never raising is deliberate: this is the screen a user opens *because*
    something is wrong, and a traceback there is less useful than a sentence.
    """
    # `all_providers()` rather than PROVIDER_ENDPOINTS: Anthropic is a known
    # provider that simply is not OpenAI-shaped, and telling a user "unknown
    # provider 'anthropic'" would be actively misleading.
    spec = all_providers().get(provider)
    base = (spec or {}).get("base_url", "")
    res = ConnectionResult(provider=provider, ok=False, base_url=base)

    if spec is None:
        res.error = (f"Unknown provider {provider!r}. Known: "
                     f"{', '.join(sorted(all_providers()))}.")
        return res
    if provider not in LISTABLE:
        res.error = (f"{provider} serves no OpenAI-shaped /models endpoint, so "
                     f"its catalogue cannot be listed here.")
        return res

    key, source = resolve_key(provider, api_key)
    res.key_source = source
    res.key_masked = mask(key)
    if not key:
        env = spec.get("api_key_env", "")
        res.error = (f"No key. Set {env} in your environment or a .env file, "
                     f"or supply one for this check.")
        return res

    t0 = time.perf_counter()
    try:
        r = requests.get(f"{base.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=timeout)
    except requests.Timeout:
        res.error = f"Timed out after {timeout:g}s."
        return res
    except requests.RequestException as e:
        # Deliberately not `str(e)`: a requests exception can embed the full
        # request URL, and for some providers the key rides in that URL.
        res.error = f"Could not reach {base}: {type(e).__name__}"
        return res

    res.latency_ms = (time.perf_counter() - t0) * 1000.0
    res.status = r.status_code
    if r.status_code != 200:
        res.error = _classify(r.status_code, r.text)
        return res

    try:
        payload = r.json()
    except ValueError:
        res.error = "Provider returned a non-JSON body for /models."
        return res

    # OpenAI's shape is {"data": [...]}, but not everyone follows it.
    # Together returns a bare array. Handle both rather than assuming, which
    # is the sort of thing only a real key against a real provider reveals.
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data") or payload.get("models") or []
    else:
        rows = []
    if not isinstance(rows, list):
        res.error = "Provider returned an unexpected /models shape."
        return res
    models = []
    for m in rows:
        if not isinstance(m, dict):
            continue
        mid = str(m.get("id") or m.get("name") or "")
        if not mid:
            continue
        models.append({
            "id": mid,
            # `provider:model` is the string models.yaml wants, so the user can
            # copy it straight across rather than reconstructing the prefix.
            "ref": f"{provider}:{mid}",
            "context": _first_int(m, ("context_length", "context_window",
                                      "max_model_len")),
            "owner": str(m.get("owned_by") or m.get("organization") or ""),
            "kind": str(m.get("type") or m.get("object") or ""),
        })
    models.sort(key=lambda x: x["id"])

    res.ok = True
    res.n_models = len(models)
    res.models = models if include_models else []
    return res


def _first_int(d: dict, keys) -> int | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, int):
                    return vv
    return None


def check_all(providers=None, *, include_models: bool = False,
              timeout: float = DEFAULT_TIMEOUT) -> list[ConnectionResult]:
    """Check several providers. Ones without a key are reported, not skipped.

    Reported rather than skipped because "no key" and "key broken" lead to
    different actions, and a list that silently omits the first is a list that
    makes the second look like the only failure mode.
    """
    out = []
    for p in (providers or LISTABLE):
        key, _ = resolve_key(p)
        if not key:
            spec = PROVIDER_ENDPOINTS.get(p, {})
            out.append(ConnectionResult(
                provider=p, ok=False, key_source="none",
                base_url=spec.get("base_url", ""),
                error=f"No key ({spec.get('api_key_env', 'no env var')} unset)."))
            continue
        out.append(check_provider(p, include_models=include_models,
                                  timeout=timeout))
    return out


@dataclass
class ModelProbe:
    """Whether a model can actually be *invoked*, not merely listed."""
    provider: str
    model: str
    invokable: bool
    latency_ms: float | None = None
    error: str = ""
    price_in: float | None = None      # USD per 1M input tokens, as published
    price_out: float | None = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def probe_model(provider: str, model: str, *, api_key: str | None = None,
                timeout: float = 45.0) -> ModelProbe:
    """Send the smallest possible chat request and report what came back.

    **Listing is not access.** Together's `/v1/models` returns 274 entries and
    nothing in the payload distinguishes the ones an account may actually call
: `running` reads false for both a working model and a dead one. The only
    honest test is a request, so this sends one: a two-word prompt capped at
    one completion token, costing a fraction of a cent.

    Found the hard way: a 540-call benchmark run where two of three models
    returned "Unable to access non-serverless model" on every single item. The
    run completed, the rows were written, and the leaderboard would have shown
    two models at zero if I7 had not kept errors out of the numerator.
    """
    spec = all_providers().get(provider)
    res = ModelProbe(provider=provider, model=model, invokable=False)
    if spec is None:
        res.error = f"Unknown provider {provider!r}."
        return res

    key, _ = resolve_key(provider, api_key)
    if not key:
        res.error = f"No key for {provider}."
        return res

    base = spec.get("base_url", "").rstrip("/")
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{base}/chat/completions", timeout=timeout,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "max_tokens": 1,
                  "messages": [{"role": "user", "content": "hi"}]})
    except requests.RequestException as e:
        res.error = f"{type(e).__name__}"
        return res

    res.latency_ms = (time.perf_counter() - t0) * 1000.0
    if r.status_code == 200:
        res.invokable = True
        return res

    # Together says so in the body; surface the vendor's own sentence rather
    # than a status code, because "non-serverless" is actionable and "400" is not.
    detail = ""
    try:
        body = r.json()
        detail = str((body.get("error") or {}).get("message") or "")[:160]
    except ValueError:
        detail = r.text[:160]
    res.error = detail or _classify(r.status_code, r.text)
    return res


def probe_models(provider: str, models, *, api_key: str | None = None,
                 timeout: float = 45.0) -> list[ModelProbe]:
    """Probe several models. Serial on purpose, this is a diagnostic, and a
    burst of parallel requests is the fastest way to get rate-limited while
    trying to find out whether you are rate-limited."""
    return [probe_model(provider, m, api_key=api_key, timeout=timeout)
            for m in models]


def catalogue_pricing(provider: str, api_key: str | None = None,
                      timeout: float = DEFAULT_TIMEOUT) -> dict:
    """Published per-1M-token prices, read from the provider's own catalogue.

    §6 forbids guessing a price, and this is the alternative: the vendor's
    listed rate, retrieved now, attributable. It still has to be written into
    `configs/pricing.yaml` by a human, pricing changes results, so §7 reserves
    the edit, but nobody has to transcribe it from a web page.
    """
    spec = all_providers().get(provider, {})
    key, _ = resolve_key(provider, api_key)
    if not key:
        return {}
    try:
        r = requests.get(f"{spec.get('base_url', '').rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {key}"},
                         timeout=timeout)
        payload = r.json()
    except (requests.RequestException, ValueError):
        return {}

    rows = payload if isinstance(payload, list) else payload.get("data", [])
    out = {}
    for m in rows:
        if not isinstance(m, dict):
            continue
        p = m.get("pricing") or {}
        if p.get("input") and m.get("id"):
            out[str(m["id"])] = {"input": round(float(p["input"]), 4),
                                 "output": round(float(p.get("output", 0)), 4)}
    return out
