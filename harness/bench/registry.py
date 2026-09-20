"""
Adapter registry, an explicit dict, populated by one decorator defined here.

§4: "No magic. Registries are explicit dicts populated by decorators defined in
one place." So there is no import-scanning, no entry points, no metaclass. If
an adapter is not in this file's registry, it does not exist, which makes the
answer to "what benchmarks can this harness run?" a five-line read rather than
an investigation.

Adding an adapter touches exactly four things (§10.3): one YAML spec, one
adapter module, one line here, one fixture. Never the runner, store, stats or
reporter.
"""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register(benchmark_id: str):
    """Register an adapter factory under a benchmark id."""
    def deco(factory: Callable) -> Callable:
        if benchmark_id in _REGISTRY:
            raise ValueError(
                f"benchmark id {benchmark_id!r} is already registered; ids are "
                f"the stable name used in specs, run rows and reports, so a "
                f"silent overwrite would mislabel stored results.")
        _REGISTRY[benchmark_id] = factory
        return factory
    return deco


def get(benchmark_id: str) -> Callable:
    if benchmark_id not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for {benchmark_id!r}. Known: "
            f"{sorted(_REGISTRY)}. Register one in harness/bench/registry.py.")
    return _REGISTRY[benchmark_id]


def known() -> list[str]:
    return sorted(_REGISTRY)


def build(spec, data_path: str | None = None):
    """Instantiate the adapter a spec names.

    For a fetched dataset the adapter is pointed at the *cached* file rather
    than the dataset name. Resolving the path here, and never fetching here,
    keeps construction offline: `bench fetch` and `bench run` do the download
    explicitly, and everything else replays.
    """
    if data_path is None and getattr(spec.source, "kind", "local") != "local":
        from .fetch import cached_path
        data_path = str(cached_path(spec))
    return get(spec.id)(spec, data_path)


def _install_builtins() -> None:
    """Import and register the shipped adapters.

    Called once at module import. Kept as a function rather than top-level
    imports so the registry module itself stays importable when an optional
    adapter's dependency is missing.
    """
    from .adapters.arc import ARCAdapter
    from .adapters.boolq import BoolQAdapter
    from .adapters.gsm8k import GSM8KAdapter
    from .adapters.hellaswag import HellaSwagAdapter
    from .adapters.ifeval import IFEvalAdapter
    from .adapters.mgsm import MGSMAdapter
    from .adapters.mmlu_pro import MMLUProAdapter
    from .adapters.truthfulqa_mc import TruthfulQAMCAdapter
    from .adapters.winogrande import WinoGrandeAdapter

    for bid, cls in (("mmlu_pro", MMLUProAdapter),
                     ("gsm8k", GSM8KAdapter),
                     ("ifeval", IFEvalAdapter),
                     ("arc_challenge", ARCAdapter),
                     # The real-dataset twins share their adapter with the
                     # synthetic fixture spec; only the source differs.
                     ("gsm8k_hf", GSM8KAdapter),
                     ("mmlu_pro_hf", MMLUProAdapter),
                     # ARC-Easy ships the same choices/label shape as
                     # ARC-Challenge, so a spec is the whole difference.
                     ("arc_easy", ARCAdapter),
                     ("hellaswag", HellaSwagAdapter),
                     ("boolq", BoolQAdapter),
                     # OpenBookQA is ARC-shaped (choices/answerKey) with the
                     # question under `question_stem`; the ARC adapter reads both.
                     ("openbookqa", ARCAdapter),
                     ("truthfulqa_mc", TruthfulQAMCAdapter),
                     ("winogrande", WinoGrandeAdapter),
                     # MGSM: one adapter, one spec per language config.
                     ("mgsm_bn", MGSMAdapter),
                     ("mgsm_te", MGSMAdapter)):
        if bid not in _REGISTRY:
            register(bid)(cls)


_install_builtins()
