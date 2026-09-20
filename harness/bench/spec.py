"""
`BenchmarkSpec`, the YAML that makes two benchmark numbers comparable.

§10.1 is the argument for this file. The same benchmark scores differently
under different prompt formats, few-shot counts and extraction rules, so a
benchmark id alone ("MMLU-Pro: 0.71") identifies almost nothing. The spec makes
the *format* part of the identity: everything that can move the number lives in
one versioned file, and `spec_hash` covers the file, its templates and the
dataset checksum together. The reporter refuses to compare rows whose
`spec_hash` differs.

Validation reports every problem at once, matching the profile loader's
behaviour, a config that fails one error at a time turns a five-minute fix
into five round trips.

One rule enforced here rather than left to reviewers: **few-shot examples may
never come from the scored split**. It is the easiest contamination to
introduce by accident and the hardest to see afterwards, because the run looks
normal and the score is merely wrong.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..cache.cache import stable_hash
from .contracts import BenchmarkFamily, ExtractionMode


class SpecError(ValueError):
    """An invalid benchmark spec, with every problem listed at once."""


_VALID_SOURCE_KINDS = ("hf", "url", "local")
_VALID_SELECTION = ("fixed", "nearest", "random")
_VALID_EXTRACTORS = ("regex", "last_capital_letter", "numeric", "boxed",
                     "first_line", "verbatim", "label_set", "judge")


@dataclass
class Source:
    kind: str = "local"
    ref: str = ""
    # HuggingFace datasets are (dataset, config, split). "default" is wrong for
    # a surprising number of benchmarks: GSM8K's is "main", ARC's is the task
    # name, and a 404 four layers down is a poor way to learn that.
    config: str = "default"
    split: str = "test"
    checksum: str = ""
    licence: str = ""
    commercial_use: bool = True
    citation: str = ""


@dataclass
class Sampling:
    limit: int | None = None
    seed: int = 1729
    stratify_by: str = ""
    samples_per_item: int = 1


@dataclass
class FewShot:
    n: int = 0
    selection: str = "fixed"
    pool_split: str = "dev"


@dataclass
class PromptSpec:
    template: str = ""
    few_shot: FewShot = field(default_factory=FewShot)
    system: str | None = None
    decoding: dict = field(default_factory=lambda: {
        "temperature": 0.0, "max_tokens": 1024, "top_p": 1.0})


@dataclass
class ScoringSpec:
    mode: str = ExtractionMode.GENERATIVE.value
    chain: tuple[dict, ...] = ()
    metric: str = "accuracy"
    chance_level: float = 0.0
    aggregate: tuple[str, ...] = ("mean",)


@dataclass
class Contamination:
    canary: bool = False
    ngram_overlap_against: str = ""
    perturbations: tuple[str, ...] = ()


@dataclass
class BenchmarkSpec:
    """One benchmark, fully specified. See §10.2 for the YAML shape."""

    id: str
    version: int = 1
    family: str = BenchmarkFamily.MULTIPLE_CHOICE.value
    task: str = "direct"
    source: Source = field(default_factory=Source)
    sampling: Sampling = field(default_factory=Sampling)
    prompt: PromptSpec = field(default_factory=PromptSpec)
    scoring: ScoringSpec = field(default_factory=ScoringSpec)
    primary_metric: str = "accuracy"
    also_report: tuple[str, ...] = ()
    contamination: Contamination = field(default_factory=Contamination)
    #: Labels for classification-family specs; empty otherwise.
    label_set: tuple[str, ...] = ()
    path: str = ""

    # ------------------------------------------------------------------ #
    def spec_hash(self) -> str:
        """`sha256(canonicalised spec + template contents + dataset checksum)`.

        The template's *contents* are hashed, not its path: editing a prompt
        file without touching the spec changes the number, and a hash that
        missed that would certify two incomparable runs as comparable.
        """
        template_text = ""
        if self.prompt.template:
            p = Path(self.prompt.template)
            if p.exists():
                template_text = p.read_text(encoding="utf-8")

        canonical = json.dumps({
            "id": self.id, "version": self.version, "family": self.family,
            "task": self.task,
            "source": {"kind": self.source.kind, "ref": self.source.ref,
                       "config": self.source.config,
                       "split": self.source.split,
                       "checksum": self.source.checksum},
            "sampling": {"limit": self.sampling.limit,
                         "seed": self.sampling.seed,
                         "stratify_by": self.sampling.stratify_by,
                         "samples_per_item": self.sampling.samples_per_item},
            "prompt": {"few_shot": {"n": self.prompt.few_shot.n,
                                    "selection": self.prompt.few_shot.selection,
                                    "pool_split": self.prompt.few_shot.pool_split},
                       "system": self.prompt.system,
                       "decoding": self.prompt.decoding},
            "scoring": {"mode": self.scoring.mode,
                        "chain": list(self.scoring.chain),
                        "metric": self.scoring.metric,
                        "chance_level": self.scoring.chance_level},
            "label_set": list(self.label_set),
        }, sort_keys=True)

        return stable_hash("spec/v1", canonical,
                           hashlib.sha256(template_text.encode()).hexdigest(),
                           self.source.checksum)

    def requires_licence_ack(self) -> bool:
        """Non-commercial sets need `--acknowledge-licence` (§10.2)."""
        return not self.source.commercial_use

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchmarkSpec:
        p = Path(path)
        if not p.exists():
            raise SpecError(f"Benchmark spec not found: {path}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, path=str(p))

    @classmethod
    def from_dict(cls, raw: dict, path: str = "") -> BenchmarkSpec:
        src = raw.get("source", {}) or {}
        smp = raw.get("sampling", {}) or {}
        pr = raw.get("prompt", {}) or {}
        fs = pr.get("few_shot", {}) or {}
        sc = raw.get("scoring", {}) or {}
        ex = sc.get("extraction", {}) or {}
        rep = raw.get("reporting", {}) or {}
        cont = raw.get("contamination", {}) or {}

        spec = cls(
            id=str(raw.get("id", "")),
            version=int(raw.get("version", 1)),
            family=str(raw.get("family", BenchmarkFamily.MULTIPLE_CHOICE.value)),
            task=str(raw.get("task", "direct")),
            source=Source(
                kind=str(src.get("kind", "local")),
                ref=str(src.get("ref", "")),
                config=str(src.get("config", "default")),
                split=str(src.get("split", "test")),
                checksum=str(src.get("checksum", "")),
                licence=str(src.get("licence", "")),
                commercial_use=bool(src.get("commercial_use", True)),
                citation=str(src.get("citation", "")),
            ),
            sampling=Sampling(
                limit=smp.get("limit"),
                seed=int(smp.get("seed", 1729)),
                stratify_by=str(smp.get("stratify_by", "") or ""),
                samples_per_item=int(smp.get("samples_per_item", 1)),
            ),
            prompt=PromptSpec(
                template=str(pr.get("template", "") or ""),
                few_shot=FewShot(
                    n=int(fs.get("n", 0)),
                    selection=str(fs.get("selection", "fixed")),
                    pool_split=str(fs.get("pool_split", "dev")),
                ),
                system=pr.get("system"),
                decoding=dict(pr.get("decoding", {}) or {
                    "temperature": 0.0, "max_tokens": 1024, "top_p": 1.0}),
            ),
            scoring=ScoringSpec(
                mode=str(sc.get("mode", ExtractionMode.GENERATIVE.value)),
                chain=tuple(ex.get("chain", ()) or ()),
                metric=str(sc.get("metric", "accuracy")),
                chance_level=float(sc.get("chance_level", 0.0) or 0.0),
                aggregate=tuple(sc.get("aggregate", ("mean",)) or ("mean",)),
            ),
            primary_metric=str(rep.get("primary_metric", "accuracy")),
            also_report=tuple(rep.get("also", ()) or ()),
            contamination=Contamination(
                canary=bool(cont.get("canary", False)),
                ngram_overlap_against=str(cont.get("ngram_overlap_against", "") or ""),
                perturbations=tuple(cont.get("perturbations", ()) or ()),
            ),
            label_set=tuple(raw.get("label_set", ()) or ()),
            path=path,
        )
        problems = spec.validate()
        if problems:
            where = path or spec.id or "<spec>"
            raise SpecError(
                f"Invalid benchmark spec '{where}':\n"
                + "\n".join(f"  - {p}" for p in problems))
        return spec

    # ------------------------------------------------------------------ #
    def validate(self) -> list[str]:
        """Every problem at once, never the first one only."""
        out: list[str] = []

        if not self.id:
            out.append("`id` is required; it names the spec everywhere else.")
        if self.version < 1:
            out.append("`version` must be >= 1; bump it on ANY change, because "
                       "it feeds spec_hash and therefore comparability.")
        if self.family not in {f.value for f in BenchmarkFamily}:
            out.append(f"unknown family {self.family!r}; expected one of "
                       f"{sorted(f.value for f in BenchmarkFamily)}")
        if self.task not in ("direct", "rag", "classify"):
            out.append(f"`task` must be direct|rag|classify, got {self.task!r}")

        if self.source.kind not in _VALID_SOURCE_KINDS:
            out.append(f"`source.kind` must be one of {list(_VALID_SOURCE_KINDS)}, "
                       f"got {self.source.kind!r}")
        if self.source.kind != "local" and not self.source.checksum:
            out.append("`source.checksum` is required for a fetched source: "
                       "without it a silently-changed dataset produces a "
                       "silently-changed score.")
        if not self.source.licence:
            out.append("`source.licence` is required, a benchmark with no "
                       "recorded licence cannot be redistributed or used "
                       "commercially with any confidence.")

        if self.sampling.samples_per_item < 1:
            out.append("`sampling.samples_per_item` must be >= 1")
        if self.sampling.limit is not None and self.sampling.limit < 1:
            out.append("`sampling.limit` must be null or >= 1")

        sel = self.prompt.few_shot.selection
        if sel not in _VALID_SELECTION:
            out.append(f"`prompt.few_shot.selection` must be one of "
                       f"{list(_VALID_SELECTION)}, never 'whatever came "
                       f"first', got {sel!r}")
        if self.prompt.few_shot.n < 0:
            out.append("`prompt.few_shot.n` must be >= 0")
        if (self.prompt.few_shot.n > 0
                and self.prompt.few_shot.pool_split == self.source.split):
            out.append(
                f"few-shot examples are drawn from {self.prompt.few_shot.pool_split!r}, "
                f"which IS the scored split. That leaks answers into the prompt "
                f"and inflates the score invisibly. Use a different pool_split.")

        if self.scoring.mode not in {m.value for m in ExtractionMode}:
            out.append(f"`scoring.mode` must be generative|loglikelihood, got "
                       f"{self.scoring.mode!r}")
        if not self.scoring.chain and self.scoring.mode == ExtractionMode.GENERATIVE.value:
            out.append("`scoring.extraction.chain` is required for generative "
                       "scoring: without it every answer is an extraction "
                       "failure, which reads as 0% accuracy.")
        for i, link in enumerate(self.scoring.chain):
            kind = (link or {}).get("kind")
            if kind not in _VALID_EXTRACTORS:
                out.append(f"extraction chain[{i}]: unknown kind {kind!r}; "
                           f"expected one of {list(_VALID_EXTRACTORS)}")
            if kind == "regex" and not (link or {}).get("pattern"):
                out.append(f"extraction chain[{i}]: a regex link needs `pattern`")
        if not 0.0 <= self.scoring.chance_level < 1.0:
            out.append(f"`scoring.chance_level` must be in [0, 1), got "
                       f"{self.scoring.chance_level}")
        if (self.family == BenchmarkFamily.MULTIPLE_CHOICE.value
                and self.scoring.chance_level == 0.0):
            out.append("a multiple-choice benchmark needs a non-zero "
                       "`scoring.chance_level`: 25% on 4-way MC is not '25% "
                       "good', and chance-adjusted accuracy is mandatory (§10.4).")

        if self.task == "classify" and not self.label_set:
            out.append("`label_set` is required when task is classify")

        decoding = self.prompt.decoding or {}
        if "max_tokens" not in decoding:
            out.append("`prompt.decoding.max_tokens` must be declared: an "
                       "undeclared budget silently truncates reasoning models "
                       "and reads as a quality finding.")
        return out


def load_spec(ref: str | Path, search_dir: str = "configs/benchmarks") -> BenchmarkSpec:
    """Accept a path or a bare benchmark id, like `_profile` does for profiles."""
    p = Path(ref)
    if not p.exists() and not str(ref).endswith((".yaml", ".yml")):
        p = Path(search_dir) / f"{ref}.yaml"
    return BenchmarkSpec.from_yaml(p)


def available_specs(search_dir: str = "configs/benchmarks") -> list[Path]:
    d = Path(search_dir)
    return sorted(d.glob("*.yaml")) if d.exists() else []
