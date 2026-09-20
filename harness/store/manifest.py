"""
The run manifest, what makes a run a *result* rather than a number.

CLAUDE.md I9 states the rule plainly: "Every run records git SHA, dirty flag,
resolved-profile hash, dataset hash, apparatus hash, spec hash, seeds, package
versions, pricing-table version. A run without a manifest is not a result."

Phase 0 found that `harness/store/schema.py:127-128` already *documented* a run
manifest: "Full configs live in the run manifest; a row stores only hashes,
pointing at an exact setup", pointing at a file that was never built. So every
`TraceRow` carried hashes that resolved to nothing, and no run this repo had
ever produced could be reproduced or even described.

Two design points worth stating, because both are load-bearing:

**Provenance that cannot be gathered is recorded as unknown, never omitted.**
A missing `git_sha` key and a `git_sha` of `"unknown"` look the same to a
careless reader but mean different things to a careful one: the first says
nobody asked, the second says we asked and the answer was unavailable. Silence
about provenance is the failure mode I9 exists to prevent, so this module is
loud about what it could not determine.

**The apparatus hash is separate from the profile hash (I2).** Embedder,
reranker and judge are the fixed instruments of the experiment, not the
variable under test. When they change, cross-run comparison is invalid even
though the profile and the dataset are untouched, a different embedder means
the models saw different retrieved passages, so the run measures the embedders.
Folding that into `profile_cfg_hash` would hide exactly the change that
invalidates the comparison.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..cache.cache import stable_hash

#: Packages whose version can change a reported number. Deliberately short: a
#: full `pip freeze` is noise that nobody diffs, while these four move results.
_TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy")


def git_provenance(root: str | Path = ".") -> tuple[str, bool | None]:
    """Return (sha, dirty). `("unknown", None)` when this is not a git checkout.

    `dirty=None` means undetermined, which is distinct from `dirty=False`. A run
    from a dirty tree is not reproducible from its SHA, and that has to be on
    the record rather than inferred later from a hunch.
    """
    root = Path(root)
    try:
        sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return sha or "unknown", bool(status)
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def package_versions() -> dict[str, str]:
    """Versions of the packages that can move a number, plus the interpreter."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version(), "implementation": sys.implementation.name}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "absent"
    return out


def apparatus_hash(embedding_model: str = "", rerank_model: str = "",
                   judge_model: str = "", judge_ensemble: tuple[str, ...] = (),
                   embedding_provider: str = "", rerank_provider: str = "",
                   judge_provider: str = "") -> str:
    """Hash of the fixed apparatus (I2).

    Changing any argument here invalidates comparison against a run with a
    different value, even when the profile, the dataset and the models under
    test are identical. Ordering of `judge_ensemble` is normalised so a
    reordered config does not read as a different apparatus.
    """
    return stable_hash(
        "apparatus/v1",
        embedding_provider, embedding_model,
        rerank_provider, rerank_model,
        judge_provider, judge_model,
        tuple(sorted(judge_ensemble)),
    )


def dataset_hash(items) -> str:
    """Hash of the evalset actually scored, by item id and gold answer.

    Item *content* is included rather than just the count, because an evalset
    edited in place keeps its length while changing what was measured, the
    quietest way for two runs to stop being comparable.
    """
    parts = []
    for it in items:
        parts.append((
            str(getattr(it, "item_id", "")),
            str(getattr(it, "query", ""))[:512],
            str(getattr(it, "gold_answer", "") or ""),
            str(getattr(it, "item_type", "")),
        ))
    return stable_hash("dataset/v1", tuple(sorted(parts)))


@dataclass
class RunManifest:
    """Everything needed to say what a run was, and to judge what it can be
    compared against.

    Written beside the traces, once per run. `aborted` matters downstream: a
    run stopped at its budget ceiling holds real rows but an incomplete matrix,
    and ranking it against a complete run compares different amounts of work.
    """

    run_id: str
    profile: str
    models: tuple[str, ...] = ()
    passes: tuple[str, ...] = ()

    # --- provenance (I9) ---------------------------------------------------
    git_sha: str = "unknown"
    git_dirty: bool | None = None
    schema_version: int = 1
    manifest_version: int = 1

    # --- the four hashes that decide comparability -------------------------
    profile_cfg_hash: str = ""
    dataset_hash: str = ""
    apparatus_hash: str = ""       # I2
    spec_hash: str = ""            # benchmark specs; empty for profile runs

    # --- the apparatus itself, in the clear so a reader need not decode ----
    embedding_model: str = ""
    rerank_model: str = ""
    judge_model: str = ""
    judge_ensemble: tuple[str, ...] = ()

    # --- reproducibility ---------------------------------------------------
    seeds: dict = field(default_factory=dict)
    package_versions: dict = field(default_factory=dict)
    pricing_as_of: str = "unknown"
    pricing_path: str = ""

    # --- outcome -----------------------------------------------------------
    started_ts: float = 0.0
    finished_ts: float | None = None
    n_rows: int = 0
    n_errors: int = 0
    aborted: bool = False
    abort_reason: str = ""
    budget_usd: float = 0.0
    total_cost_usd: float = 0.0
    usage_estimated_rows: int = 0   # I3: how much of the cost was not measured

    # ------------------------------------------------------------------ #
    def comparable_with(self, other: RunManifest) -> tuple[bool, list[str]]:
        """Whether two runs may be compared, and if not, why not.

        This is the question I2 and I9 exist to answer. It is returned as
        reasons rather than a bare bool so a report can print them: "not
        comparable" without a cause is an assertion, not a finding.
        """
        reasons = []
        if self.profile != other.profile:
            reasons.append(f"different profile ({self.profile} vs {other.profile})")
        if self.dataset_hash != other.dataset_hash:
            reasons.append("different dataset (evalset changed)")
        if self.apparatus_hash != other.apparatus_hash:
            reasons.append("different apparatus (embedder/reranker/judge changed)")
        if self.spec_hash != other.spec_hash:
            reasons.append("different benchmark spec")
        if self.aborted or other.aborted:
            reasons.append("one run was aborted and holds an incomplete matrix")
        return (not reasons), reasons

    def to_json(self) -> str:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return json.dumps(d, indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        """Write `manifest.json` into a run directory, creating it if needed."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        target = p / "manifest.json"
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    # ------------------------------------------------------------------ #
    @classmethod
    def read(cls, path: str | Path) -> RunManifest:
        p = Path(path)
        if p.is_dir():
            p = p / "manifest.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)
        # Unknown keys are dropped rather than raising: manifests are additive,
        # and a newer run must stay readable by an older reader.
        clean = {k: v for k, v in data.items() if k in known}
        for k in ("models", "passes", "judge_ensemble"):
            if k in clean and isinstance(clean[k], list):
                clean[k] = tuple(clean[k])
        return cls(**clean)

    @classmethod
    def capture(cls, run_id: str, profile_name: str, *, root: str | Path = ".",
                **kw) -> RunManifest:
        """Build a manifest, gathering the environment side automatically."""
        sha, dirty = git_provenance(root)
        return cls(run_id=run_id, profile=profile_name, git_sha=sha,
                   git_dirty=dirty, package_versions=package_versions(), **kw)
