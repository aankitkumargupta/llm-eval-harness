"""
Benchmark contracts, the types every adapter speaks.

CLAUDE.md §10 states the goal: not "run MMLU", but run *any* public or private
benchmark through the apparatus the harness already has, paired items, metered
cost, corrected significance, failure modes separated from wrongness, and
produce a decision. "No second trace format, no second cost meter, no second
significance implementation."

So this module defines types and nothing else. There is no runner here, no
scoring policy, no network. An adapter is four pure functions over these types
(§10.3), which is what lets `tests/test_solid.py` define a working benchmark
inside the test file.

The one type that carries real weight is `Extraction`.

**Why extraction has its own failure case.** The single most common way a
benchmark number becomes a lie is counting a *parse* failure as a *wrong
answer*. A model that answered "the capital is Paris" when the harness wanted
"Paris" is not wrong; the harness is. Fold that into accuracy and you have
silently measured your own regex. I7 forbids it, so `Extraction` is a sum type
: `Ok(value)` or `Failed(reason)`, and `Failed` can never become a zero
without incrementing a counter that the report prints next to the score.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

# Re-exported so an adapter imports its item type from one place. `EvalItem` is
# deliberately the SAME type the profile path uses: a benchmark item and a
# profile item are the same thing wearing different labels, and giving them two
# types is how the second trace format §10 forbids would sneak in.
from ..store.schema import EvalItem, ItemType

__all__ = [
    "BenchmarkFamily", "EvalItem", "Extraction", "ItemType", "Ok", "Failed",
    "Prompted", "ScoreSet", "BenchmarkAdapter", "ExtractionMode",
]


class BenchmarkFamily(str, Enum):
    """§10.2's `family` field. Decides which metrics are meaningful, not which
    code runs, a multiple-choice benchmark reports chance-adjusted accuracy, a
    code benchmark reports pass@k and sandbox violations."""
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_ANSWER = "short_answer"
    MATH = "math"
    CODE = "code"
    LONG_CONTEXT = "long_context"
    RAG = "rag"
    INSTRUCTION = "instruction"
    SAFETY = "safety"
    MULTILINGUAL = "multilingual"
    AGENTIC = "agentic"
    CLASSIFY = "classify"


class ExtractionMode(str, Enum):
    """How a multiple-choice answer was obtained.

    Recorded per item, never mixed inside one comparison (§10.1). The same
    benchmark scores differently under log-likelihood and generative scoring,
    so two runs that used different modes are not comparable even though the
    benchmark id matches, which is exactly the trap that makes most published
    benchmark numbers incomparable.
    """
    GENERATIVE = "generative"
    LOGLIKELIHOOD = "loglikelihood"


@dataclass(frozen=True)
class Prompted:
    """A rendered prompt, ready for the provider layer.

    Frozen because prompting is a pure function of (item, shots, spec): the
    same inputs must render the same messages or `spec_hash` is a lie.
    """
    messages: tuple[dict, ...]
    #: Set only when the spec scores by log-likelihood: the continuations whose
    #: likelihoods are to be compared. Empty for generative scoring.
    choices: tuple[str, ...] = ()
    stop: tuple[str, ...] = ()

    def as_messages(self) -> list[dict]:
        return [dict(m) for m in self.messages]


@dataclass(frozen=True)
class Extraction:
    """The result of pulling an answer out of raw model output.

    Construct through `Ok()` / `Failed()` rather than directly, the two
    constructors read at the call site, and the whole point of this type is
    that the failure case is impossible to overlook.
    """
    value: str | None
    reason: str = ""
    #: Which link of the spec's extraction chain produced the value. Recorded
    #: because "the regex matched" and "the fallback judge guessed" are very
    #: different levels of confidence in the same reported score.
    via: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def failed(self) -> bool:
        return self.value is None


def Ok(value: str, via: str = "") -> Extraction:  # noqa: N802 - constructor
    return Extraction(value=value, via=via)


def Failed(reason: str) -> Extraction:  # noqa: N802 - constructor
    return Extraction(value=None, reason=reason)


@dataclass
class ScoreSet:
    """Metrics for one item.

    `None` means *not applicable*, and is deliberately distinct from `0.0`
    (§10.3). An item whose answer could not be extracted has
    `accuracy = None`, not `accuracy = 0.0`: it contributes to
    `extraction_failure_rate` and to nothing else. Averaging a `None` into the
    accuracy numerator is the I7 violation this type exists to make awkward.
    """
    values: dict = field(default_factory=dict)

    def __getitem__(self, k: str):
        return self.values[k]

    def get(self, k: str, default=None):
        return self.values.get(k, default)

    def set(self, k: str, v) -> ScoreSet:
        self.values[k] = v
        return self

    def applicable(self) -> dict:
        """Only the metrics that actually apply to this item."""
        return {k: v for k, v in self.values.items() if v is not None}


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """§10.3. Four pure methods; adding one touches nothing else.

    "Pure" is load-bearing, not decorative: no network (the fetcher owns that),
    no clock, no global RNG, no provider calls. An adapter that calls a model
    directly would bypass the pinned `Judge` capability and break I2, because
    the apparatus would then vary with the benchmark.
    """

    spec: object

    def load(self, *, seed: int, limit: int | None) -> Sequence[EvalItem]: ...

    def prompt(self, item: EvalItem, shots: Sequence[EvalItem]) -> Prompted: ...

    def extract(self, raw: str) -> Extraction: ...

    def score(self, item: EvalItem, extraction: Extraction) -> ScoreSet: ...
