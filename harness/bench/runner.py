"""
The benchmark run loop.

§10 is emphatic that this must not become a parallel universe: "Shares the
trace store, cost metering, cache and gate with `harness run`. No parallel
universe of code." So this module writes ordinary `TraceRow`s, meters through
the ordinary `CostMeter`, and produces something `report`, `compare`, `decide`
and `gate` already understand. There is no second trace format, no second cost
meter and no second significance implementation anywhere in `harness/bench/`.

What is genuinely new here is only the middle of the loop, prompt from an
adapter, extract through a declared chain, score into a `ScoreSet`, plus the
discipline around failure that §10.1 and I7 demand:

  * an item whose answer could not be extracted scores `accuracy = None`,
    counts toward `extraction_failure_rate`, and is absent from the accuracy
    denominator;
  * a truncated answer is recorded as truncated whether or not it parsed,
    because a parse that succeeded on a cut-off answer is luck;
  * `samples_per_item > 1` produces several rows per item, from which
    `pass@k` is computed with the unbiased estimator.

Item ordering is a pure function of the spec's seed, so every model in a run
sees the identical item set in the identical order (I1).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ..store.schema import Pass, TraceRow
from .contracts import EvalItem, Prompted, ScoreSet
from .spec import BenchmarkSpec

#: Exceptions that mean the adapter or spec is wrong, not that an item failed.
#: They stop the run instead of being averaged into an error rate, a bug that
#: presents as a slightly worse score is the hardest kind to notice.
_ADAPTER_BUGS = (AttributeError, NotImplementedError, ImportError)


@dataclass
class BenchRunReport:
    run_id: str
    benchmark: str
    rows: list = field(default_factory=list)
    errors: int = 0
    aborted: bool = False
    elapsed_s: float = 0.0

    def summary(self) -> str:
        return (f"run {self.run_id} · {self.benchmark} · {len(self.rows)} rows · "
                f"{self.errors} errors · {self.elapsed_s:.0f}s"
                + (" · ABORTED (budget)" if self.aborted else ""))


def _few_shot_pool(adapter, spec: BenchmarkSpec) -> Sequence[EvalItem]:
    """Few-shot examples, never from the scored split.

    The spec loader already rejects `pool_split == source.split`, so by the
    time we get here the only way to leak is to ignore the setting. Returning
    an empty pool when `n == 0` keeps that path obvious.
    """
    if spec.prompt.few_shot.n <= 0:
        return ()
    loader = getattr(adapter, "load_pool", None)
    if loader is None:
        return ()
    return list(loader(split=spec.prompt.few_shot.pool_split))[:spec.prompt.few_shot.n]


def run_benchmark(
    *,
    adapter,
    spec: BenchmarkSpec,
    models: Sequence[str],
    client,
    run_id: str,
    store=None,
    meter=None,
    pricing=None,
    limit: int | None = None,
    seed: int | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
    checkpoint_every: int = 50,
) -> BenchRunReport:
    """Walk model x item, writing ordinary TraceRows."""
    t0 = time.time()
    items = list(adapter.load(seed=seed, limit=limit))
    shots = _few_shot_pool(adapter, spec)
    samples = max(1, spec.sampling.samples_per_item)
    spec_hash = spec.spec_hash()

    rows: list[TraceRow] = []
    errors = 0
    aborted = False
    total = len(models) * len(items) * samples
    done = 0

    for model in models:
        for item in items:
            for sample_idx in range(samples):
                if meter is not None and getattr(meter, "exceeded", False):
                    aborted = True
                    break

                row = _blank_row(run_id, spec, item, model, spec_hash, sample_idx)
                try:
                    prompted: Prompted = adapter.prompt(item, shots)
                    row.assembled_prompt = _flatten(prompted)

                    gen = client.generate(
                        model, prompted.as_messages(),
                        **_decoding(spec, sample_idx))

                    row.raw_output = gen.text
                    row.prompt_tokens = gen.prompt_tokens
                    row.completion_tokens = gen.completion_tokens
                    row.latency_ms = gen.latency_ms
                    row.ttft_ms = gen.ttft_ms
                    row.finish_reason = gen.finish_reason
                    row.truncated = gen.truncated
                    row.usage_estimated = gen.usage_estimated
                    _price_row(row, pricing, model, gen)

                    extraction = adapter.extract(gen.text)
                    scores = _score(adapter, item, extraction, gen.text)

                except _ADAPTER_BUGS:
                    # A broken adapter is not a failed item. Recording it as
                    # one would bury a programming error inside an error-rate
                    # column and let a mis-built benchmark report numbers, so
                    # it propagates and stops the run.
                    raise

                except Exception as e:                      # noqa: BLE001
                    # Recorded, never scored. An item the harness failed to
                    # run is not an item the model got wrong (I7), and it is
                    # also the case that makes the run unpaired, which the
                    # stats layer will now refuse to paper over.
                    errors += 1
                    row.error = f"{type(e).__name__}: {e}"[:500]
                    row.accuracy = None
                else:
                    # Outside the try: a ScoreSet key with no TraceRow column
                    # is an adapter bug, and must not be caught above.
                    _apply(row, scores)

                rows.append(row)
                done += 1
                if progress_cb:
                    progress_cb(done, total, model)
                if store is not None and len(rows) % checkpoint_every == 0:
                    store.write(rows[-checkpoint_every:])
            if aborted:
                break
        if aborted:
            break

    if store is not None:
        tail = len(rows) % checkpoint_every
        if tail:
            store.write(rows[-tail:])

    return BenchRunReport(run_id=run_id, benchmark=spec.id, rows=rows,
                          errors=errors, aborted=aborted,
                          elapsed_s=time.time() - t0)


# --------------------------------------------------------------------------- #
def _price_row(row: TraceRow, pricing, model: str, gen) -> None:
    """Attach this item's own cost to its own row.

    The meter tracks a run's TOTAL spend, which is the number a budget ceiling
    needs, but `cost_per_correct_answer`, the figure §10.7 calls "the number
    that actually drives procurement", is per-model, and a per-model cost
    cannot be recovered from a single running total. Without this the column
    reads 0.00 for everyone, which looks like "free" rather than "not
    recorded".

    An unpriced model leaves the cost NULL rather than zero: I3 says a
    free-looking model must not win a cost-weighted comparison, and that
    applies just as much to a gap in the pricing table as to a missing usage
    block.
    """
    if pricing is None or gen.prompt_tokens is None:
        return
    try:
        usd = pricing.generation_cost(model, gen.prompt_tokens,
                                      gen.completion_tokens)
    except Exception:                               # noqa: BLE001 - unpriced
        return
    row.gen_cost_usd = float(usd)
    row.cost_usd = float(usd)


def _blank_row(run_id, spec, item, model, spec_hash, sample_idx) -> TraceRow:
    return TraceRow(
        run_id=run_id,
        item_id=(item.item_id if sample_idx == 0
                 else f"{item.item_id}#s{sample_idx}"),
        model=model,
        profile=spec.id,          # the benchmark id occupies the profile slot
        pass_=Pass.BASELINE,
        item_type=item.item_type,
        benchmark=spec.id,
        spec_hash=spec_hash,
        sample_idx=sample_idx,
        prompt_cfg_hash=spec_hash,
    )


def _decoding(spec: BenchmarkSpec, sample_idx: int) -> dict:
    """Decoding params from the spec.

    When drawing several samples per item, temperature 0 would return the same
    completion every time and make pass@k meaningless, so a multi-sample spec
    that forgot to raise it gets a nudge, recorded here rather than silently,
    because it changes what the number means.
    """
    d = dict(spec.prompt.decoding or {})
    if sample_idx > 0 and float(d.get("temperature", 0.0)) == 0.0:
        d["temperature"] = 0.7
    return d


def _flatten(p: Prompted) -> str:
    return "\n\n".join(f"[{m['role']}] {m['content']}" for m in p.messages)


def _score(adapter, item, extraction, raw: str) -> ScoreSet:
    """Call `score`, passing `raw` only to adapters that accept it."""
    try:
        return adapter.score(item, extraction, raw)
    except TypeError:
        return adapter.score(item, extraction)


def _apply(row: TraceRow, scores: ScoreSet) -> None:
    """Copy a ScoreSet onto the row, for keys TraceRow actually declares.

    An unknown key raises rather than being ignored. Silently dropping it
    would let an adapter believe it had reported a metric that never reached
    the store, the metric would be computed, thrown away, and its absence
    from the report read as "the model scored zero on it". Adding a metric
    means declaring a nullable column in `harness/store/schema.py` (§3,
    additive evolution), not inventing schema per benchmark.
    """
    for key, value in scores.values.items():
        if not hasattr(row, key):
            raise AttributeError(
                f"ScoreSet key {key!r} has no TraceRow column. Declare it in "
                f"harness/store/schema.py as a nullable field, or the metric "
                f"is computed and then discarded.")
        setattr(row, key, value)
