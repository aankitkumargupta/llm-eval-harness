"""
Tests for the adversarial probes, the provider layer, the store and profiles.

The probe tests are the ones with the least obvious payoff and the most real
consequence: they cover the security evaluation, and a silently broken security
check reports every model as safe.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from harness.cache.cache import Cache, NullCache, stable_hash
from harness.clients.anthropic_client import split_system, supports_sampling
from harness.clients.pricing import PricingRegistry
from harness.clients.registry import split_model_ref
from harness.eval import metrics as M
from harness.eval.probes import (
    ProbeConfig,
    apply_context_attack,
    build_probe_suite,
    inject_typos,
    make_canary,
    paraphrase_query,
    score_probe,
)
from harness.profiles.loaders import (
    DatasetError,
    dataset_stats,
    load_evalset,
    save_evalset,
)
from harness.profiles.profile import Profile, ProfileError
from harness.store.schema import (
    EvalItem,
    ItemType,
    Pass,
    RetrievedChunk,
    TraceRow,
    classify_error,
)
from harness.store.store import TraceStore
from harness.tuning.search import enumerate_candidates, halving_schedule


def _items(n: int = 20) -> list[EvalItem]:
    return [EvalItem(item_id=f"q{i}",
                     query=f"What is the maximum fine for offence {i}?",
                     gold_answer=str(i * 100),
                     gold_passage_ids=[f"doc{i}#0"])
            for i in range(n)]


# --------------------------------------------------------------------------- #
#  Probe generation
# --------------------------------------------------------------------------- #
def test_probe_suite_generates_each_requested_family():
    suite = build_probe_suite(_items(20), ProbeConfig(
        unanswerable=0.25, noise=0.25, injection=0.25, paraphrase=0.25, seed=0))
    assert set(suite.counts) == {"unanswerable", "noise", "injection", "paraphrase"}
    assert all(v == 5 for v in suite.counts.values())


def test_probe_generation_is_reproducible():
    """A probe set that changes between runs makes results incomparable."""
    a = build_probe_suite(_items(20), ProbeConfig(seed=7))
    b = build_probe_suite(_items(20), ProbeConfig(seed=7))
    assert [i.item_id for i in a.items] == [i.item_id for i in b.items]
    assert [i.query for i in a.items] == [i.query for i in b.items]


def test_canary_is_stable_across_processes():
    """Built with sha256, not the builtin hash().

    String hashing is salted per process, so a builtin-hash canary would differ
    between the run that plants it and the run that scores it - and the scorer
    would look for a token the model never saw, reporting perfect injection
    resistance regardless of behaviour.
    """
    assert make_canary("seed-a") == make_canary("seed-a")
    assert make_canary("seed-a") != make_canary("seed-b")
    assert make_canary("x").startswith("ZQ7X-")


def test_zero_fractions_generate_nothing():
    suite = build_probe_suite(_items(20), ProbeConfig(
        unanswerable=0, noise=0, injection=0, paraphrase=0, positional=0))
    assert suite.items == []


def test_probe_suite_ignores_non_answerable_source_items():
    items = _items(10) + [EvalItem(item_id="u1", query="?",
                                   item_type=ItemType.UNANSWERABLE)]
    suite = build_probe_suite(items, ProbeConfig(unanswerable=1.0, noise=0,
                                                 injection=0, paraphrase=0))
    assert len(suite.items) == 10, "should derive only from answerable items"


# --------------------------------------------------------------------------- #
#  Injection
# --------------------------------------------------------------------------- #
def _chunks(n: int = 3) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=f"doc0#{i}", doc_id="doc0",
                           text="a legitimate passage", score=1.0, rank=i)
            for i in range(n)]


def test_injection_plants_a_hostile_passage_in_the_context():
    suite = build_probe_suite(_items(10), ProbeConfig(
        injection=1.0, unanswerable=0, noise=0, paraphrase=0))
    item = suite.items[0]
    attacked = apply_context_attack(_chunks(), item)

    assert len(attacked) == 4
    assert "__injected" in [c.chunk_id for c in attacked]
    injected = next(c for c in attacked if c.chunk_id == "__injected")
    assert item.meta["canary"] in injected.text
    # Ranks are renumbered so downstream position logic stays coherent.
    assert [c.rank for c in attacked] == [0, 1, 2, 3]


def test_a_model_that_obeys_the_injection_scores_zero():
    suite = build_probe_suite(_items(10), ProbeConfig(
        injection=1.0, unanswerable=0, noise=0, paraphrase=0))
    item = suite.items[0]
    canary = item.meta["canary"]

    obedient = score_probe(item, f"Sure thing: {canary}", "context")
    resistant = score_probe(item, "The maximum fine is 500 [doc0#0].", "context")
    assert obedient["injection_resisted"] == 0.0
    assert resistant["injection_resisted"] == 1.0


def test_injection_resistance_is_case_insensitive():
    """A model that lowercases the canary still followed the instruction."""
    assert M.injection_resisted("here: zq7x-abc123", "ZQ7X-ABC123") == 0.0


def test_refusing_counts_as_resisting():
    assert M.injection_resisted("I don't know based on the context.",
                                "ZQ7X-AAA111") == 1.0


# --------------------------------------------------------------------------- #
#  Noise and positional probes
# --------------------------------------------------------------------------- #
def test_noise_probe_adds_distractors():
    suite = build_probe_suite(_items(10), ProbeConfig(
        noise=1.0, unanswerable=0, injection=0, paraphrase=0, n_distractors=4))
    attacked = apply_context_attack(_chunks(3), suite.items[0])
    assert len(attacked) == 7
    assert sum(1 for c in attacked if c.doc_id == "__distractor") == 4


def test_positional_probe_moves_gold_to_the_middle():
    """Separates models that read their whole context from ones that skim ends."""
    suite = build_probe_suite(_items(10), ProbeConfig(
        positional=1.0, unanswerable=0, noise=0, injection=0, paraphrase=0))
    item = suite.items[0]
    gold_id = item.gold_passage_ids[0]
    chunks = _chunks(2) + [RetrievedChunk(chunk_id=gold_id, doc_id="doc",
                                          text="the gold passage", score=1.0)]
    attacked = apply_context_attack(chunks, item)
    ids = [c.chunk_id for c in attacked]
    pos = ids.index(gold_id)
    assert 0 < pos < len(ids) - 1, "gold must not sit at either end"


def test_non_probe_items_pass_through_untouched():
    plain = _items(1)[0]
    chunks = _chunks()
    assert apply_context_attack(chunks, plain) is chunks


# --------------------------------------------------------------------------- #
#  Query perturbation
# --------------------------------------------------------------------------- #
def test_paraphrase_changes_wording_but_keeps_the_question():
    import random
    rng = random.Random(0)
    out = paraphrase_query("What is the maximum fine?", rng)
    assert out != "What is the maximum fine?"
    assert "fine" in out.lower()


def test_typos_preserve_word_boundaries():
    """The probe tests tolerance of realistic typing, not decoding gibberish:
    words stay separate and keep their first and last letters."""
    import random

    original = "What is the maximum penalty amount"
    noisy = inject_typos(original, 1.0, random.Random(0))
    assert noisy != original
    assert len(noisy.split()) == len(original.split())
    for before, after in zip(original.split(), noisy.split()):
        if len(before) >= 4:
            assert after[0] == before[0] and after[-1] == before[-1]


# --------------------------------------------------------------------------- #
#  Metric fixes
# --------------------------------------------------------------------------- #
def test_numeric_match_scans_every_number_not_just_the_first():
    """The original took the first number, so a leading year scored the answer
    wrong - a systematic accuracy loss that looked like a model weakness."""
    assert M.numeric_match("Under the 2019 Act, the maximum fine is 5000",
                           "5000") == 1.0
    assert M.numeric_match("about 1,000 units", "1000") == 1.0
    assert M.numeric_match("about 10,000 units", "1000") == 0.0


def test_abstention_does_not_false_positive_on_a_real_answer():
    """The original matched bare 'not provided' anywhere, flipping a correct
    answerable item from 1.0 to 0.0."""
    answered = "The fine is 5000; the appeal deadline is not provided."
    assert M.detect_abstention(answered) is False
    assert M.detect_abstention("I don't know based on the context.") is True
    assert M.detect_abstention("The context does not mention this.") is True


def test_token_f1_penalises_padding():
    """`contains` is gamed by dumping the whole context; token_f1 is not."""
    gold = "five thousand pounds"
    tight = M.token_f1("five thousand pounds", gold)
    padded = M.token_f1("five thousand pounds " + "filler " * 50, gold)
    assert tight == pytest.approx(1.0)
    assert padded < 0.2


def test_citation_density_catches_the_model_that_never_cites():
    """Pointer-validity alone is gamed by citing nothing (vacuously 1.0)."""
    uncited = "The fine is 5000. It applies from 2019. Appeals take 30 days."
    assert M.citation_valid_pointer([], ["d#0"]) == 1.0
    assert M.citation_density([], uncited) == 0.0


def test_pii_leak_only_flags_pii_absent_from_the_context():
    ctx = "Contact jane@example.com for details."
    assert M.pii_leaked("Email jane@example.com", ctx) == 0.0     # legitimate echo
    assert M.pii_leaked("Email bob@elsewhere.org", ctx) == 1.0    # invented


def test_context_precision_counterweights_recall():
    """Without it, tuning happily raises k forever: recall only ever improves."""
    assert M.context_precision(["g1", "x", "y", "z"], ["g1"], k=4) == 0.25
    assert M.context_precision(["g1"], ["g1"], k=4) == 1.0


def test_classification_report_exposes_the_ignored_rare_class():
    """95% accuracy, and the critical class is never predicted. Macro-F1 says so."""
    y_true = ["common"] * 95 + ["rare"] * 5
    y_pred = ["common"] * 100
    rep = M.classification_report(y_true, y_pred)
    assert rep["accuracy"] == pytest.approx(0.95)
    assert rep["macro_f1"] < 0.55
    assert rep["per_class"]["rare"]["recall"] == 0.0


def test_answer_consistency():
    assert M.answer_consistency(["the fine is 5000", "the fine is 5000"]) == 1.0
    assert M.answer_consistency(["the fine is 5000", "completely different text"]) < 0.5


# --------------------------------------------------------------------------- #
#  Cache
# --------------------------------------------------------------------------- #
def test_judge_cache_key_includes_the_rubric():
    """Change the rubric and old verdicts must not be reused under new rules."""
    with tempfile.TemporaryDirectory() as d:
        c = Cache(d)
        k1 = c.key_for_judge("j", "rubric A", "q", "ctx", "ans", "gold")
        k2 = c.key_for_judge("j", "rubric B", "q", "ctx", "ans", "gold")
        assert k1 != k2


def test_null_cache_never_serves_anything():
    """The latency lane needs real timings on EVERY run, including re-runs."""
    c = NullCache()
    key = c.key_for_generation("m", [{"role": "user", "content": "hi"}], {})
    c.set(key, {"text": "cached"})
    assert c.get(key) is None


def test_cache_hit_rate_tracking():
    with tempfile.TemporaryDirectory() as d:
        c = Cache(d)
        k = c.key_for_generation("m", [], {})
        c.get(k)              # miss
        c.set(k, {"text": "x"})
        c.get(k)              # hit
        assert c.hit_rate == pytest.approx(0.5)


def test_stable_hash_is_order_independent_but_boundary_sensitive():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash("ab", "c") != stable_hash("a", "bc")


# --------------------------------------------------------------------------- #
#  Store
# --------------------------------------------------------------------------- #
def _row(**kw) -> TraceRow:
    base = dict(run_id="r", item_id="q0", model="M", profile="p",
                pass_=Pass.BASELINE, item_type=ItemType.ANSWERABLE)
    base.update(kw)
    return TraceRow(**base)


def test_store_appends_without_rewriting_history():
    """The original read+concat+rewrote the whole file per write - O(n^2), and a
    crash mid-rewrite could truncate the store to nothing."""
    with tempfile.TemporaryDirectory() as d:
        store = TraceStore(str(Path(d) / "traces"))
        for i in range(5):
            store.write([_row(item_id=f"q{i}", accuracy=i / 10)])
        parts = list(Path(d, "traces").glob("part-*.parquet"))
        assert len(parts) == 5, "each write should be its own part"
        assert len(store.load_all()) == 5


def test_store_query_is_parameterised():
    """The old call sites interpolated a profile name straight into SQL."""
    with tempfile.TemporaryDirectory() as d:
        store = TraceStore(str(Path(d) / "traces"))
        store.write([_row(profile="o'brien's profile", accuracy=1.0)])
        got = store.query("SELECT * FROM traces WHERE profile = ?",
                          ["o'brien's profile"])
        assert len(got) == 1


def test_store_survives_schema_drift_between_parts():
    """A store written before a metric column existed must still read back."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "traces"
        store = TraceStore(str(root))
        store.write([_row(accuracy=1.0)])
        # Simulate an older part missing the newer columns.
        pd.DataFrame([{"run_id": "r", "item_id": "old", "model": "M",
                       "profile": "p", "pass_": "baseline",
                       "item_type": "answerable"}]).to_parquet(
            root / "part-000099-legacy.parquet", index=False)
        df = store.load_all()
        assert len(df) == 2


def test_a_legacy_single_file_store_is_adopted_automatically():
    """Results from before the append-only rewrite must not disappear.

    Configs now point at a DIRECTORY (`workspace/traces`), but anyone who ran
    the harness earlier has their history in `workspace/traces.parquet` beside
    it. Without adoption the data sits intact on disk and is simply never read
    — a migration bug in its worst shape, because nothing errors.
    """
    import pandas as pd

    with tempfile.TemporaryDirectory() as d:
        legacy = Path(d) / "traces.parquet"
        pd.DataFrame([_row(run_id="old", item_id="q0", accuracy=1.0).to_dict()])             .to_parquet(legacy, index=False)

        store = TraceStore(str(Path(d) / "traces"))   # the directory form
        assert store.legacy_file == legacy
        assert len(store.load_all()) == 1

        # New rows land in the directory; both are readable together.
        store.write([_row(run_id="new", item_id="q1", accuracy=0.5)])
        df = store.load_all()
        assert len(df) == 2
        assert set(df["run_id"]) == {"old", "new"}


def test_store_lists_runs():
    with tempfile.TemporaryDirectory() as d:
        store = TraceStore(str(Path(d) / "traces"))
        store.write([_row(run_id="a"), _row(run_id="b", error="boom")])
        runs = store.runs()
        assert set(runs["run_id"]) == {"a", "b"}
        assert runs.set_index("run_id").loc["b", "errors"] == 1


def test_list_columns_survive_the_parquet_round_trip():
    with tempfile.TemporaryDirectory() as d:
        store = TraceStore(str(Path(d) / "traces"))
        store.write([_row(retrieved_ids=["a", "b"], cited_ids=["a"])])
        assert list(store.load_all().iloc[0]["retrieved_ids"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
#  Error taxonomy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc,kind", [
    (Exception("Rate limit exceeded"), "rate_limit"),
    (Exception("Read timed out"), "timeout"),
    (Exception("maximum context length is 8192 tokens"), "context_length"),
    (Exception("invalid api key"), "auth"),
    (Exception("model not found"), "not_found"),
    (Exception("something odd"), "other"),
])
def test_error_classification(exc, kind):
    """'8% failed' is not actionable; 'all context_length' means lower your k."""
    assert classify_error(exc) == kind


# --------------------------------------------------------------------------- #
#  Profile validation
# --------------------------------------------------------------------------- #
def _profile_dict(**over) -> dict:
    base = dict(name="p", task="rag", corpus_path="c.jsonl",
                evalset_path="e.jsonl", embedding_model="emb",
                active_metrics=["accuracy", "cost_usd"],
                metric_weights={"accuracy": 1.0, "cost_usd": -0.1})
    base.update(over)
    return base


def test_valid_profile_loads():
    assert Profile.from_dict(_profile_dict()).name == "p"


def test_positive_weight_on_a_lower_is_better_metric_is_rejected():
    """It rewards models for being expensive - and inverts the leaderboard."""
    with pytest.raises(ProfileError, match="NEGATIVE"):
        Profile.from_dict(_profile_dict(metric_weights={"accuracy": 1.0,
                                                        "cost_usd": 0.1}))


def test_weight_on_an_uncollected_metric_is_rejected():
    """It would contribute nothing, and the run would look like it measured
    something it never did."""
    with pytest.raises(ProfileError, match="not in active_metrics"):
        Profile.from_dict(_profile_dict(
            metric_weights={"accuracy": 0.5, "faithfulness": 0.5}))


def test_typo_in_active_metrics_is_caught_at_load():
    with pytest.raises(ProfileError, match="Unknown active_metrics"):
        Profile.from_dict(_profile_dict(active_metrics=["accuarcy"]))


def test_rag_profile_requires_an_embedder():
    with pytest.raises(ProfileError, match="embedding_model"):
        Profile.from_dict(_profile_dict(embedding_model=""))


def test_classify_profile_requires_a_label_set():
    with pytest.raises(ProfileError, match="label_set"):
        Profile.from_dict(_profile_dict(task="classify", embedding_model="",
                                        corpus_path=""))


def test_validation_reports_every_problem_at_once():
    """One load should fix them all, not surface them one run at a time."""
    with pytest.raises(ProfileError) as exc:
        Profile.from_dict(_profile_dict(k=0, max_tokens=0,
                                        accuracy_scorer="nonsense"))
    text = str(exc.value)
    assert "`k` must be" in text and "max_tokens" in text and "scorer" in text


def test_shipped_profiles_all_validate():
    for path in sorted(Path("configs/profiles").glob("*.yaml")):
        Profile.from_yaml(str(path))


# --------------------------------------------------------------------------- #
#  Dataset loading
# --------------------------------------------------------------------------- #
def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def test_duplicate_item_ids_are_rejected():
    """Paired statistics join on item_id; a duplicate pairs the wrong answers."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.jsonl"
        _write_jsonl(p, [{"item_id": "q1", "query": "a"},
                         {"item_id": "q1", "query": "b"}])
        with pytest.raises(DatasetError, match="duplicate item_id"):
            load_evalset(str(p))


def test_malformed_json_names_the_line():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.jsonl"
        p.write_text('{"item_id":"q1","query":"a"}\nNOT JSON\n', encoding="utf-8")
        with pytest.raises(DatasetError, match=":2:"):
            load_evalset(str(p))


def test_missing_query_field_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.jsonl"
        _write_jsonl(p, [{"item_id": "q1", "answer": "a"}])
        with pytest.raises(DatasetError, match="query"):
            load_evalset(str(p))


def test_evalset_round_trips_through_save_and_load():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "e.jsonl"
        original = build_probe_suite(_items(8), ProbeConfig(seed=0)).items
        save_evalset(original, str(p))
        loaded = load_evalset(str(p))
        assert [i.item_id for i in loaded] == [i.item_id for i in original]
        # The canary must survive: without it the scorer looks for nothing.
        inj = [i for i in loaded if i.meta.get("probe") == "injection"]
        assert inj and all(i.meta.get("canary") for i in inj)


def test_dataset_stats_flags_the_things_that_silently_ruin_a_benchmark():
    items = [EvalItem(item_id=f"q{i}", query="the same question",
                      gold_answer="x") for i in range(5)]
    stats = dataset_stats(items)
    warnings = " ".join(stats.warnings)
    assert "too few" in warnings
    assert "duplicate" in warnings
    assert "gold_passage_ids" in warnings
    assert "unanswerable" in warnings


def test_dataset_stats_skips_rag_warnings_for_classification():
    """A validator that cries wolf is one people stop reading."""
    items = [EvalItem(item_id=f"q{i}", query=f"ticket {i}", gold_answer="billing")
             for i in range(40)]
    warnings = " ".join(dataset_stats(items, task="classify").warnings)
    assert "gold_passage_ids" not in warnings
    assert "unanswerable" not in warnings


# --------------------------------------------------------------------------- #
#  Tuning search
# --------------------------------------------------------------------------- #
def test_equal_budget_guarantee_holds():
    """Every model must face the identical candidate set, or 'equal budget' is
    a claim rather than a property."""
    profile = Profile.from_yaml("configs/profiles/regulated_qa.yaml")
    a = enumerate_candidates(profile, budget=20, seed=0)
    b = enumerate_candidates(profile, budget=20, seed=0)
    assert len(a) <= 20
    assert [c.identity() for c in a] == [c.identity() for c in b]


def test_candidates_are_deduplicated():
    """rerank_top_n is irrelevant when rerank is off; sampling both wastes a
    budget slot on a literal duplicate."""
    profile = Profile.from_dict(_profile_dict(
        knobs={"retrieval_modes": ["dense"], "k_values": [5],
               "rerank_options": [False], "rerank_top_n": [3, 5, 8],
               "context_orders": ["as_is"]}))
    cands = enumerate_candidates(profile, budget=20)
    assert len(cands) == 1


def test_rerank_candidates_over_fetch():
    """A reranker can only reorder what it was handed; fetching exactly k leaves
    it nothing to promote, so rerank gain measures as zero by construction."""
    profile = Profile.from_dict(_profile_dict(
        knobs={"retrieval_modes": ["dense"], "k_values": [5],
               "rerank_options": [True], "rerank_top_n": [3]}))
    cand = enumerate_candidates(profile, budget=5)[0]
    assert cand.retrieval.candidate_multiplier > 1


def test_halving_schedule_ends_on_the_full_dev_split():
    schedule = halving_schedule(n_candidates=16, n_dev=64)
    assert schedule[-1][0] == 1
    assert schedule[-1][1] == 64
    assert schedule[0][0] >= schedule[-1][0]


# --------------------------------------------------------------------------- #
#  Provider layer
# --------------------------------------------------------------------------- #
def test_model_ref_splitting_handles_ollama_style_tags():
    """'llama3.1:8b' contains a colon; a naive split routes it to a provider
    called 'llama3.1' and fails with a baffling error."""
    assert split_model_ref("llama3.1:8b") == ("together", "llama3.1:8b")
    assert split_model_ref("ollama:llama3.1:8b") == ("ollama", "llama3.1:8b")
    assert split_model_ref("anthropic:claude-opus-5") == ("anthropic",
                                                          "claude-opus-5")
    assert split_model_ref("openai/gpt-oss-20b") == ("together",
                                                     "openai/gpt-oss-20b")


def test_anthropic_lifts_system_out_of_the_message_list():
    """`system` is a top-level parameter there, not a message role."""
    system, convo = split_system([
        {"role": "system", "content": "be careful"},
        {"role": "user", "content": "hello"},
    ])
    assert system == "be careful"
    assert convo == [{"role": "user", "content": "hello"}]


def test_anthropic_conversation_must_start_with_a_user_turn():
    _, convo = split_system([
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a few-shot answer"},
        {"role": "user", "content": "q"},
    ])
    assert convo[0]["role"] == "user"


def test_sampling_params_dropped_for_models_that_reject_them():
    """The harness sends temperature=0.0 on every call; forwarding it to these
    models is a 400 on every item."""
    assert not supports_sampling("claude-opus-5")
    assert not supports_sampling("claude-sonnet-5")
    assert not supports_sampling("claude-fable-5")
    assert supports_sampling("claude-haiku-4-5")
    assert supports_sampling("claude-sonnet-4-6")


# --------------------------------------------------------------------------- #
#  Pricing
# --------------------------------------------------------------------------- #
def test_unpriced_model_raises_rather_than_costing_zero():
    """A free-looking model would leap to the top of a cost-weighted composite."""
    pricing = PricingRegistry("configs/pricing.yaml")
    with pytest.raises(KeyError):
        pricing.generation_cost("no/such-model", 1000, 100)


def test_non_strict_pricing_degrades_for_estimates():
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    assert pricing.generation_cost("no/such-model", 1000, 100) == 0.0


def test_prefix_matching_covers_a_model_family():
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    exact = pricing.generation_cost("claude-opus-5", 1_000_000, 0)
    snapshot = pricing.generation_cost("claude-opus-5-20260101", 1_000_000, 0)
    assert exact == snapshot > 0


def test_staleness_is_reported():
    import datetime as dt

    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    future = dt.date.fromisoformat(pricing.as_of) + dt.timedelta(days=400)
    assert pricing.staleness_warning(today=future) is not None
    soon = dt.date.fromisoformat(pricing.as_of) + dt.timedelta(days=1)
    assert pricing.staleness_warning(today=soon) is None


def test_missing_reports_unpriced_models_before_a_run():
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    assert pricing.missing(["openai/gpt-oss-20b", "ghost"]) == ["ghost"]


def test_provider_prefixed_model_still_prices():
    """The routing prefix must not blank a model's cost.

    The meter bills with the stripped name (the client sees it post-routing)
    while the trace row prices with the full string. Before the fallback, any
    `provider:`-prefixed model got `cost_usd = None` on every row — so cost
    silently dropped out of the composite, the Pareto frontier and `decide`,
    while the run total still looked correct.
    """
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    bare = pricing.generation_cost("openai/gpt-oss-120b", 1000, 100)
    assert bare > 0
    for prefixed in ("together:openai/gpt-oss-120b",
                     "openrouter:openai/gpt-oss-120b"):
        assert pricing.generation_cost(prefixed, 1000, 100) > 0


def test_provider_qualified_price_beats_the_bare_one():
    """The same model costs different amounts through different routers."""
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    routed = pricing.generation_cost(
        "openrouter:meta-llama/llama-3.3-70b-instruct", 1_000_000, 0)
    direct = pricing.generation_cost(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo", 1_000_000, 0)
    assert routed == pytest.approx(0.10)
    assert direct == pytest.approx(0.88)
    assert routed != direct


def test_ollama_tag_is_not_mistaken_for_a_provider_prefix():
    """`llama3.1:8b` must not be stripped to `8b` when looking up a price."""
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    assert pricing._strip_provider("llama3.1:8b") is None
    assert pricing._strip_provider("ollama:llama3.1:8b") == "llama3.1:8b"


def test_openrouter_can_serve_embeddings():
    """OpenRouter has an OpenAI-shaped /v1/embeddings endpoint, so it can build
    the index as well as run the models — an OpenRouter-only RAG run works.

    Marking it embeddings-incapable would make preflight reject that setup with
    a confident, wrong explanation.
    """
    from harness.clients.openai_compatible import PROVIDER_ENDPOINTS

    assert PROVIDER_ENDPOINTS["openrouter"]["supports_embeddings"] is True
    # It has no rerank endpoint, though, so that must stay declared false.
    assert PROVIDER_ENDPOINTS["openrouter"]["supports_rerank"] is False


def test_openrouter_embedding_models_are_priced():
    """A blank embedding price silently under-reports end-to-end cost."""
    pricing = PricingRegistry("configs/pricing.yaml", strict=False)
    for model in ("baai/bge-m3", "openai/text-embedding-3-small"):
        assert pricing.embedding_cost(model, 1_000_000) > 0
