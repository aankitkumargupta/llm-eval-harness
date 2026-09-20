# in_scheme_qa

Forty-three illustrative welfare-scheme passages and the questions citizens
ask about them. Built by `build_dataset.py`, which is the source of truth;
`corpus.jsonl`, `evalset_base.jsonl` and `evalset.jsonl` are its output.

**Illustrative only, and fictional on purpose.** Every scheme name, amount,
eligibility rule, deadline and office is invented. This is not a
convenience: it is what lets the evaluation measure whether a model answers
from the passages it was given rather than from what it remembers about
real schemes with similar names. Any resemblance to a real scheme is
coincidental, and nothing here may be relied on.

**Corpus.** 43 passages of 40 to 180 words, one scheme each: who is
eligible, how much, which documents, where to apply, by when. Each is under
the profile's chunk size, so each passage is exactly one retrieval chunk
(`<doc_id>#0`) and the gold passage ids are exact.

**Evalset.** 59 answerable questions (33 English, 15 Hindi in
Devanagari, 11 Hinglish), each answerable from one passage, with the gold
answer in the language of the question (Hinglish questions carry an English
gold). Six hand-written unanswerable questions are on-topic but not covered
by any passage; the right answer is to say so. The builder then appends the
harness's derived probes with the profile's settings (unanswerable
variants, noise, injection, paraphrase), so `evalset.jsonl` is a pure
function of the script and no separate `probes --append` step is needed.

**Apparatus.** The profile pins a multilingual sentence embedder served
locally (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`),
because a Hindi question must retrieve an English passage. Dense retrieval
only; lexical matching cannot cross the script boundary.

**Regenerate and index.**

    python data/in_scheme_qa/build_dataset.py
    python main.py ingest --profile configs/profiles/in_scheme_qa.yaml
