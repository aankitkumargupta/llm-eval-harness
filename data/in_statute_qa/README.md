# in_statute_qa

Forty-eight plain-language summaries of the Right to Information framework
and the questions an RTI help desk fields about it. Built by
`build_dataset.py`, which is the source of truth; `corpus.jsonl`,
`evalset_base.jsonl` and `evalset.jsonl` are its output.

**Summaries, not the statute.** The passages summarise the Right to
Information Act, 2005 (Sections 2 to 28), the central Right to Information
Rules, 2012, the 2019 amendment, the 2023 data-protection amendment's
status, and four leading Supreme Court decisions. They were written for
this evaluation and must be verified against the Act before being relied
on. Nothing here is legal advice. Their purpose is to give the model a
corpus to answer from, so the evaluation measures grounding and citation,
not recall.

**Corpus.** 48 passages of 40 to 180 words, each under the profile's chunk
size, so each is exactly one retrieval chunk (`<doc_id>#0`) and gold
passage ids are exact.

**Evalset.** 59 answerable questions (35 English, 13 Hindi in
Devanagari, 11 Hinglish), each answerable from one passage, with the gold
answer in the language of the question (Hinglish questions carry an English
gold). Six hand-written unanswerable questions ask on-topic things the
summaries do not cover (a state's fee, the current Chief Information
Commissioner, pending-appeal counts); the right answer is to say so. The
builder then appends the harness's derived probes with the profile's
settings, so `evalset.jsonl` is a pure function of the script.

**Apparatus.** Hybrid retrieval over the multilingual sentence embedder
served locally, with the sparse (lexical) leg weighted up because section
numbers are lexical tokens; the dense leg carries the Hindi questions.

**Regenerate and index.**

    python data/in_statute_qa/build_dataset.py
    python main.py ingest --profile configs/profiles/in_statute_qa.yaml
