# in_notice_translation

Forty-eight fictional English public notices and office memoranda, each with
a reference Hindi translation. Built by `build_dataset.py`, which is the
source of truth; `evalset.jsonl` is its output.

**Illustrative only.** Every office, place, date, amount, phone number and
person is invented. No real notice is reproduced. The register mirrors what
state and municipal offices publish (water supply interruptions, exam
schedules, tax deadlines, health camps, tenders, transfer orders, hearings,
RTI replies), and the Hindi references are written in the plain official
Hindi such notices use, not literary Hindi.

**Shape.** 30 to 130 English words per notice. Every notice carries at least
one number, date or proper name that a translation must preserve exactly,
because that is what goes wrong in practice: a mistranslated deadline is a
missed deadline.

**Scoring.** A direct profile. The judge scores `accuracy` against the Hindi
reference (many renderings are equally right, so exact match would be
meaningless). `native_script_ratio` is the share of letters in the answer
that are Devanagari; it is a separate metric so that "fluent but in English"
and "in Hindi but wrong" show up as the different failures they are.

**Regenerate.**

    python data/in_notice_translation/build_dataset.py
